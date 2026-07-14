import json
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "matplotlib")
)

import folium
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from folium import Choropleth
from rasterio.features import rasterize
from rasterio.transform import from_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from b3alien import b3cube
from b3alien import griis
from b3alien import simulation


def empty_to_none(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def input_int(data, key, default=None):
    value = empty_to_none(data.get(key))
    if value is None:
        return default
    return int(value)


def input_float(data, key, default=None):
    value = empty_to_none(data.get(key))
    if value is None:
        return default
    return float(value)


def find_column(df, *candidates):
    lookup = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        lowered = candidate.lower()
        if lowered in lookup:
            return lookup[lowered]
    return None


def is_cloud_path(path):
    return str(path).startswith("gs://")


def local_path_error(label, path):
    return (
        f"{label} not found at '{path}'. Local paths are resolved inside the "
        "BIAB runner container, so host paths such as /Users/... are not visible. "
        "Use a gs:// path, or place the file in BIAB userdata and pass a path like "
        "/userdata/my_file.parquet."
    )


def ensure_local_file_visible(label, path):
    if is_cloud_path(path):
        return
    if not Path(path).exists():
        biab_error_stop(local_path_error(label, path))


def checklist_file(path):
    if is_cloud_path(path):
        biab_error_stop(
            "GRIIS checklist paths must currently be local files visible to the runner. "
            "Place merged_distr.txt in BIAB userdata and pass /userdata/merged_distr.txt."
        )
    path = Path(path)
    if path.is_dir():
        path = path / "merged_distr.txt"
    if not path.exists():
        biab_error_stop(local_path_error("GRIIS checklist", path))
    return str(path)


def dataframe_with_optional_taxon_filter(cube, taxon_rank, taxon_name):
    df = cube.df.copy()
    taxon_name = empty_to_none(taxon_name)
    if taxon_name is None:
        return df, "all taxa"

    rank_col = find_column(df, taxon_rank)
    if rank_col is None:
        biab_error_stop(
            f"Taxon filter requested, but the cube does not contain a '{taxon_rank}' column."
        )

    filtered = df[df[rank_col].astype(str).str.casefold() == str(taxon_name).casefold()].copy()
    if filtered.empty:
        biab_error_stop(
            f"No cube rows matched the taxon filter {taxon_rank}={taxon_name}."
        )
    return filtered, f"{taxon_rank}={taxon_name}"


def dataframe_within_year_window(df, start_year, end_year):
    time_col = find_column(df, "yearmonth", "time", "eventdate")
    if time_col is None:
        biab_error_stop(
            "The cube must contain a yearmonth or time column to restrict observation effort to the analysis window."
        )

    year = pd.to_datetime(df[time_col], format="%Y-%m", errors="coerce").dt.year
    windowed = df[(year >= start_year) & (year <= end_year)].copy()
    if windowed.empty:
        biab_error_stop(
            f"No cube rows fall within the analysis window {start_year}-{end_year} for observation effort."
        )
    return windowed


def build_spatial_effort(df, measure, map_scale):
    cell_col = find_column(df, "cellcode", "cellCode", "cell")
    geom_col = find_column(df, "geometry")
    occ_col = find_column(df, "occurrences", "occurrencecount", "occurrence_count")
    observer_col = find_column(df, "distinctobservers", "distinct_observers")

    if cell_col is None:
        biab_error_stop("The cube must contain a cell code column to map observation effort.")
    if geom_col is None:
        biab_error_stop("The cube must contain a geometry column to map observation effort.")

    metric_col = occ_col
    metric_label = "Total occurrences"
    if measure == "distinct_observers":
        if observer_col is None:
            biab_warning(
                "The cube has no distinct observers column; mapping total occurrences instead."
            )
        else:
            metric_col = observer_col
            metric_label = "Distinct observers"

    if metric_col is None:
        biab_error_stop(
            "The cube must contain either occurrences or distinct observers to estimate observation effort."
        )

    effort = (
        df.groupby(cell_col, observed=True)
        .agg(observation_effort=(metric_col, "sum"), geometry=(geom_col, "first"))
        .reset_index()
    )

    gdf = gpd.GeoDataFrame(effort, geometry="geometry", crs=getattr(df, "crs", None))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    if map_scale == "log10":
        gdf["map_value"] = np.log10(gdf["observation_effort"].astype(float) + 1)
        legend_label = f"{metric_label} (log10 value + 1)"
    else:
        gdf["map_value"] = gdf["observation_effort"].astype(float)
        legend_label = metric_label

    return gdf, cell_col, metric_label, legend_label


def build_yearly_effort(df, measure):
    time_col = find_column(df, "yearmonth", "time", "eventdate")
    occ_col = find_column(df, "occurrences", "occurrencecount", "occurrence_count")
    observer_col = find_column(df, "distinctobservers", "distinct_observers")

    if time_col is None:
        biab_error_stop("The cube must contain a yearmonth or time column to plot survey effort through time.")

    metric_col = occ_col
    output_col = "total_occurrences"
    if measure == "distinct_observers" and observer_col is not None:
        metric_col = observer_col
        output_col = "distinct_observers"

    if metric_col is None:
        biab_error_stop("The cube must contain an occurrences column to summarize survey effort through time.")

    yearly = df[[time_col, metric_col]].copy()
    yearly["date"] = pd.to_datetime(yearly[time_col], format="%Y-%m", errors="coerce")
    yearly = yearly.dropna(subset=["date"])
    yearly["year"] = yearly["date"].dt.year.astype(int)
    yearly = (
        yearly.groupby("year", observed=True)[metric_col]
        .sum()
        .reset_index()
        .rename(columns={metric_col: output_col})
    )
    return yearly


def save_interactive_map(gdf, cell_col, legend_label, simplify_tolerance, output_dir):
    gdf_4326 = gdf.to_crs("EPSG:4326").copy()
    if simplify_tolerance and simplify_tolerance > 0:
        gdf_4326["geometry"] = gdf_4326.geometry.simplify(
            simplify_tolerance, preserve_topology=True
        )

    bounds = gdf_4326.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    values = gdf_4326["map_value"].astype(float)
    min_value = float(np.nanmin(values)) if values.notna().any() else 0.0
    max_value = float(np.nanmax(values)) if values.notna().any() else 1.0
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    colormap = folium.LinearColormap(
        colors=["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"],
        vmin=min_value,
        vmax=max_value,
        caption=legend_label,
    )

    def fill_color(value):
        if value is None or not np.isfinite(value):
            return "#f7f7f7"
        return colormap(value)

    gdf_4326["fill_color"] = gdf_4326["map_value"].astype(float).apply(fill_color)

    # simplestyle-spec properties so the exported GeoJSON is self-styled: the
    # BIAB embedded map viewer renders this file directly, not the Folium HTML,
    # so per-feature color must travel as feature properties, not JS callbacks.
    gdf_4326["fill"] = gdf_4326["fill_color"]
    gdf_4326["fill-opacity"] = np.where(gdf_4326["fill_color"] == "#f7f7f7", 0.6, 0.8)
    gdf_4326["stroke"] = np.where(gdf_4326["fill_color"] == "#f7f7f7", "#bdbdbd", "#444444")
    gdf_4326["stroke-width"] = 0.5
    gdf_4326["stroke-opacity"] = 1.0

    def style_function(feature):
        fill = feature.get("properties", {}).get("fill_color", "#f7f7f7")
        return {
            "fillColor": fill,
            "color": "#bdbdbd" if fill == "#f7f7f7" else "#444444",
            "weight": 0.5,
            "fillOpacity": 0.6 if fill == "#f7f7f7" else 0.8,
        }

    folium.GeoJson(
        gdf_4326[[cell_col, "observation_effort", "map_value", "fill_color", "geometry"]],
        name="Observation effort",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[cell_col, "observation_effort", "map_value"],
            aliases=["Cell", "Observation effort", "Map value"],
            localize=True,
        ),
    ).add_to(fmap)

    colormap.add_to(fmap)
    folium.LayerControl().add_to(fmap)

    html_path = output_dir / "observation_effort_map.html"
    fmap.save(html_path)

    geojson_path = output_dir / "observation_effort_map.geojson"
    relevant_columns = [
        cell_col,
        "observation_effort",
        "map_value",
        "fill",
        "fill-opacity",
        "stroke",
        "stroke-width",
        "stroke-opacity",
        "geometry",
    ]
    gdf_4326[relevant_columns].to_file(geojson_path, driver="GeoJSON")

    return html_path, geojson_path


def save_raster_map(gdf, output_dir):
    gdf_4326 = gdf.to_crs("EPSG:4326")
    bounds_df = gdf_4326.geometry.bounds
    cell_width = (bounds_df["maxx"] - bounds_df["minx"]).median()
    cell_height = (bounds_df["maxy"] - bounds_df["miny"]).median()

    minx, miny, maxx, maxy = gdf_4326.total_bounds
    width = max(1, int(round((maxx - minx) / cell_width)))
    height = max(1, int(round((maxy - miny) / cell_height)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    shapes = list(zip(gdf_4326.geometry, gdf_4326["map_value"].astype("float32")))
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype="float32",
    )

    raster_path = output_dir / "observation_effort_map.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="COG",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
        compress="DEFLATE",
    ) as dst:
        dst.write(raster, 1)

    return raster_path


def save_rate_outputs(time, rate, bootstrap_iterations, output_dir):
    if len(time) == 0 or len(rate) == 0:
        biab_error_stop(
            "No annual establishment rate could be calculated. The cube/checklist overlap needs observations in at least two years."
        )

    time_series = pd.Series(time, dtype="int64")
    rate_series = pd.Series(rate, dtype="float64")
    c1, vec1 = simulation.simulate_solow_costello_scipy(
        time_series, rate_series, vis=False
    )

    rate_df = pd.DataFrame(
        {
            "year": time_series,
            "annual_establishment_rate": rate_series,
            "solow_costello_estimated_rate": np.asarray(c1, dtype=float),
            "cumulative_observed_establishments": np.cumsum(rate_series),
            "cumulative_model_establishments": np.cumsum(c1),
        }
    )

    bootstrap_summary = None
    if bootstrap_iterations and bootstrap_iterations > 0:
        results = simulation.parallel_bootstrap_solow_costello(
            time_series, rate_series, n_iterations=bootstrap_iterations
        )
        rate_df["bootstrap_cumulative_mean"] = results["c1_mean"]
        rate_df["bootstrap_cumulative_lower_95"] = results["c1_lower"]
        rate_df["bootstrap_cumulative_upper_95"] = results["c1_upper"]
        bootstrap_summary = {
            "beta1_ci_lower": float(results["beta1_ci"][0]),
            "beta1_ci_upper": float(results["beta1_ci"][1]),
            "iterations": int(bootstrap_iterations),
        }

    csv_path = output_dir / "rate_of_establishment.csv"
    rate_df.to_csv(csv_path, index=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        rate_df["year"],
        rate_df["annual_establishment_rate"],
        marker="o",
        color="#2b6cb0",
        label="Observed annual rate",
    )
    ax1.plot(
        rate_df["year"],
        rate_df["solow_costello_estimated_rate"],
        linestyle="--",
        color="#1a202c",
        label="Solow-Costello estimated rate",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual establishments")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(
        rate_df["year"],
        rate_df["cumulative_observed_establishments"],
        color="#c53030",
        label="Cumulative observed",
    )
    if "bootstrap_cumulative_mean" in rate_df.columns:
        ax2.plot(
            rate_df["year"],
            rate_df["bootstrap_cumulative_mean"],
            color="#6b46c1",
            linestyle=":",
            label="Bootstrap cumulative mean",
        )
        ax2.fill_between(
            rate_df["year"],
            rate_df["bootstrap_cumulative_lower_95"],
            rate_df["bootstrap_cumulative_upper_95"],
            color="#6b46c1",
            alpha=0.18,
            label="Bootstrap 95% interval",
        )
    ax2.set_ylabel("Cumulative establishments")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Rate of establishment")
    fig.tight_layout()
    plot_path = output_dir / "rate_of_establishment.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return csv_path, plot_path, vec1, bootstrap_summary, rate_df


data = biab_inputs()
output_dir = Path(output_folder)
output_dir.mkdir(parents=True, exist_ok=True)

cube_path = empty_to_none(data.get("cube_path"))
checklist_path = empty_to_none(data.get("checklist_path"))
google_project = empty_to_none(data.get("google_project")) or ""
start_year = input_int(data, "start_year", 1970)
end_year = input_int(data, "end_year", 2022)
measure = empty_to_none(data.get("survey_effort_measure")) or "total_occurrences"
taxon_rank = empty_to_none(data.get("taxon_rank")) or "kingdom"
taxon_name = empty_to_none(data.get("taxon_name"))
map_scale = empty_to_none(data.get("map_scale")) or "log10"
simplify_tolerance = input_float(data, "map_simplification_tolerance", 0.01)
bootstrap_iterations = input_int(data, "bootstrap_iterations", 0)

if cube_path is None:
    biab_error_stop("Please provide a GeoParquet biodiversity data cube path.")
if checklist_path is None:
    biab_error_stop("Please provide a GRIIS checklist merged_distr.txt path.")
if start_year > end_year:
    biab_error_stop("The start year must be earlier than or equal to the end year.")

ensure_local_file_visible("Biodiversity data cube", cube_path)
checklist_path = checklist_file(checklist_path)

print(f"Loading occurrence cube from {cube_path}", flush=True)
try:
    cube = b3cube.OccurrenceCube(cube_path, gproject=google_project)
except Exception as exc:
    biab_error_stop(f"Failed to load biodiversity data cube '{cube_path}': {exc}")

try:
    checklist = griis.CheckList(checklist_path)
except Exception as exc:
    biab_error_stop(f"Failed to load GRIIS checklist '{checklist_path}': {exc}")

print(f"Loaded {len(checklist.species)} checklist species.", flush=True)
print("Calculating cumulative alien species with b3alien.", flush=True)
df_sparse, df_cumulative = b3cube.cumulative_species(cube, checklist.species)
if df_cumulative.empty:
    biab_error_stop(
        "No matching alien species were found between the checklist and the occurrence cube."
    )

cumulative_path = output_dir / "cumulative_alien_species.csv"
df_cumulative.to_csv(cumulative_path, index=False)
biab_output("cumulative_species", str(cumulative_path))

print("Calculating annual establishment rate with b3alien.", flush=True)
annual_time, annual_rate = b3cube.calculate_rate(df_cumulative.copy())
rate_filter = pd.DataFrame({"year": annual_time, "rate": annual_rate})
filtered_time, filtered_rate = b3cube.filter_time_window(
    rate_filter, start_year, end_year, cols=["year", "rate"]
)

rate_csv, rate_plot, vec1, bootstrap_summary, rate_df = save_rate_outputs(
    list(filtered_time), list(filtered_rate), bootstrap_iterations, output_dir
)
biab_output("rate_table", str(rate_csv))
biab_output("rate_plot", str(rate_plot))

filtered_df, taxon_filter_label = dataframe_with_optional_taxon_filter(
    cube, taxon_rank, taxon_name
)
filtered_df = dataframe_within_year_window(filtered_df, start_year, end_year)
spatial_effort, cell_col, metric_label, legend_label = build_spatial_effort(
    filtered_df, measure, map_scale
)
map_html, map_geojson = save_interactive_map(
    spatial_effort, cell_col, legend_label, simplify_tolerance, output_dir
)
# observation_effort_map (geojson) output disabled: BIAB's embedded geo+json
# viewer doesn't apply per-feature color styling. Kept generating the file
# below in case it's reactivated later; just not registered as a pipeline output.
biab_output("observation_effort_map_html", str(map_html))

map_raster = save_raster_map(spatial_effort, output_dir)
biab_output("observation_effort_raster", str(map_raster))

yearly_effort = build_yearly_effort(filtered_df, measure)
effort_timeseries_path = output_dir / "observation_effort_timeseries.csv"
yearly_effort.to_csv(effort_timeseries_path, index=False)
# observation_effort_timeseries output disabled for now; file still generated above.

summary = {
    "cube_path": cube_path,
    "checklist_path": checklist_path,
    "analysis_window": {"start_year": start_year, "end_year": end_year},
    "checklist_species": int(len(checklist.species)),
    "matched_records": int(len(df_sparse)),
    "cumulative_species_final": int(df_cumulative["cumulative_species"].max()),
    "rate_years": int(len(rate_df)),
    "solow_costello_change_in_rate_per_year": float(vec1[1])
    if len(vec1) > 1 and math.isfinite(float(vec1[1]))
    else None,
    "bootstrap": bootstrap_summary,
    "observation_effort_measure": measure,
    "observation_effort_taxon_filter": taxon_filter_label,
    "observation_effort_metric": metric_label,
}
summary_path = output_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
biab_output("summary", str(summary_path))
