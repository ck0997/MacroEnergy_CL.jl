#!/usr/bin/env python3
"""Standalone helpers for CO2 capture, transport, and storage plots."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASIN_COORDS = {
    "Songliao": (124.2489, 45.5424),
    "TurpanHami": (91.8474, 42.7741),
    "Subei": (120.2968, 33.2802),
    "BohaiOnshore": (117.027, 38.1531),
    "Qaidam": (94.0675, 37.4565),
    "Nanxiang": (112.3891, 32.6284),
    "Sanjiang": (132.0, 47.0),
    "Hailar": (118.5403, 48.9011),
    "Jianghan": (112.9206, 30.4742),
    "Tarim": (82.2819, 39.3991),
    "Ordos": (108.7178, 37.5548),
    "YingenEjina": (101.5, 41.5),
    "Hehuai": (115.0, 34.0),
    "Qinshui": (112.4531, 36.7143),
    "Erlian": (115.3402, 44.1892),
    "Junggar": (87.0998, 44.9779),
    "SichuanBasin": (105.9742, 30.357),
    "BohaiOffshore": (119.7405, 39.0081),
    "NorthYellowSea": (123.3029, 38.3492),
    "SouthYellowSea": (123.0614, 35.6424),
    "EastChinaSea": (124.815, 29.2425),
    "PearlRiverMouth": (114.5827, 20.6148),
    "BeibuGulf": (109.1529, 20.3578),
    "BeibugGulf": (109.1529, 20.3578),
    "Qiongdongnan": (110.6, 17.9),
}

PROVINCE_RENAME = {
    "Nei Mongol": "Innermongolia",
    "Xizang": "Tibet",
    "Ningxia Hui": "Ningxia",
    "Xinjiang Uygur": "Xinjiang",
}

CO2_SECTOR_COLORS = {
    "Cement": "#e15759",
    "Steel": "#4e79a7",
    "Aluminum": "#bab0ac",
    "Power": "#f28e2b",
    "Other": "#9c755f",
}

CO2_STORAGE_COLOR = "#59a14f"
CO2_PIPELINE_COLOR = "#4a5568"
CO2_FLOW_LEGEND_VALUES = (0.25, 0.50, 1.00)
CO2_CAPTURE_LEGEND_VALUES = (1.00, 0.50, 0.25)


def import_plotting_deps():
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Wedge
        from pyproj import Transformer
    except ImportError as exc:
        raise SystemExit(
            "Missing plotting dependency. Use an environment with geopandas, "
            "matplotlib, and pyproj."
        ) from exc
    return gpd, plt, Line2D, Circle, Wedge, Transformer


def format_map_quantity(value: float) -> str:
    value = float(value)
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def legend_sizes(max_value: float | None, fractions=(0.25, 0.5, 1.0)):
    if max_value is None or max_value <= 0:
        return None
    return [max_value * f for f in fractions]


def region_label(node: str) -> str:
    match = re.match(r"^Region\d+([A-Za-z]+)$", str(node))
    return match.group(1) if match else str(node)


def sector_from_tech(tech: str) -> str:
    tech = str(tech).lower()
    if "cement" in tech:
        return "Cement"
    if any(k in tech for k in ["bfbof", "scrap", "dri", "eaf", "steel"]):
        return "Steel"
    if "aluminum" in tech:
        return "Aluminum"
    if any(k in tech for k in ["coal", "naturalgas", "nuclear", "power"]):
        return "Power"
    return "Other"


def resolve_path(path: str | Path, script_dir: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    candidate = script_dir / path
    return candidate if candidate.exists() else path


def load_inputs(results_dir: Path):
    flows_path = results_dir / "flows.csv"
    capacity_path = results_dir / "capacity.csv"
    if not flows_path.exists():
        raise FileNotFoundError(f"Missing flows.csv: {flows_path}")
    if not capacity_path.exists():
        raise FileNotFoundError(f"Missing capacity.csv: {capacity_path}")
    return pd.read_csv(flows_path), pd.read_csv(capacity_path)


def load_china_gdf(gpd, map_path: Path):
    map_path = Path(map_path)
    if not map_path.exists() and map_path.name == "gadm36_CHN_1.json":
        fallback = (
            Path(__file__).resolve().parent.parent
            / "improved_co2_pipelines"
            / "chinny_co2_pipeline_distance"
            / "gadm36_CHN_1.json"
        )
        if fallback.exists():
            map_path = fallback
    gdf = gpd.read_file(map_path)
    gdf["NAME_1"] = gdf["NAME_1"].replace(PROVINCE_RENAME)
    return gdf


def get_co2_captured_by_sector(flows_df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"^Region\d+([A-Za-z]+)_(.+)_co2_captured_edge$")
    rows = []
    for col in flows_df.columns:
        if "CO2_Injection" in col:
            continue
        match = pattern.match(col)
        if not match:
            continue
        province, tech = match.groups()
        value = flows_df[col].sum()
        if value > 0:
            rows.append(
                {"Province": province, "Sector": sector_from_tech(tech), "CO2 Captured": value}
            )
    if not rows:
        return pd.DataFrame()
    out = (
        pd.DataFrame(rows)
        .groupby(["Province", "Sector"], as_index=False)["CO2 Captured"]
        .sum()
        .pivot(index="Province", columns="Sector", values="CO2 Captured")
        .fillna(0)
    )
    out.columns.name = None
    return out.reindex(columns=[c for c in CO2_SECTOR_COLORS if c in out.columns])


def get_co2_stored_by_basin(flows_df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"^(.+)_to_(.+)_CO2_Injection_co2_storage_edge$")
    rows = []
    for col in flows_df.columns:
        match = pattern.match(col)
        if not match:
            continue
        source, sink = match.groups()
        value = flows_df[col].sum()
        if value > 0:
            rows.append(
                {"Source": region_label(source), "Storage Site": region_label(sink), "CO2 Stored": value}
            )
    if not rows:
        return pd.DataFrame(columns=["Storage Site", "CO2 Stored"])
    return (
        pd.DataFrame(rows)
        .groupby("Storage Site", as_index=False)["CO2 Stored"]
        .sum()
        .sort_values("CO2 Stored", ascending=False)
    )


def get_co2_pipeline_flows(flows_df: pd.DataFrame, flow_threshold: float) -> dict:
    pattern = re.compile(r"^(.+)_to_(.+)_CO2_Pipeline_transmission_edge$")
    flows = {}
    for col in flows_df.columns:
        match = pattern.match(col)
        if not match:
            continue
        src, dst = (region_label(value) for value in match.groups())
        value = flows_df[col].sum()
        if value > flow_threshold:
            flows[(src, dst)] = flows.get((src, dst), 0.0) + value
    return flows


def get_co2_pipeline_capacity(capacity_df: pd.DataFrame) -> dict:
    pattern = re.compile(r"^(.+)_to_(.+)_CO2_Pipeline_transmission_edge$")
    capacities = {}
    if "component_id" not in capacity_df or "capacity" not in capacity_df:
        return capacities
    for _, row in capacity_df.iterrows():
        match = pattern.match(str(row["component_id"]))
        if not match:
            continue
        value = float(row.get("capacity", 0) or 0)
        if value > 0:
            src, dst = (region_label(value) for value in match.groups())
            capacities[(src, dst)] = capacities.get((src, dst), 0.0) + value
    return capacities


def co2_node_xy(gdf_proj, basin_xy, province_col="NAME_1") -> dict:
    gdf_cent = gdf_proj.copy()
    gdf_cent["centroid"] = gdf_cent.geometry.centroid
    province_xy = dict(zip(gdf_cent[province_col], zip(gdf_cent.centroid.x, gdf_cent.centroid.y)))
    xy = dict(basin_xy)
    xy.update(province_xy)
    return xy


def normalize_route_label(name: str) -> str:
    name = str(name).strip()
    replacements = {
        "Inner Mongolia": "Innermongolia",
        "Bohai Bay Basin (offshore)": "BohaiOffshore",
        "Bohai Bay Basin (onshore)": "BohaiOnshore",
        "Beibu Gulf Basin": "BeibuGulf",
        "Sichuan Basin": "SichuanBasin",
        "Turpan-Hami Basin": "TurpanHami",
        "Yingen-Ejina Basin": "YingenEjina",
    }
    if name in replacements:
        return replacements[name]
    if name.endswith(" Basin"):
        name = name.removesuffix(" Basin")
    return name.replace(" ", "").replace("-", "")


def parse_coordinate_list(value):
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [float(item) for item in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [float(item) for item in parsed]


def load_pipeline_route_paths(route_path: Path | None, xy: dict, transformer=None) -> dict:
    if not route_path or not Path(route_path).exists():
        return {}
    routes_df = pd.read_csv(route_path)
    start_col = "start" if "start" in routes_df else "Origin_Province"
    end_col = "end" if "end" in routes_df else "Destination_Province"
    if start_col not in routes_df or end_col not in routes_df:
        return {}

    routes = {}
    has_geometry = {"path_lon", "path_lat"}.issubset(routes_df.columns) and transformer is not None
    for _, row in routes_df.iterrows():
        src = normalize_route_label(row[start_col])
        dst = normalize_route_label(row[end_col])
        if has_geometry:
            lons = parse_coordinate_list(row["path_lon"])
            lats = parse_coordinate_list(row["path_lat"])
            if len(lons) == len(lats) and len(lons) >= 2:
                xs, ys = transformer.transform(lons, lats)
                routes[(src, dst)] = list(zip(xs, ys))
                continue
        if src in xy and dst in xy:
            routes[(src, dst)] = [xy[src], xy[dst]]
    return routes


def plot_province_pies(
    gdf,
    data_df: pd.DataFrame,
    plt,
    Circle,
    Wedge,
    output_path: Path,
    province_col="NAME_1",
    categories=None,
    colors=None,
    pie_scale=2.8,
    figsize=(14, 10),
    title="",
    size_legend=None,
    unit_label="Tonnes",
    show=False,
):
    if data_df.empty:
        print("No CO2 capture data found; skipping capture pie map.")
        return
    categories = list(data_df.columns) if categories is None else categories
    colors = {cat: plt.cm.tab20(i) for i, cat in enumerate(categories)} if colors is None else colors
    gdf_proj = gdf.to_crs(epsg=3857)
    gdf_proj = gdf_proj.merge(data_df, left_on=province_col, right_index=True, how="left")
    gdf_proj[categories] = gdf_proj[categories].fillna(0)
    gdf_proj["centroid"] = gdf_proj.geometry.centroid
    gdf_proj["cx"] = gdf_proj.centroid.x
    gdf_proj["cy"] = gdf_proj.centroid.y
    gdf_proj["total_value"] = gdf_proj[categories].sum(axis=1)
    max_total = gdf_proj["total_value"].max() or 1
    gdf_proj["radius"] = (gdf_proj["total_value"] / max_total) ** 0.5 * pie_scale * 80_000

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    gdf_proj.plot(ax=ax, color="#F4F4F4", edgecolor="black", linewidth=0.7, zorder=0)
    gdf_proj.boundary.plot(ax=ax, linewidth=0.7, color="black", zorder=1)
    for _, row in gdf_proj.iterrows():
        values = row[categories].values.astype(float)
        total = values.sum()
        if total == 0:
            continue
        angles = np.cumsum(values) / total * 360
        previous_angle = 0
        for category, angle in zip(categories, angles):
            ax.add_patch(
                Wedge(
                    center=(row["cx"], row["cy"]),
                    r=row["radius"],
                    theta1=previous_angle,
                    theta2=angle,
                    facecolor=colors[category],
                    edgecolor="none",
                    zorder=1000,
                )
            )
            previous_angle = angle
    ax.legend(
        handles=[Wedge((0, 0), 1, 0, 360, facecolor=colors[cat], label=cat) for cat in categories],
        loc="upper right",
        frameon=True,
        title="Capture sector",
        fontsize=8,
    )
    if size_legend:
        _draw_circle_legend(ax, plt, Circle, size_legend, max_total, pie_scale, unit_label, "CO2 Captured")
    ax.set_title(title, fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def _draw_circle_legend(ax, plt, Circle, sizes, max_total, scale, unit_label, title):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plot_width = xlim[1] - xlim[0]
    plot_height = ylim[1] - ylim[0]

    pad_x = plot_width * 0.02
    pad_y = plot_height * 0.02

    sizes_and_radii = [
        (size, (size / max_total) ** 0.5 * scale * 80_000)
        for size in sizes
        if size > 0
    ]

    if not sizes_and_radii:
        return

    sizes_and_radii = sorted(sizes_and_radii)
    max_radius = max(radius for _, radius in sizes_and_radii)

    legend_cx = xlim[0] + pad_x + max_radius
    y_start = ylim[0] + pad_y

    # Compute where the stacked legend ends so the title can sit above it.
    legend_height = sum(2 * radius for _, radius in sizes_and_radii)
    legend_spacing = pad_y * 0.4 * (len(sizes_and_radii) - 1)
    title_y = y_start + legend_height + legend_spacing + pad_y * 0.9

    ax.text(
        legend_cx - max_radius,
        title_y,
        title,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    y = y_start
    for size, radius in sizes_and_radii:
        cy = y + radius

        ax.add_patch(
            Circle(
                (legend_cx, cy),
                radius,
                facecolor="white",
                edgecolor="#555555",
                linewidth=1.2,
                zorder=20,
            )
        )

        label = format_map_quantity(size)
        if unit_label:
            label = f"{label} {unit_label}"

        ax.text(
            legend_cx + max_radius + pad_x * 0.6,
            cy,
            label,
            va="center",
            ha="left",
            fontsize=9,
            zorder=21,
        )

        y += radius * 2 + pad_y * 0.4


def _auto_legend_values(max_value, n=3):
    if max_value <= 0:
        return []
    return [max_value * frac for frac in (0.25, 0.50, 1.00)]


def _format_capacity_label(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    if value < 0.1:
        return f"{value:.3f}"
    return f"{value:g}"


def _draw_linewidth_legend(
    ax,
    Line2D,
    legend_values,
    linewidth_fn,
    color,
    title="Built capacity",
    loc="lower left",
):
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=linewidth_fn(value),
            alpha=0.75,
            solid_capstyle="round",
            label=_format_capacity_label(value),
        )
        for value in legend_values
        if value > 0
    ]

    if handles:
        ax.legend(
            handles=handles,
            title=title,
            loc=loc,
            frameon=True,
            facecolor="white",
            edgecolor="#dddddd",
            fontsize=8,
            title_fontsize=9,
        )


def _draw_arrow_legend(ax, legend_values, shaft_width_fn, color, title="CO2 flow"):
    if not legend_values:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plot_width = xlim[1] - xlim[0]
    plot_height = ylim[1] - ylim[0]
    x0 = xlim[0] + plot_width * 0.06
    y0 = ylim[0] + plot_height * 0.10
    arrow_len = plot_width * 0.12
    y_step = plot_height * 0.055

    ax.text(x0, y0 + y_step * (len(legend_values) + 0.25), title, fontsize=9, fontweight="bold")
    for i, value in enumerate(legend_values):
        y = y0 + (len(legend_values) - 1 - i) * y_step
        width = shaft_width_fn(value)
        ax.arrow(
            x0,
            y,
            arrow_len,
            0,
            width=width,
            head_width=width * 2.5,
            head_length=arrow_len * 0.20,
            length_includes_head=True,
            fc=color,
            ec="white",
            linewidth=0.4,
            alpha=0.75,
            zorder=20,
        )
        ax.text(x0 + arrow_len + plot_width * 0.02, y, format_map_quantity(value), va="center", fontsize=9)


def plot_co2_pipeline_flows(
    flows: dict,
    capacities: dict,
    gdf,
    plt,
    Line2D,
    Transformer,
    output_path: Path,
    title="CO2 Pipeline Flows",
    color=CO2_PIPELINE_COLOR,
    figsize=(13, 11),
    max_shaft_width=80_000,
    min_shaft_width=8_000,
    head_width_ratio=2.5,
    flow_legend_values=CO2_FLOW_LEGEND_VALUES,
    show_capacity_routes=False,
    show=False,
):
    if not flows:
        print("No CO2 transport flows found above threshold; skipping pipeline map.")
        return
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    basin_xy = {name: transformer.transform(lon, lat) for name, (lon, lat) in BASIN_COORDS.items()}
    gdf_proj = gdf.copy().to_crs(epsg=3857)
    xy = co2_node_xy(gdf_proj, basin_xy)
    max_flow = max(flows.values())

    def shaft_width(flow):
        frac = min(float(flow) / max_flow, 1.0) ** 0.5
        return min_shaft_width + frac * (max_shaft_width - min_shaft_width)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    gdf_proj.plot(ax=ax, color="#F4F4F4", edgecolor="#888888", linewidth=0.6, zorder=0)
    if show_capacity_routes and capacities:
        max_capacity = max(capacities.values()) or 1
        for (src, dst), capacity in sorted(capacities.items(), key=lambda item: item[1]):
            if src in xy and dst in xy and capacity > 0:
                x0, y0 = xy[src]
                x1, y1 = xy[dst]
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color="#9ca3af",
                    linewidth=0.5 + 4.5 * (capacity / max_capacity) ** 0.5,
                    alpha=0.40,
                    zorder=2,
                )
    for (src, dst), flow in sorted(flows.items(), key=lambda item: item[1]):
        if src not in xy or dst not in xy:
            continue
        x0, y0 = xy[src]
        x1, y1 = xy[dst]
        dx, dy = x1 - x0, y1 - y0
        distance = np.hypot(dx, dy)
        width = shaft_width(flow)
        head_width = width * head_width_ratio
        ax.arrow(
            x0,
            y0,
            dx,
            dy,
            width=width,
            head_width=head_width,
            head_length=min(distance * 0.30, head_width * 1.4),
            length_includes_head=True,
            fc=color,
            ec="white",
            linewidth=0.4,
            alpha=0.30 + 0.70 * (flow / max_flow),
            zorder=5,
        )
    for name in sorted({dst for _, dst in flows if dst in basin_xy}):
        bx, by = xy[name]
        ax.plot(bx, by, "v", color="#e15759", markersize=6, zorder=10, clip_on=False)
        ax.text(bx, by + 50_000, name, fontsize=6, ha="center", va="bottom", clip_on=False)
    if show_capacity_routes and capacities:
        ax.add_line(Line2D([], [], color="#9ca3af", linewidth=3, alpha=0.55, label="Built capacity route"))
    _draw_arrow_legend(ax, flow_legend_values, shaft_width, color)
    
    ax.set_title(title, fontsize=16, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_co2_pipeline_capacity(
    capacities: dict,
    gdf,
    plt,
    Line2D,
    Transformer,
    output_path: Path,
    route_path: Path | None = None,
    title="Built CO2 Pipeline Capacity",
    color="#e15759",
    base_route_color="#e15759",
    figsize=(13, 11),
    min_linewidth=0.8,
    max_linewidth=7.0,
    capacity_legend_values=None,
    capacity_threshold=0.0,
    show_delaunay_network=True,
    show=False,
):
    positive_capacities = {key: value for key, value in capacities.items() if value > capacity_threshold}
    if not positive_capacities:
        print("No built CO2 pipeline capacity found; skipping capacity map.")
        return

    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    basin_xy = {name: transformer.transform(lon, lat) for name, (lon, lat) in BASIN_COORDS.items()}
    gdf_proj = gdf.copy().to_crs(epsg=3857)
    xy = co2_node_xy(gdf_proj, basin_xy)
    route_paths = load_pipeline_route_paths(route_path, xy, transformer)

    max_capacity = max(positive_capacities.values()) or 1

    def linewidth(capacity):
        frac = min(float(capacity) / max_capacity, 1.0) ** 0.5
        return min_linewidth + frac * (max_linewidth - min_linewidth)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    gdf_proj.plot(ax=ax, color="#F4F4F4", edgecolor="#888888", linewidth=0.6, zorder=0)

    if show_delaunay_network and route_paths:
        for (src, dst), coords in route_paths.items():
            capacity = positive_capacities.get((src, dst), positive_capacities.get((dst, src), 0.0))
            if capacity <= capacity_threshold:
                continue
            if not coords:
                continue
            xs, ys = zip(*coords)
            ax.plot(
                xs,
                ys,
                color=base_route_color,
                linewidth=0.8,
                alpha=0.28,
                solid_capstyle="round",
                zorder=2,
            )

    missing_routes = 0

    for (src, dst), capacity in sorted(positive_capacities.items(), key=lambda item: item[1]):
        coords = route_paths.get((src, dst))

        if coords is None and (dst, src) in route_paths:
            coords = list(reversed(route_paths[(dst, src)]))

        if coords is None and src in xy and dst in xy:
            coords = [xy[src], xy[dst]]
            missing_routes += 1

        if not coords:
            continue

        xs, ys = zip(*coords)

        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=linewidth(capacity),
            alpha=0.28 + 0.62 * (capacity / max_capacity),
            solid_capstyle="round",
            zorder=5,
        )

    if capacity_legend_values is None:
        capacity_legend_values = _auto_legend_values(max_capacity)

    _draw_linewidth_legend(
        ax=ax,
        Line2D=Line2D,
        legend_values=capacity_legend_values,
        linewidth_fn=linewidth,
        color=color,
        title="Built capacity",
        loc="lower left",
    )

    ax.set_title(title, fontsize=16, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")

    if route_path and Path(route_path).exists():
        print(f"Used pipeline route table: {route_path}")

    if missing_routes:
        print(f"Used straight endpoint lines for {missing_routes} capacity routes missing from the route table.")

    print(f"Saved {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_delaunay_lcp_capacity_network(
    capacities: dict,
    gdf,
    route_path: Path,
    title="Delaunay LCP CO2 Pipeline Capacity Network",
    output_html: Path | None = None,
    color="#e15759",
    min_width=1.0,
    max_width=8.0,
    capacity_threshold=0.0,
    show=True,
):
    import plotly.graph_objects as go
    import plotly.colors as plotly_colors

    routes_df = pd.read_csv(route_path)
    if not {"start", "end", "path_lon", "path_lat"}.issubset(routes_df.columns):
        raise ValueError("route_path must include start, end, path_lon, and path_lat columns.")
    if not capacities:
        print("No built CO2 pipeline capacity found; skipping Plotly LCP capacity network.")
        return None

    positive_capacities = {key: value for key, value in capacities.items() if value > capacity_threshold}
    if not positive_capacities:
        print("No built CO2 pipeline capacity found above threshold; skipping Plotly LCP capacity network.")
        return None

    max_capacity = max(positive_capacities.values()) or 1

    def route_capacity(row):
        src = normalize_route_label(row["start"])
        dst = normalize_route_label(row["end"])
        return positive_capacities.get((src, dst), positive_capacities.get((dst, src), 0.0))

    def line_width(capacity):
        frac = min(float(capacity) / max_capacity, 1.0) ** 0.5
        return min_width + frac * (max_width - min_width)

    gdf_plot = gdf.copy()
    gdf_plot["location_id"] = range(len(gdf_plot))
    geojson_data = json.loads(gdf_plot.to_json())

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            geojson=geojson_data,
            locations=gdf_plot["location_id"],
            z=[1] * len(gdf_plot),
            colorscale=[[0, "#f4f4f4"], [1, "#f4f4f4"]],
            marker_line_color="black",
            marker_line_width=0.6,
            showscale=False,
            name="Provinces",
        )
    )

    for _, row in routes_df.iterrows():
        lons = parse_coordinate_list(row["path_lon"])
        lats = parse_coordinate_list(row["path_lat"])
        if len(lons) != len(lats) or len(lons) < 2:
            continue
        capacity = route_capacity(row)
        if capacity <= capacity_threshold:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(width=line_width(capacity), color=color),
                opacity=0.75,
                hovertemplate=(
                    f"{row['start']} to {row['end']}<br>"
                    f"Built capacity: {format_map_quantity(capacity)}<extra></extra>"
                ),
                name="Built capacity",
                showlegend=False,
            )
        )

    legend_values = _auto_legend_values(max_capacity)
    for value in legend_values:
        fig.add_trace(
            go.Scattergeo(
                lon=[None],
                lat=[None],
                mode="lines",
                line=dict(width=line_width(value), color=color),
                name=f"{_format_capacity_label(value)} built capacity",
                showlegend=True,
            )
        )

    route_dir = Path(route_path).resolve().parent
    point_sources = [
        (route_dir / "capitals_data.csv", "provinces"),
        (route_dir / "basin_centroids.csv", "basins"),
    ]
    marker_colors = plotly_colors.qualitative.Plotly
    for i, (point_path, name) in enumerate(point_sources):
        if not point_path.exists():
            continue
        point_df = pd.read_csv(point_path)
        if not {"name", "latitude", "longitude"}.issubset(point_df.columns):
            continue
        fig.add_trace(
            go.Scattergeo(
                lat=point_df["latitude"],
                lon=point_df["longitude"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=marker_colors[i % len(marker_colors)],
                    opacity=0.9,
                    line=dict(color="black", width=0.5),
                ),
                text=point_df["name"],
                hovertemplate=f"{name}: %{{text}}<br>Lat %{{lat}}<br>Lon %{{lon}}<extra></extra>",
                name=name,
            )
        )

    fig.update_geos(projection_type="mercator", fitbounds="locations", visible=False)
    fig.update_layout(
        title=title,
        showlegend=True,
        width=1000,
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    if output_html is not None:
        fig.write_html(output_html)
        print(f"Saved {output_html}")
    if show:
        fig.show()
    return fig


def plot_co2_storage_circles(
    storage_df: pd.DataFrame,
    gdf,
    plt,
    Circle,
    Transformer,
    output_path: Path,
    title="CO2 Stored by Storage Site",
    figsize=(13, 11),
    circle_scale=2.2,
    storage_legend_values=(0.50, 1.00, 1.75),
    show=False,
):
    if storage_df.empty:
        print("No CO2 storage flows found; skipping storage map.")
        return
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    basin_xy = {name: transformer.transform(lon, lat) for name, (lon, lat) in BASIN_COORDS.items()}
    gdf_proj = gdf.copy().to_crs(epsg=3857)
    xy = co2_node_xy(gdf_proj, basin_xy)
    plot_df = storage_df[storage_df["Storage Site"].isin(xy)].copy()
    max_total = plot_df["CO2 Stored"].max() or 1
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    gdf_proj.plot(ax=ax, color="#F4F4F4", edgecolor="#888888", linewidth=0.6, zorder=0)
    for _, row in plot_df.iterrows():
        x, y = xy[row["Storage Site"]]
        radius = (row["CO2 Stored"] / max_total) ** 0.5 * circle_scale * 80_000
        ax.add_patch(
            Circle((x, y), radius=radius, facecolor=CO2_STORAGE_COLOR, edgecolor="white", linewidth=0.8, alpha=0.85)
        )
        ax.text(x, y + radius + 28_000, row["Storage Site"], fontsize=7, ha="center", va="bottom")
    _draw_circle_legend(ax, plt, Circle, storage_legend_values, max_total, circle_scale, "", "CO2 Stored")
    ax.set_title(title, fontsize=16, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def write_summary(output_dir: Path, capture_df: pd.DataFrame, storage_df: pd.DataFrame, flows: dict):
    if not capture_df.empty:
        capture_df.to_csv(output_dir / "co2_captured_by_sector_by_province.csv")
        capture_df.sum().sort_values(ascending=False).to_csv(
            output_dir / "co2_captured_by_sector_national.csv", header=["CO2 Captured"]
        )
    if not storage_df.empty:
        storage_df.to_csv(output_dir / "co2_stored_by_storage_site.csv", index=False)
    if flows:
        pd.DataFrame(
            [{"Source": src, "Destination": dst, "CO2 Flow": value} for (src, dst), value in flows.items()]
        ).sort_values("CO2 Flow", ascending=False).to_csv(output_dir / "co2_pipeline_flows.csv", index=False)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_008/results")
    parser.add_argument(
        "--map-path",
        default=str(
            script_dir.parent
            / "improved_co2_pipelines"
            / "chinny_co2_pipeline_distance"
            / "gadm36_CHN_1.json"
        ),
    )
    parser.add_argument(
        "--route-path",
        default=str(
            script_dir.parent
            / "improved_co2_pipelines"
            / "chinny_co2_pipeline_distance"
            / "delaunay_lcp_route_paths_output147.csv"
        ),
        help="Methodology notebook route table. If path_lon/path_lat columns exist, they are used as route geometry.",
    )
    parser.add_argument("--output-dir", default="plots/co2_results_008")
    parser.add_argument("--flow-threshold", type=float, default=0.0)
    parser.add_argument("--include-capacity-routes", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_dir = resolve_path(args.results_dir, script_dir)
    map_path = resolve_path(args.map_path, script_dir)
    route_path = resolve_path(args.route_path, script_dir)
    output_dir = resolve_path(args.output_dir, script_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    gpd, plt, Line2D, Circle, Wedge, Transformer = import_plotting_deps()
    if not args.show:
        plt.switch_backend("Agg")
    flows_df, capacity_df = load_inputs(results_dir)
    gdf = load_china_gdf(gpd, map_path)
    capture_df = get_co2_captured_by_sector(flows_df)
    storage_df = get_co2_stored_by_basin(flows_df)
    flows = get_co2_pipeline_flows(flows_df, args.flow_threshold)
    capacities = get_co2_pipeline_capacity(capacity_df)

    print(f"Loaded results from {results_dir}")
    print(f"Loaded map from {map_path}")
    print(f"Loaded route table from {route_path}" if route_path.exists() else f"Route table not found: {route_path}")
    print(f"Capture provinces: {len(capture_df)}")
    print(f"Storage sites: {storage_df['Storage Site'].nunique() if not storage_df.empty else 0}")
    print(f"Pipeline flows above threshold: {len(flows)}")

    plot_province_pies(
        gdf,
        capture_df,
        plt,
        Circle,
        Wedge,
        output_dir / "co2_captured_by_sector.png",
        colors=CO2_SECTOR_COLORS,
        title=f"CO2 Captured by Sector - {results_dir.parent.name}",
        size_legend=CO2_CAPTURE_LEGEND_VALUES if not capture_df.empty else None,
        show=args.show,
    )
    plot_co2_pipeline_flows(
        flows,
        capacities,
        gdf,
        plt,
        Line2D,
        Transformer,
        output_dir / "co2_pipeline_flows.png",
        title=f"CO2 Pipeline Flows - {results_dir.parent.name}",
        show=args.show,
    )
    if args.include_capacity_routes:
        plot_co2_pipeline_flows(
            flows,
            capacities,
            gdf,
            plt,
            Line2D,
            Transformer,
            output_dir / "co2_pipeline_flows_with_capacity_routes.png",
            title=f"CO2 Pipeline Flows and Built Capacity Routes - {results_dir.parent.name}",
            show_capacity_routes=True,
            show=args.show,
        )
    plot_co2_pipeline_capacity(
        capacities,
        gdf,
        plt,
        Line2D,
        Transformer,
        output_dir / "co2_pipeline_capacity_delaunay_lcp.png",
        route_path=route_path,
        title=f"Delaunay LCP CO2 Pipeline Capacity Network - {results_dir.parent.name}",
        capacity_threshold=0.0,
        show_delaunay_network=True,
        show=args.show,
    )
    plot_co2_storage_circles(
        storage_df,
        gdf,
        plt,
        Circle,
        Transformer,
        output_dir / "co2_stored_by_basin.png",
        title=f"CO2 Stored by Basin - {results_dir.parent.name}",
        show=args.show,
    )
    write_summary(output_dir, capture_df, storage_df, flows)


if __name__ == "__main__":
    main()
