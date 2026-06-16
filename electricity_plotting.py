#!/usr/bin/env python3
"""Helpers for visualizing MACRO electricity capacity and flow results."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROVINCE_RENAME = {
    "Nei Mongol": "Innermongolia",
    "Xizang": "Tibet",
    "Ningxia Hui": "Ningxia",
    "Xinjiang Uygur": "Xinjiang",
}

TECH_COLORS = {
    "Coal": "#4d4d4d",
    "Natural Gas": "#d98c3a",
    "Nuclear": "#8e6bbd",
    "Hydropower": "#3b82c4",
    "Solar Utility": "#f2c94c",
    "Solar Distributed": "#f6e27f",
    "Wind Onshore": "#56b870",
    "Wind Offshore": "#2f9e9e",
    "Battery": "#9aa3ad",
    "Pumped Hydro": "#6aaed6",
    "Other": "#b8b8b8",
}

TECH_ORDER = list(TECH_COLORS)
REGION_RE = re.compile(r"Region\d+([A-Za-z]+)")


def import_plotting_deps():
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge
    except ImportError as exc:
        raise SystemExit(
            "Missing plotting dependency. Use an environment with geopandas and matplotlib."
        ) from exc
    return gpd, plt, Wedge


def load_macro_results(
    capacity_path: str | Path,
    flows_path: str | Path,
    load_flows: bool = True,
):
    capacity_path = Path(capacity_path)
    flows_path = Path(flows_path)
    if not capacity_path.exists():
        raise FileNotFoundError(f"Missing capacity.csv: {capacity_path}")
    if not flows_path.exists():
        raise FileNotFoundError(f"Missing flows.csv: {flows_path}")
    flows_df = pd.read_csv(flows_path) if load_flows else None
    return pd.read_csv(capacity_path), flows_df


def load_china_gdf(gpd, map_path: str | Path):
    gdf = gpd.read_file(map_path)
    gdf["NAME_1"] = gdf["NAME_1"].replace(PROVINCE_RENAME)
    return gdf


def province_from_name(value: str) -> str | None:
    match = REGION_RE.search(str(value))
    return match.group(1) if match else None


def is_virtual_capacity_edge(value: str) -> bool:
    text = str(value).lower()
    return "hydropower" in text and ("_inflow_edge" in text or "_spill_edge" in text)


def generation_tech(value: str) -> str | None:
    text = str(value).lower()
    if "transmission" in text or "_to_" in text:
        return None
    if "_charge_edge" in text or "_inflow_edge" in text or "_spill_edge" in text:
        return None
    if "coal_elec" in text:
        return "Coal"
    if "naturalgas_power_elec" in text:
        return "Natural Gas"
    if "nuclear_elec" in text:
        return "Nuclear"
    if "hydropower_discharge" in text:
        return "Hydropower"
    if "pv_central" in text:
        return "Solar Utility"
    if "pv_distribution" in text:
        return "Solar Distributed"
    if "wind_onshore" in text:
        return "Wind Onshore"
    if "wind_offshore" in text:
        return "Wind Offshore"
    if "battery_discharge" in text:
        return "Battery"
    if "pumpedhydro_discharge" in text:
        return "Pumped Hydro"
    return None


def ordered_columns(columns):
    known = [col for col in TECH_ORDER if col in columns]
    extras = sorted(col for col in columns if col not in known)
    return known + extras


def summarize_capacity(
    capacity_df: pd.DataFrame,
    capacity_col: str = "capacity",
    min_capacity: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if capacity_col not in capacity_df:
        raise KeyError(f"capacity_df does not contain {capacity_col!r}")

    rows = []
    elec = capacity_df[capacity_df["commodity"].astype(str).str.lower().eq("electricity")]
    for _, row in elec.iterrows():
        component_id = str(row.get("component_id", ""))
        if is_virtual_capacity_edge(component_id):
            continue
        tech = generation_tech(component_id)
        province = province_from_name(component_id)
        value = float(row.get(capacity_col, 0) or 0)
        if not tech or not province or value <= min_capacity:
            continue
        rows.append({"Province": province, "Technology": tech, "Capacity": value})

    tidy = pd.DataFrame(rows, columns=["Province", "Technology", "Capacity"])
    if tidy.empty:
        return tidy, pd.DataFrame()

    tidy = tidy.groupby(["Province", "Technology"], as_index=False)["Capacity"].sum()
    wide = tidy.pivot(index="Province", columns="Technology", values="Capacity").fillna(0)
    wide = wide.reindex(columns=ordered_columns(wide.columns))
    wide.columns.name = None
    return tidy, wide


def summarize_generation(flows_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "time" not in flows_df:
        raise KeyError("flows_df does not contain 'time'")

    grouped = {"time": flows_df["time"]}
    for col in flows_df.columns:
        if col == "time":
            continue
        tech = generation_tech(col)
        if not tech:
            continue
        values = pd.to_numeric(flows_df[col], errors="coerce").fillna(0).clip(lower=0)
        grouped[tech] = grouped.get(tech, 0) + values

    wide = pd.DataFrame(grouped)
    tech_cols = ordered_columns([col for col in wide.columns if col != "time"])
    wide = wide[["time"] + tech_cols]
    tidy = wide.melt("time", var_name="Technology", value_name="Generation")
    return tidy, wide


def generation_columns(columns) -> list[str]:
    return [col for col in columns if col == "time" or generation_tech(col)]


def summarize_generation_from_csv(
    flows_path: str | Path,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flows_path = Path(flows_path)
    if not flows_path.exists():
        raise FileNotFoundError(f"Missing flows.csv: {flows_path}")

    header = pd.read_csv(flows_path, nrows=0).columns
    usecols = generation_columns(header)
    if "time" not in usecols:
        raise KeyError("flows.csv does not contain 'time'")

    chunks = []
    for chunk in pd.read_csv(flows_path, usecols=usecols, chunksize=chunksize):
        _, wide = summarize_generation(chunk)
        chunks.append(wide)

    if not chunks:
        wide = pd.DataFrame(columns=["time"])
    else:
        wide = pd.concat(chunks, ignore_index=True)

    tidy = wide.melt("time", var_name="Technology", value_name="Generation")
    return tidy, wide


def _project_for_china(gdf):
    return gdf.to_crs("EPSG:3857")


def _province_centroids(gdf_proj):
    cent = gdf_proj.copy()
    cent["point"] = cent.geometry.representative_point()
    return {row["NAME_1"]: (row["point"].x, row["point"].y) for _, row in cent.iterrows()}


def _plot_base_map(gdf_proj, ax):
    gdf_proj.plot(ax=ax, color="#f7f7f2", edgecolor="#9ca3af", linewidth=0.5)
    ax.set_axis_off()


def plot_capacity_stacked_bars(
    capacity_by_province: pd.DataFrame,
    plt,
    output_path: str | Path,
    colors: dict[str, str] | None = None,
    title: str = "Electricity Capacity by Province",
    show: bool = False,
):
    colors = colors or TECH_COLORS
    output_path = Path(output_path)

    plot_df = capacity_by_province.reindex(
        capacity_by_province.sum(axis=1).sort_values(ascending=False).index
    )
    plot_df = plot_df.reindex(columns=ordered_columns(plot_df.columns))

    fig_width = max(14, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    bottoms = np.zeros(len(plot_df))
    x = np.arange(len(plot_df))

    for tech in plot_df.columns:
        values = plot_df[tech].to_numpy()
        if np.allclose(values, 0):
            continue
        ax.bar(
            x,
            values,
            bottom=bottoms,
            label=tech,
            color=colors.get(tech, TECH_COLORS["Other"]),
            edgecolor="white",
            linewidth=0.25,
        )
        bottoms += values

    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("Province")
    ax.set_ylabel("Capacity")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=60, ha="right")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_capacity_map_bars(
    capacity_by_province: pd.DataFrame,
    gdf,
    plt,
    output_path: str | Path,
    colors: dict[str, str] | None = None,
    title: str = "Electricity Capacity by Province",
    max_bar_height: float = 420_000,
    bar_width: float = 55_000,
    show: bool = False,
):
    return plot_capacity_stacked_bars(
        capacity_by_province,
        plt,
        output_path,
        colors=colors,
        title=title,
        show=show,
    )


def plot_capacity_map_pies(
    capacity_by_province: pd.DataFrame,
    gdf,
    plt,
    Wedge,
    output_path: str | Path,
    colors: dict[str, str] | None = None,
    title: str = "Electricity Capacity Mix by Province",
    max_radius: float = 125_000,
    min_radius: float = 18_000,
    show: bool = False,
):
    colors = colors or TECH_COLORS
    output_path = Path(output_path)
    gdf_proj = _project_for_china(gdf)
    xy = _province_centroids(gdf_proj)
    totals = capacity_by_province.sum(axis=1)
    max_total = totals.max() if len(totals) else 0

    fig, ax = plt.subplots(figsize=(13, 11))
    _plot_base_map(gdf_proj, ax)

    if max_total > 0:
        for province, row in capacity_by_province.iterrows():
            total = row.sum()
            if total <= 0 or province not in xy:
                continue
            x, y = xy[province]
            radius = min_radius + (max_radius - min_radius) * np.sqrt(total / max_total)
            start = 90
            for tech in ordered_columns(row.index):
                value = row[tech]
                if value <= 0:
                    continue
                theta = 360 * value / total
                ax.add_patch(
                    Wedge(
                        (x, y),
                        radius,
                        start,
                        start + theta,
                        facecolor=colors.get(tech, TECH_COLORS["Other"]),
                        edgecolor="white",
                        linewidth=0.3,
                    )
                )
                start += theta

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors.get(col, TECH_COLORS["Other"]))
        for col in capacity_by_province.columns
    ]
    ax.legend(handles, capacity_by_province.columns, loc="lower left", frameon=False, ncol=2)
    ax.set_title(title, fontsize=16, pad=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_generation_area(
    generation_by_time: pd.DataFrame,
    plt,
    output_path: str | Path,
    colors: dict[str, str] | None = None,
    title: str = "Electricity Generation by Technology",
    show: bool = False,
):
    colors = colors or TECH_COLORS
    output_path = Path(output_path)
    tech_cols = [col for col in generation_by_time.columns if col != "time"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        generation_by_time["time"],
        [generation_by_time[col] for col in tech_cols],
        labels=tech_cols,
        colors=[colors.get(col, TECH_COLORS["Other"]) for col in tech_cols],
        alpha=0.92,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Generation")
    ax.margins(x=0)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=Path, required=True, help="Path to MACRO capacity.csv")
    parser.add_argument("--flows", type=Path, required=True, help="Path to MACRO flows.csv")
    parser.add_argument("--map", type=Path, required=True, help="Path to China province GeoJSON")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/electricity_results"))
    parser.add_argument("--capacity-col", default="capacity")
    parser.add_argument("--flow-chunksize", type=int, default=100_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir / ".cache"))

    gpd, plt, Wedge = import_plotting_deps()
    capacity_df, _ = load_macro_results(args.capacity, args.flows, load_flows=False)
    gdf = load_china_gdf(gpd, args.map)
    capacity_tidy, capacity_wide = summarize_capacity(capacity_df, args.capacity_col)
    generation_tidy, generation_wide = summarize_generation_from_csv(
        args.flows,
        chunksize=args.flow_chunksize,
    )

    capacity_tidy.to_csv(args.output_dir / "electricity_capacity_by_province_technology.csv", index=False)
    generation_tidy.to_csv(args.output_dir / "electricity_generation_by_time_technology.csv", index=False)
    plot_capacity_stacked_bars(capacity_wide, plt, args.output_dir / "electricity_capacity_stacked_bars.png")
    plot_capacity_map_pies(capacity_wide, gdf, plt, Wedge, args.output_dir / "electricity_capacity_map_pies.png")
    plot_generation_area(generation_wide, plt, args.output_dir / "electricity_generation_stacked_area.png")


if __name__ == "__main__":
    main()
