"""
update_transport_costs.py

Updates inter-provincial transport costs using:
  - ROAD (all commodities): actual OSM road-network distances via local OSRM server
                            (falls back to haversine × 1.15 if route not found)

Road distances come from a local OSRM instance routing on China's OSM road network.
The road cost rate (0.034 USD/(tonne·km)) is back-calculated from the original model inputs.

Prerequisites:
    Start the OSRM road server before running:
        docker run -d --name osrm-road -p 5000:5000 \\
          -v "<path-to-osrm-data>:/data" osrm/osrm-backend \\
          osrm-routed --algorithm mld /data/china.osrm

    To restart a stopped container: docker start osrm-road

Usage:
    python update_transport_costs.py              # dry-run: prints summary, no file writes
    python update_transport_costs.py --write      # write updated JSONs + save plot
    python update_transport_costs.py --plot-only  # re-generate plot without re-querying OSRM

Outputs (in plot_inputs/):
    transport_distance_matrix.csv   -- actual road distances (km), cached for re-use
    transport_validation.png        -- validation figure

Note: Tibet (Region26) was previously missing from PROVINCE_CAPITALS and therefore
excluded from all transport routes. It is now included.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
ASSETS       = BASE / "assets" / "assets_1"
VARIANT_ROAD = BASE / "assets" / "transport_variants" / "road_osrm"
OUT          = BASE / "plot_inputs"
OUT.mkdir(exist_ok=True)

TRANSPORT_FILES = {
    "cement":      "cement_transport.json",
    "crudesteel":  "crudesteel_transport.json",
    "dri":         "dri_transport.json",
    "ironore":     "ironore_transport.json",
    "steelscrap":  "steelscrap_transport.json",
}

# ── Vertex name template per commodity ────────────────────────────────────────
VERTEX_TEMPLATE = {
    "cement":     "cement_{r}",
    "crudesteel": "crudesteel_{r}",
    "dri":        "dri_{r}",
    "ironore":    "ironore_{r}",
    "steelscrap": "steelscrap_source_{r}",  # steelscrap nodes have _source_ infix
}

# ── Transport parameters ───────────────────────────────────────────────────────
ROAD_RATE         = 0.034   # USD/(tonne·km) — back-calculated from original model inputs
ROAD_DETOUR       = 1.15    # fallback detour factor for most provinces
ROAD_DETOUR_TIBET = 1.50    # higher detour for Tibet: mountain passes inflate road vs straight-line

# ── OSRM server — public demo server; swap for http://localhost:5000 if running locally ──
OSRM_URL     = "http://router.project-osrm.org"
MATRIX_CACHE = OUT / "transport_distance_matrix.csv"

# ── Province capitals (lat, lon) ───────────────────────────────────────────────
PROVINCE_CAPITALS = {
    "Region1Beijing":       (39.9042, 116.4074),
    "Region2Tianjin":       (39.1421, 117.1767),
    "Region3Hebei":         (38.0428, 114.5149),
    "Region4Shanxi":        (37.8706, 112.5490),
    "Region5Innermongolia": (40.8414, 111.7519),
    "Region6Liaoning":      (41.8057, 123.4315),
    "Region7Jilin":         (43.8171, 125.3235),
    "Region8Heilongjiang":  (45.7569, 126.6425),
    "Region9Shanghai":      (31.2304, 121.4737),
    "Region10Jiangsu":      (32.0603, 118.7969),
    "Region11Zhejiang":     (30.2741, 120.1551),
    "Region12Anhui":        (31.8639, 117.2808),
    "Region13Fujian":       (26.0745, 119.2965),
    "Region14Jiangxi":      (28.6820, 115.8579),
    "Region15Shandong":     (36.6512, 117.1201),
    "Region16Henan":        (34.7466, 113.6253),
    "Region17Hubei":        (30.5928, 114.3055),
    "Region18Hunan":        (28.2278, 112.9388),
    "Region19Guangdong":    (23.1291, 113.2644),
    "Region20Guangxi":      (22.8170, 108.3665),
    "Region21Hainan":       (20.0440, 110.1999),
    "Region22Chongqing":    (29.4316, 106.9123),
    "Region23Sichuan":      (30.5728, 104.0668),
    "Region24Guizhou":      (26.5983, 106.7078),
    "Region25Yunnan":       (24.8801, 102.8329),
    "Region26Tibet":        (29.6500,  91.1329),   # Lhasa; was missing — added
    "Region27Shaanxi":      (34.2658, 108.9541),
    "Region28Gansu":        (36.0611, 103.8343),
    "Region29Qinghai":      (36.6232, 101.7786),
    "Region30Ningxia":      (38.4680, 106.2734),
    "Region31Xinjiang":     (43.8256,  87.6168),
}

REGIONS = list(PROVINCE_CAPITALS.keys())
N = len(REGIONS)

# ── Helpers ───────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = math.pi / 180
    a = (0.5 - math.cos((lat2 - lat1) * p) / 2
         + math.cos(lat1 * p) * math.cos(lat2 * p)
         * (1 - math.cos((lon2 - lon1) * p)) / 2)
    return 2 * R * math.asin(math.sqrt(a))


def fetch_road_matrix(regions, capitals):
    """
    Query local OSRM road server for the full N×N distance matrix (km).
    Falls back to haversine × ROAD_DETOUR for unreachable pairs.
    Returns a dict (src, dst) -> road_km.
    """
    coord_str = ";".join(f"{capitals[r][1]},{capitals[r][0]}" for r in regions)
    url = f"{OSRM_URL}/table/v1/driving/{coord_str}?annotations=distance"
    print(f"Querying local OSRM road server for {len(regions)}×{len(regions)} matrix …")
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(3)
    else:
        raise RuntimeError("OSRM query failed. Is the road server running?\n"
                           "  docker start osrm-road")

    data = resp.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data.get('message')}")

    raw = data["distances"]
    matrix = {}
    fallbacks = []
    for i, src in enumerate(regions):
        for j, dst in enumerate(regions):
            if i == j:
                continue
            val = raw[i][j]
            if val is None or val == 0:
                slat, slon = capitals[src]
                dlat, dlon = capitals[dst]
                km = haversine_km(slat, slon, dlat, dlon) * ROAD_DETOUR
                fallbacks.append((src, dst))
            else:
                km = val / 1000.0
            matrix[(src, dst)] = km

    if fallbacks:
        print(f"  {len(fallbacks)} pairs used haversine fallback (no road route found):")
        for s, d in fallbacks[:10]:
            print(f"    {s} → {d}")
        if len(fallbacks) > 10:
            print(f"    … and {len(fallbacks)-10} more")
    return matrix


def haversine_fallback(src, dst):
    """Haversine fallback with a higher detour for Tibet routes (mountain terrain)."""
    slat, slon = PROVINCE_CAPITALS[src]
    dlat, dlon = PROVINCE_CAPITALS[dst]
    factor = ROAD_DETOUR_TIBET if src == "Region26Tibet" or dst == "Region26Tibet" else ROAD_DETOUR
    return haversine_km(slat, slon, dlat, dlon) * factor


def load_or_fetch_road_matrix():
    """
    Load cached road distance matrix, patching in any missing regions.

    If the cache exists but is missing some regions (e.g. Tibet was newly added),
    only the missing rows/columns are fetched from OSRM — the existing distances
    are preserved. Falls back to haversine if OSRM is unavailable.
    """
    matrix = {}
    cached_regions = set()

    if MATRIX_CACHE.exists():
        with open(MATRIX_CACHE) as f:
            reader = csv.reader(f)
            cols = next(reader)[1:]
            cached_regions = set(cols)
            for row in reader:
                src = row[0]
                for j, dst in enumerate(cols):
                    if src != dst and row[j + 1]:
                        matrix[(src, dst)] = float(row[j + 1])
        print(f"Loaded {len(matrix)} cached distances from {MATRIX_CACHE.name}")

    missing_regions = [r for r in REGIONS if r not in cached_regions]
    if missing_regions:
        print(f"Missing regions: {missing_regions}")
        # Try OSRM for just the missing regions (query all 31 points but extract only new pairs)
        try:
            new = fetch_road_matrix(REGIONS, PROVINCE_CAPITALS)
            for r in missing_regions:
                for dst in REGIONS:
                    if r != dst:
                        matrix[(r, dst)] = new[(r, dst)]
                        matrix[(dst, r)] = new[(dst, r)]
            print(f"  Added OSRM distances for: {missing_regions}")
        except RuntimeError as e:
            print(f"  OSRM unavailable — using haversine fallback for missing regions.\n  ({e})")
            for r in missing_regions:
                for dst in REGIONS:
                    if r != dst:
                        matrix[(r, dst)] = haversine_fallback(r, dst)
                        matrix[(dst, r)] = haversine_fallback(dst, r)

        # Write updated cache
        with open(MATRIX_CACHE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + REGIONS)
            for src in REGIONS:
                w.writerow([src] + [
                    "" if src == dst else f"{matrix.get((src, dst), '')}"
                    for dst in REGIONS
                ])
        print(f"Updated cache written to {MATRIX_CACHE.name}")

    elif not matrix:
        # No cache at all — full fetch
        matrix = fetch_road_matrix(REGIONS, PROVINCE_CAPITALS)
        with open(MATRIX_CACHE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + REGIONS)
            for src in REGIONS:
                w.writerow([src] + [
                    "" if src == dst else f"{matrix.get((src, dst), '')}"
                    for dst in REGIONS
                ])
        print(f"Road distance matrix cached to {MATRIX_CACHE.name}")

    return matrix


# Module-level cache (populated on first call to get_road_matrix)
_road_matrix = None

def get_road_matrix():
    global _road_matrix
    if _road_matrix is None:
        _road_matrix = load_or_fetch_road_matrix()
    return _road_matrix


def compute_cost(src, dst):
    """Return USD/tonne cost for src→dst road transport."""
    road_km = get_road_matrix()[(src, dst)]
    return round(road_km * ROAD_RATE, 2)


# ── Core: build updated JSON ──────────────────────────────────────────────────
def build_updated_json(original_file, commodity):
    """Rebuild the full N×(N-1) route list from scratch using current REGIONS."""
    with open(original_file) as f:
        data = json.load(f)
    top_key = list(data.keys())[0]
    tmpl = VERTEX_TEMPLATE[commodity]

    new_instances = []
    for src in REGIONS:
        for dst in REGIONS:
            if src == dst:
                continue
            new_instances.append({
                "id": f"{commodity}_transport_{src}_to_{dst}",
                "edges": {
                    "transmission_edge": {
                        "start_vertex": tmpl.format(r=src),
                        "end_vertex":   tmpl.format(r=dst),
                        "variable_om_cost": compute_cost(src, dst),
                    }
                },
            })

    data[top_key]["instance_data"] = new_instances
    print(f"  {original_file.name}: {len(new_instances)} routes")
    return data


# ── Summary stats ─────────────────────────────────────────────────────────────
def print_summary():
    print(f"\n{'─'*60}")
    print(f"  Provinces:         {N}")
    print(f"  Route pairs:       {N*(N-1)}")
    print(f"  Road rate:         {ROAD_RATE} USD/(tonne·km)")
    print(f"  Road detour:       {ROAD_DETOUR}×  (straight-line fallback only)")
    print(f"{'─'*60}")

    print(f"\n{'Commodity':<14} {'Avg cost (USD/t)':<18} {'Min':<10} {'Max'}")
    print("─" * 60)
    for commodity in TRANSPORT_FILES:
        costs = [compute_cost(s, d) for s in REGIONS for d in REGIONS if s != d]
        print(f"  {commodity:<12}  {np.mean(costs):<16.2f} {min(costs):<10.2f} {max(costs):.2f}")
    print()


# ── Plot ──────────────────────────────────────────────────────────────────────
def load_china_map():
    import geopandas as gpd
    cache = OUT / "ne_50m_admin1_china.gpkg"
    if cache.exists():
        return gpd.read_file(cache)
    try:
        url = ("https://naciscdn.org/naturalearth/50m/cultural/"
               "ne_50m_admin_1_states_provinces.zip")
        print("Downloading Natural Earth 50m province boundaries …")
        world = gpd.read_file(url)
        china = world[world["admin"] == "China"].copy()
        china.to_file(cache, driver="GPKG")
        return china
    except Exception as e:
        print(f"  Map unavailable: {e}")
        return None


def fetch_road_geometries(src_region):
    """
    Fetch actual OSRM road route geometries from src_region to all other regions.
    Returns list of (dst_region, cost, list_of_[lon,lat]_coords).
    """
    src_lat, src_lon = PROVINCE_CAPITALS[src_region]
    results = []
    for dst in REGIONS:
        if dst == src_region:
            continue
        dlat, dlon = PROVINCE_CAPITALS[dst]
        url = (f"{OSRM_URL}/route/v1/driving/"
               f"{src_lon},{src_lat};{dlon},{dlat}"
               f"?overview=full&geometries=geojson")
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=15)
                d = r.json()
                if d.get("code") == "Ok":
                    coords = d["routes"][0]["geometry"]["coordinates"]
                    results.append((dst, compute_cost(src_region, dst), coords))
                break
            except Exception:
                time.sleep(1)
        time.sleep(0.05)
    return results


def make_validation_plot(output_path):
    china = load_china_map()

    road_costs = {}
    orig_costs = {}
    for src in REGIONS:
        for dst in REGIONS:
            if src == dst:
                continue
            sl = haversine_km(*PROVINCE_CAPITALS[src], *PROVINCE_CAPITALS[dst])
            orig_costs[(src, dst)] = round(sl * ROAD_RATE, 2)
            road_costs[(src, dst)] = compute_cost(src, dst)

    print(f"Fetching actual road route geometries from Beijing ({N-1} routes) …")
    beijing_routes = fetch_road_geometries("Region1Beijing")

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Transport Cost Update: Straight-Line → OSM Road Network (31 provinces incl. Tibet)",
                 fontsize=15, fontweight="bold", y=0.98)

    # ── Panel 1: Actual road routes from Beijing ──────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    if china is not None:
        china.plot(ax=ax1, color="#f0f0f0", edgecolor="#aaaaaa", linewidth=0.5)
    else:
        ax1.set_facecolor("#e8f4f8")

    all_costs = list(road_costs.values())
    norm  = mcolors.Normalize(vmin=min(all_costs), vmax=max(all_costs))
    cmap  = cm.plasma_r

    for dst, cost, coords in beijing_routes:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        ax1.plot(lons, lats, color=cmap(norm(cost)), linewidth=1.2, alpha=0.85, zorder=2)

    for reg, (lat, lon) in PROVINCE_CAPITALS.items():
        color = "darkorange" if reg == "Region26Tibet" else "black"
        ax1.scatter(lon, lat, s=18, color=color, zorder=5)
    ax1.scatter(*PROVINCE_CAPITALS["Region1Beijing"][::-1],
                s=60, color="red", zorder=6)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, shrink=0.6, label="Transport cost (USD/tonne)")

    legend_els = [
        Line2D([0], [0], color="red",        marker="o", linewidth=0, markersize=6, label="Beijing (origin)"),
        Line2D([0], [0], color="darkorange",  marker="o", linewidth=0, markersize=6, label="Tibet (newly added)"),
        Line2D([0], [0], color="black",       marker="o", linewidth=0, markersize=4, label="Other provinces"),
    ]
    ax1.legend(handles=legend_els, fontsize=8, loc="upper left")
    ax1.set_title(f"Road Routes from Beijing (N={N} provinces)", fontsize=11)
    ax1.set_xlim(72, 137); ax1.set_ylim(17, 55)
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")

    # ── Panel 2: Cost comparison — original (SL) vs new (road OSRM) ──────────
    ax2 = fig.add_subplot(1, 2, 2)
    orig_vals  = [orig_costs[(s, d)]  for s in REGIONS for d in REGIONS if s != d]
    new_vals   = [road_costs[(s, d)]  for s in REGIONS for d in REGIONS if s != d]
    tibet_mask = np.array([s == "Region26Tibet" or d == "Region26Tibet"
                           for s in REGIONS for d in REGIONS if s != d])

    ax2.scatter(np.array(orig_vals)[~tibet_mask], np.array(new_vals)[~tibet_mask],
                alpha=0.3, s=10, color="steelblue", label="Other routes", edgecolors="none")
    ax2.scatter(np.array(orig_vals)[tibet_mask], np.array(new_vals)[tibet_mask],
                alpha=0.8, s=30, color="darkorange", label="Tibet routes (new)", edgecolors="none", zorder=5)

    lim = max(max(orig_vals), max(new_vals)) * 1.05
    ax2.plot([0, lim], [0, lim], "k--", lw=1, label="1:1 (no change)")
    ax2.set_xlabel(f"Original cost (USD/tonne)  [straight-line × {ROAD_RATE}]")
    ax2.set_ylabel("New cost (USD/tonne)  [road OSRM]")
    ax2.set_title(f"Original vs New Transport Costs\n(all {N*(N-1)} province pairs)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {output_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",     action="store_true",
                        help="Write updated JSON files (default: dry-run)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip JSON update, just regenerate the plot")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip the validation plot")
    args = parser.parse_args()

    if not args.plot_only:
        get_road_matrix()
        print_summary()
        print("Computing new transport costs …\n")
        for commodity, fname in TRANSPORT_FILES.items():
            fpath = ASSETS / fname
            updated = build_updated_json(fpath, commodity)
            if args.write:
                with open(fpath, "w") as fp:
                    json.dump(updated, fp, indent=2)
                variant = VARIANT_ROAD / fname
                if variant.exists():
                    with open(variant, "w") as fp:
                        json.dump(updated, fp, indent=2)

        if args.write:
            print(f"\nAll transport JSON files written (assets_1 + road_osrm variant).")
        else:
            print(f"\nDry-run — pass --write to update files.")

    if not args.no_plot:
        make_validation_plot(OUT / "transport_validation.png")


if __name__ == "__main__":
    main()
