"""
Build harmonized digital desert modeling features.

This pipeline enforces comparability by computing flood vulnerability and
connectivity on the same community records.

Outputs:
- processed_data/07_osm_telecom_towers.geojson
- processed_data/07b_osm_water_points.geojson
- processed_data/08_digital_desert_communities.geojson
- processed_data/09_digital_desert_summary.csv
- processed_data/10_model_thresholds.csv
- processed_data/11_province_model_metrics.csv
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"
RAW_GEO_CRS = "EPSG:32648"
MAP_CRS = "EPSG:4326"
HOUSEHOLD_SIZE_PROXY = 4.6


@dataclass
class Thresholds:
    high_risk_cutoff: float
    low_connectivity_km_cutoff: float
    high_population_cutoff: float


def load_geojson_as_gdf(path: Path) -> gpd.GeoDataFrame:
    with path.open("r", encoding="utf-8") as fh:
        gj = json.load(fh)
    gdf = gpd.GeoDataFrame.from_features(gj["features"])
    return gdf.set_crs(RAW_GEO_CRS, allow_override=True)


def load_inputs() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    communities = load_geojson_as_gdf(
        DATA_DIR / "05_indigenous_registered_lands.geojson"
    )
    flood = pd.read_csv(DATA_DIR / "02_flood_risk_analysis.csv")
    return communities, flood


def safe_num(value) -> float:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(num) if not pd.isna(num) else float("nan")


def geometry_to_latlon(geometry) -> tuple[float, float] | None:
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type == "Point":
        return (float(geometry.y), float(geometry.x))
    if hasattr(geometry, "geoms") and len(geometry.geoms) > 0:
        g0 = geometry.geoms[0]
        if g0.geom_type == "Point":
            return (float(g0.y), float(g0.x))
    return None


def minmax_norm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_v = s.min(skipna=True)
    max_v = s.max(skipna=True)
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - min_v) / (max_v - min_v)


def query_overpass(query: str) -> list[dict]:
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    payload = query.encode("utf-8")
    headers = {"User-Agent": "numpum-digital-desert-model/2.0"}
    last_err = None
    for url in endpoints:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("elements", [])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Overpass query failed on all endpoints: {last_err}")


def query_overpass_towers(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> list[dict]:
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    query = f"""
[out:json][timeout:180];
(
  node["man_made"="communications_tower"]({bbox});
  node["man_made"="tower"]["power"!="tower"]({bbox});
  node["man_made"="mast"]["power"!="tower"]({bbox});
  node["tower:type"="communication"]({bbox});
  node["communication:mobile_phone"]({bbox});
  node["telecom"]({bbox});
  way["man_made"="communications_tower"]({bbox});
  way["man_made"="tower"]["power"!="tower"]({bbox});
  way["man_made"="mast"]["power"!="tower"]({bbox});
  way["tower:type"="communication"]({bbox});
  way["communication:mobile_phone"]({bbox});
  way["telecom"]({bbox});
  relation["man_made"="communications_tower"]({bbox});
  relation["man_made"="tower"]["power"!="tower"]({bbox});
  relation["man_made"="mast"]["power"!="tower"]({bbox});
  relation["tower:type"="communication"]({bbox});
  relation["communication:mobile_phone"]({bbox});
  relation["telecom"]({bbox});
);
out center;
"""
    return query_overpass(query)


def query_overpass_water(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> list[dict]:
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    query = f"""
[out:json][timeout:180];
(
  node["waterway"~"river|stream|canal|ditch|drain"]({bbox});
  way["waterway"~"river|stream|canal|ditch|drain"]({bbox});
  relation["waterway"~"river|stream|canal|ditch|drain"]({bbox});
  node["natural"="water"]({bbox});
  way["natural"="water"]({bbox});
  relation["natural"="water"]({bbox});
);
out center;
"""
    return query_overpass(query)


def overpass_elements_to_df(elements: Iterable[dict], source: str) -> pd.DataFrame:
    rows = []
    for elem in elements:
        lat = elem.get("lat")
        lon = elem.get("lon")
        if lat is None or lon is None:
            center = elem.get("center", {})
            lat = center.get("lat")
            lon = center.get("lon")
        if lat is None or lon is None:
            continue
        tags = elem.get("tags", {})
        rows.append(
            {
                "source_type": source,
                "osm_type": elem.get("type"),
                "osm_id": elem.get("id"),
                "name": tags.get("name"),
                "man_made": tags.get("man_made"),
                "tower_type": tags.get("tower:type"),
                "mobile_phone": tags.get("communication:mobile_phone"),
                "operator": tags.get("operator"),
                "waterway": tags.get("waterway"),
                "natural": tags.get("natural"),
                "lat": float(lat),
                "lon": float(lon),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["source_type", "osm_type", "osm_id", "lat", "lon"]
    )


def add_nearest_distance_km(
    points_utm: gpd.GeoDataFrame, targets_utm: gpd.GeoDataFrame
) -> pd.Series:
    if targets_utm.empty:
        return pd.Series([math.nan] * len(points_utm), index=points_utm.index)
    target_geoms = targets_utm.geometry
    distances_m = points_utm.geometry.apply(
        lambda geom: target_geoms.distance(geom).min()
    )
    return distances_m / 1000.0


def fetch_elevation_opentopodata(coords_ll: list[tuple[float, float]]) -> list[float]:
    if not coords_ll:
        return []
    endpoint = "https://api.opentopodata.org/v1/srtm90m"
    out = []
    chunk_size = 80
    for i in range(0, len(coords_ll), chunk_size):
        chunk = coords_ll[i : i + chunk_size]
        loc = "|".join([f"{lat:.6f},{lon:.6f}" for lat, lon in chunk])
        url = f"{endpoint}?locations={urllib.parse.quote(loc, safe='|,')}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "numpum-digital-desert-model/2.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            results = payload.get("results", [])
            for r in results:
                elev = r.get("elevation")
                out.append(float(elev) if elev is not None else math.nan)
        except Exception:
            out.extend([math.nan] * len(chunk))
    return out


def classify_row(row: pd.Series, thresholds: Thresholds) -> str:
    high_risk = (
        row["risk_score"] >= thresholds.high_risk_cutoff
        if not pd.isna(row["risk_score"])
        else False
    )
    low_conn = (
        row["nearest_tower_km"] >= thresholds.low_connectivity_km_cutoff
        if not pd.isna(row["nearest_tower_km"])
        else True
    )
    high_pop = (
        row["est_population_community"] >= thresholds.high_population_cutoff
        if not pd.isna(row["est_population_community"])
        else False
    )
    if high_risk and low_conn:
        return "A: High Risk + Low Connectivity"
    if high_risk and not low_conn:
        return "B: High Risk + Better Connectivity"
    if (not high_risk) and low_conn and high_pop:
        return "C: Low Risk + Low Connectivity + High Population"
    if (not high_risk) and low_conn:
        return "D: Low Risk + Low Connectivity"
    return "E: Lower Priority (Current Data)"


def run() -> None:
    communities, flood = load_inputs()

    communities_ll = communities.to_crs(MAP_CRS)
    min_lon, min_lat, max_lon, max_lat = communities_ll.total_bounds
    pad = 0.35

    towers_elements = query_overpass_towers(
        min_lon - pad, min_lat - pad, max_lon + pad, max_lat + pad
    )
    water_elements = query_overpass_water(
        min_lon - pad, min_lat - pad, max_lon + pad, max_lat + pad
    )
    towers_df = overpass_elements_to_df(towers_elements, source="telecom")
    water_df = overpass_elements_to_df(water_elements, source="water")

    towers_ll = (
        gpd.GeoDataFrame(
            towers_df,
            geometry=gpd.points_from_xy(towers_df["lon"], towers_df["lat"]),
            crs=MAP_CRS,
        )
        if not towers_df.empty
        else gpd.GeoDataFrame(
            towers_df, geometry=gpd.GeoSeries(dtype="geometry"), crs=MAP_CRS
        )
    )
    water_ll = (
        gpd.GeoDataFrame(
            water_df,
            geometry=gpd.points_from_xy(water_df["lon"], water_df["lat"]),
            crs=MAP_CRS,
        )
        if not water_df.empty
        else gpd.GeoDataFrame(
            water_df, geometry=gpd.GeoSeries(dtype="geometry"), crs=MAP_CRS
        )
    )
    towers_utm = towers_ll.to_crs(RAW_GEO_CRS) if not towers_ll.empty else towers_ll
    water_utm = water_ll.to_crs(RAW_GEO_CRS) if not water_ll.empty else water_ll

    model = communities.copy()
    model["num_family"] = pd.to_numeric(model.get("num_family"), errors="coerce")
    model["land_size"] = pd.to_numeric(model.get("land_size"), errors="coerce")
    model["est_population_community"] = model["num_family"] * HOUSEHOLD_SIZE_PROXY
    model["nearest_tower_km"] = add_nearest_distance_km(model, towers_utm)
    model["nearest_water_km"] = add_nearest_distance_km(model, water_utm)

    # Pull free elevation data for all communities (OpenTopoData public API).
    model_ll_for_elev = model.to_crs(MAP_CRS)
    coords_ll = []
    valid_idx = []
    for idx, geom in model_ll_for_elev.geometry.items():
        ll = geometry_to_latlon(geom)
        if ll is not None:
            coords_ll.append(ll)
            valid_idx.append(idx)
    elevations = fetch_elevation_opentopodata(coords_ll)
    model["elevation_m"] = math.nan
    for idx, elev in zip(valid_idx, elevations):
        model.at[idx, "elevation_m"] = elev
    model["elevation_m"] = pd.to_numeric(model["elevation_m"], errors="coerce")
    model["elevation_m"] = model["elevation_m"].fillna(model["elevation_m"].median())

    # Existing province flood table (used as one signal, not the full definition).
    flood["risk_score"] = pd.to_numeric(flood["risk_score"], errors="coerce")
    province_flood = flood[["province", "risk_score"]].dropna().drop_duplicates()
    province_flood["province_flood_norm"] = minmax_norm(province_flood["risk_score"])
    model = model.merge(
        province_flood[["province", "province_flood_norm"]], on="province", how="left"
    )

    # Community-level flood signals from new data.
    water_closeness = 1 - minmax_norm(model["nearest_water_km"].clip(upper=40))
    low_elevation = 1 - minmax_norm(model["elevation_m"])
    province_component = model["province_flood_norm"]

    model["flood_exposure_score"] = 100 * (
        province_component.fillna(0.0) * 0.70 + water_closeness * 0.30
    )
    model["topographic_susceptibility_score"] = 100 * low_elevation
    model["flood_proxy_score"] = 100 * (
        model["flood_exposure_score"] / 100.0 * 0.70
        + model["topographic_susceptibility_score"] / 100.0 * 0.30
    )
    model["flood_data_flag"] = model["province_flood_norm"].apply(
        lambda v: (
            "blended_with_province_flood"
            if pd.notna(v)
            else "proxy_only_no_province_flood"
        )
    )
    model["flood_exposure_flag"] = model["province_flood_norm"].apply(
        lambda v: (
            "province_flood_and_water_proxy" if pd.notna(v) else "water_proxy_only"
        )
    )
    model["topographic_susceptibility_flag"] = "elevation_based"

    # Use the composite flood score as the main risk metric for ranking.
    model["risk_score"] = model["flood_proxy_score"]

    risk_non_null = model["risk_score"].dropna()
    dist_non_null = model["nearest_tower_km"].dropna()
    pop_non_null = model["est_population_community"].dropna()

    thresholds = Thresholds(
        high_risk_cutoff=float(risk_non_null.quantile(0.75))
        if not risk_non_null.empty
        else 0.0,
        low_connectivity_km_cutoff=float(dist_non_null.quantile(0.75))
        if not dist_non_null.empty
        else 0.0,
        high_population_cutoff=float(pop_non_null.quantile(0.75))
        if not pop_non_null.empty
        else 0.0,
    )

    model["digital_desert_class"] = model.apply(
        lambda row: classify_row(row, thresholds), axis=1
    )
    model["is_priority_digital_desert"] = model["digital_desert_class"].isin(
        [
            "A: High Risk + Low Connectivity",
            "C: Low Risk + Low Connectivity + High Population",
            "D: Low Risk + Low Connectivity",
        ]
    )

    model_ll = model.to_crs(MAP_CRS)

    # Persist outputs.
    towers_ll.to_file(DATA_DIR / "07_osm_telecom_towers.geojson", driver="GeoJSON")
    water_ll.to_file(DATA_DIR / "07b_osm_water_points.geojson", driver="GeoJSON")
    model_ll.to_file(
        DATA_DIR / "08_digital_desert_communities.geojson", driver="GeoJSON"
    )

    summary = (
        model_ll.groupby("digital_desert_class", dropna=False)
        .agg(
            communities=("ip_name", "count"),
            provinces=("province", lambda s: ", ".join(sorted(set(s.dropna())))),
            avg_distance_km=("nearest_tower_km", "mean"),
            avg_flood_exposure_score=("flood_exposure_score", "mean"),
            avg_topographic_susceptibility_score=(
                "topographic_susceptibility_score",
                "mean",
            ),
            avg_flood_score=("risk_score", "mean"),
            est_population_community=("est_population_community", "sum"),
        )
        .reset_index()
        .sort_values("communities", ascending=False)
    )

    province_metrics = (
        model_ll.groupby("province", dropna=False)
        .agg(
            communities=("ip_name", "count"),
            avg_flood_exposure_score=("flood_exposure_score", "mean"),
            avg_topographic_susceptibility_score=(
                "topographic_susceptibility_score",
                "mean",
            ),
            avg_flood_score=("risk_score", "mean"),
            avg_network_distance_km=("nearest_tower_km", "mean"),
            avg_water_distance_km=("nearest_water_km", "mean"),
            avg_elevation_m=("elevation_m", "mean"),
            est_population_community=("est_population_community", "sum"),
            priority_communities=("is_priority_digital_desert", "sum"),
        )
        .reset_index()
        .sort_values("avg_flood_score", ascending=False)
    )

    threshold_df = pd.DataFrame(
        [
            {
                "metric": "high_risk_cutoff",
                "value": thresholds.high_risk_cutoff,
                "unit": "composite_flood_score_0_100",
            },
            {
                "metric": "low_connectivity_km_cutoff",
                "value": thresholds.low_connectivity_km_cutoff,
                "unit": "km_to_nearest_tower",
            },
            {
                "metric": "high_population_cutoff",
                "value": thresholds.high_population_cutoff,
                "unit": "estimated_people_per_community",
            },
            {
                "metric": "osm_towers_count",
                "value": float(len(towers_ll)),
                "unit": "points",
            },
            {
                "metric": "osm_water_points_count",
                "value": float(len(water_ll)),
                "unit": "points",
            },
        ]
    )

    summary.to_csv(DATA_DIR / "09_digital_desert_summary.csv", index=False)
    threshold_df.to_csv(DATA_DIR / "10_model_thresholds.csv", index=False)
    province_metrics.to_csv(DATA_DIR / "11_province_model_metrics.csv", index=False)

    print("Generated:")
    print("- 07_osm_telecom_towers.geojson")
    print("- 07b_osm_water_points.geojson")
    print("- 08_digital_desert_communities.geojson")
    print("- 09_digital_desert_summary.csv")
    print("- 10_model_thresholds.csv")
    print("- 11_province_model_metrics.csv")
    print(f"OSM towers collected: {len(towers_ll)}")
    print(f"OSM water points collected: {len(water_ll)}")
    print(
        "Thresholds:",
        {
            "high_risk_cutoff": round(thresholds.high_risk_cutoff, 3),
            "low_connectivity_km_cutoff": round(
                thresholds.low_connectivity_km_cutoff, 3
            ),
            "high_population_cutoff": round(thresholds.high_population_cutoff, 3),
        },
    )


if __name__ == "__main__":
    run()
