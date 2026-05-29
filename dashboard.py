"""
Interpretability-first dashboard for identifying digital deserts in NE Cambodia.

Story flow:
1) Problem framing
2) Evidence and source data
3) Rule-based model logic
4) Mapped results
5) Policy actions
"""

from pathlib import Path
import json

import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Digital Desert Story Dashboard",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path(__file__).resolve().parent / "processed_data"
RAW_CRS = "EPSG:32648"
MAP_CRS = "EPSG:4326"
HOUSEHOLD_SIZE_PROXY = 4.6

CLASS_COLORS = {
    "A: High Risk + Low Connectivity": "#B91C1C",
    "B: High Risk + Better Connectivity": "#EA580C",
    "C: Low Risk + Low Connectivity + High Population": "#CA8A04",
    "D: Low Risk + Low Connectivity": "#2563EB",
    "E: Lower Priority (Current Data)": "#15803D",
}


def _get_point_coords(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type == "Point":
        return (geometry.y, geometry.x)
    if hasattr(geometry, "geoms") and len(geometry.geoms) > 0:
        first_geom = geometry.geoms[0]
        if first_geom.geom_type == "Point":
            return (first_geom.y, first_geom.x)
    return None


def _safe_number(value):
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(num) if not pd.isna(num) else None


def _infer_input_crs(gdf):
    sample = None
    for geom in gdf.geometry:
        coords = _get_point_coords(geom)
        if coords is not None:
            sample = coords
            break
    if sample is None:
        return MAP_CRS
    lat, lon = sample
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return MAP_CRS
    return RAW_CRS


def _get_threshold(threshold_df, metric):
    if threshold_df is None or threshold_df.empty:
        return None
    row = threshold_df[threshold_df["metric"] == metric]
    if row.empty:
        return None
    return _safe_number(row.iloc[0].get("value"))


@st.cache_data(show_spinner=False)
def load_data():
    required = [
        "01_provincial_summary.csv",
        "02_flood_risk_analysis.csv",
        "03_connectivity_statistics.csv",
        "04_indigenous_villages_mrd.geojson",
        "05_indigenous_registered_lands.geojson",
        "06_villages_with_risk_context.geojson",
    ]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {DATA_DIR}: {', '.join(missing)}"
        )

    def load_geojson(path):
        with open(path, "r", encoding="utf-8") as fh:
            gj = json.load(fh)
        gdf = gpd.GeoDataFrame.from_features(gj["features"])
        inferred_crs = _infer_input_crs(gdf)
        if gdf.crs is None:
            gdf = gdf.set_crs(inferred_crs, allow_override=True)
        gdf = gdf.to_crs(MAP_CRS)
        numeric_cols = {
            "num_family",
            "land_size",
            "flood_exposure_score",
            "topographic_susceptibility_score",
            "risk_score",
            "total_pop_exposed",
            "total_area_flooded_km2",
            "nearest_tower_km",
            "est_population_community",
            "est_population_need_connectivity",
        }
        for col in numeric_cols:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
        return gdf

    provincial = pd.read_csv(DATA_DIR / "01_provincial_summary.csv")
    flood = pd.read_csv(DATA_DIR / "02_flood_risk_analysis.csv")
    connectivity = pd.read_csv(DATA_DIR / "03_connectivity_statistics.csv")
    villages = load_geojson(DATA_DIR / "04_indigenous_villages_mrd.geojson")
    lands = load_geojson(DATA_DIR / "05_indigenous_registered_lands.geojson")
    villages_context = load_geojson(DATA_DIR / "06_villages_with_risk_context.geojson")

    model_communities = None
    osm_towers = None
    model_summary = None
    model_thresholds = None
    province_model_metrics = None
    osm_water_points = None

    if (DATA_DIR / "08_digital_desert_communities.geojson").exists():
        model_communities = load_geojson(
            DATA_DIR / "08_digital_desert_communities.geojson"
        )
    if (DATA_DIR / "07_osm_telecom_towers.geojson").exists():
        osm_towers = load_geojson(DATA_DIR / "07_osm_telecom_towers.geojson")
    if (DATA_DIR / "07b_osm_water_points.geojson").exists():
        osm_water_points = load_geojson(DATA_DIR / "07b_osm_water_points.geojson")
    if (DATA_DIR / "09_digital_desert_summary.csv").exists():
        model_summary = pd.read_csv(DATA_DIR / "09_digital_desert_summary.csv")
    if (DATA_DIR / "10_model_thresholds.csv").exists():
        model_thresholds = pd.read_csv(DATA_DIR / "10_model_thresholds.csv")
    if (DATA_DIR / "11_province_model_metrics.csv").exists():
        province_model_metrics = pd.read_csv(DATA_DIR / "11_province_model_metrics.csv")

    return {
        "provincial": provincial,
        "flood": flood,
        "connectivity": connectivity,
        "villages": villages,
        "lands": lands,
        "villages_context": villages_context,
        "model_communities": model_communities,
        "osm_towers": osm_towers,
        "osm_water_points": osm_water_points,
        "model_summary": model_summary,
        "model_thresholds": model_thresholds,
        "province_model_metrics": province_model_metrics,
    }


def style_dashboard():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Source+Sans+3:wght@400;600;700&display=swap');
:root {
  --surface: #0f172a;
  --surface-2: #111827;
  --panel: rgba(15, 23, 42, 0.78);
  --ink: #e5e7eb;
  --muted: #94a3b8;
  --line: #334155;
  --brand: #2b6cf3;
  --brand-2: #06b6d4;
}
.stApp {
  font-family: 'Source Sans 3', sans-serif;
  background: radial-gradient(1100px 600px at 10% -8%, #1e293b 0%, #0b1220 36%, #050b17 100%);
  color: var(--ink);
}
.story-hero, .nav-wrap, .filter-wrap, .mini-note, .insight-card, .source-card, .action-card, .phase-card, .flow-wrap, .coverage-stack {
  border: 1px solid var(--line);
  border-radius: 12px;
}
.story-hero {
  background: linear-gradient(160deg, #111b2f 0%, #0b1324 60%, #0a1020 100%);
  padding: 22px 24px;
  margin-bottom: 14px;
}
.story-pill {
  display: inline-block;
  border-radius: 10px;
  border: 1px solid #334155;
  background: #0f1a2e;
  color: #cbd5e1;
  padding: 4px 10px;
  font-size: 12px;
  margin-right: 8px;
  font-family: 'Space Grotesk', sans-serif;
}
.nav-wrap { padding: 10px 14px 2px 14px; background: var(--panel); margin-bottom: 12px; }
.filter-wrap { padding: 12px 14px 10px 14px; background: rgba(12, 19, 35, 0.86); margin-bottom: 14px; }
.mini-note { background: rgba(9, 16, 30, 0.9); padding: 10px 12px; }
.insight-card { background: rgba(12, 20, 35, 0.95); padding: 12px 14px 8px 14px; height: 100%; }
.source-card { background: rgba(13, 22, 39, 0.95); padding: 12px; min-height: 88px; height: 100%; }
.action-card, .phase-card { background: rgba(12, 20, 36, 0.95); padding: 12px; height: 100%; }
.coverage-stack { background: rgba(11, 18, 32, 0.8); padding: 10px 12px; }
.coverage-row { display: flex; justify-content: space-between; gap: 10px; padding: 8px 0; border-bottom: 1px solid rgba(51, 65, 85, 0.7); }
.coverage-row:last-child { border-bottom: none; padding-bottom: 0; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: #f8fafc !important; }
p, span, label, .stMarkdown, .stCaption { color: #cbd5e1 !important; }
div[data-testid="stMetric"] { background: rgba(16, 25, 44, 0.92); border: 1px solid #2a3c54; border-radius: 12px; padding: 10px 12px; }
</style>
        """,
        unsafe_allow_html=True,
    )


def get_story_metrics(data):
    lands = data["lands"].copy()
    families = pd.to_numeric(lands.get("num_family"), errors="coerce").fillna(0)
    est_pop = int((families * HOUSEHOLD_SIZE_PROXY).sum())
    provinces = lands["province"].dropna().nunique()
    towers = 0 if data["osm_towers"] is None else len(data["osm_towers"])
    return {
        "communities": len(lands),
        "families": int(families.sum()),
        "estimated_population": est_pop,
        "provinces": int(provinces),
        "towers": int(towers),
    }


def build_priority_frame(model_df, thresholds):
    df = model_df.copy()
    risk_cut = _get_threshold(thresholds, "high_risk_cutoff") or 1.0
    dist_cut = _get_threshold(thresholds, "low_connectivity_km_cutoff") or 1.0
    pop_cut = _get_threshold(thresholds, "high_population_cutoff") or 1.0

    df["risk_score"] = pd.to_numeric(df.get("risk_score"), errors="coerce").fillna(0.0)
    df["nearest_tower_km"] = pd.to_numeric(
        df.get("nearest_tower_km"), errors="coerce"
    ).fillna(0.0)
    df["est_population_community"] = pd.to_numeric(
        df.get("est_population_community"), errors="coerce"
    ).fillna(0.0)

    # Transparent composite score for ranking only.
    df["score_risk"] = (df["risk_score"] / risk_cut).clip(0, 3)
    df["score_distance"] = (df["nearest_tower_km"] / dist_cut).clip(0, 3)
    df["score_population"] = (df["est_population_community"] / pop_cut).clip(0, 3)
    df["priority_score"] = (
        (0.5 * df["score_risk"])
        + (0.3 * df["score_distance"])
        + (0.2 * df["score_population"])
    )
    return df


def render_global_filters(model_df):
    provinces = sorted(model_df["province"].dropna().unique().tolist())
    classes = sorted(model_df["digital_desert_class"].dropna().unique().tolist())

    if "flt_provinces" not in st.session_state:
        st.session_state.flt_provinces = provinces
    if "flt_classes" not in st.session_state:
        st.session_state.flt_classes = classes
    if "flt_show_towers" not in st.session_state:
        st.session_state.flt_show_towers = True
    if "flt_priority_only" not in st.session_state:
        st.session_state.flt_priority_only = False

    st.markdown('<div class="filter-wrap">', unsafe_allow_html=True)
    with st.form("global_filters"):
        c1, c2, c3, c4 = st.columns([2.4, 2.4, 1.1, 1.1])
        with c1:
            cur_prov = st.session_state.get("flt_provinces", provinces)
            all_prov = st.checkbox(
                "All Provinces", value=len(cur_prov) == len(provinces)
            )
            selected_prov = st.multiselect(
                "Province",
                provinces,
                default=provinces if all_prov else cur_prov,
            )
        with c2:
            cur_cls = st.session_state.get("flt_classes", classes)
            all_cls = st.checkbox("All Classes", value=len(cur_cls) == len(classes))
            selected_cls = st.multiselect(
                "Class",
                classes,
                default=classes if all_cls else cur_cls,
            )
        with c3:
            show_towers = st.checkbox(
                "Show telecom", value=st.session_state.get("flt_show_towers", True)
            )
        with c4:
            priority_only = st.checkbox(
                "Priority only", value=st.session_state.get("flt_priority_only", False)
            )
        applied = st.form_submit_button("Apply Filters", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if applied:
        st.session_state.flt_provinces = selected_prov if selected_prov else provinces
        st.session_state.flt_classes = selected_cls if selected_cls else classes
        st.session_state.flt_show_towers = show_towers
        st.session_state.flt_priority_only = priority_only

    filtered = model_df[
        model_df["province"].isin(st.session_state.get("flt_provinces", provinces))
        & model_df["digital_desert_class"].isin(
            st.session_state.get("flt_classes", classes)
        )
    ].copy()
    if st.session_state.get("flt_priority_only", False) and not filtered.empty:
        filtered = filtered[
            filtered["priority_score"] >= filtered["priority_score"].quantile(0.7)
        ]

    return filtered


def render_header():
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="story-hero">
  <div class="story-pill">Interpretability First</div>
  <div class="story-pill">Policy-Oriented</div>
  <h1 style="margin: 8px 0 4px 0; color:#f8fafc;">Digital Desert Story: North-eastern Cambodia</h1>
  <p style="margin: 0; color:#cbd5e1;">
    A guided narrative for international judges to see which indigenous communities in North-eastern Cambodia face the double burden of flood exposure, terrain risk, and weak network access.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_page_start(data):
    st.subheader("Overview")
    metrics = get_story_metrics(data)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Indigenous communities", f"{metrics['communities']:,}")
    c2.metric("Families recorded", f"{metrics['families']:,}")
    c3.metric("Estimated people", f"{metrics['estimated_population']:,}")
    c4.metric("Provinces represented", metrics["provinces"])
    c5.metric("OpenStreetMap telecom proxy points", metrics["towers"])

    st.info(
        "Goal: identify indigenous communities with high flood exposure, high terrain risk, and weak network access, then rank them for action using transparent rules."
    )

    overview_col1, overview_col2 = st.columns([1.2, 1], gap="medium")
    with overview_col1:
        st.markdown(
            """
            <div class="mini-note">
            <b>Who this dashboard describes:</b> indigenous communities recorded in Ministry of Rural Development communal land data, then enriched with flood and connectivity evidence.
            <br><br>
            <b>Why the geography changes:</b> some charts summarize communities, while others aggregate those same community records to province level so the committee can compare places consistently.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_col2:
        st.markdown(
            """
            <div class="mini-note">
            <b>How to read the story:</b> a community is the unit of analysis; a province is only a reporting layer for comparison.
            <br><br>
            <b>Main message:</b> risk is highest when flood exposure and poor telecom proximity happen together.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_page_data(data):
    st.subheader("Data Sources")
    if data["model_communities"] is None:
        st.warning(
            "Model outputs not found. Run `python build_digital_desert_model.py` first."
        )
        return

    model = data["model_communities"].copy()
    model["risk_score"] = pd.to_numeric(model["risk_score"], errors="coerce")
    model["nearest_tower_km"] = pd.to_numeric(
        model["nearest_tower_km"], errors="coerce"
    )
    model["nearest_water_km"] = pd.to_numeric(
        model["nearest_water_km"], errors="coerce"
    )
    model["elevation_m"] = pd.to_numeric(model["elevation_m"], errors="coerce")
    if "ip_name" in model.columns:
        model["community_name"] = model["ip_name"].fillna(
            "Unnamed indigenous community"
        )
    else:
        model["community_name"] = "Unnamed indigenous community"

    p = (
        data["province_model_metrics"].copy()
        if data["province_model_metrics"] is not None
        else None
    )
    if p is None:
        p = model.groupby("province", as_index=False).agg(
            avg_flood_exposure_score=("flood_exposure_score", "mean"),
            avg_topographic_susceptibility_score=(
                "topographic_susceptibility_score",
                "mean",
            ),
            avg_flood_score=("risk_score", "mean"),
            avg_network_distance_km=("nearest_tower_km", "mean"),
            avg_water_distance_km=("nearest_water_km", "mean"),
            avg_elevation_m=("elevation_m", "mean"),
            est_population_community=("est_population_community", "mean"),
        )

    st.markdown("#### Source-to-Feature Map")
    st.caption(
        "Unit of analysis: indigenous communities. Province charts below are summaries of those community records, not separate source datasets."
    )
    card_payload = [
        (
            "Ministry of Rural Development indigenous village records",
            "Community locations and indigenous context used to define the study population.",
        ),
        (
            "Registered communal lands",
            "Family counts and local population proxy for each indigenous community.",
        ),
        (
            "Flood risk inputs",
            "Province-level flood exposure signal that is later mapped back to communities.",
        ),
        (
            "OpenStreetMap water proxy points",
            "Water proximity proxy used as one flood-exposure signal.",
        ),
        (
            "OpenTopoData elevation",
            "How high or low the land is for each community location.",
        ),
        (
            "OpenStreetMap telecommunications infrastructure",
            "Nearest network-distance proxy for access comparison.",
        ),
    ]
    for row_payload in [card_payload[:3], card_payload[3:]]:
        row = st.columns(3)
        for idx, (title, desc) in enumerate(row_payload):
            with row[idx]:
                st.markdown(
                    f'<div class="source-card"><h5>{title}</h5><p>{desc}</p></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="insight-card"><div class="insight-title">Indigenous communities by province</div><div class="insight-sub">Which provinces contain the mapped indigenous communities?</div></div>',
            unsafe_allow_html=True,
        )
        by_prov = model["province"].value_counts().reset_index()
        by_prov.columns = ["province", "communities"]
        fig = px.bar(
            by_prov.sort_values("communities", ascending=True),
            x="communities",
            y="province",
            orientation="h",
            color="communities",
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(
            height=320, margin=dict(l=8, r=8, t=10, b=10), showlegend=False
        )
        fig.update_xaxes(title_text="Indigenous communities")
        fig.update_yaxes(title_text="Province")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This counts indigenous communities in scope by province. It is a summary of community records, not a count of all villages in Cambodia."
        )

    with c2:
        st.markdown(
            '<div class="insight-card"><div class="insight-title">Flood exposure vs land height</div><div class="insight-sub">A province-level bar comparison of flood exposure and raw land height</div></div>',
            unsafe_allow_html=True,
        )
        compare_df = p[
            [
                "province",
                "avg_flood_exposure_score",
                "avg_elevation_m",
            ]
        ].copy()
        compare_long = compare_df.melt(
            id_vars=["province"],
            value_vars=[
                "avg_flood_exposure_score",
                "avg_elevation_m",
            ],
            var_name="component",
            value_name="score",
        )
        compare_long["component"] = compare_long["component"].replace(
            {
                "avg_flood_exposure_score": "Flood exposure",
                "avg_elevation_m": "Land height (m)",
            }
        )
        fig = px.bar(
            compare_long.sort_values(["province", "component"]),
            x="province",
            y="score",
            color="component",
            barmode="group",
        )
        fig.update_layout(
            height=320,
            margin=dict(l=8, r=8, t=10, b=10),
            legend_title="",
            xaxis_title="Province",
            yaxis_title="Average score",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This bar chart compares flood exposure and the raw average land height side by side for each province."
        )

    with c3:
        st.markdown(
            '<div class="insight-card"><div class="insight-title">Connectivity vs land height</div><div class="insight-sub">A bar comparison of remoteness and how high or low the land is</div></div>',
            unsafe_allow_html=True,
        )
        context_df = p[
            ["province", "avg_network_distance_km", "avg_elevation_m"]
        ].copy()
        context_long = context_df.melt(
            id_vars=["province"],
            value_vars=["avg_network_distance_km", "avg_elevation_m"],
            var_name="metric",
            value_name="value",
        )
        context_long["metric"] = context_long["metric"].replace(
            {
                "avg_network_distance_km": "Distance to telecom proxy (km)",
                "avg_elevation_m": "Land height (m)",
            }
        )
        fig = px.bar(
            context_long,
            x="province",
            y="value",
            color="metric",
            barmode="group",
        )
        fig.update_layout(
            height=320,
            margin=dict(l=8, r=8, t=10, b=10),
            legend_title="",
            xaxis_title="Province",
            yaxis_title="Value",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This bar chart keeps the raw units visible, so the province differences are easier to read at a glance."
        )

    c4, c5 = st.columns(2)
    with c4:
        st.markdown(
            '<div class="insight-card"><div class="insight-title">Source availability</div><div class="insight-sub">How much supporting proxy data is available</div></div>',
            unsafe_allow_html=True,
        )
        water_count = (
            0 if data["osm_water_points"] is None else len(data["osm_water_points"])
        )
        tower_count = 0 if data["osm_towers"] is None else len(data["osm_towers"])
        m1, m2 = st.columns(2)
        with m1:
            st.metric("OpenStreetMap water proxy points", f"{water_count:,}")
        with m2:
            st.metric("OpenStreetMap telecom proxy points", f"{tower_count:,}")
        st.caption(
            "These counts show how much supporting source data exists. They are not used to rank communities by themselves."
        )

    with c5:
        st.markdown(
            '<div class="insight-card"><div class="insight-title">Model coverage</div><div class="insight-sub">How much data is modeled end-to-end</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="coverage-stack">
              <div class="coverage-row"><span>Communities ingested</span><strong>{len(model)}</strong></div>
              <div class="coverage-row"><span>Flood proxies added</span><strong>{len(model)}</strong></div>
              <div class="coverage-row"><span>Connectivity proxies added</span><strong>{len(model)}</strong></div>
              <div class="coverage-row"><span>Final scored</span><strong>{len(model)}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "The repeated 44s are intentional. They mean the same indigenous communities move through every step of the pipeline, so this is coverage, not a ranking graph."
        )


def render_page_model(data):
    st.subheader("Model Logic")
    if data["model_communities"] is None:
        st.warning(
            "Model outputs not found. Run `python build_digital_desert_model.py` first."
        )
        return

    thresholds = data["model_thresholds"]
    high_risk = _get_threshold(thresholds, "high_risk_cutoff")
    low_conn = _get_threshold(thresholds, "low_connectivity_km_cutoff")
    high_pop = _get_threshold(thresholds, "high_population_cutoff")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("High Risk Cutoff", f"{0 if high_risk is None else high_risk:,.2f}")
    with c2:
        st.metric(
            "Low Connectivity Cutoff", f"{0 if low_conn is None else low_conn:.2f} km"
        )
    with c3:
        st.metric(
            "High Population Cutoff",
            f"{0 if high_pop is None else high_pop:,.0f} people",
        )

    model_df = build_priority_frame(data["model_communities"], thresholds)
    model_df["community_name"] = model_df.get(
        "ip_name", pd.Series(index=model_df.index)
    ).fillna("Unnamed indigenous community")

    class_order = [
        "A: High Risk + Low Connectivity",
        "B: High Risk + Better Connectivity",
        "C: Low Risk + Low Connectivity + High Population",
        "D: Low Risk + Low Connectivity",
        "E: Lower Priority (Current Data)",
    ]
    class_descriptions = {
        "A: High Risk + Low Connectivity": "Highest concern: flood-prone and far from telecom proxy points.",
        "B: High Risk + Better Connectivity": "Flood-prone, but connectivity is comparatively better than class A.",
        "C: Low Risk + Low Connectivity + High Population": "Not flood-dominant, but connectivity and population pressure still matter.",
        "D: Low Risk + Low Connectivity": "Lower flood pressure, but access to telecom proxy points remains weak.",
        "E: Lower Priority (Current Data)": "Lower immediate priority in the current evidence set.",
    }
    class_counts = model_df["digital_desert_class"].value_counts().to_dict()
    visible_class_order = [
        name for name in class_order if class_counts.get(name, 0) > 0
    ]
    class_cols = st.columns(len(visible_class_order))
    for idx, class_name in enumerate(visible_class_order):
        with class_cols[idx]:
            st.markdown(
                f"""
                <div class="action-card">
                  <h5>{class_name}</h5>
                  <div style="font-size:30px; font-weight:700; color:#f8fafc; line-height:1.1; margin-bottom:6px;">{int(class_counts.get(class_name, 0))}</div>
                  <div style="font-size:13px; color:#cbd5e1;">{class_descriptions[class_name]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "These cards show only the classes that currently have communities. Classes with zero communities are hidden from the page."
    )

    support_df = model_df[
        model_df["digital_desert_class"].isin(visible_class_order)
    ].copy()
    support_df["digital_desert_class"] = pd.Categorical(
        support_df["digital_desert_class"], categories=visible_class_order, ordered=True
    )

    support_cols = st.columns(3)

    # show combined provinces as one root, then split by class (sunburst)
    support_df["province"] = support_df.get(
        "province", pd.Series(index=support_df.index)
    ).fillna("Unknown Province")
    n_provs = support_df["province"].nunique()
    root_label = f"{n_provs} provinces (combined)"
    support_df["_count"] = 1
    support_df["_root"] = root_label
    with support_cols[0]:
        sunburst = px.sunburst(
            support_df,
            path=["_root", "digital_desert_class"],
            values="_count",
            color="digital_desert_class",
            color_discrete_map=CLASS_COLORS,
            title="Communities: combined provinces → class",
            labels={"digital_desert_class": "Class"},
        )
        sunburst.update_traces(textinfo="label+value")
        sunburst.update_layout(height=320, margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(sunburst, use_container_width=True)

    risk_by_class_df = (
        support_df.groupby("digital_desert_class", observed=True)["risk_score"]
        .mean()
        .reset_index(name="avg_risk_score")
    )
    with support_cols[1]:
        risk_chart = px.bar(
            risk_by_class_df,
            x="digital_desert_class",
            y="avg_risk_score",
            color="digital_desert_class",
            color_discrete_map=CLASS_COLORS,
            title="Average composite flood score",
            labels={
                "digital_desert_class": "Class",
                "avg_risk_score": "Average composite flood score",
            },
        )
        risk_chart.update_layout(
            height=260, showlegend=False, margin=dict(t=50, l=20, r=20, b=20)
        )
        st.plotly_chart(risk_chart, use_container_width=True)

    access_by_class_df = (
        support_df.groupby("digital_desert_class", observed=True)["nearest_tower_km"]
        .mean()
        .reset_index(name="avg_nearest_tower_km")
    )
    with support_cols[2]:
        access_chart = px.bar(
            access_by_class_df,
            x="digital_desert_class",
            y="avg_nearest_tower_km",
            color="digital_desert_class",
            color_discrete_map=CLASS_COLORS,
            title="Average distance to telecom proxy",
            labels={
                "digital_desert_class": "Class",
                "avg_nearest_tower_km": "Average distance (km)",
            },
        )
        access_chart.update_layout(
            height=260, showlegend=False, margin=dict(t=50, l=20, r=20, b=20)
        )
        st.plotly_chart(access_chart, use_container_width=True)

    if high_risk is not None and low_conn is not None:
        scatter = px.scatter(
            model_df,
            x="nearest_tower_km",
            y="risk_score",
            color="digital_desert_class",
            size="est_population_community",
            color_discrete_map=CLASS_COLORS,
            hover_name="community_name",
            title="Interpretability Lens: Composite Flood Score vs Network Distance",
            labels={
                "nearest_tower_km": "Distance to Nearest Telecom Proxy (km)",
                "risk_score": "Composite flood score",
            },
        )
        scatter.add_vline(x=low_conn, line_dash="dash", line_color="#1d4ed8")
        scatter.add_hline(y=high_risk, line_dash="dash", line_color="#b91c1c")
        st.plotly_chart(scatter, use_container_width=True)

    st.caption(
        "The bubble plot below shows the actual decision logic: the top-right area combines high composite flood score and weak network access, which is the highest concern."
    )


def render_page_map(data):
    st.subheader("Digital Desert Risk Map")
    if data["model_communities"] is None:
        st.warning(
            "Model outputs not found. Run `python build_digital_desert_model.py` first."
        )
        return

    model = build_priority_frame(data["model_communities"], data["model_thresholds"])
    provinces = sorted(model["province"].dropna().unique().tolist())
    classes = sorted(model["digital_desert_class"].dropna().unique().tolist())

    if "map_prov" not in st.session_state:
        st.session_state.map_prov = provinces
    if "map_classes" not in st.session_state:
        st.session_state.map_classes = classes
    if "map_show_heat" not in st.session_state:
        st.session_state.map_show_heat = True
    if "map_show_points" not in st.session_state:
        st.session_state.map_show_points = True
    if "map_show_towers" not in st.session_state:
        st.session_state.map_show_towers = False
    if "map_priority_only" not in st.session_state:
        st.session_state.map_priority_only = False
    if "map_basemap" not in st.session_state:
        st.session_state.map_basemap = "Light"

    left, right = st.columns([1, 3], gap="medium")
    with left:
        st.markdown("### Map Controls")
        with st.form("map_controls_form"):
            all_prov = st.checkbox(
                "All Provinces", value=len(st.session_state.map_prov) == len(provinces)
            )
            prov_selected = st.multiselect(
                "Province",
                provinces,
                default=provinces if all_prov else st.session_state.map_prov,
            )
            all_cls = st.checkbox(
                "All Classes", value=len(st.session_state.map_classes) == len(classes)
            )
            cls_selected = st.multiselect(
                "Class",
                classes,
                default=classes if all_cls else st.session_state.map_classes,
            )
            show_heat = st.checkbox(
                "Flood heatmap", value=st.session_state.map_show_heat
            )
            show_points = st.checkbox(
                "Community points", value=st.session_state.map_show_points
            )
            show_towers = st.checkbox(
                "Cell towers", value=st.session_state.map_show_towers
            )
            priority_only = st.checkbox(
                "Priority hotspots only", value=st.session_state.map_priority_only
            )
            basemap = st.selectbox(
                "Basemap",
                ["Light", "Dark", "Street"],
                index=["Light", "Dark", "Street"].index(st.session_state.map_basemap),
            )
            hotspot_n = st.slider("Top hotspots", min_value=5, max_value=20, value=10)
            apply = st.form_submit_button("Update Map", use_container_width=True)

        if apply:
            st.session_state.map_prov = prov_selected if prov_selected else provinces
            st.session_state.map_classes = cls_selected if cls_selected else classes
            st.session_state.map_show_heat = show_heat
            st.session_state.map_show_points = show_points
            st.session_state.map_show_towers = show_towers
            st.session_state.map_priority_only = priority_only
            st.session_state.map_basemap = basemap
        else:
            show_heat = st.session_state.map_show_heat
            show_points = st.session_state.map_show_points
            show_towers = st.session_state.map_show_towers
            priority_only = st.session_state.map_priority_only
            basemap = st.session_state.map_basemap
            hotspot_n = 10

    filtered = model[
        model["province"].isin(st.session_state.map_prov)
        & model["digital_desert_class"].isin(st.session_state.map_classes)
    ].copy()
    if priority_only and not filtered.empty:
        filtered = filtered.sort_values("priority_score", ascending=False).head(
            hotspot_n
        )
    if filtered.empty:
        st.info("No communities match the current map filters.")
        return

    coords = [_get_point_coords(g) for g in filtered.geometry]
    coords = [c for c in coords if c is not None]
    center = (
        [
            sum(c[0] for c in coords) / len(coords),
            sum(c[1] for c in coords) / len(coords),
        ]
        if coords
        else [13.5, 106.8]
    )

    tile_name = (
        "CartoDB dark_matter"
        if basemap == "Dark"
        else "OpenStreetMap"
        if basemap == "Street"
        else "CartoDB positron"
    )
    m = folium.Map(location=center, zoom_start=8, tiles=tile_name, max_bounds=True)
    comm_layer = folium.FeatureGroup(name="Communities", show=show_points).add_to(m)
    tower_layer = folium.FeatureGroup(
        name="Telecommunications proxy points", show=show_towers
    ).add_to(m)
    heat_layer = folium.FeatureGroup(name="Flood Heatmap", show=show_heat).add_to(m)

    if show_heat:
        heat_data = []
        for _, row in filtered.iterrows():
            c = _get_point_coords(row.geometry)
            if c is None:
                continue
            heat_data.append([c[0], c[1], _safe_number(row.get("risk_score")) or 0.0])
        if heat_data:
            HeatMap(
                heat_data,
                radius=26,
                blur=20,
                min_opacity=0.35,
                gradient={
                    0.1: "#60a5fa",
                    0.35: "#22d3ee",
                    0.6: "#facc15",
                    0.85: "#fb923c",
                    1.0: "#ef4444",
                },
            ).add_to(heat_layer)

    if show_points:
        for _, row in filtered.iterrows():
            c = _get_point_coords(row.geometry)
            if c is None:
                continue
            est_pop = _safe_number(row.get("est_population_community"))
            radius = 4 if est_pop is None else max(4, min(10, 3 + est_pop / 300))
            class_color = CLASS_COLORS.get(
                row.get("digital_desert_class", ""), "#1e40af"
            )
            popup = (
                f"<b>{row.get('ip_name', 'Indigenous community')}</b><br>"
                f"Province: {row.get('province', 'N/A')}<br>"
                f"Flood score: {0 if _safe_number(row.get('risk_score')) is None else _safe_number(row.get('risk_score')):,.0f}<br>"
                f"Tower distance: {0 if _safe_number(row.get('nearest_tower_km')) is None else _safe_number(row.get('nearest_tower_km')):.1f} km<br>"
                f"Class: {row.get('digital_desert_class', 'N/A')}"
            )
            folium.CircleMarker(
                location=c,
                radius=radius,
                color="#ffffff",
                fill=True,
                fill_color=class_color,
                fill_opacity=0.95,
                weight=1.2,
                popup=folium.Popup(popup, max_width=280),
            ).add_to(comm_layer)

    if show_towers and data["osm_towers"] is not None and len(data["osm_towers"]) > 0:
        for _, row in data["osm_towers"].iterrows():
            c = _get_point_coords(row.geometry)
            if c is None:
                continue
            folium.Marker(
                location=c,
                icon=folium.Icon(color="purple", icon="signal", prefix="fa"),
                tooltip="Cell tower proxy",
            ).add_to(tower_layer)

    legend_html = """
    <div style="position: fixed; bottom: 18px; left: 18px; z-index: 9999; background: rgba(12, 18, 31, 0.92); color: #e2e8f0; border: 1px solid #334155; border-radius: 10px; padding: 10px 12px; font-size: 12px; line-height: 1.35; min-width: 220px;">
      <div style="font-weight:700; margin-bottom:6px;">Map Legend</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:99px;background:#b91c1c;margin-right:6px;"></span>Community class A (highest concern)</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:99px;background:#ea580c;margin-right:6px;"></span>Community class B</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:99px;background:#ca8a04;margin-right:6px;"></span>Community class C</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:99px;background:#2563eb;margin-right:6px;"></span>Community class D</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:99px;background:#15803d;margin-right:6px;"></span>Community class E</div>
    <div style="margin-top:6px;">Purple icon: cell tower proxy</div>
      <div>Heat layer: higher composite flood score = warmer colors</div>
    </div>
    """
    m.get_root().add_child(folium.Element(legend_html))

    with right:
        st_folium(m, width=1200, height=700)


def render_page_architecture(data):
    st.subheader("Architecture")
    st.markdown(
        "#### One End-to-End Flow (Data → Preparation → Modeling → Policy Action)"
    )

    st.markdown(
        """
        <div class="flow-wrap">
          <div class="flow-row">
            <div class="flow-node"><h5>Source Layer</h5>Ministry of Rural Development indigenous village records, communal lands, flood references, OpenStreetMap telecommunications and water proxy points, and OpenTopoData elevation.</div>
            <div class="flow-arrow">-&gt;</div>
            <div class="flow-node"><h5>Data Preparation Layer</h5>Coordinate reference system alignment, null handling, deduplication, distance enrichment, standardized community feature store.</div>
            <div class="flow-arrow">-&gt;</div>
            <div class="flow-node"><h5>Modeling Layer</h5>Rule-based thresholds for flood, low connectivity, and population pressure to produce classes A-E.</div>
            <div class="flow-arrow">-&gt;</div>
            <div class="flow-node"><h5>Decision Layer</h5>Hotspot prioritization, phased intervention packages, and transparent monitoring indicators.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data["model_communities"] is not None:
        model = build_priority_frame(
            data["model_communities"], data["model_thresholds"]
        )
        raw_count = len(model)
        etl_ready = len(
            model.dropna(
                subset=["risk_score", "nearest_tower_km", "est_population_community"]
            )
        )
        scored = len(model.dropna(subset=["priority_score"]))
        hotspots = model[
            model["priority_score"] >= model["priority_score"].quantile(0.7)
        ]
        hotspot_count = len(hotspots)

        left, right = st.columns([1.4, 1], gap="medium")
        with left:
            pipeline_df = pd.DataFrame(
                {
                    "Stage": [
                        "Source records",
                        "Data prepared",
                        "Rule scored",
                        "Priority portfolio",
                    ],
                    "Count": [raw_count, etl_ready, scored, hotspot_count],
                }
            )
            pipeline_chart = px.bar(
                pipeline_df,
                x="Stage",
                y="Count",
                color="Stage",
                text="Count",
                title="Pipeline throughput",
                color_discrete_sequence=["#1d4ed8", "#0ea5e9", "#14b8a6", "#f59e0b"],
            )
            pipeline_chart.update_layout(
                height=360, margin=dict(l=8, r=8, t=48, b=8), showlegend=False
            )
            pipeline_chart.update_traces(textposition="outside")
            st.plotly_chart(pipeline_chart, use_container_width=True)

        with right:
            class_counts = (
                model["digital_desert_class"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("class")
                .reset_index(name="count")
            )
            class_bar = px.bar(
                class_counts.sort_values("count", ascending=True),
                x="count",
                y="class",
                orientation="h",
                color="class",
                text="count",
                color_discrete_map=CLASS_COLORS,
                title="Modeled class mix",
                labels={"count": "Communities", "class": "Class"},
            )
            class_bar.update_layout(
                height=360, margin=dict(l=8, r=8, t=48, b=8), showlegend=False
            )
            class_bar.update_traces(textposition="outside")
            st.plotly_chart(class_bar, use_container_width=True)

    st.markdown(
        """
        <div class="mini-note">
        <b>Why this architecture works:</b> each policy decision can be audited back to one transformed feature and one explicit rule, which keeps the model interpretable for committee review and implementation teams.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_actions(data):
    st.subheader("Action Plan")
    if data["model_communities"] is None:
        st.warning(
            "Model outputs not found. Run `python build_digital_desert_model.py` first."
        )
        return

    model = build_priority_frame(data["model_communities"], data["model_thresholds"])
    shortlist = model.sort_values(["priority_score"], ascending=False).head(15).copy()
    if "ip_name" in shortlist.columns:
        shortlist["community"] = shortlist["ip_name"].fillna(
            "Unnamed indigenous community"
        )
    else:
        shortlist["community"] = "Unnamed indigenous community"

    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1:
        # Simplified: show normalized priority (0-100) so committee reads a single scale
        shortlist["priority_score_norm"] = (
            shortlist["priority_score"] / shortlist["priority_score"].max() * 100
        )
        display_df = shortlist.sort_values("priority_score_norm", ascending=True).copy()
        fig = px.bar(
            display_df,
            x="priority_score_norm",
            y="community",
            color="digital_desert_class",
            color_discrete_map=CLASS_COLORS,
            orientation="h",
            text=display_df["priority_score_norm"].round(0),
            labels={
                "priority_score_norm": "Priority (0-100)",
                "community": "Indigenous community",
            },
            title="Top Priority Communities (0-100)",
        )
        fig.update_layout(
            height=520,
            margin=dict(l=8, r=8, t=48, b=8),
            legend_title="Class",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # More actionable: show estimated people who could be reached by intervening in each community
        people_df = display_df[
            ["community", "est_population_community", "digital_desert_class"]
        ].copy()
        people_df["est_population_community"] = pd.to_numeric(
            people_df["est_population_community"], errors="coerce"
        ).fillna(0)
        people_chart = px.bar(
            people_df,
            x="est_population_community",
            y="community",
            orientation="h",
            color="digital_desert_class",
            color_discrete_map=CLASS_COLORS,
            labels={
                "est_population_community": "Estimated people",
                "community": "Indigenous community",
            },
            title="People potentially reached (estimated)",
        )
        people_chart.update_layout(
            height=520, margin=dict(l=8, r=8, t=48, b=8), legend_title="Class"
        )
        people_chart.update_traces(texttemplate="%{x:.0f}", textposition="outside")
        st.plotly_chart(people_chart, use_container_width=True)

    st.markdown("#### Phased Intervention Plan")
    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        st.markdown(
            """
            <div class="phase-card">
              <h5>Phase 1 (0-3 months)</h5>
                            Emergency connectivity and flood-alert readiness for top A and B indigenous communities with the longest distance to telecom proxy points.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="phase-card">
              <h5>Phase 2 (4-9 months)</h5>
                            Add shared internet points, school and health digital access hubs, and targeted digital-skills support.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """
            <div class="phase-card">
              <h5>Phase 3 (10-18 months)</h5>
                            Expand to the next-priority indigenous communities using monitored improvements in risk exposure and access distance.
            </div>
            """,
            unsafe_allow_html=True,
        )

    prov_actions = (
        shortlist.groupby("province", as_index=False)
        .agg(
            hotspot_count=("community", "count"),
            avg_priority=("priority_score", "mean"),
            avg_distance_km=("nearest_tower_km", "mean"),
        )
        .sort_values("avg_priority", ascending=False)
    )
    fig2 = px.bar(
        prov_actions,
        x="province",
        y="hotspot_count",
        color="avg_priority",
        color_continuous_scale="Sunset",
        title="Where to Start First (Province Portfolio)",
        hover_data=["avg_distance_km"],
    )
    fig2.update_layout(height=340, margin=dict(l=8, r=8, t=48, b=8))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
        <div class="mini-note">
        <b>How the solution architecture executes this plan:</b> the ETL layer refreshes community features, the rule-based model recalculates class and priority scores, and the dashboard updates province portfolios so decision-makers can re-sequence interventions using the same transparent logic every cycle.
        </div>
        """,
        unsafe_allow_html=True,
    )


style_dashboard()

try:
    data = load_data()
except Exception as exc:
    st.error(
        "Data loading failed. Please verify `processed_data/` and rerun ETL/model scripts."
    )
    st.exception(exc)
    st.stop()

render_header()
if "page" not in st.session_state:
    st.session_state.page = "Overview"

st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
nav_items = [
    ("Overview", "Overview"),
    ("Data Sources", "Data Sources"),
    ("Model Logic", "Model Logic"),
    ("Risk Map", "Risk Map"),
    ("Action Plan", "Action Plan"),
    ("Architecture", "Architecture"),
]
nav_cols = st.columns(len(nav_items))
for idx, (label, value) in enumerate(nav_items):
    current_page = st.session_state.get("page", "Overview")
    btn_type = "primary" if current_page == value else "secondary"
    if nav_cols[idx].button(
        label, key=f"nav_btn_{idx}", use_container_width=True, type=btn_type
    ):
        st.session_state.page = value
st.markdown("</div>", unsafe_allow_html=True)

current_page = st.session_state.get("page", "Overview")
if current_page == "Overview":
    render_page_start(data)
elif current_page == "Data Sources":
    render_page_data(data)
elif current_page == "Model Logic":
    render_page_model(data)
elif current_page == "Risk Map":
    render_page_map(data)
elif current_page == "Architecture":
    render_page_architecture(data)
else:
    render_page_actions(data)

st.divider()
st.caption(
    "Hackathon project: Bridging the Digital Divide by Uncovering Digital Deserts. "
    "Sources include Ministry of Rural Development records, flood-risk inputs, OpenStreetMap telecommunications proxy points, and team data pipeline outputs."
)
