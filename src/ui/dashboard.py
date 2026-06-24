"""
Streamlit Dashboard for Airbnb Tokyo Analytics
"""

import os
import re
import sys
import glob
import json
import time
import html as _html
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import folium_static
from textblob import TextBlob

# Translation dependencies — optional but required for auto-translate feature
try:
    from langdetect import detect as _detect_lang
    from deep_translator import GoogleTranslator
    _TRANSLATION_AVAILABLE = True
except ImportError:
    _TRANSLATION_AVAILABLE = False

from src.analytics.eda import EDAAnalyzer
from src.analytics.clustering import GeospatialClusterer
from src.agent.agent import AirbnbAgent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb Tokyo Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
CORAL = "#FF5A5F"
TEAL  = "#00A699"
AMBER = "#FFB400"


def _chart(fig: go.Figure) -> go.Figure:
    """Transparent dark styling applied to every Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="rgba(240,240,240,0.85)", size=12),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.12)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.12)",
        ),
        margin=dict(t=10, b=10, l=0, r=0),
    )
    return fig


def _clean_review(text: str) -> str:
    """Strip HTML tags, decode entities, and normalise whitespace."""
    text = _html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)   # remove <br/>, <b>, etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner=False)
def _translate(text: str) -> tuple[str, str, bool]:
    """Clean HTML, detect language, translate to English if needed.

    Returns (display_text, iso_lang_code, was_translated).
    """
    clean = _clean_review(text)
    if not _TRANSLATION_AVAILABLE or not clean:
        return clean, "en", False
    try:
        lang = _detect_lang(clean[:500])
        if lang != "en":
            translated = GoogleTranslator(source="auto", target="en").translate(
                clean[:500]
            )
            return _clean_review(translated or clean), lang, True
    except Exception:
        pass
    return clean, "en", False


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* ── Metric cards ───────────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background    : rgba(255,255,255,0.05);
    border        : 1px solid rgba(255,255,255,0.09);
    border-radius : 13px;
    padding       : 1.3rem;
    box-shadow    : 0 4px 21px rgba(0,0,0,0.35);
    transition    : box-shadow 0.2s;
  }
  [data-testid="stMetric"]:hover { box-shadow: 0 6px 34px rgba(255,90,95,0.15); }
  [data-testid="stMetricLabel"] {
    color: rgba(240,240,240,0.5); font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.07em;
  }
  [data-testid="stMetricValue"] { font-weight: 700; font-size: 1.7rem; }
  [data-testid="stMetricDelta"]  { font-size: 0.75rem; }

  /* ── Sidebar nav buttons ────────────────────────────────────────────── */
  /* inactive */
  [data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background    : transparent !important;
    color         : rgba(240,240,240,0.55) !important;
    border        : none !important;
    text-align    : left !important;
    font-weight   : 400 !important;
    padding-left  : 0.9rem !important;
    border-radius : 8px !important;
    transition    : background 0.15s, color 0.15s;
  }
  [data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
    background : rgba(255,255,255,0.07) !important;
    color      : rgba(240,240,240,0.95) !important;
  }
  /* active */
  [data-testid="stSidebar"] [data-testid="baseButton-primary"] {
    background    : rgba(255,90,95,0.12) !important;
    color         : #FF5A5F !important;
    border        : none !important;
    border-left   : 3px solid #FF5A5F !important;
    text-align    : left !important;
    font-weight   : 600 !important;
    border-radius : 0 8px 8px 0 !important;
  }
  [data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
    background: rgba(255,90,95,0.2) !important;
  }

  /* ── Download + primary buttons ─────────────────────────────────────── */
  .stDownloadButton > button {
    background: #FF5A5F !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; width: 100% !important;
    letter-spacing: 0.02em !important;
  }
  .stDownloadButton > button:hover { background: #e04a4f !important; }

  /* ── Section micro-labels ────────────────────────────────────────────── */
  .section-label {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: rgba(240,240,240,0.3);
    margin: 1.2rem 0 0.3rem 0;
  }

  hr { border-color: rgba(255,255,255,0.08) !important; margin: 1.3rem 0; }
  div[data-testid="stChatMessage"] { border-radius: 13px; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet("data/processed/listings_processed.parquet")


@st.cache_data
def load_reviews_with_sentiment() -> pd.DataFrame:
    try:
        reviews = pd.read_parquet("data/processed/reviews_processed.parquet")
        if len(reviews) > 5000:
            reviews = reviews.sample(5000, random_state=42)

        def _polarity(text: str) -> float:
            try:
                return TextBlob(str(text)).sentiment.polarity
            except Exception:
                return 0.0

        reviews["sentiment"] = reviews["comments"].apply(_polarity)
        reviews["sentiment_label"] = pd.cut(
            reviews["sentiment"],
            bins=[-1, -0.1, 0.1, 1],
            labels=["Negative", "Neutral", "Positive"],
        )
        return reviews
    except Exception as e:
        st.warning(f"Could not load reviews: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_agent() -> AirbnbAgent:
    return AirbnbAgent()


# ── Session state ──────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_data()
eda = EDAAnalyzer(df)
clusterer = GeospatialClusterer(df)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏠 Tokyo Airbnb")
    st.markdown("---")

    # Icon-button navigation — Material icons (Streamlit 1.36+)
    NAV_ITEMS = [
        (":material/dashboard:",    "Dashboard"),
        (":material/smart_toy:",    "AI Assistant"),
        (":material/map:",          "Maps"),
        (":material/table_chart:",  "Data Explorer"),
        (":material/reviews:",      "Sentiment Analysis"),
    ]
    for icon, label in NAV_ITEMS:
        is_active = st.session_state.page == label
        if st.button(
            label,
            icon=icon,
            key=f"nav_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = label
            st.rerun()

    st.markdown("---")
    st.markdown('<p class="section-label">Filters</p>', unsafe_allow_html=True)

    price_cap = min(int(df["price"].quantile(0.98)), 200_000)
    price_range = st.slider(
        "Price (JPY)",
        int(df["price"].min()),
        price_cap,
        (int(df["price"].min()), price_cap),
        format="¥%d",
    )
    room_types = st.multiselect(
        "Room type",
        options=df["room_type"].unique().tolist(),
        default=df["room_type"].unique().tolist(),
    )
    with st.expander("More filters"):
        all_neighborhoods = sorted(df["neighbourhood_cleansed"].unique().tolist())
        selected_neighborhoods = st.multiselect("Neighbourhood", all_neighborhoods)

    filtered_df = df[df["price"].between(*price_range) & df["room_type"].isin(room_types)]
    if selected_neighborhoods:
        filtered_df = filtered_df[
            filtered_df["neighbourhood_cleansed"].isin(selected_neighborhoods)
        ]

    st.caption(f"**{len(filtered_df):,}** listings match filters")

    st.markdown("---")
    st.markdown('<p class="section-label">Export</p>', unsafe_allow_html=True)

    @st.cache_data
    def to_csv(data: pd.DataFrame) -> bytes:
        return data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered CSV",
        data=to_csv(filtered_df),
        file_name=f"airbnb_tokyo_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
    st.markdown("---")
    st.caption("Source: Inside Airbnb Tokyo")


page = st.session_state.page


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Tokyo Airbnb Market")
    st.caption(f"Showing **{len(filtered_df):,}** of {len(df):,} listings")

    # ── KPIs ───────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings", f"{len(filtered_df):,}")
    c2.metric(
        "Average Price",
        f"¥{filtered_df['price'].mean():,.0f}",
        delta=f"¥{filtered_df['price'].mean() - df['price'].mean():+,.0f} vs all",
    )
    c3.metric("Median Price", f"¥{filtered_df['price'].median():,.0f}")
    c4.metric("Neighbourhoods", filtered_df["neighbourhood_cleansed"].nunique())

    st.markdown("---")

    # ── Section 1: Market Overview ─────────────────────────────────────────────
    st.markdown("#### Market Overview")

    # [1.618 : 1] — histogram is the primary story, wider
    col_l, col_r = st.columns([1.618, 1])

    with col_l:
        st.subheader("Price Distribution")
        plot_data = filtered_df[filtered_df["price"] < 100_000]
        mean_p   = plot_data["price"].mean()
        median_p = plot_data["price"].median()
        fig = px.histogram(
            plot_data, x="price", nbins=60,
            color_discrete_sequence=[CORAL],
            labels={"price": "Price (JPY)", "count": "Listings"},
        )
        fig.add_vline(
            x=mean_p, line_dash="dash", line_color=AMBER, line_width=1.5,
            annotation_text=f"Mean  ¥{mean_p:,.0f}",
            annotation_font_color=AMBER,
            annotation_position="top right",
        )
        fig.add_vline(
            x=median_p, line_dash="dot", line_color=TEAL, line_width=1.5,
            annotation_text=f"Median  ¥{median_p:,.0f}",
            annotation_font_color=TEAL,
            annotation_position="top left",
        )
        st.plotly_chart(_chart(fig), use_container_width=True)

    with col_r:
        st.subheader("Avg Price by Room Type")
        room_avg = (
            filtered_df.groupby("room_type")["price"]
            .mean().sort_values().reset_index()
        )
        fig = px.bar(
            room_avg, x="price", y="room_type", orientation="h",
            color_discrete_sequence=[CORAL],
            labels={"price": "Average Price (JPY)", "room_type": ""},
        )
        st.plotly_chart(_chart(fig), use_container_width=True)

    st.markdown("---")

    # ── Section 2: Neighbourhood Analysis ─────────────────────────────────────
    st.markdown("#### Neighbourhood Analysis")

    # [1 : 1.618] — scatter (richer info) gets more space
    col_a, col_b = st.columns([1, 1.618])

    with col_a:
        st.subheader("Listing Volume & Price")
        st.caption("Box size = number of listings · Colour = avg price")
        nb_tree = (
            filtered_df.groupby("neighbourhood_cleansed")
            .agg(avg_price=("price", "mean"), listings=("price", "count"))
            .reset_index()
        )
        fig = px.treemap(
            nb_tree,
            path=["neighbourhood_cleansed"],
            values="listings",
            color="avg_price",
            color_continuous_scale=[[0, "#2c1415"], [0.5, "#c43b3f"], [1, CORAL]],
            custom_data=["avg_price", "listings"],
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Avg Price: ¥%{customdata[0]:,.0f}<br>"
                "Listings: %{customdata[1]:,}"
                "<extra></extra>"
            ),
            textfont_size=11,
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Price vs Rating")
        st.caption(
            "Each bubble = one neighbourhood · Size = listing count · "
            "Top-left quadrant = best value"
        )
        scatter_data = (
            filtered_df.groupby("neighbourhood_cleansed")
            .agg(
                avg_price=("price", "mean"),
                avg_rating=("review_scores_rating", "mean"),
                listings=("id", "count"),
            )
            .dropna(subset=["avg_rating"])
            .reset_index()
        )
        med_price  = scatter_data["avg_price"].median()
        med_rating = scatter_data["avg_rating"].median()

        fig = px.scatter(
            scatter_data,
            x="avg_price", y="avg_rating", size="listings",
            color="avg_price",
            hover_name="neighbourhood_cleansed",
            color_continuous_scale=[[0, "#7a1f22"], [1, CORAL]],
            size_max=50,
            labels={
                "avg_price":  "Avg Price (JPY)",
                "avg_rating": "Avg Rating",
                "listings":   "Listings",
            },
        )
        fig.add_hline(
            y=med_rating, line_dash="dot", line_color="rgba(255,255,255,0.18)",
            annotation_text="Median rating",
            annotation_font_color="rgba(255,255,255,0.35)",
            annotation_position="bottom right",
        )
        fig.add_vline(
            x=med_price, line_dash="dot", line_color="rgba(255,255,255,0.18)",
            annotation_text="Median price",
            annotation_font_color="rgba(255,255,255,0.35)",
            annotation_position="top right",
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(_chart(fig), use_container_width=True)

    st.markdown("---")

    # ── Section 3: Price Spread ────────────────────────────────────────────────
    st.markdown("#### Price Spread — Top Neighbourhoods")
    st.caption("Whiskers = min/max · Box = IQR · Line = median · Hover for values")

    top_n_box = st.slider("Neighbourhoods to show", 5, 20, 10, key="box_slider")
    top_nbs = (
        filtered_df.groupby("neighbourhood_cleansed")["price"]
        .mean().sort_values(ascending=False).head(top_n_box).index.tolist()
    )
    box_data = filtered_df[
        filtered_df["neighbourhood_cleansed"].isin(top_nbs)
        & (filtered_df["price"] < 100_000)
    ].copy()
    order = (
        box_data.groupby("neighbourhood_cleansed")["price"]
        .median().sort_values(ascending=False).index.tolist()
    )
    fig = px.box(
        box_data,
        x="neighbourhood_cleansed", y="price",
        category_orders={"neighbourhood_cleansed": order},
        color_discrete_sequence=[CORAL],
        labels={"neighbourhood_cleansed": "", "price": "Price (JPY)"},
        points=False,
    )
    fig.update_layout(xaxis_tickangle=-34)
    st.plotly_chart(_chart(fig), use_container_width=True)

    st.markdown("---")

    # ── Section 4: Price Predictor ─────────────────────────────────────────────
    st.markdown("#### Price Predictor")
    st.caption("Rule-based estimate from listing attributes")

    pc1, pc2, pc3 = st.columns([1, 1.618, 1])
    with pc1:
        accommodates = st.number_input("Accommodates", 1, 10, 2)
        bathrooms    = st.number_input("Bathrooms", 0.5, 5.0, 1.0, 0.5)
    with pc2:
        bedrooms     = st.number_input("Bedrooms", 0, 5, 1)
        beds         = st.number_input("Beds", 0, 10, 2)
        availability = st.slider("Availability (days/year)", 0, 365, 200)
    with pc3:
        pred_room_type = st.selectbox("Room Type", df["room_type"].unique())
        st.markdown("")
        predict = st.button("Estimate Price", type="primary", use_container_width=True)

    if predict:
        base  = 15_000
        base += (accommodates - 2) * 3_000
        base += (bathrooms - 1)    * 5_000
        base += bedrooms           * 8_000
        base += beds               * 2_000
        base += (365 - availability) * 20
        multipliers = {
            "Entire home/apt": 1.0, "Private room": 0.6,
            "Shared room": 0.3,     "Hotel room": 0.8,
        }
        base *= multipliers.get(pred_room_type, 1.0)
        final_price = int(base * (0.9 + 0.2 * random.random()))
        st.success(f"Estimated price: **¥{final_price:,} per night**")
        st.caption("Based on Tokyo median adjusted for listing attributes")


# ═══════════════════════════════════════════════════════════════════════════════
# AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "AI Assistant":
    st.title("AI Assistant")
    st.caption("Ask questions about the Tokyo Airbnb market in plain English.")

    agent = load_agent()

    PRESETS = [
        "What's the average price in Tokyo?",
        "Which neighbourhoods are most expensive?",
        "What are the cheapest neighbourhoods?",
        "Recommend neighbourhoods under ¥20,000",
        "Show the price distribution by room type",
        "What's the average review score?",
    ]

    st.markdown("**Quick questions**")
    cols = st.columns(3)
    preset_clicked = None
    for i, q in enumerate(PRESETS):
        if cols[i % 3].button(q, key=f"preset_{i}", use_container_width=True):
            preset_clicked = q

    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for plot_path in msg.get("plots", []):
                if os.path.exists(plot_path):
                    st.image(plot_path, use_container_width=True)

    user_input = st.chat_input("Ask a question...") or preset_clicked

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                call_time = time.time()
                response  = agent.ask(user_input)
            st.markdown(response)

            # Show any plots the agent generated during this call
            new_plots = sorted(
                [
                    p for p in glob.glob("outputs/*.png")
                    if os.path.getmtime(p) >= call_time - 1
                ],
                key=os.path.getmtime,
            )
            for plot_path in new_plots:
                st.image(plot_path, use_container_width=True)

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "plots": new_plots}
        )

    if st.session_state.messages:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAPS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Maps":
    from folium.plugins import HeatMap, MarkerCluster

    st.title("Geospatial Map")

    # ── KPIs ───────────────────────────────────────────────────────────────────
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Listings in view", f"{len(filtered_df):,}")
    mc2.metric("Neighbourhoods", filtered_df["neighbourhood_cleansed"].nunique())
    mc3.metric("Avg Price", f"¥{filtered_df['price'].mean():,.0f}")

    st.caption(
        "**Layers** — toggle in the top-right corner of the map: "
        "Choropleth (avg price by district) · Density heatmap · Individual listings"
    )

    # ── Build map ───────────────────────────────────────────────────────────────
    with st.spinner("Building map…"):
        center = [df["latitude"].mean(), df["longitude"].mean()]
        m = folium.Map(
            location=center,
            zoom_start=11,
            tiles="CartoDB dark_matter",
        )

        # Layer 1 — Neighbourhood choropleth (avg price) with hover tooltip
        nb_avg = df.groupby("neighbourhood_cleansed")["price"].mean().reset_index()
        nb_avg.columns = ["neighbourhood", "avg_price"]
        nb_count = df.groupby("neighbourhood_cleansed")["id"].count().rename("count")
        price_lookup = nb_avg.set_index("neighbourhood")["avg_price"].to_dict()
        count_lookup = nb_count.to_dict()

        try:
            with open("data/raw/neighbourhoods.geojson") as _f:
                geo_data = json.load(_f)

            # Enrich GeoJSON features so the tooltip can read them
            for feat in geo_data["features"]:
                nb = feat["properties"].get("neighbourhood", "")
                avg_p = price_lookup.get(nb)
                feat["properties"]["avg_price_fmt"] = (
                    f"¥{avg_p:,.0f}" if avg_p is not None else "No data"
                )
                feat["properties"]["listing_count"] = (
                    f"{count_lookup.get(nb, 0):,}"
                )

            choropleth = folium.Choropleth(
                geo_data=geo_data,
                data=nb_avg,
                columns=["neighbourhood", "avg_price"],
                key_on="feature.properties.neighbourhood",
                fill_color="YlOrRd",
                fill_opacity=0.65,
                line_opacity=0.3,
                legend_name="Avg Nightly Price (JPY)",
                name="Avg Price by Neighbourhood",
                highlight=True,
            )
            choropleth.add_to(m)

            choropleth.geojson.add_child(
                folium.features.GeoJsonTooltip(
                    fields=["neighbourhood", "avg_price_fmt", "listing_count"],
                    aliases=["District", "Avg Price", "Listings"],
                    style=(
                        "background:#1c1c1c;color:#f0f0f0;font-family:Arial;"
                        "font-size:12px;padding:8px 12px;border-radius:8px;"
                        "border:1px solid rgba(255,255,255,0.2);"
                    ),
                    localize=True,
                )
            )
        except Exception as e:
            st.warning(f"Choropleth unavailable: {e}")

        # Layer 2 — Listing density heatmap (off by default)
        heat_coords = (
            filtered_df.dropna(subset=["latitude", "longitude"])
            [["latitude", "longitude"]].values.tolist()
        )
        HeatMap(
            heat_coords,
            name="Listing Density",
            radius=12,
            blur=10,
            show=False,
        ).add_to(m)

        # Layer 3 — MarkerCluster with rich popups (off by default)
        marker_group = folium.FeatureGroup(name="Individual Listings", show=False)
        mc = MarkerCluster().add_to(marker_group)

        sample = (
            filtered_df.sample(n=min(2_000, len(filtered_df)), random_state=42)
            .dropna(subset=["latitude", "longitude"])
        )
        for _, row in sample.iterrows():
            rating     = row.get("review_scores_rating", None)
            rating_str = f"⭐ {rating:.1f}" if pd.notna(rating) else "No rating yet"
            avail      = int(row["availability_365"]) if pd.notna(row.get("availability_365")) else "—"
            popup_html = f"""
                <div style="font-family:Arial;font-size:12px;min-width:190px;padding:4px">
                    <b style="font-size:14px;color:#FF5A5F">
                        ¥{row['price']:,.0f}
                        <span style="font-size:11px;color:#888"> / night</span>
                    </b><br>
                    <span style="color:#666;font-size:11px">{row.get('room_type', '')}</span>
                    <hr style="margin:5px 0;border-color:#ddd">
                    {rating_str}&nbsp;·&nbsp;📍 {row.get('neighbourhood_cleansed', '')}<br>
                    <span style="color:#888">📅 {avail} days available / year</span>
                </div>
            """
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                popup=folium.Popup(popup_html, max_width=240),
                color="#FF5A5F",
                fill=True,
                fill_color="#FF5A5F",
                fill_opacity=0.7,
                weight=1,
            ).add_to(mc)

        marker_group.add_to(m)

        # Layer control
        folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # ── Guide panel + map side by side ─────────────────────────────────────────
    guide_col, map_col = st.columns([1, 2.618])

    with guide_col:
        st.markdown("""
<div style="
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 13px;
    padding: 1.3rem 1.1rem;
    font-family: Arial, sans-serif;
    color: #f0f0f0;
    height: 100%;
">

  <p style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;
            color:rgba(240,240,240,0.35);margin:0 0 1rem 0">Map Layers</p>

  <!-- Layer 1 -->
  <p style="font-size:0.82rem;font-weight:600;margin:0 0 0.35rem 0">
    ① Avg Price by Neighbourhood
  </p>
  <p style="font-size:0.72rem;color:rgba(240,240,240,0.55);margin:0 0 0.4rem 0">
    Filled district boundaries. Darker red = more expensive.
  </p>
  <div style="height:10px;border-radius:5px;
              background:linear-gradient(to right,#ffffb2,#fecc5c,#fd8d3c,#e31a1c);
              margin-bottom:0.25rem"></div>
  <div style="display:flex;justify-content:space-between;
              font-size:0.68rem;color:rgba(240,240,240,0.4);margin-bottom:1.2rem">
    <span>Budget</span><span>Luxury</span>
  </div>

  <!-- Layer 2 -->
  <p style="font-size:0.82rem;font-weight:600;margin:0 0 0.35rem 0">
    ② Listing Density
  </p>
  <p style="font-size:0.72rem;color:rgba(240,240,240,0.55);margin:0 0 0.4rem 0">
    Heat glow shows where listings are concentrated.
  </p>
  <div style="height:10px;border-radius:5px;
              background:linear-gradient(to right,#313695,#4575b4,#74add1,#abd9e9,
                          #fee090,#fdae61,#f46d43,#d73027);
              margin-bottom:0.25rem"></div>
  <div style="display:flex;justify-content:space-between;
              font-size:0.68rem;color:rgba(240,240,240,0.4);margin-bottom:1.2rem">
    <span>Sparse</span><span>Dense hotspot</span>
  </div>

  <!-- Layer 3 -->
  <p style="font-size:0.82rem;font-weight:600;margin:0 0 0.35rem 0">
    ③ Individual Listings
  </p>
  <div style="display:flex;align-items:flex-start;gap:0.55rem;margin-bottom:1.5rem">
    <div style="width:13px;height:13px;border-radius:50%;background:#FF5A5F;
                flex-shrink:0;margin-top:2px"></div>
    <p style="font-size:0.72rem;color:rgba(240,240,240,0.55);margin:0">
      Each dot = one listing.<br>
      Clustered numbers show count — <b>zoom in</b> to expand.<br>
      <b>Click</b> any dot for price, rating, room type &amp; availability.
    </p>
  </div>

  <hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:0 0 1rem 0">

  <p style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;
            color:rgba(240,240,240,0.35);margin:0 0 0.6rem 0">How to Interact</p>
  <ul style="font-size:0.72rem;color:rgba(240,240,240,0.55);
             padding-left:1.1rem;margin:0;line-height:2">
    <li>Scroll / pinch to zoom</li>
    <li>Toggle layers — top-right panel</li>
    <li>Hover a district to highlight it</li>
    <li>Click a dot to see listing details</li>
  </ul>

</div>
""", unsafe_allow_html=True)

    with map_col:
        folium_static(m, width=820, height=580)

    # ── Cluster summary ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("K-Means Cluster Summary", expanded=False):
        if "cluster" not in df.columns:
            with st.spinner("Running K-Means clustering…"):
                clusterer.cluster(n_clusters=5, include_price=True)
        st.dataframe(clusterer.get_cluster_summary(), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.title("Data Explorer")

    # ── Column selector ────────────────────────────────────────────────────────
    st.subheader("Browse Listings")

    default_cols = [
        c for c in [
            "id", "neighbourhood_cleansed", "room_type", "price",
            "bedrooms", "bathrooms", "accommodates",
            "review_scores_rating", "availability_365", "number_of_reviews",
        ] if c in filtered_df.columns
    ]
    selected_cols = st.multiselect(
        "Columns to display",
        options=filtered_df.columns.tolist(),
        default=default_cols,
    )
    row_count = st.slider("Rows to show", 50, 500, 100, 50, key="row_slider")

    if selected_cols:
        st.dataframe(
            filtered_df[selected_cols].head(row_count),
            use_container_width=True,
            height=340,
        )
    else:
        st.info("Select at least one column above.")

    st.markdown("---")

    # ── Distribution viewer ────────────────────────────────────────────────────
    st.subheader("Distribution Viewer")
    st.caption("Select any numeric column to inspect its distribution and key statistics")

    numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()
    default_idx  = numeric_cols.index("price") if "price" in numeric_cols else 0
    sel_col = st.selectbox("Numeric column", numeric_cols, index=default_idx)

    col_series = filtered_df[sel_col].dropna()
    skewness   = col_series.skew()
    skew_label = (
        "right-skewed (long tail of high values)"  if skewness >  0.5 else
        "left-skewed (long tail of low values)"    if skewness < -0.5 else
        "roughly symmetric"
    )

    chart_col, stats_col = st.columns([1.618, 1])

    with chart_col:
        fig = px.histogram(
            filtered_df.dropna(subset=[sel_col]),
            x=sel_col, nbins=60,
            color_discrete_sequence=[CORAL],
            labels={sel_col: sel_col.replace("_", " ").title()},
        )
        fig.add_vline(
            x=col_series.mean(), line_dash="dash", line_color=AMBER, line_width=1.5,
            annotation_text=f"Mean  {col_series.mean():,.2f}",
            annotation_font_color=AMBER, annotation_position="top right",
        )
        fig.add_vline(
            x=col_series.median(), line_dash="dot", line_color=TEAL, line_width=1.5,
            annotation_text=f"Median  {col_series.median():,.2f}",
            annotation_font_color=TEAL, annotation_position="top left",
        )
        st.plotly_chart(_chart(fig), use_container_width=True)

    with stats_col:
        st.markdown("**Statistics**")
        null_count = filtered_df[sel_col].isna().sum()
        null_pct   = null_count / len(filtered_df) * 100
        stats_rows = [
            ("Count",    f"{len(col_series):,}"),
            ("Mean",     f"{col_series.mean():,.2f}"),
            ("Median",   f"{col_series.median():,.2f}"),
            ("Std Dev",  f"{col_series.std():,.2f}"),
            ("Min",      f"{col_series.min():,.2f}"),
            ("Q1",       f"{col_series.quantile(0.25):,.2f}"),
            ("Q3",       f"{col_series.quantile(0.75):,.2f}"),
            ("Max",      f"{col_series.max():,.2f}"),
            ("Skewness", f"{skewness:.2f} — {skew_label}"),
            ("Nulls",    f"{null_count:,} ({null_pct:.1f}%)"),
        ]
        st.dataframe(
            pd.DataFrame(stats_rows, columns=["Stat", "Value"]),
            hide_index=True,
            use_container_width=True,
            height=385,
        )

    st.markdown("---")

    # ── Data quality ───────────────────────────────────────────────────────────
    st.subheader("Data Quality — Missing Values")
    st.caption("Only columns with at least one missing value are shown")

    missing = (
        filtered_df.isnull().mean()
        .mul(100).round(1)
        .reset_index()
    )
    missing.columns = ["column", "pct_missing"]
    missing = missing[missing["pct_missing"] > 0].sort_values("pct_missing")

    if missing.empty:
        st.success("No missing values in the filtered dataset.")
    else:
        fig = px.bar(
            missing,
            x="pct_missing", y="column", orientation="h",
            color="pct_missing",
            color_continuous_scale=[[0, TEAL], [0.5, AMBER], [1, CORAL]],
            labels={"pct_missing": "% Missing", "column": ""},
            text="pct_missing",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            height=max(300, len(missing) * 28),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(_chart(fig), use_container_width=True)

    st.markdown("---")
    st.subheader("Neighbourhood Comparison")

    nb_counts = filtered_df["neighbourhood_cleansed"].value_counts()
    valid_nbs = sorted(nb_counts[nb_counts >= 10].index.tolist())
    compare_nbs = st.multiselect(
        "Select neighbourhoods to compare (min 10 listings each)",
        valid_nbs,
        valid_nbs[:4] if len(valid_nbs) >= 4 else valid_nbs,
    )

    if compare_nbs:
        cmp = (
            filtered_df[filtered_df["neighbourhood_cleansed"].isin(compare_nbs)]
            .groupby("neighbourhood_cleansed")
            .agg(
                Listings=("id", "count"),
                Avg_Price=("price", "mean"),
                Median_Price=("price", "median"),
                Min_Price=("price", "min"),
                Max_Price=("price", "max"),
                Avg_Rating=("review_scores_rating", "mean"),
                Avg_Availability=("availability_365", "mean"),
            )
            .round(1)
        )
        st.dataframe(cmp, use_container_width=True)

        fig = px.bar(
            cmp.reset_index(),
            x="neighbourhood_cleansed", y="Avg_Price",
            color="Avg_Price",
            color_continuous_scale=[[0, "#7a1f22"], [0.5, "#c43b3f"], [1, CORAL]],
            labels={"neighbourhood_cleansed": "", "Avg_Price": "Avg Price (JPY)"},
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(_chart(fig), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Sentiment Analysis":
    st.title("Review Sentiment")
    st.caption("TextBlob polarity scores on a 5,000-review sample.")

    if not _TRANSLATION_AVAILABLE:
        st.warning(
            "Auto-translation is disabled. "
            "Run `python -m pip install langdetect deep-translator` and restart."
        )

    reviews = load_reviews_with_sentiment()

    if len(reviews) == 0:
        st.info("No review data found. Run `scripts/save_processed_data.py` first.")
        st.stop()

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Reviews Analysed", f"{len(reviews):,}")
    sc2.metric("Average Sentiment", f"{reviews['sentiment'].mean():.3f}")
    sc3.metric("Positive Reviews", f"{(reviews['sentiment'] > 0).mean() * 100:.1f}%")

    st.markdown("---")

    col_l, col_r = st.columns([1, 1.618])

    with col_l:
        st.subheader("Sentiment Breakdown")
        counts = reviews["sentiment_label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        color_map = {"Positive": TEAL, "Neutral": AMBER, "Negative": CORAL}
        fig = px.bar(
            counts, x="label", y="count", color="label",
            color_discrete_map=color_map,
            labels={"label": "", "count": "Reviews"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(_chart(fig), use_container_width=True)

    with col_r:
        st.subheader("Polarity Score Distribution")
        fig = px.histogram(
            reviews, x="sentiment", nbins=40,
            color_discrete_sequence=[CORAL],
            labels={"sentiment": "Polarity (−1 to 1)"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(_chart(fig), use_container_width=True)

    st.markdown("---")
    st.subheader("Sample Reviews")
    st.caption("Non-English reviews are automatically translated. Expand to see the original.")
    tab_pos, tab_neg = st.tabs(["Most Positive", "Most Negative"])

    def _render_review(row, positive: bool) -> None:
        original   = str(row["comments"])
        translated, lang, was_translated = _translate(original)
        score_tag  = f"**Score {row['sentiment']:.2f}**"
        display    = translated[:260]

        if positive:
            st.success(f"{score_tag} — {display}")
        else:
            st.error(f"{score_tag} — {display}")

        if was_translated:
            with st.expander(f"Original · {lang.upper()}"):
                st.write(original[:400])

    with tab_pos:
        for _, row in reviews.nlargest(5, "sentiment").iterrows():
            _render_review(row, positive=True)

    with tab_neg:
        for _, row in reviews.nsmallest(5, "sentiment").iterrows():
            _render_review(row, positive=False)
