"""
Intelligent Data Discovery and Integration Agent.

Phase 1 — Discovery: auto-detects patterns, anomalies, and market gaps
           in the Airbnb listings without the user asking any questions.
Phase 2 — Integration: profiles an uploaded external dataset and uses an
           LLM to suggest how to join/enrich it with the Airbnb data.
"""

import difflib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
logger = logging.getLogger(__name__)

# ── Design tokens (mirror dashboard.py) ───────────────────────────────────────
CORAL      = "#FF5A5F"
TEAL       = "#00A699"
AMBER      = "#FFB400"
PURPLE     = "#A78BFA"
DARK_BG    = "rgba(0,0,0,0)"
PLOT_BG    = "rgba(255,255,255,0.03)"
FONT_COLOR = "#F0F0F0"
GRID_COLOR = "rgba(255,255,255,0.08)"

CATEGORY_COLOR = {
    "anomaly":     CORAL,
    "correlation": AMBER,
    "gap":         PURPLE,
    "pattern":     TEAL,
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Insight:
    title:     str
    narrative: str
    category:  str            # "anomaly" | "correlation" | "gap" | "pattern"
    severity:  str            # "high" | "medium" | "low"
    chart:     object         # Plotly Figure or None
    data:      dict = field(default_factory=dict)


@dataclass
class DataProfile:
    shape:           tuple
    columns:         list     # list of column-stat dicts
    suggested_joins: list     # column names that could join with Airbnb data
    summary:         str      # LLM-generated 1–2 sentence description


# ── Agent ──────────────────────────────────────────────────────────────────────

class DataDiscoveryAgent:
    """
    Two-phase intelligent agent:
    - run_full_discovery()  → list[Insight]
    - profile_dataset()     → DataProfile
    - suggest_column_mapping() → dict
    - merge_datasets()      → (enriched_df, report)
    """

    def __init__(
        self,
        listings_df: pd.DataFrame,
        reviews_df: Optional[pd.DataFrame] = None,
    ):
        self.listings = listings_df.copy()
        self.reviews  = reviews_df
        self._llm: Optional[ChatOpenAI] = None

    @property
    def llm(self) -> Optional[ChatOpenAI]:
        if self._llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self._llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0.1,
                        api_key=api_key,
                    )
                except Exception as e:
                    logger.warning(f"LLM init failed: {e}")
        return self._llm

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _theme(self, fig) -> object:
        """Apply transparent dark theme to a Plotly figure."""
        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COLOR, size=11),
            margin=dict(t=32, b=24, l=0, r=0),
            xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    def _narrate(self, context: str) -> str:
        """Ask GPT-4o-mini for a 1–2 sentence plain-English explanation."""
        if not self.llm:
            return ""
        try:
            resp = self.llm.invoke([
                SystemMessage(content=(
                    "You are a concise data analyst. Write 1–2 plain-English sentences "
                    "explaining the finding. No bullet points, no markdown, under 45 words."
                )),
                HumanMessage(content=context),
            ])
            return resp.content.strip()
        except Exception as e:
            logger.warning(f"LLM narration error: {e}")
            return ""

    # ── Discovery methods ─────────────────────────────────────────────────────

    def _discover_price_anomalies(self) -> Optional[Insight]:
        """Neighbourhoods where mean price greatly exceeds the median."""
        df = self.listings.dropna(subset=["price", "neighbourhood_cleansed"])
        nb = (
            df.groupby("neighbourhood_cleansed")["price"]
            .agg(mean="mean", median="median", count="count")
            .reset_index()
        )
        nb = nb[nb["count"] >= 20]
        nb["skew_ratio"] = (nb["mean"] - nb["median"]) / nb["median"]
        top = nb.nlargest(8, "skew_ratio")

        fig = go.Figure()
        fig.add_bar(x=top["neighbourhood_cleansed"], y=top["mean"],
                    name="Mean", marker_color=CORAL)
        fig.add_bar(x=top["neighbourhood_cleansed"], y=top["median"],
                    name="Median", marker_color=TEAL)
        fig.update_layout(
            barmode="group",
            title="Mean vs Median Price — Most Skewed Districts",
            xaxis_tickangle=-30,
        )
        self._theme(fig)

        worst = top.iloc[0]
        fallback = (
            f"{worst['neighbourhood_cleansed']} has a "
            f"{worst['skew_ratio']:.0%} gap between mean (¥{worst['mean']:,.0f}) "
            "and median price, driven by a small number of luxury outliers."
        )
        return Insight(
            title=f"Outliers skew prices in {worst['neighbourhood_cleansed']}",
            narrative=self._narrate(
                f"In {worst['neighbourhood_cleansed']}, mean price is ¥{worst['mean']:,.0f} "
                f"but median is ¥{worst['median']:,.0f} — a {worst['skew_ratio']:.0%} gap "
                "caused by a small number of very expensive listings."
            ) or fallback,
            category="anomaly",
            severity="high",
            chart=fig,
            data=top.to_dict("records"),
        )

    def _discover_value_gems(self) -> Optional[Insight]:
        """Listings with above-median rating but below-median price."""
        df = self.listings.dropna(subset=["price", "review_scores_rating"])
        med_price  = df["price"].median()
        med_rating = df["review_scores_rating"].median()
        gems = df[
            (df["price"] < med_price) & (df["review_scores_rating"] >= med_rating)
        ]
        nb_gems = (
            gems.groupby("neighbourhood_cleansed")
            .agg(count=("id", "count"),
                 avg_price=("price", "mean"),
                 avg_rating=("review_scores_rating", "mean"))
            .reset_index()
            .nlargest(12, "count")
        )

        fig = px.scatter(
            nb_gems,
            x="avg_price", y="avg_rating", size="count",
            text="neighbourhood_cleansed",
            color_discrete_sequence=[TEAL],
            labels={"avg_price": "Avg Price (JPY)", "avg_rating": "Avg Rating",
                    "count": "Listings"},
            title=f"Hidden Gems — Below ¥{med_price:,.0f} & Rated ≥ {med_rating:.0f}",
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        self._theme(fig)

        top_nb = nb_gems.iloc[0]
        fallback = (
            f"{len(gems):,} listings are priced below the market median "
            f"(¥{med_price:,.0f}) yet rated above average. "
            f"{top_nb['neighbourhood_cleansed']} leads with {top_nb['count']} such listings."
        )
        return Insight(
            title=f"{len(gems):,} hidden-gem listings identified across Tokyo",
            narrative=self._narrate(
                f"{len(gems):,} listings are priced below the median (¥{med_price:,.0f}) "
                f"but rated above the median ({med_rating:.0f}/100). "
                f"{top_nb['neighbourhood_cleansed']} has the most ({top_nb['count']})."
            ) or fallback,
            category="pattern",
            severity="high",
            chart=fig,
            data=nb_gems.to_dict("records"),
        )

    def _discover_market_gaps(self) -> Optional[Insight]:
        """Room-type × neighbourhood combos with high ratings but low supply."""
        df = self.listings.dropna(
            subset=["neighbourhood_cleansed", "room_type", "review_scores_rating"]
        )
        grid = (
            df.groupby(["neighbourhood_cleansed", "room_type"])
            .agg(supply=("id", "count"),
                 avg_rating=("review_scores_rating", "mean"))
            .reset_index()
        )
        p20 = grid["supply"].quantile(0.20)
        p70 = grid["avg_rating"].quantile(0.70)
        gaps = grid[
            (grid["supply"] <= p20) & (grid["avg_rating"] >= p70)
        ].nlargest(12, "avg_rating")

        fig = px.scatter(
            gaps,
            x="supply", y="avg_rating", color="room_type",
            text="neighbourhood_cleansed",
            labels={"supply": "# Listings", "avg_rating": "Avg Rating",
                    "room_type": "Room Type"},
            title="Market Gaps — High Demand Signal, Low Supply",
            color_discrete_sequence=[CORAL, TEAL, AMBER, PURPLE],
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        self._theme(fig)

        top = gaps.iloc[0]
        fallback = (
            f"'{top['room_type']}' in {top['neighbourhood_cleansed']} has only "
            f"{int(top['supply'])} listings but rates {top['avg_rating']:.1f}/100 — "
            "high guest satisfaction signals unmet demand."
        )
        return Insight(
            title=f"{len(gaps)} under-supplied but highly-rated market segments",
            narrative=self._narrate(
                f"'{top['room_type']}' listings in {top['neighbourhood_cleansed']}: "
                f"only {int(top['supply'])} available but avg rating {top['avg_rating']:.1f}/100. "
                "Low supply and high satisfaction point to a gap in the market."
            ) or fallback,
            category="gap",
            severity="medium",
            chart=fig,
            data=gaps.to_dict("records"),
        )

    def _discover_correlations(self) -> Optional[Insight]:
        """Numeric features most correlated with nightly price."""
        candidates = [
            "accommodates", "bathrooms", "bedrooms", "beds",
            "number_of_reviews", "review_scores_rating",
            "availability_365", "minimum_nights",
        ]
        cols = [c for c in candidates if c in self.listings.columns]
        corr = (
            self.listings[cols + ["price"]]
            .dropna()
            .corr()["price"]
            .drop("price")
            .sort_values(key=abs, ascending=False)
            .head(8)
        )

        fig = go.Figure(go.Bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            marker_color=[CORAL if v > 0 else TEAL for v in corr.values],
        ))
        fig.update_layout(
            title="Feature Correlation with Nightly Price (Pearson r)",
            xaxis_title="Pearson r",
            yaxis=dict(autorange="reversed"),
        )
        self._theme(fig)

        top_pos = corr[corr > 0].idxmax() if (corr > 0).any() else corr.idxmax()
        top_neg = corr[corr < 0].idxmin() if (corr < 0).any() else corr.idxmin()
        fallback = (
            f"'{top_pos}' is the strongest positive predictor of price "
            f"(r={corr[top_pos]:.2f}); '{top_neg}' is the strongest negative "
            f"(r={corr[top_neg]:.2f})."
        )
        return Insight(
            title=f"'{top_pos}' is the strongest price predictor",
            narrative=self._narrate(
                f"Strongest positive predictor: '{top_pos}' (r={corr[top_pos]:.2f}). "
                f"Strongest negative: '{top_neg}' (r={corr[top_neg]:.2f})."
            ) or fallback,
            category="correlation",
            severity="medium",
            chart=fig,
            data=corr.to_dict(),
        )

    def _discover_availability_pattern(self) -> Optional[Insight]:
        """Districts that command high prices despite low availability."""
        df = self.listings.dropna(
            subset=["availability_365", "price", "neighbourhood_cleansed"]
        )
        nb = (
            df.groupby("neighbourhood_cleansed")
            .agg(avg_avail=("availability_365", "mean"),
                 avg_price=("price", "mean"),
                 count=("id", "count"))
            .reset_index()
        )
        nb = nb[nb["count"] >= 20]

        fig = px.scatter(
            nb,
            x="avg_avail", y="avg_price", size="count",
            text="neighbourhood_cleansed",
            color_discrete_sequence=[AMBER],
            labels={"avg_avail": "Avg Availability (days/yr)",
                    "avg_price": "Avg Price (JPY)"},
            title="Availability vs Price by District",
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        self._theme(fig)

        scarce = nb[nb["avg_avail"] <= nb["avg_avail"].quantile(0.25)]
        top = (
            scarce.nlargest(1, "avg_price").iloc[0]
            if len(scarce) else nb.nlargest(1, "avg_price").iloc[0]
        )
        fallback = (
            f"{top['neighbourhood_cleansed']} averages only {top['avg_avail']:.0f} "
            "available days/year while maintaining premium prices — evidence of "
            "sustained, high occupancy demand."
        )
        return Insight(
            title=f"{top['neighbourhood_cleansed']} — scarce and expensive",
            narrative=self._narrate(
                f"{top['neighbourhood_cleansed']}: only {top['avg_avail']:.0f} days/yr "
                f"available on average yet commands ¥{top['avg_price']:,.0f}/night, "
                "suggesting high and sustained occupancy demand."
            ) or fallback,
            category="pattern",
            severity="low",
            chart=fig,
            data=top.to_dict(),
        )

    def run_full_discovery(self) -> list:
        """Run all discovery methods and return a list of Insight objects."""
        results = []
        for method in [
            self._discover_price_anomalies,
            self._discover_value_gems,
            self._discover_market_gaps,
            self._discover_correlations,
            self._discover_availability_pattern,
        ]:
            try:
                insight = method()
                if insight:
                    results.append(insight)
            except Exception as e:
                logger.warning(f"{method.__name__} failed: {e}")
        return results

    # ── Integration methods ───────────────────────────────────────────────────

    def profile_dataset(self, df: pd.DataFrame) -> DataProfile:
        """Statistical + LLM profile of an external DataFrame."""
        AIRBNB_KEYS = {
            "neighbourhood", "neighbourhood_cleansed", "id", "host_id",
            "latitude", "longitude", "name", "price",
        }
        columns = []
        for col in df.columns:
            sample = [str(v) for v in df[col].dropna().head(3).tolist()]
            columns.append({
                "Column":      col,
                "Type":        str(df[col].dtype),
                "Unique":      int(df[col].nunique()),
                "Missing (%)": round(df[col].isna().mean() * 100, 1),
                "Samples":     " | ".join(sample),
            })

        join_candidates = [
            c["Column"] for c in columns
            if any(k in c["Column"].lower() for k in AIRBNB_KEYS)
        ]

        col_text = ", ".join(
            f"{c['Column']} ({c['Type']}, {c['Unique']} unique)"
            for c in columns[:12]
        )
        summary = f"Dataset with {df.shape[0]:,} rows and {df.shape[1]} columns."
        if self.llm:
            try:
                resp = self.llm.invoke([
                    SystemMessage(content=(
                        "You are a data engineer. In 2 sentences describe what this dataset "
                        "contains and whether it could enrich an Airbnb listings dataset. "
                        "No bullet points. Under 50 words."
                    )),
                    HumanMessage(
                        content=f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols. "
                                f"Columns: {col_text}."
                    ),
                ])
                summary = resp.content.strip()
            except Exception as e:
                logger.warning(f"LLM profile summary failed: {e}")

        return DataProfile(
            shape=df.shape,
            columns=columns,
            suggested_joins=join_candidates,
            summary=summary,
        )

    def _fuzzy_match(self, new_col: str, airbnb_cols: list) -> Optional[str]:
        """String-similarity fallback for column matching."""
        normalise = lambda s: s.lower().replace("_", " ").replace("-", " ")
        matches = difflib.get_close_matches(
            normalise(new_col),
            [normalise(c) for c in airbnb_cols],
            n=1,
            cutoff=0.55,
        )
        if not matches:
            return None
        normalised_airbnb = [normalise(c) for c in airbnb_cols]
        return airbnb_cols[normalised_airbnb.index(matches[0])]

    def suggest_column_mapping(
        self, new_df: pd.DataFrame, context: str = ""
    ) -> dict:
        """
        Return {new_col: airbnb_col_or_None} for all columns in new_df.
        Uses LLM when available, falls back to fuzzy string matching.
        """
        airbnb_cols = list(self.listings.columns)
        new_cols    = list(new_df.columns)

        if self.llm:
            sample = new_df.head(3).to_dict("records")
            prompt = (
                f"Airbnb listings columns: {airbnb_cols}\n\n"
                f"New dataset columns: {new_cols}\n\n"
                f"Sample rows from new dataset: {json.dumps(sample, default=str)}\n\n"
                f"Context: {context}\n\n"
                "Return a JSON object mapping each new column to the closest Airbnb "
                "column it could join/enrich on (or null if no match). "
                'Example: {"ward": "neighbourhood_cleansed", "usd_price": null}'
            )
            try:
                resp = self.llm.invoke([
                    SystemMessage(content=(
                        "You are a data integration expert. "
                        "Return only valid JSON, no explanation or markdown fences."
                    )),
                    HumanMessage(content=prompt),
                ])
                raw = resp.content.strip().strip("`")
                if raw.startswith("json"):
                    raw = raw[4:].strip()
                return json.loads(raw)
            except Exception as e:
                logger.warning(f"LLM column mapping failed, falling back to fuzzy: {e}")

        # Fuzzy fallback
        return {col: self._fuzzy_match(col, airbnb_cols) for col in new_cols}

    def merge_datasets(
        self,
        new_df: pd.DataFrame,
        join_key_new: str,
        join_key_airbnb: str,
        import_cols: list,
        how: str = "left",
    ) -> tuple:
        """
        Merge selected columns from new_df into self.listings.
        Returns (enriched_df, report_dict).
        """
        if join_key_new not in new_df.columns:
            return self.listings.copy(), {
                "status": "error",
                "reason": f"Column '{join_key_new}' not found in uploaded dataset.",
            }
        if join_key_airbnb not in self.listings.columns:
            return self.listings.copy(), {
                "status": "error",
                "reason": f"Column '{join_key_airbnb}' not found in Airbnb data.",
            }

        cols_to_bring = [join_key_new] + [c for c in import_cols if c in new_df.columns]
        subset = new_df[cols_to_bring].rename(columns={join_key_new: join_key_airbnb})

        # Drop duplicate join-key rows in new data (keep first)
        subset = subset.drop_duplicates(subset=[join_key_airbnb])

        enriched = self.listings.merge(subset, on=join_key_airbnb, how=how)

        match_rate = (
            round(enriched[import_cols[0]].notna().mean() * 100, 1)
            if import_cols else 0.0
        )
        self.listings = enriched

        return enriched, {
            "status": "success",
            "join_key": join_key_airbnb,
            "original_rows": len(self.listings),
            "merged_rows": len(enriched),
            "new_columns": import_cols,
            "match_rate": match_rate,
        }
