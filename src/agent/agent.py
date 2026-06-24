"""
LLM Agent for Airbnb Tokyo Data Analysis.
Uses LangGraph create_react_agent (LangGraph 1.x / LangChain 1.x compatible).
"""

import os
import json
import logging

import pandas as pd
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.analytics.eda import EDAAnalyzer
from src.analytics.clustering import GeospatialClusterer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful data analyst assistant for the Airbnb Tokyo market.

You have tools that provide summary statistics, generate plots, and analyse market segments.

When answering:
1. Always call a tool to back up your answer with data.
2. Be specific — include numbers, percentages, and neighbourhood names.
3. Provide one actionable insight after presenting the data.
4. Keep responses concise and easy to read."""


class AirbnbAgent:
    """LLM agent for answering plain-English questions about Airbnb Tokyo data."""

    def __init__(self, listings_path: str = "data/processed/listings_processed.parquet"):
        self.listings  = pd.read_parquet(listings_path)
        self.eda       = EDAAnalyzer(self.listings)
        self.clusterer = GeospatialClusterer(self.listings)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Add it to Streamlit Cloud Secrets or your local .env file."
            )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
        )
        self.agent = self._create_agent()

    # ── Tools ─────────────────────────────────────────────────────────────────

    def _create_tools(self) -> list:
        @tool
        def get_summary_stats() -> str:
            """Get summary statistics of the Tokyo Airbnb market."""
            report = self.eda.create_summary_report()
            return json.dumps({
                "total_listings":  report["total_listings"],
                "avg_price":       f"¥{report['avg_price']:,.0f}",
                "median_price":    f"¥{report['median_price']:,.0f}",
                "top_5_expensive": report["top_5_expensive"],
                "top_5_cheapest":  report["top_5_cheapest"],
            }, indent=2)

        @tool
        def plot_price_distribution() -> str:
            """Generate a price distribution plot and save it to disk."""
            path = self.eda.plot_price_distribution()
            return f"Price distribution plot saved to: {path}"

        @tool
        def plot_price_by_neighborhood(order: str = "expensive") -> str:
            """Generate a bar chart of average prices by neighbourhood.
            Pass order='expensive' for the most expensive neighbourhoods (default),
            or order='cheapest' for the cheapest neighbourhoods."""
            path = self.eda.plot_price_by_neighborhood(order=order)
            return f"Price by neighbourhood ({order}) plot saved to: {path}"

        @tool
        def plot_price_by_room_type() -> str:
            """Generate a chart of prices broken down by room type."""
            path = self.eda.plot_price_by_room_type()
            return f"Price by room type plot saved to: {path}"

        @tool
        def get_cluster_summary() -> str:
            """Get a summary of the 5 geospatial market segments (K-Means clusters)."""
            if "cluster" not in self.listings.columns:
                self.clusterer.cluster(n_clusters=5, include_price=True)
            return self.clusterer.get_cluster_summary().to_string()

        @tool
        def recommend_neighborhood(budget: int = 20000, room_type: str = "Entire home/apt") -> str:
            """Recommend neighbourhoods that fit a given budget and room type."""
            df = self.listings[
                (self.listings["price"] <= budget) &
                (self.listings["room_type"] == room_type)
            ]
            rec = (
                df.groupby("neighbourhood_cleansed")
                .agg(count=("price", "count"),
                     avg_price=("price", "mean"),
                     avg_rating=("review_scores_rating", "mean"))
                .round(2)
            )
            rec = rec[rec["count"] >= 10].sort_values("avg_rating", ascending=False).head(10)
            return rec.to_string()

        @tool
        def get_review_stats() -> str:
            """Get review score statistics for Tokyo Airbnb listings.
            Returns average and median for the overall rating and all six
            sub-category scores (accuracy, cleanliness, check-in,
            communication, location, value), plus rating distribution."""
            score_cols = [
                "review_scores_rating",
                "review_scores_accuracy",
                "review_scores_cleanliness",
                "review_scores_checkin",
                "review_scores_communication",
                "review_scores_location",
                "review_scores_value",
            ]
            result = {}
            for col in score_cols:
                if col not in self.listings.columns:
                    continue
                s = self.listings[col].dropna()
                label = col.replace("review_scores_", "")
                result[label] = {
                    "mean":   round(float(s.mean()), 2),
                    "median": round(float(s.median()), 2),
                    "count":  int(len(s)),
                }
            # Overall rating distribution (bands on 1-5 scale)
            if "review_scores_rating" in self.listings.columns:
                r = self.listings["review_scores_rating"].dropna()
                result["rating_distribution"] = {
                    "5.0 (excellent)":    int((r == 5.0).sum()),
                    "4.5–4.9 (great)":    int(((r >= 4.5) & (r < 5.0)).sum()),
                    "4.0–4.4 (good)":     int(((r >= 4.0) & (r < 4.5)).sum()),
                    "below 4.0 (fair)":   int((r < 4.0).sum()),
                }
                result["reviewed_listings"] = int(len(r))
                result["total_listings"]    = int(len(self.listings))
            return json.dumps(result, indent=2)

        return [
            get_summary_stats,
            get_review_stats,
            plot_price_distribution,
            plot_price_by_neighborhood,
            plot_price_by_room_type,
            get_cluster_summary,
            recommend_neighborhood,
        ]

    # ── Agent ─────────────────────────────────────────────────────────────────

    def _create_agent(self):
        """Build a LangGraph ReAct agent."""
        return create_react_agent(
            model=self.llm,
            tools=self._create_tools(),
            prompt=SystemMessage(content=_SYSTEM_PROMPT),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Ask a question and return the agent's plain-text answer."""
        try:
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=question)]}
            )
            messages = response.get("messages", [])
            if not messages:
                return "No response from agent."
            answer = messages[-1].content

            # Flag hedged language so the user knows the answer is approximate
            if any(w in answer.lower() for w in ("maybe", "might", "could", "approximately")):
                answer += "\n\n*Based on available data — actual values may vary.*"

            return answer
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return f"Error: {str(e)}"
