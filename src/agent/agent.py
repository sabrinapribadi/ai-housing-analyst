"""
LLM Agent for Airbnb Tokyo Data Analysis.
Uses LangGraph create_react_agent (LangGraph 1.x / LangChain 1.x compatible).
"""

import glob
import os
import json
import logging
import time

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

MODEL = "gpt-4o-mini"
# USD per 1M tokens, configs/rag_eval.yaml prices (read 2026-11-11)
PRICES = {"input": 0.15, "cached_input": 0.075, "output": 0.60}

_SYSTEM_PROMPT = """You are a helpful data analyst assistant for the Airbnb Tokyo market.

You have tools that provide summary statistics, generate plots, and analyse market segments.

When answering:
1. Always call a tool to back up your answer with data.
2. Be specific — include numbers, percentages, and neighbourhood names.
3. Provide one actionable insight after presenting the data.
4. Keep responses concise and easy to read.
5. Only name neighbourhoods and state figures that appear in a tool result; never describe places from general knowledge.
6. Plot tools return only a file path, so also call get_summary_stats (or another data tool) for any number or ranking you state.
7. Do not link to file paths; the app shows the chart under your answer."""


class AirbnbAgent:
    """LLM agent for answering plain-English questions about Airbnb Tokyo data."""

    def __init__(self, listings_path: str = "data/processed/listings_processed.parquet", api_key: str | None = None):
        """api_key: the viewer's own key in the app's live mode (R26); otherwise OPENAI_API_KEY from the environment."""
        self.listings  = pd.read_parquet(listings_path)
        self.eda       = EDAAnalyzer(self.listings)
        self.clusterer = GeospatialClusterer(self.listings)
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI key: pass api_key or set OPENAI_API_KEY in your local .env file.")
        self.llm = ChatOpenAI(
            model=MODEL,
            temperature=0,
            api_key=api_key,
            timeout=60,
            max_retries=1,
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

    def ask_detailed(self, question: str) -> dict:
        """One question with its trace (R13, R26): answer, tool calls and their outputs, tokens, cost, latency, charts drawn."""
        t0, started = time.perf_counter(), time.time()
        response = self.agent.invoke({"messages": [HumanMessage(content=question)]})
        msgs = response.get("messages", [])
        calls, results, tin, tout, rounds = [], [], 0, 0, 0
        for m in msgs:
            usage = getattr(m, "usage_metadata", None) or {}
            if usage:
                rounds += 1
                tin += int(usage.get("input_tokens", 0) or 0)
                tout += int(usage.get("output_tokens", 0) or 0)
            for tc in getattr(m, "tool_calls", None) or []:
                calls.append({"name": tc["name"], "arguments": tc.get("args", {}) or {}, "ok": True})
            if getattr(m, "type", "") == "tool":
                try:
                    results.append(json.loads(str(m.content)))     # JSON tools: keep values, not escaped text (¥ as \u00a5)
                except ValueError:
                    results.append(str(m.content))
                if isinstance(results[-1], str) and results[-1].lower().startswith("error"):
                    calls[len(results) - 1]["ok"] = False
        images = sorted((p for p in glob.glob("outputs/*.png") if os.path.getmtime(p) >= started - 1), key=os.path.getmtime)
        return {"question": question, "answer": str(msgs[-1].content) if msgs else "", "tool_calls": calls, "tool_results": results,
                "images": images, "steps": rounds, "model": MODEL, "tokens": {"input": tin, "cached_input": 0, "output": tout},
                "cost_usd": round((tin * PRICES["input"] + tout * PRICES["output"]) / 1e6, 8),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

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
