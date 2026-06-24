"""
LLM Agent for Airbnb Tokyo Data Analysis.
Uses LangChain AgentExecutor (compatible with LangChain/LangGraph 1.x).
"""

import os
import json
import logging

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

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
        self.listings = pd.read_parquet(listings_path)
        self.eda      = EDAAnalyzer(self.listings)
        self.clusterer = GeospatialClusterer(self.listings)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
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
        def plot_price_by_neighborhood() -> str:
            """Generate a bar chart of average prices by neighbourhood."""
            path = self.eda.plot_price_by_neighborhood()
            return f"Price by neighbourhood plot saved to: {path}"

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

        return [
            get_summary_stats,
            plot_price_distribution,
            plot_price_by_neighborhood,
            plot_price_by_room_type,
            get_cluster_summary,
            recommend_neighborhood,
        ]

    # ── Agent ─────────────────────────────────────────────────────────────────

    def _create_agent(self) -> AgentExecutor:
        """Build an AgentExecutor using LangChain's tool-calling agent."""
        tools = self._create_tools()
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Ask a question and return the agent's plain-text answer."""
        try:
            result = self.agent.invoke({"input": question})
            answer = result["output"]

            # Flag hedged language so the user knows the answer is approximate
            if any(w in answer.lower() for w in ("maybe", "might", "could", "approximately")):
                answer += "\n\n*Based on available data — actual values may vary.*"

            return answer
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return f"Error: {str(e)}"
