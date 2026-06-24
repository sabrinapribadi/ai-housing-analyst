"""
LLM Agent for Airbnb Tokyo Data Analysis.
Uses LangGraph for agent functionality (LangChain v1.x compatible).
"""

import os
import pandas as pd
import json
from typing import List, Dict, Any
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import logging

# Import our analytics modules
from src.analytics.eda import EDAAnalyzer
from src.analytics.clustering import GeospatialClusterer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AirbnbAgent:
    """LLM agent for answering questions about Airbnb Tokyo data."""

    def __init__(self, listings_path: str = "data/processed/listings_processed.parquet"):
        """Initialize the agent with data."""
        self.listings = pd.read_parquet(listings_path)
        self.eda = EDAAnalyzer(self.listings)
        self.clusterer = GeospatialClusterer(self.listings)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.agent = self._create_agent()

    def _create_tools(self):
        """Create the tools for the agent."""
        
        @tool
        def get_summary_stats() -> str:
            """Get summary statistics of Tokyo Airbnb market."""
            report = self.eda.create_summary_report()
            return json.dumps({
                'total_listings': report['total_listings'],
                'avg_price': f"¥{report['avg_price']:,.0f}",
                'median_price': f"¥{report['median_price']:,.0f}",
                'top_5_expensive': report['top_5_expensive'],
                'top_5_cheapest': report['top_5_cheapest']
            }, indent=2)

        @tool
        def plot_price_distribution() -> str:
            """Generate price distribution plot."""
            path = self.eda.plot_price_distribution()
            return f"Price distribution plot saved to: {path}"

        @tool
        def plot_price_by_neighborhood() -> str:
            """Generate plot of average prices by neighborhood."""
            path = self.eda.plot_price_by_neighborhood()
            return f"Price by neighborhood plot saved to: {path}"

        @tool
        def plot_price_by_room_type() -> str:
            """Generate plot of prices by room type."""
            path = self.eda.plot_price_by_room_type()
            return f"Price by room type plot saved to: {path}"

        @tool
        def get_cluster_summary() -> str:
            """Get summary of geospatial clusters."""
            if 'cluster' not in self.listings.columns:
                self.clusterer.cluster(n_clusters=5, include_price=True)
            summary = self.clusterer.get_cluster_summary()
            return summary.to_string()

        @tool
        def recommend_neighborhood(budget: int = 20000, room_type: str = "Entire home/apt") -> str:
            """Recommend neighborhoods based on budget and room type."""
            df = self.listings[
                (self.listings['price'] <= budget) &
                (self.listings['room_type'] == room_type)
            ]
            
            recommendations = df.groupby('neighbourhood_cleansed').agg({
                'price': ['count', 'mean'],
                'review_scores_rating': 'mean'
            }).round(2)
            
            recommendations.columns = ['count', 'avg_price', 'avg_rating']
            recommendations = recommendations[recommendations['count'] >= 10]
            recommendations = recommendations.sort_values('avg_rating', ascending=False).head(10)
            
            return recommendations.to_string()

        return [get_summary_stats, plot_price_distribution, plot_price_by_neighborhood, 
                plot_price_by_room_type, get_cluster_summary, recommend_neighborhood]

    def _create_agent(self):
        """Create the LangGraph agent with tools."""
        tools = self._create_tools()
        
        # Create the agent using LangGraph's create_react_agent
        agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=SystemMessage(content="""You are a helpful data analyst assistant for Airbnb Tokyo market data.

You have access to tools that can provide summary statistics, generate plots, and analyze market segments.

When answering:
1. Use data from the tools to support your answers
2. Be specific with numbers and percentages
3. Provide actionable insights
4. Mention neighborhood names when relevant""")
        )
        
        return agent

    def ask(self, question: str) -> str:
        """Ask a question to the agent."""
        try:
            # Use the agent with the question
            response = self.agent.invoke({
                "messages": [HumanMessage(content=question)]
            })
            # Extract the final response
            messages = response.get('messages', [])
            if messages:
                return messages[-1].content
            return "No response from agent"
        except Exception as e:
            return f"Error: {str(e)}"
