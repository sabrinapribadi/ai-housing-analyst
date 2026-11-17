"""The two agents behind the app's chat (R26): the data agent (tools over the listings) and the review search (RAG).

Each exposes MODEL, QUESTIONS and live(question, key) -> run, the shape app chat tabs replay and display.
Runs are checked like every portfolio agent: each number in the answer must appear in the question or a tool result.
"""
from __future__ import annotations

import time
from pathlib import Path

import agent as A                                   # shared chat engine (src/ui/agent.py): number check, replay matching

REVIEW_SNIPPET = 600                                # characters kept per retrieved review in a stored run


class DataAgent:
    MODEL = "gpt-4o-mini"
    QUESTIONS = [
        "What's the average price in Tokyo?",
        "Which neighbourhoods are most expensive?",
        "What are the cheapest neighbourhoods?",
        "Recommend neighbourhoods under ¥20,000",
        "Show the price distribution by room type",
        "What's the average review score?",
    ]

    @staticmethod
    def live(question: str, key: str | None) -> dict:
        from src.agent.agent import AirbnbAgent
        run = AirbnbAgent(api_key=key).ask_detailed(question)
        run["images"] = [str(Path(p).resolve()) for p in run["images"]]          # charts the agent drew, shown under the answer
        bad = A.ungrounded_numbers(run["answer"], [question, run["tool_results"]])
        return {**run, "grounded": not bad and bool(run["answer"]), "ungrounded_numbers": bad}


class ReviewAgent:
    MODEL = "gpt-6-luna"
    QUESTIONS = [
        "What do guests complain about most?",
        "What do guests love about staying in Tokyo?",
        "What are common issues with cleanliness?",
        "What do people say about the location and transport?",
        "What makes a highly rated listing?",
        "What do guests say about communication with hosts?",
    ]

    @staticmethod
    def live(question: str, key: str | None, n_results: int = 10) -> dict:
        from src.agent.rag_agent import ReviewRAGAgent
        t0 = time.perf_counter()
        out = ReviewRAGAgent(api_key=key).query(question, n_results=n_results)
        u = out.get("usage", {})
        sources = [{"id": s["id"], "neighbourhood": s["neighbourhood"], "similarity": s["similarity"], "cited": s["cited"],
                    "text": s["text"][:REVIEW_SNIPPET]} for s in out["sources"]]
        bad = A.ungrounded_numbers(out["answer"], [question, [s["text"] for s in out["sources"]]])
        cited = sum(s["cited"] for s in sources)
        return {"question": question, "answer": out["answer"], "images": [],
                "tool_calls": [{"name": "search_reviews", "arguments": {"n_results": n_results}, "ok": bool(sources)},
                               {"name": "cited_answer", "arguments": {"cited_reviews": cited}, "ok": cited > 0}],
                "tool_results": sources, "steps": 1, "model": u.get("model", ReviewAgent.MODEL),
                "grounded": not bad and cited > 0, "ungrounded_numbers": bad,
                "tokens": {"input": int(u.get("tokens_in", 0)), "cached_input": 0, "output": int(u.get("tokens_out", 0))},
                "cost_usd": float(u.get("cost_usd", 0.0)), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
