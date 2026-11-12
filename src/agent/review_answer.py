"""Review answers for the app (ADR-009 decision): gpt-6-luna with the study's cited-answer prompt.

Chosen on 300 test questions (data/outputs/rag_eval/answer_results.json): 78% fully grounded answers and 96% of cited
sentences confirmed, vs 1% and 77% for gpt-4o-mini, at $0.178 vs $0.265 per 1,000 answers.
R13: every call returns tokens, latency and cost; if the API fails, a no-cost fallback quotes the top reviews.
"""
from __future__ import annotations

import time

from src.rag_eval.answers import ABSTAIN, ANSWER_SYSTEM, answer_user, citations

ANSWER_MODEL = "gpt-6-luna"
# USD per 1M tokens, developers.openai.com/api/docs/pricing (read 2026-11-11); same values as configs/rag_eval.yaml
PRICE_IN, PRICE_OUT = 0.10, 0.50
BUDGET_PER_1000 = 0.30          # ADR-009 R13 budget
MAX_OUTPUT_TOKENS = 256


def fallback_answer(docs: list[tuple[str, str]], n: int = 3) -> str:
    """No-cost answer when the model is unavailable: the top reviews, quoted and cited."""
    if not docs:
        return ABSTAIN
    quotes = [f"“{text[:160].strip()}…” [{doc_id}]" for doc_id, text in docs[:n]]
    return "The assistant is unavailable, so here are the most relevant reviews: " + " ".join(quotes)


def answer(client, question: str, docs: list[tuple[str, str]], timeout_s: float = 30) -> dict:
    """docs: (review id, text) in retrieval order. Returns answer, cited ids, tokens, latency, cost and the model used."""
    t0 = time.time()
    try:
        r = client.responses.create(model=ANSWER_MODEL, instructions=ANSWER_SYSTEM, input=answer_user(question, docs),
                                    reasoning={"effort": "none"}, max_output_tokens=MAX_OUTPUT_TOKENS, timeout=timeout_s)
        text = r.output_text.strip()
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
        model = ANSWER_MODEL
    except Exception:  # noqa: BLE001 - any API failure falls back to the free extractive answer
        text, tin, tout, model = fallback_answer(docs), 0, 0, "fallback"
    valid = {d for d, _ in docs}
    return {"answer": text, "cited_ids": [c for c in citations(text) if c in valid], "model": model,
            "tokens_in": tin, "tokens_out": tout, "latency_s": round(time.time() - t0, 3),
            "cost_usd": (tin * PRICE_IN + tout * PRICE_OUT) / 1e6}
