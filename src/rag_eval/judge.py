"""Relevance labels for the retrieval pool (ADR-008 step 4). Answer judging (ADR-009) is added in P10.

Blind: reviews are shuffled and shown as d1..dN; the judge never sees system names or review ids.
"""
from __future__ import annotations

import json
import random

RELEVANCE_SYSTEM = """You grade how well guest reviews of Tokyo short-term rentals answer a traveller's question.
Grade each review on this scale:
0 = unrelated to the question's topic.
1 = mentions the topic but says nothing about the guest's experience of it.
2 = describes the guest's experience of the asked topic (positive or negative).
3 = a direct, specific answer to the question (clear details about the asked topic).
Reviews may be in any language; judge the meaning. Ignore the area name in the question: every review shown is
already from the right area. Return JSON: {"grades": {"d1": <0-3>, "d2": <0-3>, ...}} with one grade per review."""


def relevance_messages(question: str, docs: list[tuple[str, str]], seed: int) -> tuple[str, dict[str, str]]:
    """docs: (doc_id, text). Returns the user prompt and the blind key -> doc_id map."""
    order = docs[:]
    random.Random(seed).shuffle(order)
    keys = {f"d{i + 1}": doc_id for i, (doc_id, _) in enumerate(order)}
    body = "\n\n".join(f"[d{i + 1}] {text[:1500]}" for i, (_, text) in enumerate(order))
    return f"Question: {question}\n\nReviews:\n{body}\n\nReturn the grades as JSON.", keys


def parse_grades(output_text: str, keys: dict[str, str]) -> dict[str, int]:
    """Map blind keys back to doc ids; a missing or invalid grade raises so the batch is retried."""
    g = json.loads(output_text)["grades"]
    out = {}
    for k, doc_id in keys.items():
        v = int(g[k])
        if v not in (0, 1, 2, 3):
            raise ValueError(f"grade {v} for {k}")
        out[doc_id] = v
    return out
