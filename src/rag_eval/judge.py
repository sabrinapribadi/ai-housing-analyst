"""Relevance labels for the retrieval pool (ADR-008 step 4). Answer judging (ADR-009) is added in P10.

Blind: reviews are shuffled and shown as d1..dN; the judge never sees system names or review ids.
"""
from __future__ import annotations

import json
import random
import re

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


# ---- answer judging (ADR-009 step 6b-c); same rubric for the primary and the cross-check judge ----
ANSWER_JUDGE_SYSTEM = """You check answers about Tokyo short-term rentals against the guest reviews they were written from.
You get the question, the reviews (each with an id such as rev_123), and the answer split into numbered sentences.
For each sentence decide "supported":
- true if every factual claim in it is stated or clearly implied by the reviews it cites in square brackets;
- false if it cites nothing but makes a new factual claim, cites reviews that do not back it, or overstates them.
A sentence that only says the reviews do not cover the question is supported when no review addresses the asked topic.
Then decide "answers_question": true if the answer addresses the asked topic using the reviews, or correctly says the
reviews do not cover it when none of them address it; false otherwise.
Return JSON: {"supported": [true/false for each sentence in order], "answers_question": true/false}."""



def split_sentences(answer: str) -> list[str]:
    """Sentence split that keeps trailing citations with their sentence."""
    parts, start = [], 0
    for m in re.finditer(r"[.!?](?:\s*\[rev_\d+\])*(?=\s+[A-Z0-9\"'(]|\s*$)", answer.strip()):
        parts.append(answer.strip()[start:m.end()].strip())
        start = m.end()
    tail = answer.strip()[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def answer_judge_input(question: str, docs: list[tuple[str, str]], answer: str) -> tuple[str, int]:
    sents = split_sentences(answer)
    reviews = "\n\n".join(f"[{doc_id}] {text[:1500]}" for doc_id, text in docs)
    numbered = "\n".join(f"S{i + 1}: {s}" for i, s in enumerate(sents))
    return (f"Question: {question}\n\nReviews:\n{reviews}\n\nAnswer sentences:\n{numbered}\n\n"
            f"Return JSON with {len(sents)} booleans in \"supported\"."), len(sents)


def parse_answer_judgement(output_text: str, n_sentences: int) -> dict:
    j = json.loads(output_text)
    sup = [bool(x) for x in j["supported"]]
    if len(sup) != n_sentences:
        raise ValueError(f"{len(sup)} verdicts for {n_sentences} sentences")
    return {"supported": sup, "answers_question": bool(j["answers_question"])}
