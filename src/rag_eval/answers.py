"""One answer prompt for every system (A0, A1, S0, S1, teacher), and code-only answer checks (ADR-009).

All systems see the same top-10 reviews from the retrieval chosen in ADR-008, with their real ids.
"""
from __future__ import annotations

import re

ABSTAIN = "The reviews do not cover this."

ANSWER_SYSTEM = f"""You answer travellers' questions about Tokyo short-term rentals using only the guest reviews provided.
Rules:
1. Write in English, 2 to 5 sentences.
2. After each claim, cite the supporting review ids in square brackets, for example [rev_123] or [rev_12][rev_40].
3. Use only what the reviews say. If guests disagree, say so.
4. Do not name hosts or guests.
5. If the reviews say nothing about the question, reply exactly: {ABSTAIN}"""

_CITE = re.compile(r"\[(rev_\d+)\]")


def answer_user(question: str, docs: list[tuple[str, str]]) -> str:
    body = "\n\n".join(f"[{doc_id}] {text[:1500]}" for doc_id, text in docs)
    return f"Question: {question}\n\nReviews:\n{body}"


def citations(answer: str) -> list[str]:
    return _CITE.findall(answer)


def is_abstention(answer: str) -> bool:
    return answer.strip().rstrip(".").lower() == ABSTAIN.rstrip(".").lower()


def code_checks(answer: str, context_ids: list[str]) -> dict:
    """Citation validity (ADR-009 step 6a): at least one citation and every cited id is in the context."""
    cited = citations(answer)
    ctx = set(context_ids)
    abstained = is_abstention(answer)
    return {
        "abstained": abstained,
        "n_citations": len(cited),
        "citations_valid": (not abstained) and len(cited) > 0 and all(c in ctx for c in cited),
        "invalid_ids": sorted({c for c in cited if c not in ctx}),
        "words": len(answer.split()),
    }
