"""First screen of the app (R22, R26 item 7): headline results from committed files and the brief as compact cards."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/sabrinapribadi/ai-housing-analyst"
PAGES = "https://sabrinapribadi.github.io/ai-housing-analyst/"


def headline() -> dict:
    """Retrieval and answer study results (ADR-008, ADR-009), confirmation set and primary judge, as committed."""
    r = json.loads((ROOT / "data/outputs/rag_eval/retrieval_results.json").read_text())
    a = json.loads((ROOT / "data/outputs/rag_eval/answer_results.json").read_text())
    return {"ndcg_before": r["systems"]["R0"]["confirmation"]["ndcg@10"], "ndcg_after": r["systems"][r["decision"]]["confirmation"]["ndcg@10"],
            "grounded_new": a["judges"]["primary"]["A1"]["grounded_answer_rate"], "grounded_old": a["judges"]["primary"]["A0"]["grounded_answer_rate"]}


def brief() -> list[tuple[str, list[str]]]:
    import frame as F
    return F.load_brief(ROOT, REPO)


def render(st, listings: int, reviews: int) -> None:
    """Hero, KPI strip, the 5W1H brief and links (shared app frame)."""
    import frame as F
    h = headline()
    F.hero(st, f"Inside Airbnb Tokyo · {listings:,} listings · {reviews:,} reviews", "AI Housing Analyst",
           "Prices, places and guest sentiment for the Tokyo Airbnb market, with a data agent and cited answers from reviews.")
    F.kpis(st, [("Search nDCG@10", f"{h['ndcg_after']:.3f}", f"was {h['ndcg_before']:.3f}"),
                ("Grounded answers", f"{h['grounded_new']:.0%}", f"previous model: {h['grounded_old']:.0%}"),
                ("Listings", f"{listings:,}", "Tokyo, Inside Airbnb"),
                ("Reviews", f"{reviews:,}", "25,000 indexed for search")])
    F.brief(st, brief())
    F.links_line(st, {"Results page": PAGES, "Technical walkthrough": PAGES + "build.html", "Source": REPO})
