"""First screen of the app (R22, R26 item 7): headline results from committed files and the brief as compact cards."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import yaml

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
    b = yaml.safe_load((ROOT / "docs/brief.yaml").read_text())
    fix = lambda t: re.sub(r"\]\((?!https?://)([^)]+)\)", lambda m: f"]({REPO}/blob/main/{m.group(1)})", t)  # noqa: E731
    return [(i["w"], [fix(str(x)) for x in i["lines"]]) for i in b["brief"]]


def md_html(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" target="_blank">\1</a>', t)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)


CSS = """<style>
.hero-eyebrow{font-size:.76rem;letter-spacing:.09em;text-transform:uppercase;font-weight:600;opacity:.65;margin:0}
.hero-title{font-size:clamp(2rem,4.2vw,3rem);font-weight:800;line-height:1.05;margin:.2rem 0 .4rem}
.hero-sub{font-size:1.02rem;opacity:.8;max-width:70ch;margin:0 0 .4rem}
.brief-w{font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;font-weight:700;color:#ff8a8d;margin-bottom:.2rem}
.brief-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px;margin:.2rem 0 .3rem}
.bcard{border:1px solid #2a2a2a;border-radius:10px;padding:8px 12px;font-size:.86rem;line-height:1.4;background:#161616}
.bcard a{color:#ff8a8d}
</style>"""


def render(st, listings: int, reviews: int) -> None:
    h = headline()
    st.html(CSS + f"""<p class="hero-eyebrow">Inside Airbnb Tokyo · {listings:,} listings · {reviews:,} reviews</p>
<h1 class="hero-title">AI Housing Analyst</h1>
<p class="hero-sub">Prices, places and guest sentiment for the Tokyo Airbnb market, with a data agent and cited answers from reviews.</p>""")
    k = st.columns(4)
    k[0].metric("Search nDCG@10", f"{h['ndcg_after']:.3f}", f"was {h['ndcg_before']:.3f}", delta_color="off", delta_arrow="off", border=True)
    k[1].metric("Grounded answers", f"{h['grounded_new']:.0%}", f"previous model: {h['grounded_old']:.0%}", delta_color="off", delta_arrow="off", border=True)
    k[2].metric("Listings", f"{listings:,}", "Tokyo, Inside Airbnb", delta_color="off", delta_arrow="off", border=True)
    k[3].metric("Reviews", f"{reviews:,}", "25,000 indexed for search", delta_color="off", delta_arrow="off", border=True)
    items = brief()
    st.html('<div class="brief-grid">' + "".join(f'<div class="bcard"><div class="brief-w">{w}</div><div>{md_html(lines[0])}</div></div>'
                                               for w, lines in items) + "</div>")
    with st.expander("Full brief"):
        for w, lines in items:
            st.markdown(f"**{w}:** " + " ".join(lines))
    st.caption(f"[Results page]({PAGES}) · [Technical walkthrough]({PAGES}build.html) · [Source]({REPO}) · Free Streamlit app: it sleeps "
               f"after 12 hours without visits; the [results page]({PAGES}) is always on and carries every result.")
