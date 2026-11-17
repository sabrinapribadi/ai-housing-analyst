"""Record both agents' answers for the app's replay chat (R26), with tools, number check, cost, latency and charts.

    set -a; . ./.env; set +a; python scripts/record_app_replays.py      # needs OPENAI_API_KEY; prints cost, never the key
Writes src/ui/replays_data_agent.json, src/ui/replays_reviews.json and the charts under src/ui/replay_plots/.
Uses the committed listings and review index (Inside Airbnb Tokyo, CC BY 4.0) only.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "src" / "ui"
sys.path[:0] = [str(UI), str(ROOT)]
import aha_agents as AG  # noqa: E402

SOURCES = {"data_agent": "the committed Tokyo listings (27,945)", "reviews": "the committed review index (25,000 reviews)"}


def record(name: str, agent) -> dict:
    runs = []
    plots = UI / "replay_plots"
    plots.mkdir(exist_ok=True)
    for i, q in enumerate(agent.QUESTIONS):
        r = agent.live(q, None)
        kept = []
        for j, img in enumerate(r.get("images", [])):
            dst = plots / f"{name}_{i}_{j}.png"
            shutil.copyfile(img, dst)
            kept.append(str(dst.relative_to(UI)))
        r["images"] = kept
        runs.append(r)
        print(f"{r['latency_ms']:>7.0f} ms  ${r['cost_usd']:.6f}  grounded={r['grounded']}  {q[:60]}  {r['ungrounded_numbers'] or ''}")
    lat = sorted(r["latency_ms"] for r in runs)
    out = {"recorded_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), "model": agent.MODEL,  # noqa: UP017 (3.8+)
           "source": SOURCES[name], "runs": runs,
           "summary": {"questions": len(runs), "grounded": sum(r["grounded"] for r in runs),
                       "cost_usd_total": round(sum(r["cost_usd"] for r in runs), 6),
                       "cost_per_1000_usd": round(1000 * sum(r["cost_usd"] for r in runs) / len(runs), 4),
                       "latency_ms_p50": lat[len(lat) // 2], "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))]}}
    (UI / f"replays_{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(name, json.dumps(out["summary"]))
    return out


if __name__ == "__main__":
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    os.chdir(ROOT)                                   # the agents read data/ and write charts to outputs/ relative to the repo
    which = sys.argv[1:] or ["data_agent", "reviews"]
    if "data_agent" in which:
        record("data_agent", AG.DataAgent)
    if "reviews" in which:
        record("reviews", AG.ReviewAgent)
