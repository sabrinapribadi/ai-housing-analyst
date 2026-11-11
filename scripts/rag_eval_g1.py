"""Gate G1 (ADR-008): choose the best of R1-R4 on the 40 selection queries, confirm against R0 on the 80 others.

    ~/.venvs/portfolio/bin/python scripts/rag_eval_g1.py
Writes data/outputs/rag_eval/retrieval_results.json (committed) with per-system metrics, the choice and the gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_eval.config import load  # noqa: E402
from src.rag_eval.retrieval import paired_bootstrap, per_query_metrics  # noqa: E402


def main() -> None:
    cfg = load()
    d = cfg["paths"]["data_dir"]
    r = cfg["retrieval"]
    gate = r["gate_G1"]
    q = pd.read_parquet(d / "questions.parquet")
    qrels = pd.read_parquet(d / "qrels.parquet")
    sets = {name: set(q.qid[q.retrieval_set == name]) for name in ("selection", "confirmation")}
    assert set(qrels.qid) == sets["selection"] | sets["confirmation"], "labels must cover exactly the 120 retrieval queries"

    per = {}
    summary = {}
    for s in r["systems"]:
        run = pd.read_parquet(d / "runs" / f"{s}.parquet")
        m = per_query_metrics(run, qrels, k=r["k"], relevant_at=r["relevant_at"])
        per[s] = m.set_index("qid")
        summary[s] = {name: {k: round(float(m[m.qid.isin(ids)][k].mean()), 4) for k in ("ndcg@10", "recall@10", "mrr@10")}
                      for name, ids in sets.items()}

    challengers = [s for s in r["systems"] if s != "R0"]
    chosen = max(challengers, key=lambda s: summary[s]["selection"][gate["metric"]])
    conf = sorted(sets["confirmation"])
    bs = paired_bootstrap(per["R0"].loc[conf, gate["metric"]].to_numpy(), per[chosen].loc[conf, gate["metric"]].to_numpy(),
                          n=gate["bootstrap"], ci=gate["ci"])
    passed = bs["mean_gain"] >= gate["min_gain"] and bs["ci_low"] > 0
    all_bs = {s: paired_bootstrap(per["R0"].loc[conf, gate["metric"]].to_numpy(), per[s].loc[conf, gate["metric"]].to_numpy(),
                                  n=gate["bootstrap"], ci=gate["ci"]) for s in challengers}
    grades = qrels.grade.value_counts().sort_index()
    out = {
        "gate": "G1", "metric": gate["metric"], "min_gain": gate["min_gain"],
        "chosen_on_selection": chosen, "confirmation": bs, "passed": bool(passed),
        "decision": chosen if passed else "R0",
        "systems": summary,
        "all_vs_R0_on_confirmation (reported, not used for the choice)": all_bs,
        "labels": {"n": int(len(qrels)), "grades": {str(k): int(v) for k, v in grades.items()},
                   "relevant_share": round(float((qrels.grade >= r["relevant_at"]).mean()), 4),
                   "queries_with_no_relevant": int((qrels.groupby("qid").grade.max() < r["relevant_at"]).sum()),
                   "judge": cfg["judge"]["primary"]["model"], "human_audit": cfg["judge"]["human_audit"]["status"]},
    }
    p = cfg["paths"]["results_dir"] / "retrieval_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("chosen_on_selection", "confirmation", "passed", "decision")}, indent=1))
    print(pd.DataFrame({s: v["confirmation"] for s, v in summary.items()}).T.to_string())


if __name__ == "__main__":
    main()
