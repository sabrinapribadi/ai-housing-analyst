"""Gates G2-G4 (ADR-009): grounded answers per system and judge, paired bootstrap, cost per 1,000 and p95 latency.

    ~/.venvs/portfolio/bin/python scripts/rag_eval_gates.py
Inputs: <data_dir>/answers/*.parquet, <data_dir>/judgements/{primary,cross}_*.parquet, <data_dir>/bench.json.
Writes data/outputs/rag_eval/answer_results.json (committed). A gate passes only if it passes under BOTH judges.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_eval.answers import code_checks  # noqa: E402
from src.rag_eval.config import load  # noqa: E402
from src.rag_eval.retrieval import paired_bootstrap  # noqa: E402

SYSTEMS = ["A1", "A0", "S0", "S1_seed0", "S1_seed1", "S1_seed2"]


def per_question(d: Path, system: str, judge: str, q: pd.DataFrame) -> pd.DataFrame | None:
    a, j = d / "answers" / f"{system}.parquet", d / "judgements" / f"{judge}_{system}.parquet"
    if not (a.exists() and j.exists()):
        return None
    ans = pd.read_parquet(a)
    chk = pd.DataFrame([code_checks(x, list(c)) for x, c in zip(ans.answer, ans.context_ids)])
    ans = pd.concat([ans[["qid"]].reset_index(drop=True), chk], axis=1)
    jj = pd.read_parquet(j)[["qid", "supported", "answers_question"]]
    m = ans.merge(jj, on="qid", how="left").join(q[["answerable"]], on="qid")
    m["all_supported"] = m.supported.map(lambda s: bool(len(s)) and all(s) if s is not None else np.nan)
    m["supported_rate"] = m.supported.map(lambda s: float(np.mean(s)) if s is not None and len(s) else np.nan)
    m["grounded"] = (m.citations_valid & (m.all_supported == True) & (m.answers_question == True)).astype(float)  # noqa: E712
    return m.set_index("qid")


def summarise(m: pd.DataFrame) -> dict:
    a, o = m[m.answerable], m[~m.answerable]
    return {"n_answerable": int(len(a)), "judged": int(a.answers_question.notna().sum()),
            "grounded_answer_rate": round(float(a.grounded.mean()), 4),
            "citation_validity": round(float(a.citations_valid.mean()), 4),
            "supported_sentence_rate": round(float(a.supported_rate.mean()), 4),
            "answers_question_rate": round(float(a.answers_question.astype(float).mean()), 4),
            "abstained_answerable": round(float(a.abstained.mean()), 4),
            "abstention_out_of_scope": round(float(o.abstained.mean()), 4) if len(o) else None}


def main() -> None:
    cfg = load()
    d = cfg["paths"]["data_dir"]
    g = cfg["gates"]
    q = pd.read_parquet(d / "questions.parquet").set_index("qid")
    q = q[q.set == "test"]
    out = {"judges": {}, "gates": {}}
    per = {}
    for judge in ("primary", "cross"):
        per[judge] = {s: m for s in SYSTEMS if (m := per_question(d, s, judge, q)) is not None}
        out["judges"][judge] = {s: summarise(m) for s, m in per[judge].items()}

    def diff(judge, a, b):
        ma, mb = per[judge].get(a), per[judge].get(b)
        if ma is None or mb is None:
            return None
        ids = sorted(q.index[q.answerable])
        return paired_bootstrap(ma.loc[ids, "grounded"].to_numpy(), mb.loc[ids, "grounded"].to_numpy(),
                                n=10000, ci=g["G2_finetune_helps"]["ci"])

    g2, g3 = {}, {}
    for judge in ("primary", "cross"):
        b2 = diff(judge, "S0", "S1_seed0")
        g2[judge] = None if b2 is None else {**b2, "pass": b2["mean_gain"] * 100 >= g["G2_finetune_helps"]["min_gain_points"] and b2["ci_low"] > 0}
        b3 = diff(judge, "A1", "S1_seed0")
        s1 = out["judges"][judge].get("S1_seed0")
        g3[judge] = None if b3 is None else {**b3, "pass": b3["ci_low"] * 100 >= g["G3_non_inferior_to_api"]["margin_points"]
                                             and s1["citation_validity"] >= g["G3_non_inferior_to_api"]["min_citation_validity"]
                                             and s1["abstention_out_of_scope"] >= g["G3_non_inferior_to_api"]["min_abstention"]}
    both = lambda x: all(v is not None and v["pass"] for v in x.values())   # noqa: E731
    out["gates"]["G2"] = {"by_judge": g2, "pass": both(g2)}
    out["gates"]["G3"] = {"by_judge": g3, "pass": both(g3)}

    # G4: latency and cost
    bench_p = d / "bench.json"
    led = pd.DataFrame([json.loads(x) for x in (d / "openai_ledger.jsonl").read_text().splitlines()])
    a1 = pd.read_parquet(d / "answers" / "A1.parquet")
    a1_cost = 1000 * float(a1.cost_usd.mean())
    g4 = {"A1": {"cost_per_1000": round(a1_cost, 4), "p50_s": round(float(a1.latency_s.quantile(.5)), 3),
                 "p95_s": round(float(a1.latency_s.quantile(.95)), 3), "measured_from": "development Mac over the internet, one call at a time"}}
    if bench_p.exists():
        bench = json.loads(bench_p.read_text())
        price = cfg["prices"]["t4_on_demand_per_hour"]
        for name in ("S1_seed0", "S1_merged", "S0"):
            if name in bench:
                b = bench[name]
                aps = b["concurrency_8"]["answers_per_s"]
                cost = {str(u): round(price * (1000 / aps) / 3600 / u, 4) for u in [g["G4_operations"]["utilisation_central"]]
                        + g["G4_operations"]["utilisation_sensitivity"]}
                g4[name] = {"p95_s_c1": b["concurrency_1"]["p95_s"], "p50_s_c1": b["concurrency_1"]["p50_s"],
                            "p95_s_c8": b["concurrency_8"]["p95_s"], "answers_per_s_c8": aps, "cost_per_1000_by_utilisation": cost}
        s1 = g4.get("S1_seed0")
        if s1:
            c = s1["cost_per_1000_by_utilisation"][str(g["G4_operations"]["utilisation_central"])]
            g4["pass"] = (s1["p95_s_c1"] <= g["G4_operations"]["p95_s_concurrency_1"] and s1["p95_s_c8"] <= g["G4_operations"]["p95_s_concurrency_8"]
                          and c <= a1_cost)
            g4["break_even_utilisation_vs_A1"] = round(price * (1000 / s1["answers_per_s_c8"]) / 3600 / a1_cost, 4)
    out["gates"]["G4"] = g4
    out["decision"] = ("serve S1" if out["gates"]["G3"]["pass"] and g4.get("pass") else
                       "keep the API model; S1 documented as the offline / no-key option")
    out["openai_spend_usd"] = round(float(led.cost_usd.sum()), 4)
    p = cfg["paths"]["results_dir"] / "answer_results.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
