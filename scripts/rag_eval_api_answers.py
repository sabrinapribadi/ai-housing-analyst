"""API answers for the 300 test questions (ADR-009 systems A1 = gpt-6-luna, A0 = gpt-4o-mini).

Same prompt (src/rag_eval/answers.py) and the same R4 top-10 contexts as every other system. Calls run one at a time
so latency is measured cleanly (R13): tokens, latency and cost are logged per call.
    ~/.venvs/portfolio/bin/python scripts/rag_eval_api_answers.py A1|A0
Writes <data_dir>/answers/<system>.parquet (resumable).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_eval.answers import ANSWER_SYSTEM, answer_user  # noqa: E402
from src.rag_eval.config import load  # noqa: E402
from src.rag_eval.ledger import Ledger  # noqa: E402


def main() -> None:
    system = sys.argv[1]
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    cfg = load()
    d = cfg["paths"]["data_dir"]
    spec = cfg["generation"]["systems"][system]
    model = spec["model"]
    ret = json.loads((cfg["paths"]["results_dir"] / "retrieval_results.json").read_text())["decision"]
    q = pd.read_parquet(d / "questions.parquet")
    q = q[q.set == "test"]
    text = dict(pd.read_parquet(d / "corpus.parquet")[["doc_id", "text"]].itertuples(index=False))
    run = pd.read_parquet(d / "runs" / f"{ret}.parquet")
    ctx = {qid: g.sort_values("rank").doc_id.head(cfg["retrieval"]["k"]).tolist() for qid, g in run[run.qid.isin(q.qid)].groupby("qid")}

    out_path = d / "answers" / f"{system}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    todo = q[~q.qid.isin(set(done.qid))] if len(done) else q
    ledger = Ledger(cfg)
    est = ledger.cost(model, 1600 * len(todo), 200 * len(todo))
    print(f"{system} ({model}): {len(todo)} to answer, estimated ${est:.3f}; spent ${ledger.spent():.4f}")
    ledger.check(est * 1.5)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60, max_retries=2)
    kw = {"reasoning": {"effort": spec["reasoning_effort"]}} if "reasoning_effort" in spec else {"temperature": cfg["generation"]["temperature"]}

    rows = []
    for r in todo.itertuples():
        user = answer_user(r.question, [(i, text[i]) for i in ctx[r.qid]])
        t0 = time.time()
        resp = client.responses.create(model=model, instructions=ANSWER_SYSTEM, input=user,
                                       max_output_tokens=cfg["generation"]["max_output_tokens"], **kw)
        lat = time.time() - t0
        u = resp.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        cost = ledger.log(f"answers {system}", model, u.input_tokens, u.output_tokens, cached_in=cached, latency_s=round(lat, 3))
        rows.append({"qid": r.qid, "system": system, "model": model, "context_ids": ctx[r.qid], "answer": resp.output_text.strip(),
                     "tokens_in": u.input_tokens, "cached_in": cached, "tokens_out": u.output_tokens, "latency_s": lat, "cost_usd": cost})
        if len(rows) % 25 == 0:
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(out_path, index=False)
    allr = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    allr.to_parquet(out_path, index=False)
    lat = allr.latency_s
    print(f"{system}: n={len(allr)} p50={lat.quantile(.5):.2f}s p95={lat.quantile(.95):.2f}s "
          f"cost per 1,000 = ${1000 * allr.cost_usd.mean():.3f}; spent ${ledger.spent():.4f}")


if __name__ == "__main__":
    main()
