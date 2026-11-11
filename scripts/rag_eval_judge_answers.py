"""Primary answer judge (ADR-009 step 7): gpt-6.1-terra, blind, one answer per call, same rubric for every system.

    ~/.venvs/portfolio/bin/python scripts/rag_eval_judge_answers.py A1 A0 S0 S1_seed0 ... [--dry-run]
Reads <data_dir>/answers/<system>.parquet; writes <data_dir>/judgements/primary_<system>.parquet (resumable).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_eval.config import load  # noqa: E402
from src.rag_eval.judge import ANSWER_JUDGE_SYSTEM, answer_judge_input, parse_answer_judgement  # noqa: E402
from src.rag_eval.ledger import Ledger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    cfg = load()
    d = cfg["paths"]["data_dir"]
    model = cfg["judge"]["primary"]["model"]
    q = pd.read_parquet(d / "questions.parquet").set_index("qid")
    text = dict(pd.read_parquet(d / "corpus.parquet")[["doc_id", "text"]].itertuples(index=False))
    ledger = Ledger(cfg)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60, max_retries=2)
    (d / "judgements").mkdir(exist_ok=True)

    for system in args.systems:
        ans = pd.read_parquet(d / "answers" / f"{system}.parquet")
        out_path = d / "judgements" / f"primary_{system}.parquet"
        done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["qid"])
        todo = ans[~ans.qid.isin(set(done.qid))]
        inputs = {r.qid: answer_judge_input(q.loc[r.qid, "question"], [(i, text[i]) for i in r.context_ids], r.answer)
                  for r in todo.itertuples()}
        est = ledger.cost(model, sum(len(u) for u, _ in inputs.values()) // 3, 30 * len(inputs))
        print(f"{system}: {len(todo)} to judge, estimated ${est:.3f}; spent ${ledger.spent():.4f}")
        if args.dry_run or todo.empty:
            continue
        ledger.check(est * 1.3)

        def work(qid):
            user, n = inputs[qid]
            err = ""
            for _ in range(3):
                try:
                    t0 = time.time()
                    r = client.responses.create(model=model, instructions=ANSWER_JUDGE_SYSTEM, input=user,
                                                reasoning={"effort": "none"}, text={"format": {"type": "json_object"}},
                                                max_output_tokens=300)
                    j = parse_answer_judgement(r.output_text, n)
                    ledger.log(f"answer judge {system}", model, r.usage.input_tokens, r.usage.output_tokens,
                               latency_s=round(time.time() - t0, 3))
                    return {"qid": qid, "system": system, "n_sentences": n, "supported": j["supported"],
                            "answers_question": j["answers_question"]}
                except Exception as e:  # noqa: BLE001 - retried, then reported
                    err = str(e)[:200]
            return {"qid": qid, "system": system, "error": err}

        ok = []
        with ThreadPoolExecutor(max_workers=8) as ex:            # save every 25 so an interruption loses little paid work
            for i, r in enumerate(ex.map(work, list(inputs)), start=1):
                if "error" in r:
                    print("failed:", r, flush=True)
                else:
                    ok.append(r)
                if i % 25 == 0 or i == len(inputs):
                    pd.concat([done, pd.DataFrame(ok)], ignore_index=True).to_parquet(out_path, index=False)
        print(f"{system}: judged {len(ok)}; spent ${ledger.spent():.4f}", flush=True)


if __name__ == "__main__":
    main()
