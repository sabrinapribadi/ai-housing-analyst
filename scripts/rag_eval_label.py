"""Grade the pooled top 10 of R0-R4 for the 120 retrieval questions with gpt-6.1-terra (ADR-008 step 4).

    ~/.venvs/portfolio/bin/python scripts/rag_eval_label.py [--dry-run]
Resumable: finished questions are cached in <data_dir>/labels_raw.jsonl. Writes <data_dir>/qrels.parquet.
Stops before the OpenAI cap (src/rag_eval/ledger.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from src.rag_eval.judge import RELEVANCE_SYSTEM, parse_grades, relevance_messages  # noqa: E402
from src.rag_eval.ledger import Ledger  # noqa: E402

CHUNK = 10   # reviews per call


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    cfg = load()
    d = cfg["paths"]["data_dir"]
    model = cfg["judge"]["primary"]["model"]
    depth = cfg["retrieval"]["pool_depth"]
    q = pd.read_parquet(d / "questions.parquet")
    q = q[q.retrieval_set != ""].set_index("qid")
    text = dict(pd.read_parquet(d / "corpus.parquet")[["doc_id", "text"]].itertuples(index=False))
    runs = {s: pd.read_parquet(d / "runs" / f"{s}.parquet") for s in cfg["retrieval"]["systems"]}
    pool = pd.concat([r[(r["rank"] <= depth) & r.qid.isin(q.index)][["qid", "doc_id"]] for r in runs.values()]).drop_duplicates()
    print(f"pool: {len(pool):,} (question, review) pairs for {pool.qid.nunique()} questions; mean {len(pool) / pool.qid.nunique():.1f} per question")

    raw_path = d / "labels_raw.jsonl"
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            r = json.loads(line)
            done[(r["qid"], r["chunk"])] = r["grades"]

    jobs = []
    for qid, g in pool.groupby("qid"):
        ids = sorted(g.doc_id)
        for c in range(0, len(ids), CHUNK):
            if (qid, c // CHUNK) not in done:
                jobs.append((qid, c // CHUNK, ids[c:c + CHUNK]))
    est_in = sum(len(RELEVANCE_SYSTEM) + sum(len(text[i][:1500]) for i in ids) for _, _, ids in jobs) / 3   # ~3 chars per token
    ledger = Ledger(cfg)
    est = ledger.cost(model, int(est_in), 15 * len(jobs))
    print(f"{len(jobs)} calls to make; estimated ${est:.3f}; spent so far ${ledger.spent():.4f}")
    if args.dry_run or not jobs:
        return
    ledger.check(est * 1.3)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60, max_retries=2)

    def work(job):
        qid, chunk, ids = job
        seed = int(hashlib.sha256(f"{qid}:{chunk}".encode()).hexdigest()[:8], 16)   # stable across runs
        user, keys = relevance_messages(q.loc[qid, "question"], [(i, text[i]) for i in ids], seed=seed)
        for attempt in range(3):
            try:
                t0 = time.time()
                r = client.responses.create(model=model, instructions=RELEVANCE_SYSTEM, input=user,
                                            reasoning={"effort": "none"}, text={"format": {"type": "json_object"}},
                                            max_output_tokens=200)
                grades = parse_grades(r.output_text, keys)
                ledger.log("relevance labels", model, r.usage.input_tokens, r.usage.output_tokens,
                           latency_s=round(time.time() - t0, 3))
                return {"qid": qid, "chunk": chunk, "grades": grades}
            except Exception as e:  # noqa: BLE001 - retried, then reported
                err = str(e)[:200]
        return {"qid": qid, "chunk": chunk, "error": err}

    with ThreadPoolExecutor(max_workers=8) as ex, raw_path.open("a") as f:
        for res in ex.map(work, jobs):
            if "grades" in res:
                f.write(json.dumps(res) + "\n")
                f.flush()
            else:
                print("failed:", res)

    rows = []
    if not raw_path.exists() or not raw_path.read_text().strip():
        print("no labels written")
        return
    for line in raw_path.read_text().splitlines():
        r = json.loads(line)
        rows += [{"qid": r["qid"], "doc_id": k, "grade": v} for k, v in r["grades"].items()]
    qrels = pd.DataFrame(rows).drop_duplicates(["qid", "doc_id"])
    qrels.to_parquet(d / "qrels.parquet", index=False)
    print(f"qrels: {len(qrels):,} grades; grade counts {qrels.grade.value_counts().sort_index().to_dict()}; spent ${ledger.spent():.4f}")


if __name__ == "__main__":
    main()
