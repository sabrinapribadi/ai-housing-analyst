"""R0 (ADR-008): today's production search — OpenAI text-embedding-3-small @ 256 dims, ChromaDB HNSW, ward filter.

Runs against a COPY of data/chroma (a newer chromadb may migrate the file). Environment: ~/.venvs/aha-chroma.
    ~/.venvs/aha-chroma/bin/python scripts/rag_eval_r0.py [--sets retrieval|test|val|train]
Writes <data_dir>/runs/R0.parquet: qid, rank, doc_id, score (top 100, for fusion).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import chromadb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_eval.config import load  # noqa: E402
from src.rag_eval.ledger import Ledger  # noqa: E402

DEPTH = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="retrieval", help="retrieval (120 labelled queries), test, val, train or all")
    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    cfg = load()
    d = cfg["paths"]["data_dir"]
    q = pd.read_parquet(d / "questions.parquet")
    if args.sets == "retrieval":
        q = q[q.retrieval_set != ""]
    elif args.sets != "all":
        q = q[q.set == args.sets]
    out_path = d / "runs" / "R0.parquet"
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["qid", "rank", "doc_id", "score"])
    q = q[~q.qid.isin(set(done.qid))]
    if q.empty:
        print("nothing to do")
        return

    r0 = cfg["retrieval"]["systems"]["R0"]
    ledger = Ledger(cfg)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    col = chromadb.PersistentClient(path=str(d / "chroma_copy")).get_collection("tokyo_reviews")

    texts = q.question.tolist()
    ledger.check(ledger.cost(r0["model"], sum(len(t) for t in texts)))       # chars >= tokens: safe upper bound
    embs, tokens = [], 0
    for i in range(0, len(texts), 100):
        t0 = time.time()
        r = client.embeddings.create(model=r0["model"], input=texts[i:i + 100], dimensions=r0["dims"])
        embs += [e.embedding for e in r.data]
        tokens += r.usage.prompt_tokens
        ledger.log("R0 query embeddings", r0["model"], r.usage.prompt_tokens, latency_s=round(time.time() - t0, 3),
                   n_calls=len(texts[i:i + 100]))

    rows = []
    for (_, row), emb in zip(q.iterrows(), embs):
        where = None if row.scope == "ALL" else {"neighbourhood": row.scope}
        res = col.query(query_embeddings=[emb], n_results=DEPTH, where=where, include=["distances"])
        for rank, (doc_id, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), start=1):
            rows.append({"qid": row.qid, "rank": rank, "doc_id": doc_id, "score": -float(dist)})
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"R0: {len(q)} queries, {tokens} embedding tokens, ledger total ${ledger.spent():.5f} -> {out_path}")


if __name__ == "__main__":
    main()
