"""R1 (ADR-008): BM25 over the 25,000-review corpus. Writes <data_dir>/runs/R1.parquet (top 100 per query)."""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_eval.config import load  # noqa: E402
from src.rag_eval.retrieval import bm25_run  # noqa: E402

sets = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
cfg = load()
d = cfg["paths"]["data_dir"]
q = pd.read_parquet(d / "questions.parquet")
q = q[q.retrieval_set != ""] if sets == "retrieval" else (q if sets == "all" else q[q.set == sets])
t0 = time.time()
run = bm25_run(pd.read_parquet(d / "corpus.parquet"), q)
out = d / "runs" / "R1.parquet"
if out.exists():
    old = pd.read_parquet(out)
    run = pd.concat([old[~old.qid.isin(set(run.qid))], run], ignore_index=True)
run.to_parquet(out, index=False)
print(f"R1: {q.qid.nunique()} queries in {time.time() - t0:.0f}s -> {out}")
