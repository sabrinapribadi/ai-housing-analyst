"""Retrieval systems R1 (BM25) and R3 (fusion), metrics and the paired bootstrap (ADR-008).

R0 runs in scripts/rag_eval_r0.py (production index); R2 (bge-m3) and R4 (reranker) run on a Kaggle GPU.
Every run is a table: qid, rank, doc_id, score. Ward-scoped questions only rank reviews from that ward.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]+")
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Latin words (lower case, English stop words removed) plus character pairs for Chinese, Japanese and Korean."""
    text = str(text).lower()
    toks = [w for w in _WORD.findall(text) if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    for run in _CJK.findall(text):
        toks += [run[i:i + 2] for i in range(len(run) - 1)] if len(run) > 1 else [run]
    return toks


def bm25_run(corpus: pd.DataFrame, questions: pd.DataFrame, depth: int = 100) -> pd.DataFrame:
    bm25 = BM25Okapi([tokenize(t) for t in corpus.text])
    wards = corpus.ward.to_numpy()
    ids = corpus.doc_id.to_numpy()
    rows = []
    for _, q in questions.iterrows():
        s = bm25.get_scores(tokenize(q.question))
        if q.scope != "ALL":
            s = np.where(wards == q.scope, s, -np.inf)
        top = np.argsort(-s, kind="stable")[:depth]
        top = top[np.isfinite(s[top])]
        rows += [{"qid": q.qid, "rank": r, "doc_id": ids[i], "score": float(s[i])} for r, i in enumerate(top, start=1)]
    return pd.DataFrame(rows)


def rrf(runs: list[pd.DataFrame], k: int = 60, depth: int = 100) -> pd.DataFrame:
    """Reciprocal rank fusion (Cormack et al. 2009): score = sum over runs of 1 / (k + rank)."""
    allr = pd.concat(runs, ignore_index=True)
    allr["rrf"] = 1.0 / (k + allr["rank"])
    fused = allr.groupby(["qid", "doc_id"], as_index=False)["rrf"].sum()
    fused = fused.sort_values(["qid", "rrf", "doc_id"], ascending=[True, False, True])
    fused["rank"] = fused.groupby("qid").cumcount() + 1
    fused = fused[fused["rank"] <= depth].rename(columns={"rrf": "score"})
    return fused[["qid", "rank", "doc_id", "score"]].reset_index(drop=True)


def per_query_metrics(run: pd.DataFrame, qrels: pd.DataFrame, k: int = 10, relevant_at: int = 2) -> pd.DataFrame:
    """nDCG@k (gain 2^grade - 1), recall@k and MRR@k per query; unjudged documents count as grade 0."""
    grades = {(r.qid, r.doc_id): int(r.grade) for r in qrels.itertuples()}
    out = []
    for qid, g in qrels.groupby("qid"):
        ideal = sorted(g.grade.astype(int), reverse=True)[:k]
        idcg = sum((2 ** gr - 1) / math.log2(i + 2) for i, gr in enumerate(ideal))
        top = run[(run.qid == qid) & (run["rank"] <= k)].sort_values("rank")
        got = [grades.get((qid, d), 0) for d in top.doc_id]
        dcg = sum((2 ** gr - 1) / math.log2(i + 2) for i, gr in enumerate(got))
        n_rel = int((g.grade >= relevant_at).sum())
        hits = [gr >= relevant_at for gr in got]
        mrr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
        out.append({"qid": qid, "ndcg@10": dcg / idcg if idcg else 0.0,
                    "recall@10": sum(hits) / n_rel if n_rel else np.nan, "mrr@10": mrr, "n_relevant": n_rel})
    return pd.DataFrame(out)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    """Mean of b - a with a percentile interval from resampling queries (Sakai 2006)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[~np.isnan(d)]
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean_gain": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_queries": int(len(d))}
