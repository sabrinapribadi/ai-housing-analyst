"""Rebuild the exact 25,000-review corpus behind the production index (ADR-008 design step 1).

Same steps as ReviewRAGAgent.build_index: merge ward, keep comments longer than 30 characters, sample with seed 42.
Ids are rev_0 … rev_24999 in sample order, matching the Chroma ids. Reviewer names are never kept (privacy).
"""
from __future__ import annotations

import pandas as pd

from src.rag_eval.config import load


def build(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load()
    c = cfg["corpus"]
    reviews = pd.read_parquet(cfg["paths"]["reviews"])[["listing_id", "comments"]]
    listings = pd.read_parquet(cfg["paths"]["listings"])[["id", "neighbourhood_cleansed"]]
    df = reviews.merge(listings, left_on="listing_id", right_on="id", how="left")
    df = df[df["comments"].notna() & (df["comments"].str.strip().str.len() >= c["min_chars"])]
    df = df.dropna(subset=["neighbourhood_cleansed"]).reset_index(drop=True)
    df = df.sample(c["n_docs"], random_state=c["sample_seed"]).reset_index(drop=True)
    out = pd.DataFrame({"doc_id": [f"{c['id_prefix']}{i}" for i in range(len(df))],
                        "listing_id": df["listing_id"].astype("int64").to_numpy(),
                        "ward": df["neighbourhood_cleansed"].astype(str).to_numpy(),
                        "text": df["comments"].astype(str).to_numpy()})
    return out


if __name__ == "__main__":
    cfg = load()
    corpus = build(cfg)
    path = cfg["paths"]["data_dir"] / "corpus.parquet"
    corpus.to_parquet(path, index=False)
    print(f"wrote {path} ({len(corpus):,} reviews, {corpus.ward.nunique()} wards)")
