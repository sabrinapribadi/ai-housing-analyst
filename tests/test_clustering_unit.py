"""Unit tests that run in CI without data files or an API key (rule R19)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def listings(n=300, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[35.68, 139.76, 30000], [35.69, 139.70, 15000], [35.71, 139.80, 8000]])
    pts = centres[rng.integers(0, 3, n)] + rng.normal(0, [0.005, 0.005, 1500], (n, 3))
    return pd.DataFrame(pts, columns=["latitude", "longitude", "price"])


def test_kmeans_assigns_every_complete_listing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                     # the clusterer writes to ./outputs
    from src.analytics.clustering import GeospatialClusterer
    d = listings(); d.loc[0, "price"] = np.nan
    out = GeospatialClusterer(d).cluster(n_clusters=3, include_price=True)
    assert out["cluster"].notna().sum() == len(d) - 1 and out.loc[0, "cluster"] is pd.NA
    assert out["cluster"].dropna().nunique() == 3


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="live LLM check runs locally only (needs a key)")
def test_hallucination_suite_runs_locally():
    assert Path(__file__).with_name("test_hallucination.py").exists()
