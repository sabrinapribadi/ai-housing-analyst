"""Question sets never leak test (facet, ward) pairs or the test-only phrasing into training (ADR-008/009)."""
import copy

import pandas as pd

from src.rag_eval import questions
from src.rag_eval.config import load


def _cfg(tmp_path):
    cfg = copy.deepcopy(load())
    wards = [f"Ward{i} Ku" for i in range(22)]
    pd.DataFrame({"doc_id": [f"rev_{i}" for i in range(22 * 250)], "listing_id": 0,
                  "ward": [w for w in wards for _ in range(250)], "text": "x"}).to_parquet(tmp_path / "corpus.parquet")
    cfg["paths"]["data_dir"] = tmp_path
    return cfg


def test_sizes_and_no_leak(tmp_path):
    allq = questions.build(_cfg(tmp_path))
    questions.check_no_leak(allq)
    sizes = allq.groupby(["set", "answerable"]).size().to_dict()
    assert sizes == {("test", False): 30, ("test", True): 270, ("train", True): 2000, ("val", True): 100}
    assert allq.question.is_unique
    assert (allq.retrieval_set == "selection").sum() == 40 and (allq.retrieval_set == "confirmation").sum() == 80


def test_deterministic(tmp_path):
    cfg = _cfg(tmp_path)
    assert questions.build(cfg).equals(questions.build(cfg))
