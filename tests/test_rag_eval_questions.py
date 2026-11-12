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


def test_answer_code_checks():
    from src.rag_eval.answers import ABSTAIN, code_checks
    ok = code_checks("Guests found it quiet [rev_1]. Some heard trains [rev_2][rev_1].", ["rev_1", "rev_2"])
    assert ok["citations_valid"] and ok["n_citations"] == 3 and not ok["abstained"]
    bad = code_checks("Quiet [rev_9].", ["rev_1"])
    assert not bad["citations_valid"] and bad["invalid_ids"] == ["rev_9"]
    none = code_checks("Quiet and clean.", ["rev_1"])
    assert not none["citations_valid"]
    ab = code_checks(ABSTAIN, ["rev_1"])
    assert ab["abstained"] and not ab["citations_valid"]


def test_review_answer_cost_and_fallback():
    """App answers (ADR-009): tokens and cost reported per call; an API failure falls back to quoted reviews at no cost."""
    from types import SimpleNamespace

    from src.agent.review_answer import answer

    docs = [("rev_1", "Quiet street, five minutes to the station."), ("rev_2", "Thin walls, heard neighbours.")]

    class Ok:
        responses = SimpleNamespace(create=lambda **kw: SimpleNamespace(
            output_text="Quiet street [rev_1]. Thin walls [rev_2][rev_9].", usage=SimpleNamespace(input_tokens=1000, output_tokens=100)))

    r = answer(Ok(), "Is it quiet?", docs)
    assert r["model"] == "gpt-6-luna" and r["cited_ids"] == ["rev_1", "rev_2"]
    assert abs(r["cost_usd"] - (1000 * 0.10 + 100 * 0.50) / 1e6) < 1e-12

    class Down:
        responses = SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("down")))

    f = answer(Down(), "Is it quiet?", docs)
    assert f["model"] == "fallback" and f["cost_usd"] == 0 and "[rev_1]" in f["answer"]
