"""Guest-experience question sets (ADR-006, ADR-008, ADR-009), generated from templates: no model, no cost.

- Aspects: Airbnb's six rated categories plus ten aspects hosts manage; each aspect has facets (what a guest asks about).
- A question is (facet, scope, phrasing). Scope is all of Tokyo or one ward with >= ward_min_docs indexed reviews.
- Split by (facet, scope) pair, so a test question never shares its pair with training; phrasing 3 is test-only.
- Out-of-scope questions ask about things Tokyo short-term rentals do not have, to test abstention.
"""
from __future__ import annotations

import hashlib
import random

import pandas as pd

from src.rag_eval.config import load

FACETS: dict[str, list[str]] = {
    # Airbnb's six rated categories (4 facets each)
    "cleanliness": ["overall cleanliness", "bathroom cleanliness", "clean bedding and towels", "smells in the room"],
    "accuracy": ["whether the place matched the photos", "whether the listing description was accurate",
                 "room size compared with the listing", "amenities that were listed but missing"],
    "check-in": ["self check-in", "finding the building", "check-in instructions", "early check-in or late check-out"],
    "communication": ["how responsive the host was", "the host's local recommendations",
                      "help from the host during the stay", "house rules and how they were explained"],
    "location": ["convenience of the location", "shops and restaurants nearby", "convenience stores nearby", "how safe the area felt"],
    "value": ["value for money", "price compared with hotels", "extra fees and cleaning charges", "whether guests would stay again"],
    # aspects hosts manage (Minpaku duties and common complaints; 4 facets each)
    "noise": ["noise from the street", "thin walls and noise from neighbours", "trains or traffic noise at night",
              "quiet hours and noise rules"],
    "transport access": ["walking distance to the station", "airport access", "access to major sightseeing areas",
                         "bus and train connections"],
    "wifi and workspace": ["wifi speed and reliability", "working from the apartment", "pocket wifi provided by the host",
                           "a desk or place to work"],
    "kitchen and amenities": ["the kitchen and cooking", "the washing machine and laundry", "toiletries and amenities provided",
                              "the fridge and microwave"],
    "space and layout": ["how much space there was", "room for luggage", "stairs and lifts", "the balcony or view"],
    "families with children": ["staying with children", "space for a family or group", "baby cots and child-friendly items",
                               "parks and play areas nearby"],
    "luggage": ["storing luggage before check-in or after check-out", "carrying luggage to the apartment",
                "coin lockers and luggage services nearby", "space to open suitcases"],
    "bathroom and shower": ["water pressure and hot water", "the size of the bathroom", "the bathtub", "the toilet"],
    "bed comfort": ["bed and futon comfort", "pillows and sleep quality", "the number of beds", "blackout curtains and light at night"],
    "air conditioning and heating": ["air conditioning in summer", "heating in winter", "ventilation and humidity",
                                     "how easy the air conditioner was to use"],
}

TEMPLATES = [
    "What do guests say about {facet}{where}?",
    "How do guests describe {facet}{where}?",
    "Are there any complaints or praise about {facet}{where}?",   # test-only phrasing
]

OUT_OF_SCOPE = ["the on-site casino", "the private helipad", "ski-in ski-out access", "the yacht mooring",
                "horse riding from the property", "the private golf course"]


def _bucket(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def build(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load()
    q = cfg["questions"]
    seed = q["seed"]
    corpus = pd.read_parquet(cfg["paths"]["data_dir"] / "corpus.parquet")
    counts = corpus["ward"].value_counts()
    wards = sorted(counts[counts >= q["ward_min_docs"]].index)

    rows = []
    for aspect, facets in FACETS.items():
        for facet in facets:
            for scope in ["ALL"] + wards:
                b = _bucket(f"{facet}|{scope}", seed)
                split = "train" if b < 0.80 else ("val" if b < 0.85 else "test")
                for t, tpl in enumerate(TEMPLATES):
                    if t == 2 and split != "test":
                        continue
                    where = "" if scope == "ALL" else f" in {scope.replace(' Ku', '').replace(' Shi', '')}"
                    rows.append({"aspect": aspect, "facet": facet, "scope": scope, "phrasing": t + 1, "split": split,
                                 "answerable": True, "question": tpl.format(facet=facet, where=where)})
    pool = pd.DataFrame(rows)

    rng = random.Random(seed)

    def sample(split: str, n: int) -> pd.DataFrame:
        """Deviation 1 (ADR-008/009): Tokyo-wide questions are capped by the unique (facet, phrasing) pool,
        so the Tokyo-wide share is as large as the pool allows (target 1 - scoped_share) and is reported."""
        part = pool[pool.split == split]
        scoped, unscoped = part[part.scope != "ALL"], part[part.scope == "ALL"]
        n_unscoped = min(round(n * (1 - q["scoped_share"])), len(unscoped))
        idx_u = rng.sample(list(unscoped.index), n_unscoped)
        idx_s = rng.sample(list(scoped.index), n - n_unscoped)
        return pool.loc[idx_s + idx_u]

    train = sample("train", q["n_train"])
    val = sample("val", q["n_val"])
    test = sample("test", q["n_test_answerable"])

    combos = [(f, sc, t) for f in OUT_OF_SCOPE for sc in ["ALL"] + wards for t in range(len(TEMPLATES))]
    oos = []
    for facet, scope, t in rng.sample(combos, q["n_test_out_of_scope"]):     # unique combinations
        where = "" if scope == "ALL" else f" in {scope.replace(' Ku', '').replace(' Shi', '')}"
        oos.append({"aspect": "out of scope", "facet": facet, "scope": scope, "phrasing": t + 1, "split": "test",
                    "answerable": False, "question": TEMPLATES[t].format(facet=facet, where=where)})
    test = pd.concat([test, pd.DataFrame(oos)], ignore_index=True)

    # retrieval subsets (ADR-008): 40 selection + 80 confirmation, drawn from answerable test questions
    ans = test[test.answerable].reset_index(drop=True)
    order = rng.sample(range(len(ans)), q["retrieval_val"] + q["retrieval_test"])
    ret_sel = set(order[: q["retrieval_val"]])
    ret_conf = set(order[q["retrieval_val"]:])

    out = []
    for name, df in (("train", train), ("val", val), ("test", test)):
        df = df.reset_index(drop=True).copy()
        df["set"] = name
        out.append(df)
    allq = pd.concat(out, ignore_index=True)
    allq["qid"] = [f"q{i:05d}" for i in range(len(allq))]
    test_ans_qids = allq[(allq.set == "test") & allq.answerable].qid.tolist()
    allq["retrieval_set"] = ""
    allq.loc[allq.qid.isin([test_ans_qids[i] for i in ret_sel]), "retrieval_set"] = "selection"
    allq.loc[allq.qid.isin([test_ans_qids[i] for i in ret_conf]), "retrieval_set"] = "confirmation"
    return allq[["qid", "set", "retrieval_set", "answerable", "aspect", "facet", "scope", "phrasing", "question"]]


def check_no_leak(allq: pd.DataFrame) -> None:
    """Test pairs never appear in train or val; phrasing 3 never appears outside test."""
    pair = allq.facet + "|" + allq.scope
    test_pairs = set(pair[(allq.set == "test") & allq.answerable])
    assert not test_pairs & set(pair[allq.set != "test"]), "test (facet, scope) pair leaked into train/val"
    assert not ((allq.phrasing == 3) & (allq.set != "test")).any(), "test-only phrasing leaked"


if __name__ == "__main__":
    cfg = load()
    allq = build(cfg)
    check_no_leak(allq)
    path = cfg["paths"]["data_dir"] / "questions.parquet"
    allq.to_parquet(path, index=False)
    print(f"wrote {path}")
    print(allq.groupby(["set", "answerable"]).size().to_string())
    print(allq.retrieval_set.value_counts().to_string())
    t = allq[allq.set == "test"]
    print("test scoped share:", round((t.scope != "ALL").mean(), 2), "| facets in test:", t.facet.nunique())
