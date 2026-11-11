"""Loads configs/rag_eval.yaml and resolves paths (data outside git, results inside)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "rag_eval.yaml").read_text())
    cfg["paths"]["data_dir"] = (ROOT / cfg["paths"]["data_dir"]).resolve()
    cfg["paths"]["results_dir"] = ROOT / cfg["paths"]["results_dir"]
    for k in ("reviews", "listings"):
        cfg["paths"][k] = ROOT / cfg["paths"][k]
    return cfg
