"""OpenAI spend ledger (R13; ADR-009 budgets): every call is logged with tokens, latency and cost,
and a call is refused if the project total would cross the cap in configs/rag_eval.yaml.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.rag_eval.config import load


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load()
        self.path: Path = self.cfg["paths"]["data_dir"] / "openai_ledger.jsonl"
        self.cap = float(self.cfg["budgets"]["project_openai_cap_usd"])

    def spent(self) -> float:
        if not self.path.exists():
            return 0.0
        return sum(json.loads(line)["cost_usd"] for line in self.path.read_text().splitlines() if line.strip())

    def cost(self, model: str, tokens_in: int, tokens_out: int = 0, cached_in: int = 0, batch: bool = False) -> float:
        p = self.cfg["prices"][model]
        pin = p.get("batch_input", p["input"] / 2) if batch else p["input"]
        pout = p.get("batch_output", p.get("output", 0) / 2) if batch else p.get("output", 0)
        pcache = p.get("cached_input", p["input"])
        return ((tokens_in - cached_in) * pin + cached_in * pcache + tokens_out * pout) / 1e6

    def check(self, estimate_usd: float) -> None:
        if self.spent() + estimate_usd > self.cap:
            raise BudgetExceeded(f"spent ${self.spent():.4f} + estimate ${estimate_usd:.4f} would cross the ${self.cap:.2f} cap")

    def log(self, purpose: str, model: str, tokens_in: int, tokens_out: int = 0, cached_in: int = 0,
            latency_s: float | None = None, batch: bool = False, n_calls: int = 1) -> float:
        c = self.cost(model, tokens_in, tokens_out, cached_in, batch)
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": purpose, "model": model, "n_calls": n_calls,
               "tokens_in": tokens_in, "cached_in": cached_in, "tokens_out": tokens_out, "batch": batch,
               "latency_s": latency_s, "cost_usd": round(c, 8)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return c
