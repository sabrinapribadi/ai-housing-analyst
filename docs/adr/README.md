# Architecture Decision Records

ADR-001 to ADR-005 were written retrospectively on 2026-11-11 from the code and PRD (the decisions date from June 2026). ADR-006 to ADR-009 were written before their work (P7 pre-registration, 2026-11-11).

| ADR | Decision | Status | Summary |
|---|---|---|---|
| [ADR-001](ADR-001-framing-and-data.md) | Framing and data | Accepted | One Inside Airbnb Tokyo snapshot (Sep 2025): 27,945 listings, 444,182 reviews; aggregates only on pages. |
| [ADR-002](ADR-002-tool-using-agent.md) | Tool-using agent | Accepted | gpt-4o-mini with 7 tools; numbers computed by tools, not the model. |
| [ADR-003](ADR-003-review-retrieval.md) | Retrieval over reviews | Accepted | 25,000 reviews embedded (256 dimensions) in ChromaDB; answers cite sources. |
| [ADR-004](ADR-004-hallucination-checks.md) | Hallucination checks | Accepted | 6 fixed questions vs Parquet ground truth; 0 in the recorded run; limits stated. |
| [ADR-005](ADR-005-analytics-methods.md) | Analytics methods | Accepted | TextBlob sentiment and K-Means areas, not validated against labels; Prophet forecast on hold (calendar has no prices). |
| [ADR-006](ADR-006-industry-alignment.md) | Industry alignment | Accepted | Questions follow Airbnb's six rated categories plus aspects hosts manage; Japan's 180-night Minpaku rules. |
| [ADR-007](ADR-007-technology-assessment.md) | Technology assessment | Accepted | Adopt Qwen3-1.7B (student), Qwen3-8B (teacher), LoRA, vLLM on a free Kaggle T4; hold gated-licence models and paid GPU hosts. |
| [ADR-008](ADR-008-retrieval-evaluation.md) | Retrieval evaluation | Accepted (G1 passed) | Reranked fusion lifts nDCG@10 from 0.459 to 0.545 (+0.087, interval +0.021 to +0.153); app keeps today's search until a CPU reranker is measured. |
| [ADR-009](ADR-009-small-model-vs-api.md) | Small model vs API | Accepted (G2–G4 failed) | LoRA Qwen3-1.7B on vLLM failed on grounding, abstention, latency and cost; app answers switch from gpt-4o-mini to gpt-6-luna (grounded +77 points strict, +8 lenient; 33% cheaper). |

## Decision Dependencies

```
ADR-001 (framing and data)
  ├── ADR-002 (tool-using agent) ── ADR-004 (hallucination checks)
  ├── ADR-003 (retrieval over reviews)
  │     └── ADR-006 (industry alignment) ── ADR-007 (technology)
  │            └── ADR-008 (retrieval evaluation) ── ADR-009 (small model vs API)
  └── ADR-005 (analytics methods)
```
