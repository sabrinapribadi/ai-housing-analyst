# ADR-008: Retrieval Evaluation — Is the Current Review Search Good Enough?

**Status:** Accepted — pre-registered 2026-11-11 before any run; **G1 passed** (results below)
**Date:** 2026-11-11
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

- Review search (ADR-003) embeds 25,000 reviews with OpenAI text-embedding-3-small at 256 dimensions and returns the 10 nearest; its quality was never measured.
- About a third of reviews are Chinese, Japanese or Korean (ADR-006), and questions often name a ward.
- **Question:** does a keyword, multilingual, hybrid or reranked search find better evidence than the current one?

## Decision Drivers

- Answers can only be as good as the reviews retrieved.
- Free to run (R20); judged within the $5 cap (ADR-007).
- Choose on one query set, confirm on another, to avoid picking the luckiest system.

## Considered Options

| ID | System | Tested here | Why |
|---|---|---|---|
| R0 | **Current:** OpenAI text-embedding-3-small, 256 dims | Yes (baseline) | Production today |
| R1 | BM25 (Robertson and Zaragoza 2009); Latin words plus Chinese/Japanese/Korean character pairs | Yes | Strong on exact terms such as "wifi" or a station name |
| R2 | bge-m3 dense, multilingual (Chen et al. 2024) | Yes | Built for many languages |
| R3 | Reciprocal rank fusion of R0, R1, R2 (Cormack et al. 2009, k = 60) | Yes | Combines exact-term and meaning matches |
| R4 | R3 top 50 reranked by bge-reranker-v2-m3 | Yes | Cross-encoder reads question and review together |
| — | Qwen3-Embedding / Qwen3-Reranker | Cited only | One multilingual family is enough for the question |
| — | Larger OpenAI embeddings (3-large) | Rejected | Costs more; the question is whether the method, not the size, is the gap |

## Pre-registered design

1. **Corpus:** the same 25,000 reviews as production (filter > 30 characters, random sample, seed 42); ids `rev_0 … rev_24999`.
2. **Queries:** 120 answerable test questions from the templates in `configs/rag_eval.yaml` (40% limited to one ward; every system applies the same ward filter).
3. **Split:** 40 queries choose the best of R1–R4 (selection); the other 80 confirm it against R0 (gate).
4. **Labels:** pool = union of each system's top 10 per query; gpt-6.1-terra grades each pooled review 0–3 (0 unrelated, 1 topic only, 2 describes the asked aspect, 3 direct answer), blind to the system (Thomas et al. 2024). Relevant = grade ≥ 2.
5. **Metrics:** nDCG@10 (primary; Järvelin and Kekäläinen 2002), recall@10, MRR@10.
6. **Test:** paired bootstrap over queries, 10,000 resamples (Sakai 2006).
7. **Audit:** 50 random labels checked by a person; until done, labels are reported as model-judged.

## Gate G1 (falsifiable)

- **Pass:** the system chosen on the 40 selection queries beats R0 on the 80 confirmation queries by **≥ 0.03 nDCG@10**, with the 95% interval of the gain above 0.
- **Fail:** keep R0 and report all five systems.

| Result | Diagnosis | Next action |
|---|---|---|
| Pass | Current search misses evidence another method finds | Adopt the winner in the app and in ADR-009's contexts |
| Gain > 0 but < 0.03 or interval includes 0 | Difference too small to matter | Keep R0 |
| R0 best | Current search is adequate | Keep R0; retrieval is not the bottleneck |

## Consequences

- **Positive:** the first measured retrieval quality for this app; contexts for ADR-009 are fixed by this decision.
- **Negative:** labels come from a model until the human audit is done; one snapshot, one city.

## Deviations from the pre-registration

1. **Tokyo-wide question share (2026-11-11, before any retrieval, answer or label existed).** The registered `scoped_share: 0.4` (60% Tokyo-wide) cannot be met: each facet has one Tokyo-wide pair but 22 ward pairs, so unique Tokyo-wide questions run out. Facets were widened to 64 (4 per aspect; aspects unchanged) and Tokyo-wide questions are drawn as far as the pool allows. Actual shares: train 5.6%, validation 12%, test 2.3% Tokyo-wide; the rest name one ward. Counts (2,000 / 100 / 270 + 30) are unchanged.

2. **App adoption deferred (2026-11-11, after G1).** The registered next action on a pass was "adopt the winner in the app". R4 reranks 50 reviews per question with a 0.6B cross-encoder: 0.69 s per question on a T4 GPU (1,646 s for 2,400 questions), and the free app host has no GPU (R20). R4 is adopted for ADR-009's contexts; the live app keeps R0 until a lighter reranker is measured on CPU.

## Empirical Results (2026-11-11)

- **Labels:** 3,248 pooled (question, review) pairs, mean 27 per question; grades 0 / 1 / 2 / 3 = 1,969 / 171 / 372 / 736; 34% relevant (grade ≥ 2); 4 of 120 questions had no relevant review in the pool. Judge gpt-6.1-terra, blind; cost $0.50. Human audit of 50 labels: **open**.
- **Choice on the 40 selection queries (nDCG@10):** R0 0.445, R1 0.214, R2 0.188, R3 0.341, **R4 0.471** → R4 chosen.
- **Confirmation on the 80 other queries:**

| System | nDCG@10 | Recall@10 | MRR@10 | Gain vs R0 (95% interval) |
|---|---|---|---|---|
| R0 current (OpenAI 256-d) | 0.459 | 0.435 | 0.591 | — |
| R1 BM25 | 0.389 | 0.364 | 0.565 | −0.070 (−0.146 to +0.013) |
| R2 bge-m3 dense | 0.300 | 0.290 | 0.431 | −0.159 (−0.218 to −0.099) |
| R3 fusion R0+R1+R2 | 0.414 | 0.396 | 0.569 | −0.044 (−0.097 to +0.015) |
| **R4 fusion + reranker** | **0.545** | **0.521** | **0.669** | **+0.087 (+0.021 to +0.153)** |

- **G1: passed** (+0.087 ≥ 0.03; interval above 0). Decision: R4 for ADR-009 contexts.
- **What the numbers mean:**
  - **nDCG@10:** how close the top 10 is to the best possible order, weighting strongly relevant reviews at the top; 1 is perfect.
  - **Recall@10:** of the reviews that answer the question, the share found in the top 10. R4 finds about 52 in 100, R0 about 44. A host reading 10 reviews sees roughly one more useful review per question.
  - **So what:** retrieval was a real bottleneck; the gain comes from reading question and review together (the reranker), not from a different embedding model or from fusion alone.
- **Surprises:** bge-m3 alone is the weakest system, and fusing it in lowers R0; the reranker recovers and passes R0. Ward names dominate short questions, which may hurt dense models; not tested further.
- **Files:** `data/outputs/rag_eval/retrieval_results.json`, `data/outputs/rag_eval/qrels.csv`.

## Related Decisions

- ADR-003 (review retrieval), ADR-006 (industry alignment), ADR-007 (technology), ADR-009 (small model vs API).

## Evidence (OpenAlex, retrieved 2026-11-11; generated by `python scripts/evidence.py`)

| Reference | Venue | Peer-reviewed | Year | Citations | Per year | Used for |
|---|---|---|---|---|---|---|
| The Probabilistic Relevance Framework: BM25 and Beyond | Foundations and Trends® in Information Retrieval | yes | 2009 | 4,230 | 235.0 | BM25 keyword retrieval (baseline) |
| Reciprocal rank fusion outperforms condorcet and individual rank learning methods | International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) | yes | 2009 | 1,115 | 61.9 | reciprocal rank fusion for hybrid retrieval |
| M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation | Findings of ACL 2024 (OpenAlex primary copy is arXiv) | yes | 2024 | 2,230 | 743.3 | multilingual dense retrieval and reranking models |
| Cumulated gain-based evaluation of IR techniques | ACM Transactions on Information Systems | yes | 2002 | 5,137 | 205.5 | nDCG, graded-relevance ranking metric |
| Large Language Models can Accurately Predict Searcher Preferences | International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) | yes | 2024 | 294 | 98.0 | LLM relevance labels agree with human searcher preferences |
| Evaluating evaluation metrics based on the bootstrap | International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) | yes | 2006 | 227 | 10.8 | paired bootstrap tests for retrieval and system comparisons |
