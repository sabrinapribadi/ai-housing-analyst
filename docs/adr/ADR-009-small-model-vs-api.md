# ADR-009: Fine-Tuned Small Open Model vs API Model for Review Answers

**Status:** Proposed — pre-registered 2026-11-11 before any run; results appended below
**Date:** 2026-11-11
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

- Review answers use gpt-4o-mini through the API (ADR-003); every question needs a key and sends reviews to a vendor.
- A small open model could run on one's own GPU: no key, data stays in-house, cost fixed per hour.
- **Question:** can Qwen3-1.7B, fine-tuned with LoRA and served by vLLM, give grounded, cited answers no worse than the smallest current API model, at lower cost and acceptable latency?

## Decision Drivers

- Grounding first: a host must be able to check every claim against a review (ADR-006).
- R13: cost per 1,000 answers and p95 latency measured for every system.
- R20: free GPU (Kaggle T4) for training and the benchmark; OpenAI spend ≤ $5.

## Considered Options

| ID | System | Tested here | Why |
|---|---|---|---|
| A1 | gpt-6-luna (API) | Yes | Smallest current API model; the bar to meet |
| A0 | gpt-4o-mini (API) | Yes (reference) | Today's production model |
| S0 | Qwen3-1.7B, no fine-tuning (vLLM) | Yes | Shows what fine-tuning adds |
| S1 | Qwen3-1.7B + LoRA (Hu et al. 2022), served by vLLM (Kwon et al. 2023) | Yes | The candidate |
| — | QLoRA (Dettmers et al. 2023) | Cited | Saves memory not needed at 1.7B on 16 GB |
| — | Full fine-tuning | Rejected | Does not fit a T4; LoRA is the standard low-cost route |
| — | Larger students (3–4B) | Cited | Kept for a follow-up if S1 fails narrowly |

## Pre-registered design

1. **Task:** "What do guests say about <aspect> [in <ward>]?"; the model sees the top 10 reviews from the retrieval chosen in ADR-008 and must answer in English, 2–5 sentences, with `[rev_id]` after each claim, or say the reviews do not cover it.
2. **Same contexts for all systems:** retrieved once, cached, so only the generator differs.
3. **Training data (S1):** 2,000 training questions answered by the open teacher Qwen3-8B (distillation, Hinton et al. 2015); kept only if 25–180 words, ≥ 1 citation, every cited id in the context. No API output is used for training.
4. **Splits:** test questions use (aspect, ward) pairs and one phrasing per aspect never seen in training; 100 validation questions for checkpoint choice.
5. **Test set:** 270 answerable + 30 out-of-scope questions.
6. **Per-answer checks:**
   - citation validity (code): ≥ 1 citation and every cited id in the context;
   - supported sentences (judge): each sentence backed by its cited reviews (Gao et al. 2023);
   - answers the question (judge);
   - **grounded answer** (primary) = all three true; out-of-scope: correct when it abstains without claims.
7. **Judges:** gpt-6.1-terra (primary, blind) and Qwen3-8B (cross-check). LLM judges favour their own family (Zheng et al. 2023; Panickssery et al. 2024): the API judge may favour A1, the teacher judge may favour S1, so **a gate passes only if both judges agree**. 50 judgments audited by a person; open until done.
8. **Seeds:** S1 trained with seeds 0, 1, 2; seed 0 is primary; spread > 5 points is flagged.
9. **Serving benchmark:** vLLM on one T4, 300 requests at concurrency 1 and 8, max 256 output tokens; API latency measured from the development machine.
10. **Cost per 1,000 answers:** API = measured tokens × price (configs/rag_eval.yaml); self-hosted = $0.526 per T4-hour × hours per 1,000 at measured throughput ÷ utilisation (0.5 central; 0.1 and 1.0 sensitivity).
11. **Statistics:** paired bootstrap over questions, 10,000 resamples (Sakai 2006).

## Gates (falsifiable)

| Gate | Pass when |
|---|---|
| **G2 Fine-tuning helps** | S1 − S0 grounded-answer rate ≥ +10 points, 95% interval above 0 |
| **G3 No worse than the API** | lower 95% bound of S1 − A1 ≥ −5 points; S1 citation validity ≥ 95%; S1 abstains on ≥ 80% of out-of-scope questions |
| **G4 Operations** | S1 p95 ≤ 3.0 s at concurrency 1 and ≤ 6.0 s at concurrency 8 on a T4; S1 cost per 1,000 (central) ≤ A1's measured cost per 1,000 |
| **Decision** | Recommend S1 only if G3 and G4 pass; otherwise keep the API model and document S1 as the offline, no-key option |

| Result | Diagnosis | Next action |
|---|---|---|
| G2 fails | Teacher answers did not teach grounding | Inspect filtered data; try the 4B student (cited) |
| G2 passes, G3 fails | Fine-tuning helps but 1.7B is too small | Report the gap; keep the API |
| G3 passes, G4 fails on cost | Low utilisation makes a GPU dearer than tokens | Report the break-even utilisation |
| All pass | A small open model is a credible replacement | Offer it where keys or data sharing are a problem |

## Budgets (R13)

- A1: ≤ $0.30 per 1,000 answers; p95 ≤ 3.0 s.
- Project OpenAI spend (answers, judging, relevance labels): ≤ $5, enforced by a ledger that stops the run before the cap.

## Consequences

- **Positive:** a measured answer to "API or own model?" with grounding, cost and latency side by side.
- **Negative:** no free host keeps a GPU online, so the live app shows precomputed answers for S1; judges are models until the audit is done.

## Deviations from the pre-registration

1. **Tokyo-wide question share (2026-11-11, before any retrieval, answer or label existed).** The registered `scoped_share: 0.4` (60% Tokyo-wide) cannot be met: each facet has one Tokyo-wide pair but 22 ward pairs, so unique Tokyo-wide questions run out. Facets were widened to 64 (4 per aspect; aspects unchanged) and Tokyo-wide questions are drawn as far as the pool allows. Actual shares: train 5.6%, validation 12%, test 2.3% Tokyo-wide; the rest name one ward. Counts (2,000 / 100 / 270 + 30) are unchanged.

## Empirical Results

_Appended after the run._

## Related Decisions

- ADR-003, ADR-004 (hallucination checks), ADR-006, ADR-007, ADR-008.

## Evidence (OpenAlex, retrieved 2026-11-11; generated by `python scripts/evidence.py`)

| Reference | Venue | Peer-reviewed | Year | Citations | Per year | Used for |
|---|---|---|---|---|---|---|
| Evaluating evaluation metrics based on the bootstrap | International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) | yes | 2006 | 227 | 10.8 | paired bootstrap tests for retrieval and system comparisons |
| LoRA: Low-Rank Adaptation of Large Language Models | ICLR 2022 (OpenAlex primary copy is arXiv) | yes | 2021 | 14,446 | 2407.7 | low-rank adaptation (LoRA) fine-tuning |
| QLoRA: Efficient Finetuning of Quantized LLMs | NeurIPS 2023 (OpenAlex primary copy is arXiv) | yes | 2023 | 4,042 | 1010.5 | quantised LoRA, the memory-saving alternative considered |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | ACM SIGOPS Symposium on Operating Systems Principles (SOSP) | yes | 2023 | 3,869 | 967.2 | vLLM and PagedAttention serving |
| Distilling the Knowledge in a Neural Network | arXiv 1503.02531 | no (preprint) | 2015 | 26,195 | 2182.9 | knowledge distillation from a larger teacher model |
| Qwen3 Technical Report | arXiv 2505.09388 | no (preprint) | 2025 | 17,653 | 8826.5 | Qwen3 open-weight models (student and teacher) |
| Enabling Large Language Models to Generate Text with Citations | EMNLP 2023 (OpenAlex primary copy is arXiv) | yes | 2023 | 585 | 146.2 | answers with citations, citation precision and recall |
| Ragas: Automated Evaluation of Retrieval Augmented Generation | EACL 2024 demos (OpenAlex primary copy is arXiv) | yes | 2023 | 677 | 169.2 | reference-free RAG faithfulness metrics (alternative considered) |
| Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | NeurIPS 2023 Datasets and Benchmarks (OpenAlex primary copy is arXiv) | yes | 2023 | 5,250 | 1312.5 | LLM-as-a-judge agreement and biases |
| LLM Evaluators Recognize and Favor Their Own Generations | NeurIPS 2024 (OpenAlex primary copy is arXiv) | yes | 2024 | 583 | 194.3 | LLM judges favour their own outputs (bias to control) |
