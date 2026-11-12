# ADR-009: Fine-Tuned Small Open Model vs API Model for Review Answers

**Status:** Accepted — pre-registered 2026-11-11 before any run; **G2, G3 and G4 failed**; the app switches from gpt-4o-mini to gpt-6-luna (results below)
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

## Empirical Results (2026-11-12)

**Setup as registered:** 300 test questions (270 answerable, 30 out of scope); every system saw the same top 10 reviews from R4 (ADR-008). Teacher: 2,100 Qwen3-8B answers, 1,863 train + 94 validation kept by the filter (106 abstentions, 14 invalid citations, 23 off-length dropped). LoRA: 2 epochs on 2× T4, about 2.2 hours per seed; validation loss 0.469 / 0.472 / 0.469 (seeds 0 / 1 / 2), epoch 2 kept for all.

**Answer quality (answerable questions; primary judge gpt-6.1-terra / cross-check judge Qwen3-8B):**

| System | Grounded answers | Supported sentences | Answers the question | Valid citations (code) | Abstains when reviews are silent (code) |
|---|---|---|---|---|---|
| A1 gpt-6-luna | **77.8% / 90.4%** | 92.0% / 99.3% | 96.3% / 99.6% | 91.8% | 96.7% |
| A0 gpt-4o-mini (old production) | 1.1% / 82.6% | 47.3% / 97.9% | 91.1% / 100% | 88.9% | 100% |
| S0 Qwen3-1.7B, no tuning | 0.7% / 23.0% | 23.6% / 80.2% | 67.8% / 97.4% | 70.7% | 53.3% |
| S1 Qwen3-1.7B + LoRA, seed 0 | 3.3% / 72.2% | 47.4% / 92.9% | 89.6% / 100% | 95.9% | **0%** |
| S1 seeds 1 / 2 | 1.8%, 4.1% / 70.0%, 71.5% | 43.0%, 44.6% / 92.4%, 92.8% | — | 94.8%, 95.2% | 0%, 0% |

- **What "grounded answer" means:** at least one citation, every cited id in the context, every sentence backed by the reviews it cites, and the question answered. One uncited claim fails the answer.
- **Why the judges differ:** they agree on 47% of gpt-4o-mini's sentence verdicts. The primary judge applies the registered rule that an uncited new claim is unsupported (it marks 54% of gpt-4o-mini's sentences unsupported); the cross-check judge marks 2%. gpt-4o-mini and S1 open 94% and 81% of answers with an uncited summary sentence; gpt-6-luna 11%. Human audit of 50 judgements: **open**, and it decides which judge to trust on this rule.
- **Seed stability:** spread of 2.3 points (primary) and 2.2 points (cross) across seeds, under the 5-point flag.

**Speed and cost (one T4 for S0/S1; API from the development machine):**

| System | p50 / p95, one at a time | p95, 8 at a time | Answers per second (8 at a time) | Cost per 1,000 answers |
|---|---|---|---|---|
| A1 gpt-6-luna | 2.76 s / 4.50 s | — | — | **$0.178** (measured tokens) |
| A0 gpt-4o-mini | 2.47 s / 3.63 s (partly during a network outage) | — | — | $0.265 |
| S1 LoRA adapter (gated) | 6.57 s / 9.32 s | 24.3 s | 0.51 | $0.571 at 50% GPU use ($2.85 at 10%, $0.285 at 100%) |
| S1 merged weights | 2.87 s / 3.94 s | 14.8 s | 0.82 | $0.356 at 50% ($0.178 at 100%) |
| S0 base model | 2.50 s / 3.89 s | 16.1 s | 0.95 | $0.309 at 50% |

Self-hosted cost = $0.526 per T4-hour × hours for 1,000 answers at the measured throughput ÷ utilisation.

**Gates:**

| Gate | Primary judge | Cross-check judge | Result |
|---|---|---|---|
| G2 fine-tuning helps (S1 − S0 ≥ +10 points) | +2.6 (+0.4 to +5.2) | +49.3 (+42.6 to +55.9) | **Fail** (judges disagree) |
| G3 no worse than the API (S1 − A1 lower bound ≥ −5; citations ≥ 95%; abstention ≥ 80%) | −74.4 (−80.0 to −68.5) | −18.1 (−24.1 to −12.2) | **Fail** (both judges; abstention 0%) |
| G4 operations (p95 ≤ 3.0 s / 6.0 s; cost ≤ A1) | 9.3 s / 24.3 s; $0.571 vs $0.178 | — | **Fail**: break-even needs 160% GPU use |

**Decision:**
1. Keep an API model; do not serve S1. S1 stays documented as an offline, no-key option, but it must learn to abstain first.
2. **Switch the app's review answers from gpt-4o-mini to gpt-6-luna**, with the study's cited-answer prompt. gpt-6-luna beats gpt-4o-mini under both judges: grounded answers +76.7 points (+71.1 to +81.9) strict, +7.8 points (+4.1 to +11.5) lenient; and it is 33% cheaper ($0.178 vs $0.265 per 1,000). Its p95 of 4.5 s misses the 3.0 s budget, like gpt-4o-mini's 3.6 s.
3. The 7-tool data agent keeps gpt-4o-mini: this study did not test it.

**Why the small model failed:**
- **No abstention:** the registered filter required a citation, so all 106 teacher abstentions were removed; the student never saw one. Next time, keep abstentions as their own class.
- **Copied style:** the teacher opened with an uncited summary sentence and the student learned it. Filter on per-sentence citations.
- **Hardware:** a T4 decodes about 17 tokens per second per request with a LoRA adapter; merging the adapter halves latency, but a GPU only beats per-token pricing at very high, steady load.

**Spend:** OpenAI $4.02 of the $5 cap (relevance labels $0.50, answers $0.08 + $0.09, judging $3.07 including $0.29 lost to an interrupted unsaved run, plus probes). Kaggle GPU: about 11 hours of the free weekly quota.

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
