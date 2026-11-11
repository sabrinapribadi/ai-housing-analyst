# Changelog

Newest first. Each entry uses STAR (Situation, Task, Action, Result) so it can be told as a story.

## [5.3.0] — 2026-11-11 — CI on every push
**Files:** tests/test_clustering_unit.py, .github/workflows/ci.yml
- **Situation:** No CI; the existing checks needed local data files or a live OpenAI key.
- **Task:** CI on every push (rule R19) that runs without data or keys.
- **Action:** Unit test of the K-Means clusterer on synthetic Tokyo-like listings (incomplete rows skipped, every complete row assigned); the live hallucination suite is skipped in CI and runs locally; error-level lint.
- **Result:** Unit test passes in CI; live checks stay local, so CI never spends tokens.
- **Talking point:** CI checks the analytics on every push without spending a cent on the LLM.

## [5.2.1] — 2026-10-12 — MIT license
**Files:** LICENSE, CHANGELOG.md (new)
- **Situation:** The repository had no license file, so by default nobody else could legally reuse the code.
- **Task:** Make the portfolio code reusable with credit, consistently across all portfolio repositories.
- **Action:** Added the standard MIT license (© 2026 Sabrina Pribadi). Data sources keep their own terms; the license covers code only.
- **Result:** GitHub now detects the license as MIT; no code changed.
- **Talking point:** "All my portfolio code is MIT-licensed, so anyone can reuse it with attribution."

