# Changelog

Newest first. Each entry uses STAR (Situation, Task, Action, Result) so it can be told as a story.

## [5.5.0] — 2026-11-11 — Results page, walkthrough, ADRs, two corrections
**Files:** docs/index.html, docs/build.html, docs/assets/ (new); docs/adr/ (new, ADR-001 to ADR-005); docs/references.yaml, scripts/evidence.py (new); PRD.md → docs/PRD.md; tests/browser/ (new); docs/brief.yaml, README.md, .github/workflows/ci.yml
- **Situation:** Decisions lived only in the PRD; no results page; the brief said 8 tools and listed a Prophet forecast.
- **Task:** Results page and walkthrough (R12, R14), retrospective ADRs with evidence (R2, R4), browser tests and Pages gated on CI (R19), and every stated number checked against the data.
- **Action:** Pages generated from the listings file (DuckDB); five ADRs with OpenAlex evidence tables; browser checks for hovers, sliders, themes and phone width; Pages deploy only after tests pass. Profiled every column a claim relies on.
- **Result:** 7 tools (not 8). The calendar has no prices (0 of 10,199,937 rows), so the Prophet forecast is on hold and removed from the brief. K-Means uses a fixed k = 5; the elbow search is not called. Hidden gems are 6,593, not about 6,000. 81 results-page and 189 walkthrough checks pass.
- **Talking point:** "Writing the records found a feature the data could not support, so I took the claim down."

## [5.4.0] — 2026-11-11 — At-a-glance brief (R22), free-tier notes
**Files:** docs/brief.yaml (new), scripts/brief.py (new), README.md, .github/workflows/ci.yml, CHANGELOG.md
- **Situation:** A reader had to scroll and piece together what the Tokyo Airbnb app is, why, who, where, when it fails and how.
- **Task:** Put one numbered 5W1H brief first, with the skills applied and links to proof, without repeating it below (R22).
- **Action:** One source (docs/brief.yaml) written into the README by script and checked in CI. Removed the intro line the brief repeats.
- **Result:** The first screen answers the six questions; CI fails if the brief and its source drift apart.
- **Talking point:** "Six lines tell you what it is, why, who uses it, where it runs, how it fails and how it works."

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

