PRODUCT REQUIREMENT DOCUMENT (PRD)
Project: AI Housing Market Analyst – An LLM-Powered Data Agent
Version: 5.1 (RAG UX Polish)
Author: Sabrina Pribadi
Date: June 25, 2026
Status: Completed


1. EXECUTIVE SUMMARY

Problem: Real estate investors, policymakers, and analysts lack an intuitive way to explore
housing data. They either write complex SQL/Python or rely on static dashboards that can't
answer ad-hoc questions — and they have no way to integrate their own external datasets
without engineering support.

Solution: An LLM agentic system that lets users ask natural-language questions about Tokyo's
Airbnb housing data, automatically surfaces hidden patterns without prompting, integrates
external datasets via an AI-guided schema-mapping workflow, and validates its own outputs
against ground truth through an automated hallucination detection suite.

Value Proposition: Reduce time-to-insight from days to minutes. Democratise data access for
non-technical stakeholders. Demonstrate end-to-end data science, MLOps, LLM engineering,
product-quality UI/UX, and AI safety practices.

Success Metrics:
- Agent correctly answers 5+ test questions with inline chart display
- Hallucination test suite passes 6/6 questions against ground truth
- Data Discovery surfaces 5 distinct insight types on first run
- External dataset integrates with > 80% row match rate on neighbourhood key
- Dashboard loads in < 3 seconds


2. PROJECT CONTEXT & BACKGROUND

This project was built as a portfolio piece to demonstrate capabilities in:
1. Data Engineering: Ingesting, cleaning, and aggregating multi-table datasets (10M+ rows).
2. Data Analytics & ML: Performing EDA, geospatial clustering (K-Means), sentiment analysis.
3. LLM & Agentic Design: Building a LangChain agent, a discovery agent, and an integration
   agent — each with fallback strategies when the LLM is unavailable.
4. Software Engineering: Modular, testable codebase with centralised helpers and clean UI.
5. MLOps Thinking: Production pipeline patterns, Parquet for fast loading, Agg backend safety.
6. UI/UX Design: Golden ratio layout, dark theme, Material icons, inline chart display.
7. AI Safety: Automated hallucination detection against Parquet ground truth.

Data Source: Inside Airbnb Tokyo (https://insideairbnb.com/get-the-data/)
- listings.csv: 27,945 rows, 79 columns
- calendar.csv: 10,199,937 rows (no price data, used for availability)
- reviews.csv: 444,182 rows (filtered to last 2 years, 2024–2025)
- neighbourhoods.geojson: Geospatial boundaries for mapping


3. SCOPE

In-Scope:
- Data: Tokyo Airbnb dataset (Inside Airbnb)
- Analytics: EDA on listings, price trends by neighbourhood, geospatial clustering
- Modelling: Geospatial clustering (K-Means), sentiment analysis, price prediction
- LLM Agent: Answer natural language questions, generate and display visualisations inline
- Data Discovery: Proactively surface 5 insight types without user prompting
- Data Integration: Upload external CSV/Excel, AI-guided schema mapping, left-join merge
- UI: Streamlit dashboard with 6 pages, dark theme, golden ratio layout, Plotly charts
- Maps: Multi-layer Folium map with choropleth, heatmap, and marker clusters
- Translation: Auto-translate non-English reviews to English
- Testing: Automated hallucination detection suite against Parquet ground truth
- Deployment: GitHub repository with comprehensive README and Streamlit config

Out-of-Scope:
- User authentication or multi-tenancy
- Real-time data streaming
- Mobile app development
- Advanced LLM fine-tuning
- Cloud-based orchestration (Kubeflow/Vertex AI)


4. USER PERSONAS & STORIES

| Persona              | Goal                                                             | Pain Point                        |
|----------------------|------------------------------------------------------------------|-----------------------------------|
| Alex (Analyst)       | Deep dives on price trends, seasonality, geographic clusters.    | Wastes time writing code.         |
| Maria (Investor)     | Identify underpriced neighbourhoods with high growth potential.  | Doesn't know SQL/Python.          |
| James (Policymaker)  | Understand affordability and supply-demand dynamics.             | Needs quick, trustworthy answers. |
| Dana (Data Engineer) | Enrich the Airbnb dataset with ward-level demographic data.      | Schema mapping takes days.        |

User Stories Implemented:
- As a user, I can type a question in plain English and see a visual answer with inline charts.
- As a user, I can ask for price predictions based on listing attributes.
- As a user, I can view a pre-built dashboard with key market metrics and 5 chart types.
- As a user, I can filter data by price, room type, and neighbourhood.
- As a user, I can download filtered reports as CSV.
- As a user, I can analyse review sentiment with automatic translation of non-English reviews.
- As a user, I can explore listing data with a column selector and distribution viewer.
- As a user, I can view a multi-layer interactive map with hover tooltips.
- As a user, I can see automatically surfaced insights — anomalies, hidden gems, market gaps —
  without writing a single query.
- As a data engineer, I can upload a CSV, review AI-suggested column mappings, and merge
  external data into the Airbnb dataset in one workflow.


5. FUNCTIONAL REQUIREMENTS

Module A: Data Pipeline
- A.1 Data Ingestion: Load listings.csv, calendar.csv, reviews.csv, and neighbourhoods.geojson.
- A.2 Data Cleaning: Handle nulls, parse price (string → float), standardise column names,
      strip HTML tags from review text.
- A.3 Feature Engineering: Create price_per_night, price_category, cluster labels.
- A.4 Date Filtering: Reviews filtered to last 2 years (2024–2025).
- A.5 Data Export: Processed data saved as Parquet for fast loading.

Module B: Analytics & Modelling
- B.1 EDA Automation: Annotated price distribution (mean/median lines), price by room type,
      price by neighbourhood (bar, treemap, box plot), price vs rating scatter.
- B.2 Geospatial Clustering: K-Means on coordinates + price to identify 5 market segments.
- B.3 Sentiment Analysis: TextBlob analysis on up to 100,000 reviews per filter combination,
      with auto-translation of non-English text. Results cached per unique filter set.
- B.4 Price Predictor: Rule-based prediction tool with user inputs.
- B.5 Summary Reports: Automated market summary with key statistics.

Module C: Q&A Agent (AirbnbAgent) — Chat with Agent tab
- C.1 Agent Framework: LangChain with LangGraph (v1.3 compatible).
- C.2 Tools Provided: 8 tools —
      · get_summary_stats: total listings, avg/median price, top-5 expensive/cheapest districts
      · get_review_stats: mean + median for all 7 review score dimensions (overall, accuracy,
        cleanliness, check-in, communication, location, value) and rating distribution bands
      · plot_price_distribution: annotated price histogram
      · plot_price_by_neighborhood(order): bar chart of top-10 most expensive or cheapest
        neighbourhoods; order='expensive' (default) or order='cheapest' — chart title,
        bar colour, and output filename all reflect the requested direction
      · plot_price_by_room_type: price breakdown by room type
      · get_cluster_summary: K-Means market segment summary
      · recommend_neighborhood(budget, room_type): top-10 districts by rating within budget
- C.3 Model: GPT-4o-mini.
- C.4 Prompting: System prompt with data analyst persona.
- C.5 Safety: Error handling, max iterations = 5.
- C.6 Inline Charts: Plots generated by agent tools are displayed directly in the chat.

Module G: RAG Review Search Agent (ReviewRAGAgent) — Ask the Reviews tab
- G.1 Vector Store: ChromaDB PersistentClient storing 25,000 review embeddings in
      data/chroma/ (pre-built, committed to git at 85 MB).
- G.2 Embedding Model: OpenAI text-embedding-3-small with matryoshka reduction to
      256 dimensions. One-time index build cost ~$0.05; query cost ~$0.0001 per search.
- G.3 Retrieval: Cosine-similarity search returns top-10 most semantically relevant
      review documents. Optional neighbourhood metadata filter (ChromaDB $eq operator).
- G.4 Synthesis: Retrieved reviews passed as context to GPT-4o-mini with a strict
      grounding prompt — model instructed to cite only provided text, reflect mixed
      sentiment honestly, and never fabricate details.
- G.5 Index Build Script: scripts/build_review_index.py — samples 25,000 reviews
      proportionally from 444K, embeds in batches of 100, upserts to ChromaDB.
      Supports --rebuild flag to force a full re-index.
- G.6 UI — Ask the Reviews tab:
      - 6 quick-question presets (complaints, cleanliness, host communication, etc.);
        clicking a preset writes directly to session state and auto-triggers the search
        (no need to click "Search Reviews" separately)
      - Free-text question input + optional neighbourhood dropdown filter
      - Synthesised answer with source review cards (expandable, show similarity score)
      - Source reviews auto-translated to English (langdetect + deep-translator);
        non-English cards display the translated text with the language tag in the
        expander label; original text accessible in a nested expander
      - Graceful fallback message if index is not present

Module D: Web Interface
- D.1 Framework: Streamlit with dark theme enforced via .streamlit/config.toml.
- D.2 Pages: Dashboard, AI Assistant, Maps, Data Explorer, Sentiment Analysis,
      Data Discovery.
- D.3 Navigation: Sidebar with Material icon buttons; active page highlighted with coral
      left-border accent; neighbourhood filter in expandable panel.
- D.4 Layout: Golden ratio column splits (1:1.618) applied throughout. Fibonacci-scale
      spacing (8→13→21→34px).
- D.5 Charts: All charts use Plotly with transparent dark backgrounds.
- D.6 Dashboard Charts:
      - Annotated price histogram with mean (amber) and median (teal) reference lines
      - Average price by room type (horizontal bar)
      - Neighbourhood treemap (size = listings, colour = avg price)
      - Price vs rating scatter (bubble size = listing count, quadrant reference lines)
      - Box plot of price spread for top N neighbourhoods
      - Rule-based price predictor
- D.7 Maps:
      - Layer 1: Neighbourhood choropleth (GeoJSON filled by avg price, YlOrRd scale)
        with hover tooltips showing district name, avg price, and listing count
      - Layer 2: Listing density heatmap (HeatMap plugin, hidden by default)
      - Layer 3: MarkerCluster of individual listings with rich popups (hidden by default)
      - Dark map tiles (CartoDB dark_matter)
      - Folium LayerControl for toggling layers
      - Left-side guide panel explaining each layer, colour scale, and interaction tips
- D.8 Data Explorer:
      - Column selector (79 → user-chosen columns)
      - Configurable row count slider
      - Smart number formatting via pandas Styler:
          · Integer columns → thousands separator (e.g. 197,677)
          · Large float columns with no decimal part → {:,.0f} (e.g. ¥12,600)
          · Large float columns with decimals → {:,.2f}
          · Small float columns (ratings, bathrooms) → {:.2f} or {:.0f} if whole numbers
      - Distribution viewer: annotated histogram + statistics table for any numeric column
      - Data quality chart: % missing values per column (colour-scaled bar chart)
      - Neighbourhood comparison table and chart
- D.9 Sentiment Analysis:
      - Sample up to 100,000 reviews per unique filter combination (capped from 444K total).
        First load shows a spinner ("~15 s"); subsequent loads for the same filter are instant
        thanks to @st.cache_data keyed on the tuple of matching listing IDs.
      - Scoring legend displayed above tabs: Positive > 0.1, Neutral ±0.1, Negative < −0.1.
      - Most Positive tab: filters to sentiment_label == "Positive" before selecting top 5.
      - Most Negative tab: filters to sentiment_label == "Negative" before selecting bottom 5.
        Displays "No clearly negative reviews in this selection" when none qualify.
        Includes an info banner explaining two known causes of apparent score/text mismatch:
          (1) translation artefacts — TextBlob trained on native English, not translated text;
          (2) mixed-sentiment phrasing — positive + negative elements lower the overall score.
      - Auto-detect language with langdetect; translate to English with deep-translator.
      - HTML tag stripping from raw review text.
      - Original language shown in collapsible expander.
      - Warning banner if translation packages are not installed.
- D.10 Data Discovery page — see Module E below.

Module E: Data Discovery & Integration Agent (DataDiscoveryAgent)
- E.1 Auto-Discovery: run_full_discovery() executes 5 independent analysis methods against
      the current filtered DataFrame and returns a list of Insight objects, each containing
      a title, LLM-narrated description, category badge, severity level, and Plotly chart.

  Discovery methods:
  - Price Anomalies: neighbourhoods where mean price greatly exceeds the median,
    indicating a small number of outlier luxury listings.
  - Hidden Gems: listings rated above the market median but priced below it —
    best-value options by neighbourhood.
  - Market Gaps: room-type × neighbourhood combinations with low supply (≤ 20th pctl)
    but high avg rating (≥ 70th pctl), signalling unmet demand.
  - Feature Correlations: Pearson r of all numeric listing attributes against price,
    identifying the strongest positive and negative predictors.
  - Availability Patterns: districts that command premium prices despite low annual
    availability, indicating sustained high-occupancy demand.

- E.2 Dataset Profiling: profile_dataset() accepts any Pandas DataFrame and returns a
      DataProfile with column types, unique counts, missing-value percentages, sample
      values, suggested join keys, and an LLM-generated plain-English summary.

- E.3 Column Mapping: suggest_column_mapping() uses GPT-4o-mini to match external
      dataset columns to Airbnb schema columns. Falls back to difflib fuzzy string
      matching when the LLM is unavailable.

- E.4 Dataset Merge: merge_datasets() left-joins selected columns from the external
      dataset into self.listings on the confirmed join key. Returns the enriched
      DataFrame and a report (join key, match rate, new columns added).

- E.5 Post-Merge Discovery: after a successful integration, the UI re-runs
      run_full_discovery() on the enriched dataset and updates session state so
      the Auto-Discovery tab reflects the new combined data.

- E.6 UI — Auto-Discovery Tab:
      - Runs on first page visit; "Re-run" button clears cache and repeats.
      - Category filter pills (All / anomaly / correlation / gap / pattern).
      - 2-column card grid: colour-coded category badge, severity label, title,
        LLM narrative, inline Plotly chart.

- E.7 UI — Data Integration Tab:
      - CSV/Excel file uploader.
      - LLM dataset summary and column profile table.
      - Join key selectors (new dataset key ↔ Airbnb column).
      - Import column multiselect (up to all non-key columns).
      - LLM mapping suggestions shown in collapsible expander.
      - Merge button with success banner showing match rate and new columns.

Module F: Hallucination Detection (tests/test_hallucination.py)
- F.1 Ground Truth Loading: _load_ground_truth() reads the processed Parquet file
      and derives authoritative values for: total listings, avg/median price, most
      and least expensive neighbourhoods, room type counts.
- F.2 Price Error Detection: detect_price_errors() extracts numbers from the agent
      response and flags deviations > 10% from ground truth (high severity > 20%).
- F.3 Fabrication Detection: detect_fabricated_facts() flags overconfidence language
      ("exactly", "precisely") and false negatives (claiming no review data when the
      Parquet file has rows).
- F.4 Test Runner: run_hallucination_tests() executes 6 standard questions, prints
      PASS/FAIL per question, and summarises total pass rate.


6. NON-FUNCTIONAL REQUIREMENTS

- Performance: Agent Q&A response < 10 seconds; Dashboard loads < 3 seconds;
  Auto-Discovery completes < 20 seconds (5 methods + LLM narration).
- Cost: LLM calls minimised — GPT-4o-mini throughout; narration cached in session state.
- Resilience: All LLM calls have fallbacks (hardcoded narratives, fuzzy column matching)
  so the app functions without an API key.
- Code Quality: Modular structure (src/analytics/, src/agent/, src/data/, src/ui/).
  Helper functions (_chart, _clean_review, _translate, _theme, _narrate) centralised.
- Theme: Consistent dark theme via config.toml; primaryColor #FF5A5F (Airbnb coral).
- Documentation: Complete README.md and PRD.md kept in sync with each sprint.
- Version Control: Clean Git history on GitHub.


7. DATA STRATEGY

Data Dictionary (Key Fields):

| Field                   | Type   | Description                         | Example          |
|-------------------------|--------|-------------------------------------|------------------|
| id                      | int    | Unique listing ID                   | 197677           |
| host_id                 | int    | Unique host ID                      | 964081           |
| neighbourhood_cleansed  | string | Official neighbourhood name         | "Sumida Ku"      |
| latitude                | float  | Coordinate                          | 35.6595          |
| longitude               | float  | Coordinate                          | 139.7004         |
| room_type               | string | Type of listing                     | "Entire home/apt"|
| price                   | float  | Nightly price (JPY)                 | 16200            |
| minimum_nights          | int    | Minimum nights required             | 2                |
| availability_365        | int    | Days available per year             | 200              |
| number_of_reviews       | int    | Total reviews                       | 45               |
| review_scores_rating    | float  | Average rating (0–100)              | 94.5             |
| cluster                 | int    | Geospatial cluster ID               | 0                |
| cluster_name            | string | Human-readable cluster name         | "Premium Central"|
| sentiment               | float  | Review sentiment score (−1 to 1)    | 0.25             |

Data Pipeline:
1. Load raw files → clean price column → handle nulls → strip HTML from reviews
2. Feature engineering (price_category, listing_score)
3. Reviews filtered to last 2 years
4. Geospatial clustering (K-Means on coordinates + price)
5. Sentiment analysis on reviews
6. Save processed data as Parquet


8. TECHNICAL ARCHITECTURE

+------------------------------------------------------------------------+
|                            STREAMLIT UI (6 pages)                      |
|  +----------+ +----------+ +-------+ +--------+ +--------+ +--------+ |
|  |Dashboard | |AI Agent  | | Maps  | |DataExp.| |Sentimt.| |Discvry.| |
|  +----------+ +----------+ +-------+ +--------+ +--------+ +--------+ |
+------------------------------------+-----------------------------------+
                                     |
          +--------------------------+--------------------------+
          |                          |                          |
+---------+---------+    +-----------+-----------+    +--------+--------+
| AirbnbAgent       |    | DataDiscoveryAgent    |    | HallucinationD. |
| (Q&A, 8 tools)    |    | (Discovery+Integrate) |    | (Test suite)    |
| LangChain/LangGrph|    | GPT-4o-mini + difflib |    | Ground truth vs |
+---+---+---+---+---+    +-----------+-----------+    | Parquet values  |
    |   |   |   |                    |                 +-----------------+
    v   v   v   v                    v
+------+ +-------+ +------+   +----------+  +----------+
|Pandas| |Plotly | |Folium|   | Pandas   |  |deep-     |
|Query | |Charts | | Maps |   | Merge /  |  |translator|
+------+ +-------+ +------+   | Profile  |  +----------+
    |       |        |         +----------+
    +-------+--------+
                |
      +---------+----------+
      |  Processed Data    |
      |  (Parquet Files)   |
      +--------------------+


9. IMPLEMENTATION PLAN

| Sprint | Duration | Deliverables                                                  | Status    |
|--------|----------|---------------------------------------------------------------|-----------|
| 1      | 3 days   | Data loading, cleaning, feature engineering                   | Completed |
| 2      | 3 days   | EDA functions, geospatial clustering                          | Completed |
| 3      | 4 days   | LangChain Q&A agent with 6 tools                              | Completed |
| 8b     | 0.5 days | Agent tool completeness: get_review_stats (7 dimensions),     | Completed |
|        |          | plot_price_by_neighborhood order param (expensive/cheapest)   |           |
| 4      | 3 days   | Streamlit dashboard with 5 pages                              | Completed |
| 5      | 2 days   | Deployment, README, documentation                             | Completed |
| 6      | 2 days   | UI/UX: dark theme, golden ratio, Plotly charts, enhanced maps | Completed |
| 7      | 2 days   | Data Discovery agent, Data Integration workflow,              | Completed |
|        |          | hallucination detection test suite                            |           |
| 8      | 1 day    | Polish: Data Explorer number formatting, sentiment sample     | Completed |
|        |          | raised to 100K, genuine label filtering for review tabs,      |           |
|        |          | scoring methodology disclaimer with known-limitations note    |           |
| 9      | 2 days   | Full RAG pipeline: ReviewRAGAgent, ChromaDB vector store,     | Completed |
|        |          | OpenAI text-embedding-3-small (256-dim), Ask the Reviews tab  |           |
| 10     | 0.5 days | RAG UX polish: preset buttons auto-trigger search via session | Completed |
|        |          | state; source reviews auto-translated (langdetect +           |           |
|        |          | deep-translator) with original in nested expander             |           |


10. TESTING STRATEGY

- Unit Tests: Data cleaning functions validated.
- Integration Tests: Agent tools tested with sample questions.
- Hallucination Detection (automated): tests/test_hallucination.py loads ground truth
  from the Parquet file and runs 6 questions through the live agent, checking reported
  numbers against authoritative values and flagging overconfidence language.
- Manual QA: All 5 test questions answered correctly with inline chart display.
- Dashboard Testing: All 6 pages and features functional on Python 3.12.
- Translation Testing: Spanish, French, German reviews translated correctly.
- Map Testing: All three layers toggle correctly; hover tooltips display district data.
- Discovery Testing: All 5 insight methods return Insight objects with charts.
- Integration Testing: CSV upload → profile → mapping → merge flow completes end-to-end.


11. RISKS & MITIGATIONS

| Risk                            | Mitigation                                              | Status   |
|---------------------------------|---------------------------------------------------------|----------|
| Large calendar file (10M rows)  | Sampled for dashboard; full file for processing only    | Resolved |
| LLM hallucination               | Automated test suite (test_hallucination.py) validates  | Resolved |
|                                 | agent answers against Parquet ground truth              |          |
| API costs                       | GPT-4o-mini throughout; narrations cached in session    | Resolved |
| Deployment memory limit         | Parquet files; raw data excluded from Git               | Resolved |
| GitHub file size limits         | Large files excluded via .gitignore                     | Resolved |
| Translation package unavailable | Graceful fallback with visible warning banner           | Resolved |
| Matplotlib thread safety        | Agg backend; plt.close(fig) after every save            | Resolved |
| LLM unavailable (no API key)    | Discovery narration falls back to hardcoded text;       | Resolved |
|                                 | column mapping falls back to difflib fuzzy matching     |          |
| External dataset schema mismatch| LLM mapping + user-editable selectors + fuzzy fallback  | Resolved |


12. SUCCESS CRITERIA

Agent test questions (all passing):
1. "Show me the average price per neighbourhood on a bar chart."
2. "Plot the distribution of prices by room type." — chart displays inline in chat.
3. "How many listings are there in Tokyo?" — matches Parquet ground truth.
4. "Which neighbourhood has the highest average prices?"
5. "What's the most popular room type?"

Hallucination test suite:
- 6/6 questions pass price-error and fabrication checks against ground truth.

Data Discovery:
- 5 insight cards generated on first run without any user query.
- Each card has a non-empty LLM narrative and an interactive Plotly chart.

Data Integration:
- CSV upload → profiling → column mapping → merge completes in < 30 seconds.
- Merged DataFrame has all new columns and a match rate reported in the success banner.

Additional criteria met:
- Streamlit app fully functional with 6 pages and consistent dark theme.
- GitHub repo has complete README.md with setup, run, and test instructions.
- Code is modular; all LLM interactions have working offline fallbacks.
- Maps: choropleth with hover tooltips, heatmap, MarkerCluster, layer guide panel.


13. KEY INSIGHTS FROM DATA

- Total Listings: 27,945
- Average Price: ¥24,782/night
- Median Price: ¥16,200/night
- Most Expensive Neighbourhood: Chuo Ku (¥44,342/night avg)
- Cheapest Neighbourhood: Higashiyamato Shi (¥5,136/night avg)
- Top Room Type: Entire home/apt (23,759 listings)
- Market Segments: 5 K-Means clusters (Luxury Central, Premium Central, Premium South, etc.)
- Review Languages: Multi-lingual (Japanese, French, Spanish, German, Korean, etc.)
- Hidden Gems: ~6,000 listings rated above median but priced below median


14. APPENDIX

- Data Source: https://insideairbnb.com/get-the-data/
- GitHub Repository: https://github.com/sabrinapribadi/ai-housing-analyst
- Tech Stack: Streamlit, LangChain, LangGraph, OpenAI GPT-4o-mini,
  text-embedding-3-small, ChromaDB, scikit-learn, GeoPandas, Folium, Plotly,
  langdetect, deep-translator, TextBlob, difflib
- Agents: AirbnbAgent (Q&A, 8 tools), ReviewRAGAgent (RAG search),
  DataDiscoveryAgent (discovery + integration)
- Pages: 6 dashboard pages
- Agent Tools: 8 Q&A tools, 5 discovery methods, 3 integration methods
- RAG Index: 25,000 reviews, text-embedding-3-small 256-dim, ChromaDB cosine similarity
- Test Suite: 6 hallucination tests against Parquet ground truth
