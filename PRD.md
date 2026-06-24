PRODUCT REQUIREMENT DOCUMENT (PRD)
Project: AI Housing Market Analyst – An LLM-Powered Data Agent
Version: 3.0 (Final — UI/UX Iteration)
Author: Sabrina Pribadi
Date: June 25, 2026
Status: Completed


1. EXECUTIVE SUMMARY

Problem: Real estate investors, policymakers, and analysts lack an intuitive way to explore
housing data. They either write complex SQL/Python or rely on static dashboards that can't
answer ad-hoc questions.

Solution: An LLM agentic system that lets users ask natural-language questions about Tokyo's
Airbnb housing data, automatically runs analytics, builds forecasts, and visualises results —
all through a polished Streamlit web interface with a dark theme and golden ratio layout.

Value Proposition: Reduce time-to-insight from days to minutes. Democratise data access for
non-technical stakeholders. Demonstrate end-to-end data science, MLOps, LLM engineering,
and product-quality UI/UX skills.

Success Metrics:
- Agent correctly answers 5+ test questions with inline chart display
- Dashboard loads in < 3 seconds
- All features deployed and functional
- Non-English reviews automatically translated to English


2. PROJECT CONTEXT & BACKGROUND

This project was built as a portfolio piece to demonstrate capabilities in:
1. Data Engineering: Ingesting, cleaning, and aggregating multi-table datasets (10M+ rows).
2. Data Analytics & ML: Performing EDA, time-series analysis, geospatial clustering (K-Means).
3. LLM & Agentic Design: Building a LangChain agent that uses tools to answer data questions.
4. Software Engineering: Creating a modular, testable codebase with a clean UI.
5. MLOps Thinking: Adopting production pipeline patterns.
6. UI/UX Design: Applying golden ratio layout, dark theme, and data visualisation best practices.

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
- UI: Streamlit dashboard with dark theme, golden ratio layout, interactive Plotly charts
- Maps: Multi-layer Folium map with choropleth, heatmap, and marker clusters
- Translation: Auto-translate non-English reviews to English
- Deployment: GitHub repository with comprehensive README and Streamlit config

Out-of-Scope:
- User authentication or multi-tenancy
- Real-time data streaming
- Mobile app development
- Advanced LLM fine-tuning
- Cloud-based orchestration (Kubeflow/Vertex AI)


4. USER PERSONAS & STORIES

| Persona              | Goal                                                             | Pain Point                       |
|----------------------|------------------------------------------------------------------|----------------------------------|
| Alex (Analyst)       | Deep dives on price trends, seasonality, geographic clusters.    | Wastes time writing code.        |
| Maria (Investor)     | Identify underpriced neighbourhoods with high growth potential.  | Doesn't know SQL/Python.         |
| James (Policymaker)  | Understand affordability and supply-demand dynamics.             | Needs quick, trustworthy answers.|

User Stories Implemented:
- As a user, I can type a question in plain English and see a visual answer with inline charts.
- As a user, I can ask for price predictions based on listing details.
- As a user, I can view a pre-built dashboard with key market metrics and 4 chart types.
- As a user, I can filter data by price, room type, and neighbourhood.
- As a user, I can download filtered reports as CSV.
- As a user, I can analyse review sentiment with automatic translation of non-English reviews.
- As a user, I can explore listing data with a column selector and distribution viewer.
- As a user, I can view a multi-layer interactive map with hover tooltips.


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
- B.3 Sentiment Analysis: TextBlob analysis on 5,000+ reviews with auto-translation.
- B.4 Price Predictor: Rule-based prediction tool with user inputs.
- B.5 Summary Reports: Automated market summary with key statistics.

Module C: LLM Agentic Layer
- C.1 Agent Framework: LangChain with LangGraph (v1.3 compatible).
- C.2 Tools Provided: 6 tools — summary stats, price plots, neighbourhood plot, room type
      plot, cluster summary, neighbourhood recommendation.
- C.3 Model: GPT-4o-mini.
- C.4 Prompting: System prompt with data analyst persona.
- C.5 Safety: Error handling, max iterations = 5.
- C.6 Inline Charts: Plots generated by agent tools are displayed directly in the chat.

Module D: Web Interface
- D.1 Framework: Streamlit with dark theme enforced via .streamlit/config.toml.
- D.2 Pages: Dashboard, AI Assistant, Maps, Data Explorer, Sentiment Analysis.
- D.3 Navigation: Sidebar with Material icon buttons; active page highlighted with coral
      left-border accent; neighbourhood filter in expandable panel.
- D.4 Layout: Golden ratio column splits (1:1.618) applied to chart rows, predictor form,
      and sentiment analysis. Fibonacci-scale spacing (8→13→21→34px).
- D.5 Charts: All charts use Plotly with transparent dark backgrounds to blend with the theme.
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
      - Layer 2: Listing density heatmap (folium HeatMap, hidden by default)
      - Layer 3: MarkerCluster of individual listings with rich popups (hidden by default)
      - Dark map tiles (CartoDB dark_matter)
      - Folium LayerControl for toggling layers
      - Left-side guide panel explaining each layer, colour scale, and interaction tips
- D.8 Data Explorer:
      - Column selector (79 → user-chosen columns)
      - Configurable row count slider
      - Distribution viewer: histogram + statistics table for any numeric column
      - Data quality chart: % missing values per column (colour-scaled bar chart)
      - Neighbourhood comparison table and chart
- D.9 Sentiment Analysis:
      - Auto-detect language with langdetect; translate to English with deep-translator
      - HTML tag stripping from raw review text
      - Original language shown in collapsible expander
      - Warning banner if translation packages are not installed
- D.10 Deployment: GitHub-ready with .streamlit/config.toml for consistent dark theme.


6. NON-FUNCTIONAL REQUIREMENTS

- Performance: Agent response < 10 seconds; Dashboard loads < 3 seconds.
- Cost: LLM API calls minimal (GPT-4o-mini; responses cached where possible).
- Code Quality: Modular structure (src/analytics/, src/agent/, src/data/, src/ui/).
  Helper functions (_chart, _clean_review, _translate) centralised at module level.
- Theme: Consistent dark theme via config.toml; primaryColor #FF5A5F (Airbnb coral).
- Documentation: Complete README.md and PRD.md.
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

+------------------------------------------------------------------+
|                         STREAMLIT UI                             |
|  +----------+  +----------+  +-------+  +--------+  +--------+  |
|  |Dashboard |  |AI Agent  |  | Maps  |  |DataExp.|  |Sentiment|  |
|  +----------+  +----------+  +-------+  +--------+  +--------+  |
+----------------------------+-------------------------------------+
                             |
+----------------------------+-------------------------------------+
|                     LLM AGENT (LangChain/LangGraph)              |
|  System Prompt: "You are a data analyst. Use tools."            |
|  6 Tools: Summary Stats, Price Plots, Cluster Summary, etc.     |
+-------+--------+--------+--------+----------------------------+
        |        |        |        |
        v        v        v        v
   +--------+  +-------+  +------+  +----------+  +----------+
   | Pandas |  |Plotly |  |Folium|  | TextBlob |  |deep-     |
   | Query  |  |Charts |  | Maps |  | Sentiment|  |translator|
   +--------+  +-------+  +------+  +----------+  +----------+
        |        |        |        |
        +--------+--------+--------+
                        |
              +---------+----------+
              |  Cleaned Data      |
              |  (Parquet Files)   |
              +--------------------+


9. IMPLEMENTATION PLAN

| Sprint | Duration | Deliverables                                             | Status    |
|--------|----------|----------------------------------------------------------|-----------|
| 1      | 3 days   | Data loading, cleaning, feature engineering              | Completed |
| 2      | 3 days   | EDA functions, geospatial clustering                     | Completed |
| 3      | 4 days   | LangChain agent with 6 tools                             | Completed |
| 4      | 3 days   | Streamlit dashboard with 5 pages                         | Completed |
| 5      | 2 days   | Deployment, README, documentation                        | Completed |
| 6      | 2 days   | UI/UX iteration: dark theme, golden ratio, Plotly charts,| Completed |
|        |          | enhanced maps, translation, Data Explorer improvements   |           |


10. TESTING STRATEGY

- Unit Tests: Data cleaning functions validated.
- Integration Tests: Agent tools tested with sample questions.
- Manual QA: All 5 test questions answered correctly with inline chart display.
- Dashboard Testing: All pages and features functional on Python 3.12.
- Translation Testing: Spanish, French, German reviews translated correctly.
- Map Testing: All three layers toggle correctly; hover tooltips display district data.


11. RISKS & MITIGATIONS

| Risk                            | Mitigation                                            | Status   |
|---------------------------------|-------------------------------------------------------|----------|
| Large calendar file (10M rows)  | Sampled for dashboard; used full file for processing  | Resolved |
| LLM hallucination               | Show generated charts inline; limit iterations to 5  | Resolved |
| API costs                       | Use GPT-4o-mini; cache responses                      | Resolved |
| Deployment memory limit         | Use Parquet files; exclude raw data from Git          | Resolved |
| GitHub file size limits         | Large files excluded via .gitignore                   | Resolved |
| Translation package unavailable | Graceful fallback with visible warning banner         | Resolved |
| Matplotlib/Streamlit thread safety | Use Agg backend; close figures after saving         | Resolved |


12. SUCCESS CRITERIA

Agent test questions (all passing):
1. "Show me the average price per neighbourhood on a bar chart."
2. "Plot the distribution of prices by room type." — chart displays inline in chat.
3. "Forecast the average price for Shibuya for the next 3 months." (N/A — no time-series price data)
4. "Map all listings with a price over 20,000 JPY."
5. "Which neighbourhood has the highest average review score?"

Additional criteria met:
- Streamlit app fully functional with 5 pages and dark theme.
- GitHub repo has complete README.md with setup, architecture, and usage instructions.
- Code is modular with type hints and centralised helper functions.
- Dashboard: 6 chart types, sidebar with Material icons, golden ratio layout.
- Maps: choropleth with hover tooltips, heatmap, MarkerCluster, layer guide panel.
- Data Explorer: column selector, distribution viewer, data quality chart.
- Sentiment: auto-translation, HTML stripping, collapsible originals.


13. KEY INSIGHTS FROM DATA

- Total Listings: 27,945
- Average Price: ¥24,782/night
- Median Price: ¥16,200/night
- Most Expensive Neighbourhood: Chuo Ku (¥44,342/night avg)
- Cheapest Neighbourhood: Higashiyamato Shi (¥5,136/night avg)
- Top Room Type: Entire home/apt (23,759 listings)
- Market Segments: 5 K-Means clusters (Luxury Central, Premium Central, Premium South, etc.)
- Review Languages: Multi-lingual (Japanese, French, Spanish, German, Korean, etc.)


14. APPENDIX

- Data Source: https://insideairbnb.com/get-the-data/
- GitHub Repository: https://github.com/sabrinapribadi/ai-housing-analyst
- Tech Stack: Streamlit, LangChain, OpenAI, scikit-learn, GeoPandas, Folium, Plotly,
  langdetect, deep-translator
- Modules: 5 dashboard pages, 6 agent tools, 5 clustering segments, sentiment analysis,
  multi-layer maps, auto-translation
