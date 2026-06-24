PRODUCT REQUIREMENT DOCUMENT (PRD)
Project: AI Housing Market Analyst – An LLM-Powered Data Agent
Version: 1.0
Author: Sabrina Pribadi
Date: June 24, 2026
Status: Finalized


1. EXECUTIVE SUMMARY
Problem: Real estate investors, policymakers, and analysts lack an intuitive way to explore housing data. They either write complex SQL/Python or rely on static dashboards that can't answer ad-hoc questions.

Solution: An LLM agentic system that lets users ask natural-language questions about Tokyo's Airbnb housing data, automatically runs analytics, builds forecasts, and visualizes results—all through a simple web interface.

Value Proposition: Reduce time-to-insight from days to minutes. Democratize data access for non-technical stakeholders. Showcase my end-to-end data science, MLOps, and LLM engineering skills.

Success Metrics:
- Agent correctly answers 5 test questions (see Section 12).
- Dashboard loads in < 3 seconds.
- The entire project is deployable on Hugging Face Spaces for free.


2. PROJECT CONTEXT & BACKGROUND
This project is a portfolio piece designed to demonstrate my capabilities in:
1.  Data Engineering: Ingesting, cleaning, and aggregating multi-table datasets (10M+ rows).
2.  Data Analytics & ML: Performing EDA, time-series forecasting (Prophet/ARIMA), and geospatial clustering (K-Means on coordinates).
3.  LLM & Agentic Design: Building a LangChain agent that uses tools to answer data questions.
4.  Software Engineering: Creating a modular, testable codebase with a clean UI.
5.  MLOps Thinking: Adopting the pattern of a production pipeline (inspired by my company's Kubeflow template).


3. SCOPE
In-Scope:
- Data: Tokyo Airbnb dataset (Inside Airbnb).
- Analytics: EDA on listings, price trends by neighborhood, seasonality.
- Modeling: Time-series forecasting (Prophet), K-Means clustering of listings.
- LLM Agent: Answer natural language questions, generate Python code and visualizations.
- UI: A Streamlit dashboard with a chat interface and static visualizations.
- Deployment: Hugging Face Spaces.

Out-of-Scope:
- User authentication or multi-tenancy.
- Real-time data streaming.
- Mobile app development.
- Advanced LLM fine-tuning.
- Cloud-based orchestration (Kubeflow/Vertex AI) for this iteration.


4. USER PERSONAS & STORIES
Persona          | Goal                                                                  | Pain Point
-----------------|-----------------------------------------------------------------------|-----------------------------
Alex (Analyst)   | Run deep dives on price trends, seasonality, and geographic clusters. | Wastes time writing code.
Maria (Investor) | Identify underpriced neighborhoods with high growth potential.        | Doesn't know SQL/Python.
James (Policymaker) | Understand affordability and supply-demand dynamics.              | Needs quick, trustworthy answers.

User Stories:
- As a user, I can type a question in plain English and see a visual answer.
- As a user, I can ask for a forecast of prices for a specific neighborhood.
- As a user, I can view a pre-built dashboard with key market metrics.


5. FUNCTIONAL REQUIREMENTS

Module A: Data Pipeline
-------------------------
A.1 Data Ingestion: Load listings.csv, calendar.csv, and neighbourhoods.geojson using Pandas/GeoPandas.
A.2 Data Cleaning: Handle nulls, parse price (string -> float), standardize column names.
A.3 Feature Engineering: Create price_per_sqft, neighborhood_group (from geo-join), listing_age.

Module B: Analytics & Modeling
-------------------------------
B.1 EDA Automation: Agent can generate histograms, boxplots, and correlation heatmaps.
B.2 Time-Series Forecast: Use FB Prophet to forecast average price for a selected neighborhood.
B.3 Geospatial Clustering: Apply K-Means on coordinates to identify listing clusters (e.g., tourist vs. residential).

Module C: LLM Agentic Layer
----------------------------
C.1 Agent Framework: LangChain with ReAct reasoning.
C.2 Tools: Provide tools for: Pandas query, Visualization (Matplotlib), Forecasting (Prophet), Geospatial Map (Folium).
C.3 Model: GPT-4o-mini (cheap, capable).
C.4 Prompting: System prompt: "You are a data analyst. Write Python code in a tool and show the output."

Module D: Web Interface
------------------------
D.1 Framework: Streamlit (multi-page).
D.2 Pages: Chat (interact with agent), Dashboard (pre-built EDA, forecast, and map), About (project details).
D.3 Deployment: Hugging Face Spaces.


6. NON-FUNCTIONAL REQUIREMENTS
- Performance: Agent response < 10 sec; Dashboard loads < 3 sec.
- Cost: LLM API calls < $5 for the entire demo phase (use caching).
- Code Quality: Modular structure (data/, models/, agent/, ui/), type hints, docstrings, and a README.md.
- Testing: Unit tests for data cleaning functions and forecast model.


7. DATA STRATEGY
Data Sources:
- listings.csv (80.7 MB, 27,945 rows) – Main table.
- calendar.csv (394.7 MB, 10.2M rows) – For time-series aggregation.
- reviews.csv (292 MB, 1M rows) – For text analysis (sampled).
- neighbourhoods.geojson – For mapping.

Data Dictionary (Key Fields):
Field                      | Type   | Description                        | Example
---------------------------|--------|------------------------------------|----------
id                         | int    | Unique listing ID                  | 197677
host_id                    | int    | Unique host ID                     | 964081
neighbourhood_cleansed     | string | Official neighbourhood name        | "Sumida Ku"
latitude                   | float  | Coordinate                         | 35.6595
longitude                  | float  | Coordinate                         | 139.7004
room_type                  | string | Type of listing                    | "Entire home/apt"
price                      | float  | Nightly price (JPY)                | 12000
minimum_nights             | int    | Min nights required                | 2
availability_365           | int    | Days available in year             | 200
number_of_reviews          | int    | Total reviews                      | 45
review_scores_rating       | float  | Avg rating (0–100)                 | 94.5

Data Pipeline:
1. Load & Clean: Load raw files, clean price, handle nulls.
2. Feature Engineering: Create price_per_night, log_price, listing_age.
3. Aggregate: Create a daily df_calendar_agg (mean price by date and neighborhood).
4. For Analytics: Use df_listings_clean for modeling and agent RAG.


8. TECHNICAL ARCHITECTURE
+--------------------------------------------------------------+
|                        STREAMLIT UI                           |
|  +-------------------+  +----------------------------------+  |
|  |  Dashboard Page   |  |  Chat Interface Page             |  |
|  | (Pre-built EDA,   |  |  [Input: "Show me price trends"]|  |
|  |  Forecast, Map)   |  |  [Output: Chart + Explanation]  |  |
|  +-------------------+  +----------------------------------+  |
+----------------------------+---------------------------------+
                             |
+----------------------------+---------------------------------+
|                    LLM AGENT (LangChain)                     |
|  +--------------------------------------------------------+ |
|  |  System Prompt: "You are a data analyst. Use tools."   | |
|  |  ReAct Loop: Thought -> Action -> Observation          | |
|  +--------------------------------------------------------+ |
+-------+--------+--------+--------+--------------------------+
        |        |        |        |
        v        v        v        v
+--------+  +--------+  +--------+  +--------+
| Pandas |  | Prophet|  |Matplot-|  | Folium |  <- Tools
| Query  |  |Forecast|  | lib    |  | Map    |
+--------+  +--------+  +--------+  +--------+
        |        |        |        |
        +--------+--------+--------+
                        |
              +---------+----------+
              |  Cleaned Data      |
              |  (Pandas DataFrame)|
              +--------------------+


9. IMPLEMENTATION PLAN (MILESTONES)
Sprint | Duration | Deliverables
-------|----------|-------------
1      | 3 days   | Load, clean, and engineer features for listings and aggregated calendar. Save as data/processed/.
2      | 3 days   | Build Prophet forecast model and K-Means clustering. Create EDA functions.
3      | 4 days   | Build LangChain agent with 4 tools. Test prompts with 5 core questions.
4      | 3 days   | Build Streamlit app (Dashboard + Chat). Integrate agent.
5      | 2 days   | Deploy to Hugging Face. Write README.md.
Buffer | 2 days   | Polish, fine-tune prompts, handle edge cases.


10. TESTING STRATEGY
- Unit Tests: pytest for data cleaning functions (e.g., clean_price).
- Integration Test: Test the agent's ability to run a tool and return a valid output.
- Manual QA: Run the 5 key questions (Section 12) and verify outputs.


11. RISKS & MITIGATIONS
Risk                           | Mitigation
-------------------------------|-------------------------------------------------------------
Large calendar file slows down.| Sample or aggregate calendar.csv upfront.
LLM hallucination              | Show generated code; limit agent iterations; strong prompt.
API costs.                     | Cache LLM responses; use gpt-4o-mini.
Deployment memory limit        | Use smaller processed dataset (keep only key features).


12. SUCCESS CRITERIA (DEFINITION OF DONE)
- Agent correctly answers these 5 test questions:
  1. "Show me the average price per neighborhood on a bar chart."
  2. "Plot the distribution of prices by room type."
  3. "Forecast the average price for Shibuya for the next 3 months."
  4. "Map all listings with a price over 20,000 JPY."
  5. "Which neighborhood has the highest average review score?"

- Streamlit app is live on Hugging Face Spaces.
- GitHub repo has a clean README.md with setup, architecture, and a demo GIF.
- Code is modular, with type hints and docstrings.


13. APPENDIX
- Data Source: https://insideairbnb.com/get-the-data/
- Data Dictionary (Full): https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?gid=1322284596#gid=1322284596
- Company Template Reference: PIPELINE_TEMPLATE.md (for architectural inspiration)