# Airbnb Tokyo Analytics Platform

An end-to-end data analytics and AI platform for exploring the Tokyo Airbnb market. Ask questions in plain English, get interactive charts and maps, discover hidden patterns automatically, and merge in your own external datasets — no SQL or Python required.

## Features

| Page | What it does |
|------|-------------|
| **Dashboard** | Key market metrics, annotated price histogram, neighbourhood treemap, price vs rating scatter, box plot, rule-based price predictor |
| **AI Assistant** | Two tabs: **Chat with Agent** — natural language Q&A via LangChain + GPT-4o-mini (8 tools, inline charts); **Ask the Reviews (RAG)** — semantic search over 25,000 embedded reviews via ChromaDB + OpenAI text-embedding-3-small, answers grounded in retrieved guest review text |
| **Maps** | Choropleth by avg price with hover tooltips, listing density heatmap, MarkerCluster with rich popups — dark tile base with a layer guide panel |
| **Sentiment Analysis** | TextBlob scoring on up to 100,000 reviews per filter (cached); Most Positive / Most Negative tabs filter by genuine label (> 0.1 / < −0.1); inline disclaimer explains translation artefacts and mixed-sentiment scoring |
| **Data Explorer** | Column selector, configurable row count, smart number formatting (thousands separators, smart decimals), distribution viewer, data quality chart, neighbourhood comparison |
| **Data Discovery** | Auto-Discovery tab: surfaces 5 insight types (anomalies, gems, gaps, correlations, patterns) with LLM narration and Plotly charts. Data Integration tab: upload any CSV/Excel, AI-guided column mapping, left-join merge into the Airbnb dataset |

## Project Structure

```
ai-housing-analyst/
├── .streamlit/
│   └── config.toml              # Dark theme + Airbnb coral accent
├── src/
│   ├── agent/
│   │   ├── agent.py             # LangChain Q&A agent (8 tools)
│   │   ├── rag_agent.py         # ReviewRAGAgent: ChromaDB + OpenAI embeddings RAG
│   │   └── discovery_agent.py   # DataDiscoveryAgent: auto-discovery + data integration
│   ├── analytics/
│   │   ├── eda.py               # EDA plot functions (Matplotlib)
│   │   └── clustering.py        # K-Means geospatial clustering
│   ├── data/                    # Loader, cleaner, feature engineer
│   ├── models/                  # Forecast model (Prophet)
│   └── ui/
│       └── dashboard.py         # Streamlit 6-page app
├── scripts/
│   ├── save_processed_data.py   # Run once to build Parquet files
│   ├── build_review_index.py    # Run once to build ChromaDB RAG index (~$0.05)
│   └── train_forecast.py        # Train and serialise the forecast model
├── data/
│   └── chroma/                  # Pre-built ChromaDB vector store (85 MB, committed)
├── tests/
│   ├── test_hallucination.py    # Automated hallucination detection suite
│   └── hallucination_checklist.md
├── data/                        # Raw CSV / GeoJSON (not committed — see .gitignore)
├── outputs/                     # Agent-generated plots (auto-created at runtime)
├── notebooks/
└── pyproject.toml
```

## Setup

**Prerequisites:** Python 3.10–3.12, [Poetry](https://python-poetry.org/), an OpenAI API key.

```bash
# Clone the repository
git clone https://github.com/sabrinapribadi/ai-housing-analyst.git
cd ai-housing-analyst

# Install core dependencies
poetry install

# Install UI / map / translation extras
pip install streamlit-folium langdetect deep-translator
```

Copy and fill in the environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...
```

## Data Setup

Download the Tokyo dataset from [Inside Airbnb](https://insideairbnb.com/get-the-data/) and place the files under `data/raw/`:

```
data/
└── raw/
    ├── listings.csv
    ├── calendar.csv
    ├── reviews.csv
    └── neighbourhoods.geojson
```

Run the pipeline once to build the processed Parquet files:

```bash
python scripts/save_processed_data.py
```

## Running the App

```bash
python -m streamlit run src/ui/dashboard.py
```

Opens at `http://localhost:8501`.

> **Note:** Use `python -m streamlit run` (not `poetry run streamlit run`) to ensure all
> pip-installed extras are resolved from the same Python environment.

## Rebuilding the RAG Index

The pre-built ChromaDB index (`data/chroma/`) is committed to the repo and works out of the box. To rebuild it (e.g. after adding more reviews):

```bash
python scripts/build_review_index.py          # build (skips if already exists)
python scripts/build_review_index.py --rebuild # force full rebuild
```

Cost: ~$0.05 one-time (25,000 reviews × ~100 tokens × text-embedding-3-small pricing).

## Running Tests

The hallucination detection suite fires 6 standard questions at the live agent and compares
reported numbers against ground truth loaded directly from the Parquet file:

```bash
python tests/test_hallucination.py
```

Sample output:

```
============================================================
HALLUCINATION DETECTION TEST SUITE
============================================================

   Testing: What's the average price of an Airbnb in Tokyo?
   Response: The average price of an Airbnb listing in Tokyo is approximately ¥24,782...
   Status: PASS

...

============================================================
SUMMARY
============================================================
Tests:                  6
Passed:                 6 (100.0%)
Hallucinations detected:0
```

## Theme

Defined in `.streamlit/config.toml` and applied automatically:

```toml
[theme]
base                     = "dark"
primaryColor             = "#FF5A5F"   # Airbnb coral
backgroundColor          = "#111111"
secondaryBackgroundColor = "#1C1C1C"
textColor                = "#F0F0F0"
```

## Tech Stack

| Layer           | Library / Tool                                                            |
|-----------------|---------------------------------------------------------------------------|
| Frontend        | Streamlit 1.58, Material Icons                                            |
| Charts          | Plotly Express / Graph Objects                                            |
| Maps            | Folium, streamlit-folium, folium.plugins (HeatMap, MarkerCluster)         |
| AI / Agent      | LangChain, LangGraph, OpenAI GPT-4o-mini                                  |
| RAG             | ChromaDB (vector store), OpenAI text-embedding-3-small (256-dim)          |
| ML              | scikit-learn (K-Means), Prophet                                           |
| Data            | Pandas, NumPy, GeoPandas                                                  |
| NLP             | TextBlob (sentiment), langdetect, deep-translator                         |
| Schema matching | difflib (fuzzy fallback for column mapping without LLM)                   |
| Code quality    | Black, Ruff, mypy, pre-commit                                             |

## Data Source

[Inside Airbnb — Tokyo](https://insideairbnb.com/get-the-data/)

| File                   | Rows       | Purpose                            |
|------------------------|------------|------------------------------------|
| listings.csv           | 27,945     | Prices, room types, neighbourhoods |
| calendar.csv           | 10,200,000 | Availability (no price data)       |
| reviews.csv            | 444,182    | Sentiment (filtered 2024–2025)     |
| neighbourhoods.geojson | —          | Geospatial district boundaries     |

## Key Market Insights

- **Total listings:** 27,945
- **Average price:** ¥24,782/night · **Median:** ¥16,200/night
- **Most expensive district:** Chuo Ku (¥44,342 avg)
- **Cheapest district:** Higashiyamato Shi (¥5,136 avg)
- **Top room type:** Entire home/apt (23,759 listings)
- **Market segments:** 5 K-Means clusters (Luxury Central, Premium Central, etc.)
- **Hidden gems:** ~6,000 listings rated above median but priced below median
