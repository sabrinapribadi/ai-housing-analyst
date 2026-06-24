# Airbnb Tokyo Analytics Platform

An end-to-end data analytics and AI platform for exploring the Tokyo Airbnb market. Ask questions in plain English and get interactive charts, maps, and predictions — no SQL or Python required.

## Features

| Page | What it does |
|------|-------------|
| **Dashboard** | Key market metrics, annotated price histogram, neighbourhood treemap, price vs rating scatter, box plot, rule-based price predictor |
| **AI Assistant** | Natural language Q&A powered by LangChain + GPT-4o-mini; generated charts appear inline in the chat |
| **Maps** | Choropleth by avg price with hover tooltips, listing density heatmap, MarkerCluster with rich popups — all on a dark tile base with a layer guide panel |
| **Sentiment Analysis** | TextBlob scoring on 5,000+ reviews; non-English reviews auto-translated to English via `deep-translator` |
| **Data Explorer** | Column selector, configurable row count, distribution viewer, data quality chart, neighbourhood comparison |

## Project Structure

```
ai-housing-analyst/
├── .streamlit/
│   └── config.toml          # Dark theme + Airbnb coral accent
├── src/
│   ├── agent/               # LangChain ReAct agent and 6 tool definitions
│   ├── analytics/           # EDA plots and K-Means geospatial clustering
│   ├── data/                # Loader, cleaner, feature engineer
│   ├── models/              # Forecast model (Prophet)
│   └── ui/
│       └── dashboard.py     # Streamlit multi-page app (single file)
├── scripts/
│   ├── save_processed_data.py   # Run once to build Parquet files
│   └── train_forecast.py        # Train and serialise the forecast model
├── data/                    # Raw CSV / GeoJSON (not committed — see .gitignore)
├── outputs/                 # Saved plots (auto-created at runtime)
├── tests/
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

# Install UI / map extras into the active Python environment
pip install streamlit-folium langdetect deep-translator
```

Copy and fill in the environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your key:

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

The app opens at `http://localhost:8501`.

> **Note:** Use `python -m streamlit run` (not `poetry run streamlit run`) to ensure all
> pip-installed extras (`streamlit-folium`, `langdetect`, `deep-translator`) are resolved
> from the same Python environment.

## Theme

The dark theme is defined in `.streamlit/config.toml` and applied automatically:

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
| ML              | scikit-learn (K-Means), Prophet                                           |
| Data            | Pandas, NumPy, GeoPandas                                                  |
| NLP             | TextBlob (sentiment), langdetect, deep-translator                         |
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
