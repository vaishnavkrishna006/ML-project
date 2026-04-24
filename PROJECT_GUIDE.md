# Smart Movie Recommendation System (TMDb + NLP + Hybrid Ranking)

This project implements a production-style movie recommendation backend with:

- natural-language query understanding (`genre`, `mood`, `count`, `rating`, `year`, `like <movie>`)
- TMDb API integration (with cache + mock fallback)
- hybrid ranking (semantic similarity + genre match + rating + popularity + keyword match)
- diversity-aware top-N recommendations
- analytics APIs (genre trends, top genres per year, highest rated movies, summary stats)

## Architecture

- `app_new.py`: Flask API app and system initialization.
- `app.py`: compatibility entrypoint (runs `app_new.py`).
- `tmdb_data_fetcher.py`: TMDb integration, schema normalization, cache.
- `nlp_query_processor.py`: intent extraction from natural language.
- `recommendation_engine.py`: scoring, ranking, and diversity.
- `analytics.py`: analytics and aggregated insights.

## Why this model design

- **Sentence-BERT (optional)**: better semantic understanding of open queries.
- **TF-IDF fallback**: works fast when embeddings are unavailable.
- **Hybrid score**: combines relevance and quality (`semantic + genre + rating + popularity + keywords`), better than single-signal rankers.
- **Diversity layer**: prevents near-duplicate recommendation lists from one dominant genre.

## Run Steps

1) Create and activate environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Set TMDb API key:

```bash
set TMDB_API_KEY=your_tmdb_api_key
```

If key is missing, the app still runs using mock data.

4) Start server:

```bash
python app.py
```

API base: `http://127.0.0.1:5000`

## Key API examples

### 1) Recommendations (natural language)

`POST /api/recommend`

```json
{
  "query": "Give me 5 crime thriller movies",
  "n_recommendations": 5
}
```

### 2) Recommendations using count from text

```json
{
  "query": "I need action + romance movies"
}
```

### 3) Similar intent style

```json
{
  "query": "Suggest emotional movies like Titanic"
}
```

### 4) Analytics

- `GET /api/analytics`
- `GET /api/analytics/genres`
- `GET /api/analytics/summary`

## Frontend integration suggestion

- React client can call `/api/recommend` directly and render cards with:
  - `title`, `rating`, `popularity`, `genres`, `overview`, `score`.
- Add filters in UI (genre chips, year slider, min rating).
- Add analytics dashboard (bar chart for popular genres per year).

## Scalability notes

- move cache to Redis for multi-instance deployments
- precompute embeddings for fetched movie corpus
- add async background refresh for TMDb movie pools
- persist fetched movies in PostgreSQL for repeat analytics and history
