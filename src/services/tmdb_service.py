import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
BASE_URL = "https://api.themoviedb.org/3"

# Basic in-memory cache to avoid repeated API calls
_cache = {}

def fetch_popular_movies(page=1):
    """Fetches popular movies to seed our database."""
    if not TMDB_API_KEY:
        logger.warning("TMDB_API_KEY is not set.")
        return []

    url = f"{BASE_URL}/movie/popular"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": page}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('results', [])
    except Exception as e:
        logger.error(f"Error fetching popular movies: {e}")
        return []

def get_movie_details(movie_id):
    """Fetches details for a specific TMDB movie ID."""
    cache_key = f"tmdb_movie_{movie_id}"
    if cache_key in _cache:
        return _cache[cache_key]

    if not TMDB_API_KEY:
        return None

    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        _cache[cache_key] = data
        return data
    except Exception as e:
        logger.error(f"Error fetching details for {movie_id}: {e}")
        return None

def search_movies(query):
    """Searches TMDB for a movie."""
    if not TMDB_API_KEY:
        return []

    url = f"{BASE_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "query": query, "page": 1}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('results', [])
    except Exception as e:
        logger.error(f"Error searching movies: {e}")
        return []
