import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")
BASE_URL = "http://www.omdbapi.com/"

_cache = {}

def get_imdb_ratings(imdb_id):
    """
    Fetches ratings for a given IMDb ID using OMDb API.
    Returns a dictionary with 'imdb_rating', 'rotten_tomatoes'.
    """
    if not imdb_id:
        return {}
        
    cache_key = f"omdb_{imdb_id}"
    if cache_key in _cache:
        return _cache[cache_key]

    if not OMDB_API_KEY:
        logger.warning("OMDB_API_KEY is not set.")
        return {}

    params = {"apikey": OMDB_API_KEY, "i": imdb_id}
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        result = {
            "imdb_rating": data.get("imdbRating", "N/A"),
            "rotten_tomatoes": "N/A"
        }
        
        # Extract Rotten Tomatoes if available
        ratings = data.get("Ratings", [])
        for r in ratings:
            if r.get("Source") == "Rotten Tomatoes":
                result["rotten_tomatoes"] = r.get("Value", "N/A")
                break
                
        _cache[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"Error fetching OMDb data for {imdb_id}: {e}")
        return {}
