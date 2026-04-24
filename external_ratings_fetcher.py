"""External movie metadata fetcher (IMDb + Rotten Tomatoes via OMDb)."""

import os
import time
import logging
from typing import Dict, Optional

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalRatingsFetcher:
    """
    Fetches IMDb and Rotten Tomatoes metadata through OMDb API.
    Docs: http://www.omdbapi.com/
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OMDB_API_KEY", "")
        self.base_url = "http://www.omdbapi.com/"
        self.session = requests.Session()
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl_seconds = 86400

        if self.api_key:
            logger.info("OMDb integration enabled.")
        else:
            logger.warning("OMDb key missing. IMDb/Rotten Tomatoes enrichment disabled.")

    def get_external_metadata(self, title: str, year: Optional[str] = None) -> Dict:
        """
        Return canonical external metadata for a movie title.
        """
        if not self.api_key or not title:
            return self._empty_payload()

        cache_key = f"{title.lower().strip()}::{(year or '').strip()}"
        cached = self.cache.get(cache_key)
        if cached and (time.time() - cached["ts"] <= self.cache_ttl_seconds):
            return cached["data"]

        params = {
            "apikey": self.api_key,
            "t": title,
            "type": "movie",
        }
        if year:
            params["y"] = str(year)[:4]

        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if payload.get("Response") != "True":
                data = self._empty_payload()
            else:
                data = self._parse_payload(payload)
        except Exception as exc:
            logger.warning(f"OMDb lookup failed for '{title}': {exc}")
            data = self._empty_payload()

        self.cache[cache_key] = {"ts": time.time(), "data": data}
        return data

    def _parse_payload(self, payload: Dict) -> Dict:
        rotten_score = None
        for rating in payload.get("Ratings", []):
            if rating.get("Source") == "Rotten Tomatoes":
                rotten_score = rating.get("Value")
                break

        return {
            "imdb_title": payload.get("Title"),
            "imdb_id": payload.get("imdbID"),
            "imdb_rating": payload.get("imdbRating"),
            "rotten_tomatoes_rating": rotten_score,
        }

    @staticmethod
    def _empty_payload() -> Dict:
        return {
            "imdb_title": None,
            "imdb_id": None,
            "imdb_rating": None,
            "rotten_tomatoes_rating": None,
        }

