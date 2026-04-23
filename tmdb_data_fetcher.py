"""
TMDb Data Fetcher Module
Handles all interactions with The Movie Database (TMDb) API
Includes caching, error handling, and data preprocessing
"""

import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from functools import lru_cache
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TMDbFetcher:
    """
    Fetches movie data from TMDb API with intelligent caching
    """
    
    def __init__(self, api_key=None):
        """
        Initialize TMDb API client
        
        Args:
            api_key (str): TMDb API key. If None, reads from TMDB_API_KEY env var
                          For free key: https://www.themoviedb.org/settings/api
        """
        self.api_key = api_key or os.getenv('TMDB_API_KEY', '')
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        
        # Cache settings
        self.cache_file = 'tmdb_cache.json'
        self.cache_timeout = 86400  # 24 hours
        self.movie_cache = self._load_cache()
        
        # Session for API requests
        self.session = requests.Session()
        self.session.timeout = 10
        
        if self.api_key:
            logger.info(f"✓ TMDb API initialized (key: {self.api_key[:5]}...)")
        else:
            logger.warning("⚠ TMDb API key not found. Using mock data.")
    
    def _load_cache(self):
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.movie_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def discover_movies(self, sort_by='popularity.desc', genre_ids=None, 
                       year=None, min_rating=0, max_results=100):
        """
        Discover movies with filtering
        
        Args:
            sort_by (str): Sorting criterion (popularity.desc, vote_average.desc, etc.)
            genre_ids (list): Filter by genre IDs
            year (int): Filter by release year
            min_rating (float): Minimum vote average
            max_results (int): Maximum number of results
            
        Returns:
            pd.DataFrame: Movie data
        """
        all_movies = []
        pages_needed = (max_results // 20) + 1
        
        if not self.api_key:
            return self._mock_discover_movies(max_results)
        
        try:
            for page in range(1, pages_needed + 1):
                params = {
                    'api_key': self.api_key,
                    'sort_by': sort_by,
                    'page': page,
                    'vote_average.gte': min_rating,
                }
                
                if genre_ids:
                    params['with_genres'] = ','.join(map(str, genre_ids))
                if year:
                    params['primary_release_year'] = year
                
                response = self.session.get(
                    f"{self.base_url}/discover/movie",
                    params=params
                )
                response.raise_for_status()
                
                movies = response.json().get('results', [])
                all_movies.extend(movies)
                
                if len(all_movies) >= max_results:
                    break
            
            # Convert to DataFrame and clean
            df = pd.DataFrame(all_movies[:max_results])
            return self._clean_movie_data(df)
        
        except Exception as e:
            logger.error(f"Discover movies error: {e}")
            return self._mock_discover_movies(max_results)
    
    def search_movies(self, query, max_results=20):
        """
        Search for movies by title
        
        Args:
            query (str): Movie title or search query
            max_results (int): Maximum results
            
        Returns:
            pd.DataFrame: Search results
        """
        if not query or len(query) < 2:
            return pd.DataFrame()
        
        cache_key = f"search_{query}"
        if cache_key in self.movie_cache:
            return pd.DataFrame(self.movie_cache[cache_key])
        
        if not self.api_key:
            return self._mock_search_results(query, max_results)
        
        try:
            params = {
                'api_key': self.api_key,
                'query': query,
                'page': 1
            }
            
            response = self.session.get(
                f"{self.base_url}/search/movie",
                params=params
            )
            response.raise_for_status()
            
            movies = response.json().get('results', [])[:max_results]
            df = pd.DataFrame(movies)
            df = self._clean_movie_data(df)
            
            self.movie_cache[cache_key] = movies
            self._save_cache()
            
            return df
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self._mock_search_results(query, max_results)
    
    def get_movie_details(self, movie_id):
        """
        Get detailed information for a specific movie
        
        Args:
            movie_id (int): TMDb movie ID
            
        Returns:
            dict: Movie details
        """
        cache_key = f"movie_{movie_id}"
        if cache_key in self.movie_cache:
            return self.movie_cache[cache_key]
        
        if not self.api_key:
            return self._mock_movie_details(movie_id)
        
        try:
            params = {
                'api_key': self.api_key,
                'append_to_response': 'credits,recommendations'
            }
            
            response = self.session.get(
                f"{self.base_url}/movie/{movie_id}",
                params=params
            )
            response.raise_for_status()
            
            movie = response.json()
            details = self._format_movie_details(movie)
            
            self.movie_cache[cache_key] = details
            self._save_cache()
            
            return details
        
        except Exception as e:
            logger.error(f"Details error: {e}")
            return self._mock_movie_details(movie_id)
    
    def get_genres(self):
        """Get all available genres"""
        if not self.api_key:
            return self._mock_genres()
        
        try:
            params = {'api_key': self.api_key}
            response = self.session.get(
                f"{self.base_url}/genre/movie/list",
                params=params
            )
            response.raise_for_status()
            genres = response.json().get('genres', [])
            return {g['id']: g['name'] for g in genres}
        except Exception as e:
            logger.error(f"Genres error: {e}")
            return self._mock_genres()
    
    def _clean_movie_data(self, df):
        """Clean and standardize movie data"""
        if df.empty:
            return df
        
        # Rename columns for consistency
        rename_map = {
            'vote_average': 'rating',
            'vote_count': 'votes',
            'release_date': 'date',
            'genre_ids': 'genres'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Select important columns
        important_cols = ['id', 'title', 'genres', 'rating', 'votes', 
                         'popularity', 'overview', 'date', 'poster_path']
        available_cols = [col for col in important_cols if col in df.columns]
        df = df[available_cols]
        
        # Fill nulls
        df = df.fillna({
            'rating': 0,
            'votes': 0,
            'popularity': 0,
            'overview': '',
            'date': '',
            'genres': []
        })
        
        return df
    
    def _format_movie_details(self, movie):
        """Format movie details response"""
        return {
            'id': movie.get('id'),
            'title': movie.get('title', 'N/A'),
            'year': movie.get('release_date', '')[:4],
            'date': movie.get('release_date', ''),
            'rating': movie.get('vote_average', 0),
            'votes': movie.get('vote_count', 0),
            'popularity': movie.get('popularity', 0),
            'overview': movie.get('overview', 'No description available'),
            'genres': [g['name'] for g in movie.get('genres', [])],
            'runtime': movie.get('runtime', 0),
            'budget': movie.get('budget', 0),
            'revenue': movie.get('revenue', 0),
            'tagline': movie.get('tagline', ''),
            'status': movie.get('status', ''),
            'poster': self.image_base_url + movie['poster_path'] if movie.get('poster_path') else None,
            'backdrop': self.image_base_url + movie['backdrop_path'] if movie.get('backdrop_path') else None,
            'cast': self._extract_cast(movie.get('credits', {})),
            'director': self._extract_director(movie.get('credits', {})),
            'recommendations': movie.get('recommendations', {}).get('results', [])[:5],
            'production_countries': [c['name'] for c in movie.get('production_countries', [])],
        }
    
    def _extract_cast(self, credits, limit=5):
        """Extract top cast members"""
        cast_list = credits.get('cast', [])
        return [
            {'name': actor['name'], 'character': actor.get('character', 'Unknown')}
            for actor in cast_list[:limit]
        ]
    
    def _extract_director(self, credits):
        """Extract director(s)"""
        crew = credits.get('crew', [])
        directors = [c['name'] for c in crew if c.get('job') == 'Director']
        return ', '.join(directors) if directors else 'Unknown'
    
    # Mock methods for testing without API key
    def _mock_discover_movies(self, max_results):
        """Return mock movie data for testing"""
        mock_movies = [
            {'id': i, 'title': f'Mock Movie {i}', 'genres': [28, 35], 
             'rating': 7.5 + (i % 3), 'votes': 1000 * i, 'popularity': 50 + i,
             'overview': 'Mock movie for testing', 'release_date': '2023-01-01',
             'poster_path': '/mockposter.jpg'}
            for i in range(1, max_results + 1)
        ]
        return self._clean_movie_data(pd.DataFrame(mock_movies))
    
    def _mock_search_results(self, query, max_results):
        """Return mock search results"""
        mock_results = [
            {'id': i, 'title': f'{query} #{i}', 'genres': [28, 35],
             'rating': 7.0 + (i % 3), 'votes': 500 * i, 'popularity': 30 + i,
             'overview': f'Mock result for "{query}"', 'release_date': '2023-01-01',
             'poster_path': '/mockposter.jpg'}
            for i in range(1, max_results + 1)
        ]
        return self._clean_movie_data(pd.DataFrame(mock_results))
    
    def _mock_movie_details(self, movie_id):
        """Return mock movie details"""
        return {
            'id': movie_id,
            'title': f'Mock Movie {movie_id}',
            'year': '2023',
            'rating': 7.5,
            'votes': 5000,
            'popularity': 50,
            'overview': 'Mock movie details for testing',
            'genres': ['Action', 'Drama'],
            'runtime': 120,
            'budget': 100000000,
            'revenue': 500000000,
            'tagline': 'Mock tagline',
            'status': 'Released',
            'poster': None,
            'backdrop': None,
            'cast': [{'name': 'Actor Name', 'character': 'Character'}],
            'director': 'Director Name',
            'production_countries': ['United States'],
        }
    
    def _mock_genres(self):
        """Return mock genres"""
        return {
            28: 'Action',
            35: 'Comedy',
            18: 'Drama',
            27: 'Horror',
            10749: 'Romance',
            878: 'Science Fiction',
            53: 'Thriller',
            80: 'Crime',
        }


def get_tmdb_fetcher(api_key=None):
    """Factory function to get TMDb fetcher singleton"""
    if not hasattr(get_tmdb_fetcher, '_instance'):
        get_tmdb_fetcher._instance = TMDbFetcher(api_key)
    return get_tmdb_fetcher._instance
