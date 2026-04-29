import pandas as pd
import os
import logging
from src.services.tmdb_service import fetch_popular_movies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path='data/movies.csv'):
    """
    Loads data. If file doesn't exist, fetches real movies from TMDB
    and generates simulated ratings to bootstrap the ML model.
    """
    if os.path.exists(file_path):
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        return df
    
    logger.warning(f"File {file_path} not found. Attempting to fetch real data from TMDB.")
    
    # Try fetching real popular movies from TMDB
    popular_movies = []
    for page in range(1, 3):  # Fetch 2 pages (~40 movies)
        movies = fetch_popular_movies(page)
        if movies:
            popular_movies.extend(movies)
            
    movie_ids = [m['id'] for m in popular_movies] if popular_movies else []
    
    if not movie_ids:
        logger.warning("TMDB fetch failed or no API key. Using hardcoded real TMDB IDs.")
        # Fallback real TMDB IDs: Inception, Interstellar, Dark Knight, Matrix, Avatar, etc.
        movie_ids = [27205, 157336, 155, 603, 19995, 24428, 98, 597, 475557, 550]
        
    # Generate simulated interaction data for these real movies
    # 10 users, each rating random movies
    import random
    random.seed(42)
    
    data = []
    for user_id in range(1, 11):
        # Each user rates 5-10 random movies
        num_ratings = random.randint(5, min(10, len(movie_ids)))
        rated_movies = random.sample(movie_ids, num_ratings)
        
        for movie_id in rated_movies:
            rating = random.randint(1, 5)
            data.append({"user_id": user_id, "movie_id": movie_id, "rating": rating})
            
    df = pd.DataFrame(data)
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved generated interaction data to {file_path} for {len(movie_ids)} unique real movies.")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        
    return df
