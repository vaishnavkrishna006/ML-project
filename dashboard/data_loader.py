import pandas as pd
import numpy as np
import os

def load_dataset():
    """
    Attempts to load the actual dataset. If not available, creates a robust mock dataset
    for demonstration purposes.
    """
    # Adjust this path based on where the actual data resides
    data_path = "../data/sample_data.csv"
    
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            
            # If the CSV doesn't have genre or title (like sample_data.csv), add mock ones
            if 'genre' not in df.columns or 'title' not in df.columns:
                np.random.seed(42)
                genres_list = ["Action", "Sci-Fi", "Drama", "Comedy", "Thriller", "Horror", "Romance", "Crime"]
                unique_movies = df['movie_id'].unique()
                movie_genres = {m: np.random.choice(genres_list) for m in unique_movies}
                movie_titles = {m: f"Movie Title {m}" for m in unique_movies}
                
                if 'genre' not in df.columns:
                    df['genre'] = df['movie_id'].map(movie_genres)
                if 'title' not in df.columns:
                    df['title'] = df['movie_id'].map(movie_titles)
                    
            return df
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            
    # Mock data generation for robust dashboard testing
    np.random.seed(42)
    users = np.random.randint(1, 1000, 5000)
    movies = np.random.randint(1, 500, 5000)
    ratings = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.4, 0.25], size=5000)
    
    genres_list = ["Action", "Sci-Fi", "Drama", "Comedy", "Thriller", "Horror", "Romance", "Crime"]
    movie_genres = {m: np.random.choice(genres_list) for m in set(movies)}
    movie_titles = {m: f"Movie Title {m}" for m in set(movies)}
    
    df = pd.DataFrame({
        "user_id": users,
        "movie_id": movies,
        "rating": ratings,
        "genre": [movie_genres[m] for m in movies],
        "title": [movie_titles[m] for m in movies]
    })
    return df

def get_model_embeddings(num_items=500, embedding_dim=32):
    """
    Simulates getting item embeddings from the trained LightFM model.
    """
    # Normally this would be: model.item_embeddings
    np.random.seed(42)
    return np.random.randn(num_items, embedding_dim)

def get_recommendation_scores(num_items=500):
    """
    Simulates recommendation scores from the model.
    """
    np.random.seed(42)
    return np.random.normal(loc=0.5, scale=0.2, size=num_items)
