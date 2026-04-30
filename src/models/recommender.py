import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class MovieRecommender:
    """
    A Content-Based Recommender using TF-IDF and Cosine Similarity.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.movies_df = None
        self.is_fitted = False

    def fit(self, df):
        """
        Fits the recommender on a dataframe containing 'title' and 'features'.
        'features' should be a combined string of genres, overviews, etc.
        """
        if 'title' not in df.columns or 'features' not in df.columns:
            raise ValueError("Dataframe must contain 'title' and 'features' columns.")
        
        self.movies_df = df.copy().reset_index(drop=True)
        # Handle missing values
        self.movies_df['features'] = self.movies_df['features'].fillna('')
        
        print("Computing TF-IDF matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['features'])
        self.is_fitted = True
        print("Model fitted successfully.")

    def recommend(self, title, top_n=10):
        """
        Recommends movies similar to the given title.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        
        # Case insensitive search
        idx_matches = self.movies_df[self.movies_df['title'].str.lower() == title.lower()].index
        
        if len(idx_matches) == 0:
            return None # Or raise meaningful error
        
        idx = idx_matches[0]
        
        # Compute similarity between this movie and all others
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        # Get indices of top movies (excluding the movie itself)
        related_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]
        
        recommendations = self.movies_df.iloc[related_indices].copy()
        recommendations['similarity_score'] = sim_scores[related_indices]
        
        return recommendations[['title', 'similarity_score']]

    def search_by_description(self, description, top_n=10):
        """
        Bonus: Recommends movies based on a natural language description.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        
        # Transform the description into the TF-IDF space
        desc_vec = self.vectorizer.transform([description])
        
        # Compute similarity with all movies
        sim_scores = cosine_similarity(desc_vec, self.tfidf_matrix).flatten()
        
        # Get top indices
        top_indices = sim_scores.argsort()[-top_n:][::-1]
        
        results = self.movies_df.iloc[top_indices].copy()
        results['similarity_score'] = sim_scores[top_indices]
        
        return results[['title', 'similarity_score']]

    def save(self, path):
        """Saves the model artifacts."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'movies_df': self.movies_df
        }, path)

    @classmethod
    def load(cls, path):
        """Loads the model artifacts."""
        data = joblib.load(path)
        instance = cls()
        instance.vectorizer = data['vectorizer']
        instance.tfidf_matrix = data['tfidf_matrix']
        instance.movies_df = data['movies_df']
        instance.is_fitted = True
        return instance
