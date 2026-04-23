"""
Hybrid Recommendation Engine Module
Combines semantic similarity, ratings, popularity, and genre matching
for intelligent movie recommendations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRecommendationEngine:
    """
    Hybrid recommendation system combining multiple signals:
    - Semantic similarity (NLP embeddings)
    - Genre matching
    - Rating scores
    - Popularity scores
    """
    
    def __init__(self, nlp_processor=None, tmdb_fetcher=None):
        """
        Initialize recommendation engine
        
        Args:
            nlp_processor: NLP query processor instance
            tmdb_fetcher: TMDb data fetcher instance
        """
        self.nlp_processor = nlp_processor
        self.tmdb_fetcher = tmdb_fetcher
        self.genre_map = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.overview_vectors = None
        
        # Weighting parameters (can be tuned)
        self.weights = {
            'semantic_similarity': 0.30,
            'genre_match': 0.20,
            'rating': 0.25,
            'popularity': 0.15,
            'keyword_match': 0.10,
        }
    
    def set_weights(self, **kwargs):
        """Update recommendation weights"""
        for key, value in kwargs.items():
            if key in self.weights:
                self.weights[key] = value
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        logger.info(f"Updated weights: {self.weights}")
    
    def recommend(self, query: str, movies_df: pd.DataFrame, 
                  n_recommendations: int = 5, genres_dict: Dict = None) -> pd.DataFrame:
        """
        Get movie recommendations for a query
        
        Args:
            query (str): Natural language query
            movies_df (pd.DataFrame): Available movies
            n_recommendations (int): Number of recommendations
            genres_dict (Dict): Genre ID to name mapping
            
        Returns:
            pd.DataFrame: Top N recommended movies with scores
        """
        if movies_df.empty:
            logger.warning("Empty movies dataframe")
            return pd.DataFrame()
        
        # Parse query to extract intent
        query_intent = self.nlp_processor.process_query(query) if self.nlp_processor else self._default_intent(query)
        
        # Filter movies based on extracted criteria
        filtered_movies = self._filter_by_intent(movies_df, query_intent)
        
        if filtered_movies.empty:
            logger.warning("No movies match criteria, expanding search")
            filtered_movies = movies_df.copy()
        
        # Calculate recommendation scores
        scores_df = self._calculate_scores(
            filtered_movies, 
            query_intent, 
            query,
            genres_dict
        )
        
        # Sort and return top N
        scores_df = scores_df.sort_values('final_score', ascending=False)
        result = scores_df.head(n_recommendations).copy()
        result['rank'] = range(1, len(result) + 1)
        
        return result[['rank', 'id', 'title', 'rating', 'popularity', 
                       'final_score', 'genres', 'overview']].reset_index(drop=True)
    
    def _filter_by_intent(self, movies_df: pd.DataFrame, intent: Dict) -> pd.DataFrame:
        """Filter movies based on extracted intent"""
        filtered = movies_df.copy()
        
        # Filter by minimum rating
        if intent.get('min_rating', 0) > 0:
            filtered = filtered[filtered['rating'] >= intent['min_rating']]
        
        # Filter by year if specified
        if intent.get('year'):
            target_year = int(intent['year'])
            filtered['year'] = pd.to_datetime(filtered.get('date', '2023'), errors='coerce').dt.year
            filtered = filtered[(filtered['year'] >= target_year - 2) & (filtered['year'] <= target_year + 2)]
        
        return filtered if not filtered.empty else movies_df
    
    def _calculate_scores(self, movies_df: pd.DataFrame, intent: Dict, 
                         query: str, genres_dict: Dict = None) -> pd.DataFrame:
        """Calculate hybrid recommendation scores"""
        scores_df = movies_df.copy()
        scores_df['semantic_score'] = 0.0
        scores_df['genre_score'] = 0.0
        scores_df['rating_score'] = 0.0
        scores_df['popularity_score'] = 0.0
        scores_df['keyword_score'] = 0.0
        
        # 1. Semantic Similarity Score
        if self.nlp_processor and hasattr(self.nlp_processor, 'use_semantic') and self.nlp_processor.use_semantic:
            scores_df['semantic_score'] = self._calculate_semantic_similarity_scores(
                movies_df['overview'].values, 
                query
            )
        else:
            scores_df['semantic_score'] = self._calculate_tfidf_similarity_scores(
                movies_df['overview'].values, 
                query
            )
        
        # 2. Genre Match Score
        if intent.get('genres'):
            scores_df['genre_score'] = self._calculate_genre_match_scores(
                movies_df['genres'].values, 
                intent['genres']
            )
        
        # 3. Rating Score (normalized to 0-1)
        scores_df['rating_score'] = scores_df['rating'].fillna(0) / 10.0
        
        # 4. Popularity Score (normalized)
        if scores_df['popularity'].max() > 0:
            scores_df['popularity_score'] = scores_df['popularity'] / scores_df['popularity'].max()
        
        # 5. Keyword Match Score
        if intent.get('keywords'):
            scores_df['keyword_score'] = self._calculate_keyword_match_scores(
                movies_df['overview'].values,
                intent['keywords']
            )
        
        # Combine scores with weights
        scores_df['final_score'] = (
            self.weights['semantic_similarity'] * scores_df['semantic_score'] +
            self.weights['genre_match'] * scores_df['genre_score'] +
            self.weights['rating'] * scores_df['rating_score'] +
            self.weights['popularity'] * scores_df['popularity_score'] +
            self.weights['keyword_match'] * scores_df['keyword_score']
        )
        
        return scores_df
    
    def _calculate_semantic_similarity_scores(self, texts: np.ndarray, query: str) -> np.ndarray:
        """Calculate semantic similarity using embeddings"""
        if not self.nlp_processor or not self.nlp_processor.use_semantic:
            return np.zeros(len(texts))
        
        try:
            query_embedding = self.nlp_processor.get_semantic_embedding(query)
            if query_embedding is None:
                return np.zeros(len(texts))
            
            text_embeddings = np.array([
                self.nlp_processor.get_semantic_embedding(text) for text in texts
            ])
            
            similarities = cosine_similarity([query_embedding], text_embeddings)[0]
            return np.maximum(similarities, 0)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Semantic similarity error: {e}")
            return np.zeros(len(texts))
    
    def _calculate_tfidf_similarity_scores(self, texts: np.ndarray, query: str) -> np.ndarray:
        """Calculate TF-IDF similarity scores"""
        try:
            all_texts = list(texts) + [query]
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            query_vector = tfidf_matrix[-1]
            text_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(query_vector, text_vectors)[0]
            return np.maximum(similarities, 0)
        except Exception as e:
            logger.warning(f"TF-IDF similarity error: {e}")
            return np.zeros(len(texts))
    
    def _calculate_genre_match_scores(self, movie_genres: np.ndarray, 
                                     query_genres: List[str]) -> np.ndarray:
        """Calculate genre matching scores"""
        if not query_genres:
            return np.zeros(len(movie_genres))
        
        scores = np.zeros(len(movie_genres))
        
        for idx, genres in enumerate(movie_genres):
            if isinstance(genres, list):
                # Count matching genres
                matches = sum(1 for g in genres if any(qg.lower() in str(g).lower() for qg in query_genres))
                scores[idx] = min(matches / max(len(query_genres), 1), 1.0)
            elif isinstance(genres, str):
                # Handle string representation of genres
                genre_list = str(genres).lower().split(',')
                matches = sum(1 for g in genre_list if any(qg.lower() in g for qg in query_genres))
                scores[idx] = min(matches / max(len(query_genres), 1), 1.0)
        
        return scores
    
    def _calculate_keyword_match_scores(self, texts: np.ndarray, 
                                       keywords: List[str]) -> np.ndarray:
        """Calculate keyword matching scores"""
        scores = np.zeros(len(texts))
        
        for idx, text in enumerate(texts):
            if isinstance(text, str):
                text_lower = text.lower()
                matches = sum(1 for kw in keywords if kw.lower() in text_lower)
                scores[idx] = min(matches / max(len(keywords), 1), 1.0)
        
        return scores
    
    def _default_intent(self, query: str) -> Dict:
        """Default intent when NLP processor not available"""
        return {
            'genres': [],
            'mood': 'general',
            'count': 5,
            'min_rating': 0,
            'year': None,
            'keywords': query.split()[:5],
        }


def create_recommendation_engine(nlp_processor=None, tmdb_fetcher=None):
    """Factory function to create recommendation engine"""
    return HybridRecommendationEngine(nlp_processor, tmdb_fetcher)
