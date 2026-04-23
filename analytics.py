"""
Analytics Module
Provides analytics features like genre trends, ratings analysis, and statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieAnalytics:
    """
    Analyzes movie data and provides analytics features
    """
    
    def __init__(self, movies_df: pd.DataFrame = None):
        """
        Initialize analytics engine
        
        Args:
            movies_df (pd.DataFrame): Movie data
        """
        self.movies_df = movies_df if movies_df is not None else pd.DataFrame()
    
    def update_data(self, movies_df: pd.DataFrame):
        """Update movie data"""
        self.movies_df = movies_df
        logger.info(f"Analytics data updated: {len(movies_df)} movies")
    
    def get_genre_trends(self, year_range: Tuple[int, int] = None) -> Dict:
        """
        Analyze genre trends over years
        
        Returns:
            dict: Genre counts by year
        """
        if self.movies_df.empty:
            return {}
        
        df = self.movies_df.copy()
        
        # Extract year from date
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
        else:
            df['year'] = 2023
        
        # Filter by year range if specified
        if year_range:
            df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
        trends = {}
        
        # Count movies per year
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            trends[int(year)] = {
                'total_movies': len(year_data),
                'avg_rating': float(year_data['rating'].mean()) if 'rating' in year_data.columns else 0,
                'total_votes': int(year_data['votes'].sum()) if 'votes' in year_data.columns else 0,
            }
        
        return trends
    
    def get_popular_genres(self, limit: int = 10) -> Dict:
        """
        Get most popular genres by movie count
        
        Args:
            limit (int): Number of top genres
            
        Returns:
            dict: Genre popularity statistics
        """
        if self.movies_df.empty:
            return {}
        
        genre_counts = {}
        genre_ratings = {}
        
        # Flatten genres list
        for genres in self.movies_df['genres']:
            if isinstance(genres, list):
                for genre in genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            elif isinstance(genres, str):
                for genre in str(genres).split(','):
                    g = genre.strip()
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        
        # Sort by count
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        
        result = {}
        for genre, count in sorted_genres[:limit]:
            result[genre] = {
                'count': count,
                'percentage': round(count / len(self.movies_df) * 100, 2),
            }
        
        return result
    
    def get_highest_rated_movies(self, limit: int = 10, min_votes: int = 100) -> pd.DataFrame:
        """
        Get highest rated movies
        
        Args:
            limit (int): Number of results
            min_votes (int): Minimum number of votes
            
        Returns:
            pd.DataFrame: Top movies
        """
        if self.movies_df.empty:
            return pd.DataFrame()
        
        df = self.movies_df.copy()
        
        # Filter by minimum votes
        if 'votes' in df.columns:
            df = df[df['votes'] >= min_votes]
        
        # Sort by rating and popularity
        df = df.sort_values(
            by=['rating', 'popularity'] if 'popularity' in df.columns else ['rating'],
            ascending=False
        )
        
        result = df[['title', 'rating', 'votes', 'popularity', 'genres', 'overview']].head(limit)
        return result.reset_index(drop=True)
    
    def get_genre_statistics(self) -> Dict:
        """
        Get comprehensive genre statistics
        
        Returns:
            dict: Genre statistics
        """
        if self.movies_df.empty:
            return {}
        
        stats = {}
        
        # Get genre list
        all_genres = {}
        for genres in self.movies_df['genres']:
            if isinstance(genres, list):
                for genre in genres:
                    if genre not in all_genres:
                        all_genres[genre] = {'count': 0, 'ratings': []}
                    all_genres[genre]['count'] += 1
                    
                    # Add rating if available
                    if 'rating' in self.movies_df.columns:
                        idx = self.movies_df[self.movies_df['genres'] == genres].index[0]
                        all_genres[genre]['ratings'].append(self.movies_df.loc[idx, 'rating'])
        
        # Calculate statistics
        for genre, data in all_genres.items():
            avg_rating = np.mean(data['ratings']) if data['ratings'] else 0
            stats[genre] = {
                'count': data['count'],
                'avg_rating': round(avg_rating, 2),
                'percentage': round(data['count'] / len(self.movies_df) * 100, 2),
            }
        
        return dict(sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True))
    
    def get_rating_distribution(self, bins: int = 10) -> Dict:
        """
        Get distribution of ratings
        
        Args:
            bins (int): Number of bins for distribution
            
        Returns:
            dict: Rating distribution
        """
        if self.movies_df.empty or 'rating' not in self.movies_df.columns:
            return {}
        
        ratings = self.movies_df['rating'].dropna()
        
        # Create bins
        bin_edges = np.linspace(0, 10, bins + 1)
        hist, _ = np.histogram(ratings, bins=bin_edges)
        
        distribution = {}
        for i in range(len(hist)):
            bin_label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
            distribution[bin_label] = {
                'count': int(hist[i]),
                'percentage': round(hist[i] / len(ratings) * 100, 2),
            }
        
        return distribution
    
    def get_top_directors(self, limit: int = 10) -> Dict:
        """
        Get most prolific directors (if director data available)
        
        Args:
            limit (int): Number of top directors
            
        Returns:
            dict: Director statistics
        """
        if self.movies_df.empty or 'director' not in self.movies_df.columns:
            return {}
        
        director_counts = {}
        director_ratings = {}
        
        for idx, director in enumerate(self.movies_df['director']):
            if isinstance(director, str) and director.lower() != 'unknown':
                # Handle multiple directors
                for d in director.split(','):
                    d = d.strip()
                    if d:
                        director_counts[d] = director_counts.get(d, 0) + 1
                        
                        # Add rating
                        if 'rating' in self.movies_df.columns:
                            rating = self.movies_df.iloc[idx]['rating']
                            if d not in director_ratings:
                                director_ratings[d] = []
                            director_ratings[d].append(rating)
        
        # Sort by count and calculate averages
        result = {}
        for director, count in sorted(director_counts.items(), key=lambda x: x[1], reverse=True)[:limit]:
            avg_rating = np.mean(director_ratings.get(director, [0]))
            result[director] = {
                'count': count,
                'avg_rating': round(avg_rating, 2),
            }
        
        return result
    
    def get_top_production_countries(self, limit: int = 10) -> Dict:
        """
        Get top production countries
        
        Args:
            limit (int): Number of countries
            
        Returns:
            dict: Country statistics
        """
        if self.movies_df.empty or 'production_countries' not in self.movies_df.columns:
            return {}
        
        country_counts = {}
        
        for countries in self.movies_df['production_countries']:
            if isinstance(countries, list):
                for country in countries:
                    country_counts[country] = country_counts.get(country, 0) + 1
            elif isinstance(countries, str):
                for country in str(countries).split(','):
                    c = country.strip()
                    if c:
                        country_counts[c] = country_counts.get(c, 0) + 1
        
        result = {}
        for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:limit]:
            result[country] = {
                'count': count,
                'percentage': round(count / len(self.movies_df) * 100, 2),
            }
        
        return result
    
    def get_summary_statistics(self) -> Dict:
        """
        Get overall summary statistics
        
        Returns:
            dict: Summary statistics
        """
        if self.movies_df.empty:
            return {}
        
        df = self.movies_df
        
        stats = {
            'total_movies': len(df),
            'avg_rating': round(df['rating'].mean() if 'rating' in df.columns else 0, 2),
            'highest_rating': float(df['rating'].max() if 'rating' in df.columns else 0),
            'lowest_rating': float(df['rating'].min() if 'rating' in df.columns else 0),
            'avg_popularity': round(df['popularity'].mean() if 'popularity' in df.columns else 0, 2),
            'total_votes': int(df['votes'].sum() if 'votes' in df.columns else 0),
        }
        
        # Add year range if date info available
        if 'date' in df.columns:
            years = pd.to_datetime(df['date'], errors='coerce').dt.year
            stats['earliest_year'] = int(years.min())
            stats['latest_year'] = int(years.max())
        
        return stats
    
    def export_analytics(self) -> Dict:
        """
        Export all analytics data
        
        Returns:
            dict: Complete analytics data
        """
        return {
            'summary': self.get_summary_statistics(),
            'genre_trends': self.get_genre_trends(),
            'popular_genres': self.get_popular_genres(),
            'genre_statistics': self.get_genre_statistics(),
            'rating_distribution': self.get_rating_distribution(),
            'top_rated_movies': self.get_highest_rated_movies().to_dict('records'),
            'top_directors': self.get_top_directors(),
            'top_countries': self.get_top_production_countries(),
        }


def create_analytics(movies_df: pd.DataFrame = None):
    """Factory function to create analytics engine"""
    return MovieAnalytics(movies_df)
