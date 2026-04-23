import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    """Load the dataset from CSV files"""
    print("Loading data...")
    
    # Load main data - movie watches dataset
    movie_watches = pd.read_csv('../datasets/data/movie_watches.csv')
    users = pd.read_csv('../datasets/web/user_details.csv')
    users.columns = users.columns.str.lower()
    
    print(f"Movie watches shape: {movie_watches.shape}")
    print(f"Users shape: {users.shape}")
    print("Data loaded successfully!\n")
    
    return movie_watches, users

# ============================================================================
# 1. COLLABORATIVE FILTERING
# ============================================================================

class CollaborativeFiltering:
    """
    Collaborative Filtering using user-item interaction matrix
    Includes both User-based and Item-based approaches
    """
    
    def __init__(self, movie_watches):
        """Initialize with movie watch counts data"""
        # Create user-movie interaction matrix
        self.interaction_matrix = movie_watches.pivot_table(
            index='user_id',
            columns='movie_id',
            values='watch_count',
            fill_value=0
        )
        print(f"Interaction matrix shape: {self.interaction_matrix.shape}")
        
    def user_based_cf(self, user_id, n_recommendations=5, n_similar_users=10):
        """
        User-based Collaborative Filtering
        Find similar users and recommend their favorite movies
        """
        # Calculate user similarity using cosine similarity
        user_similarity = cosine_similarity(self.interaction_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=self.interaction_matrix.index,
            columns=self.interaction_matrix.index
        )
        
        if user_id not in user_similarity_df.index:
            return pd.DataFrame()
        
        # Find similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:n_similar_users+1]
        similar_users_list = similar_users.index.tolist()
        
        # Get movies liked by similar users but not by target user
        target_user_movies = set(self.interaction_matrix.loc[user_id][self.interaction_matrix.loc[user_id] > 0].index)
        
        # Aggregate recommendations from similar users
        recommendations = {}
        for similar_user in similar_users_list:
            similar_user_movies = self.interaction_matrix.loc[similar_user][self.interaction_matrix.loc[similar_user] > 0]
            for movie, watch_count in similar_user_movies.items():
                if movie not in target_user_movies:
                    recommendations[movie] = recommendations.get(movie, 0) + watch_count * similar_users[similar_user]
        
        # Sort and return top recommendations
        recommendations_df = pd.DataFrame(list(recommendations.items()), columns=['movie_id', 'score'])
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(n_recommendations)
        
        return recommendations_df
    
    def item_based_cf(self, user_id, n_recommendations=5):
        """
        Item-based Collaborative Filtering
        Find similar movies to those the user already watches
        """
        # Calculate movie similarity
        movie_similarity = cosine_similarity(self.interaction_matrix.T)
        movie_similarity_df = pd.DataFrame(
            movie_similarity,
            index=self.interaction_matrix.columns,
            columns=self.interaction_matrix.columns
        )
        
        # Get movies the user has watched
        user_movies = self.interaction_matrix.loc[user_id][self.interaction_matrix.loc[user_id] > 0]
        
        if len(user_movies) == 0:
            return pd.DataFrame()
        
        recommendations = {}
        for movie, watch_count in user_movies.items():
            # Find similar movies
            similar_movies = movie_similarity_df[movie].sort_values(ascending=False)[1:6]
            for similar_movie, similarity in similar_movies.items():
                if similar_movie not in user_movies.index:
                    recommendations[similar_movie] = recommendations.get(similar_movie, 0) + similarity * watch_count
        
        # Sort and return top recommendations
        recommendations_df = pd.DataFrame(list(recommendations.items()), columns=['movie_id', 'score'])
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(n_recommendations)
        
        return recommendations_df

# ============================================================================
# 2. CONTENT-BASED FILTERING
# ============================================================================

class ContentBasedFiltering:
    """
    Content-Based Filtering using user demographics and viewing preferences
    """
    
    def __init__(self, users, movie_watches):
        """Initialize with user and movie watch data"""
        self.users = users
        self.movie_watches = movie_watches
        
        # Create user profile
        self.user_profiles = self._create_user_profiles()
        print(f"User profiles shape: {self.user_profiles.shape}")
    
    def _create_user_profiles(self):
        """Create user profiles based on demographics and watching habits"""
        user_profiles = self.users.copy()
        
        # Add watching statistics per user
        user_stats = self.movie_watches.groupby('user_id').agg({
            'watch_count': ['sum', 'mean', 'std', 'count']
        }).reset_index()
        user_stats.columns = ['user_id', 'total_watches', 'avg_watches', 'std_watches', 'num_movies']
        
        user_profiles = user_profiles.merge(user_stats, on='user_id', how='left')
        user_profiles = user_profiles.fillna(0)
        
        return user_profiles
    
    def _get_user_features(self, user_id):
        """Extract feature vector for a user"""
        if user_id not in self.user_profiles['user_id'].values:
            return None
        
        user = self.user_profiles[self.user_profiles['user_id'] == user_id].iloc[0]
        
        # Create feature vector (normalize)
        features = [
            user.get('age', 0) if 'age' in user.index else 0,
            user.get('total_plays', 0),
            user.get('num_artists', 0),
        ]
        
        return np.array(features).reshape(1, -1)
    
    def find_similar_users(self, user_id, n_similar=5):
        """Find similar users based on demographics and listening patterns"""
        
        # Prepare features for similarity calculation
        feature_cols = [col for col in self.user_profiles.columns if col not in ['user_id']]
        features = self.user_profiles[feature_cols].fillna(0).values
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Calculate similarity
        similarity = cosine_similarity(features_scaled)
        similarity_df = pd.DataFrame(
            similarity,
            index=self.user_profiles['user_id'],
            columns=self.user_profiles['user_id']
        )
        
        if user_id not in similarity_df.index:
            return []
        
        similar_users = similarity_df[user_id].sort_values(ascending=False)[1:n_similar+1]
        return similar_users.index.tolist()
    
    def recommend_based_on_profile(self, user_id, n_recommendations=5):
        """Recommend movies based on user profile similarity"""
        similar_users = self.find_similar_users(user_id, n_similar=10)
        
        if not similar_users:
            return pd.DataFrame()
        
        # Get movies watched by similar users
        user_movies = set(self.movie_watches[self.movie_watches['user_id'] == user_id]['movie_id'])
        
        recommendations = {}
        for similar_user in similar_users:
            similar_user_watches = self.movie_watches[self.movie_watches['user_id'] == similar_user]
            for _, row in similar_user_watches.iterrows():
                if row['movie_id'] not in user_movies:
                    recommendations[row['movie_id']] = recommendations.get(row['movie_id'], 0) + row['watch_count']
        
        recommendations_df = pd.DataFrame(list(recommendations.items()), columns=['movie_id', 'score'])
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(n_recommendations)
        
        return recommendations_df

# ============================================================================
# 3. HYBRID MODEL
# ============================================================================

class HybridRecommender:
    """
    Hybrid Recommendation System combining:
    - User-based Collaborative Filtering
    - Item-based Collaborative Filtering
    - Content-Based Filtering
    """
    
    def __init__(self, movie_watches, users):
        """Initialize all three models"""
        print("\nInitializing Hybrid Model...")
        
        self.cf_model = CollaborativeFiltering(movie_watches)
        self.cb_model = ContentBasedFiltering(users, movie_watches)
        self.movie_watches = movie_watches
        
        print("Hybrid model initialized!\n")
    
    def recommend(self, user_id, n_recommendations=5, weights=None):
        """
        Generate hybrid recommendations by combining all three models
        
        Parameters:
        -----------
        user_id : int
            User ID to get recommendations for
        n_recommendations : int
            Number of recommendations to return
        weights : dict
            Weights for each model (default: equal weight)
            Example: {'user_cf': 0.3, 'item_cf': 0.3, 'content': 0.4}
        """
        
        if weights is None:
            weights = {'user_cf': 0.33, 'item_cf': 0.33, 'content': 0.34}
        
        # Get recommendations from each model
        user_cf_recs = self.cf_model.user_based_cf(user_id, n_recommendations=20)
        item_cf_recs = self.cf_model.item_based_cf(user_id, n_recommendations=20)
        content_recs = self.cb_model.recommend_based_on_profile(user_id, n_recommendations=20)
        
        # Combine scores
        all_recommendations = {}
        
        # Add user-based CF scores
        for _, row in user_cf_recs.iterrows():
            artist = row['artist_id']
            all_recommendations[artist] = all_recommendations.get(artist, 0) + row['score'] * weights['user_cf']
        
        # Add item-based CF scores
        for _, row in item_cf_recs.iterrows():
            artist = row['artist_id']
            all_recommendations[artist] = all_recommendations.get(artist, 0) + row['score'] * weights['item_cf']
        
        # Add content-based scores
        for _, row in content_recs.iterrows():
            artist = row['artist_id']
            all_recommendations[artist] = all_recommendations.get(artist, 0) + row['score'] * weights['content']
        
        # Sort and return top N
        final_recommendations = pd.DataFrame(
            list(all_recommendations.items()),
            columns=['artist_id', 'hybrid_score']
        )
        final_recommendations = final_recommendations.sort_values('hybrid_score', ascending=False).head(n_recommendations)
        
        return final_recommendations
    
    def recommend_with_breakdown(self, user_id, n_recommendations=5):
        """
        Generate recommendations with breakdown of each model's contribution
        """
        print(f"\n{'='*80}")
        print(f"Generating Hybrid Recommendations for User: {user_id}")
        print(f"{'='*80}\n")
        
        # Get recommendations from each model
        print("1. User-Based Collaborative Filtering:")
        user_cf_recs = self.cf_model.user_based_cf(user_id, n_recommendations=10)
        if len(user_cf_recs) > 0:
            print(user_cf_recs.head(3))
        else:
            print("   No recommendations available")
        
        print("\n2. Item-Based Collaborative Filtering:")
        item_cf_recs = self.cf_model.item_based_cf(user_id, n_recommendations=10)
        if len(item_cf_recs) > 0:
            print(item_cf_recs.head(3))
        else:
            print("   No recommendations available")
        
        print("\n3. Content-Based Filtering:")
        content_recs = self.cb_model.recommend_based_on_profile(user_id, n_recommendations=10)
        if len(content_recs) > 0:
            print(content_recs.head(3))
        else:
            print("   No recommendations available")
        
        print("\n" + "="*80)
        print("HYBRID RECOMMENDATIONS (Combined):")
        print("="*80 + "\n")
        
        hybrid_recs = self.recommend(user_id, n_recommendations=n_recommendations)
        print(hybrid_recs)
        
        return hybrid_recs

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Load data
    play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()
    
    # Initialize Hybrid Recommender
    hybrid = HybridRecommender(play_counts, users, user_id_map, artist_id_map)
    
    # Get a sample of user IDs
    sample_users = play_counts['user_id'].unique()[:5]
    
    print("\n" + "="*80)
    print("TESTING RECOMMENDATION MODELS")
    print("="*80)
    
    # Test on sample users
    for user_id in sample_users:
        recommendations = hybrid.recommend_with_breakdown(user_id, n_recommendations=5)
        print("\n")
    
    # Test individual models
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL TEST")
    print("="*80)
    
    test_user = sample_users[0]
    
    cf_model = CollaborativeFiltering(play_counts)
    cb_model = ContentBasedFiltering(users, play_counts, user_id_map, artist_id_map)
    
    print(f"\nUser: {test_user}")
    
    print("\n--- User-Based CF ---")
    print(cf_model.user_based_cf(test_user, n_recommendations=5))
    
    print("\n--- Item-Based CF ---")
    print(cf_model.item_based_cf(test_user, n_recommendations=5))
    
    print("\n--- Content-Based ---")
    print(cb_model.recommend_based_on_profile(test_user, n_recommendations=5))

if __name__ == "__main__":
    main()
