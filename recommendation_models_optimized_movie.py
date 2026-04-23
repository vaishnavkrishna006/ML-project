import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data(sample_size=None):
    """Load movie watch dataset - with optional sampling for memory efficiency"""
    print("Loading movie watch data...")
    
    # Load main data
    movie_watches = pd.read_csv('../datasets/data/movie_watches.csv')
    users = pd.read_csv('../datasets/web/user_details.csv')
    users.columns = users.columns.str.lower()
    
    # Sample data if needed (for testing with large datasets)
    if sample_size:
        print(f"Sampling {sample_size} records for testing...")
        movie_watches = movie_watches.sample(n=min(sample_size, len(movie_watches)), random_state=42)
    
    print(f"Movie watches shape: {movie_watches.shape}")
    print(f"Users shape: {users.shape}\n")
    
    return movie_watches, users

# ============================================================================
# COLLABORATIVE FILTERING - OPTIMIZED WITH SPARSE MATRICES
# ============================================================================

class OptimizedCollaborativeFiltering:
    """
    Optimized Collaborative Filtering using sparse CSR matrices
    Memory-efficient for large datasets
    """
    
    def __init__(self, movie_watches):
        """Initialize with sparse matrix representation"""
        print("Building optimized sparse interaction matrix...")
        
        unique_users = movie_watches['user_id'].unique()
        unique_movies = movie_watches['movie_id'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_movie_map = {idx: mid for mid, idx in self.movie_id_map.items()}
        
        rows = movie_watches['user_id'].map(self.user_id_map).values
        cols = movie_watches['movie_id'].map(self.movie_id_map).values
        data = movie_watches['watch_count'].values
        
        self.interaction_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_users), len(unique_movies))
        )
        print(f"✓ Matrix created: {self.interaction_matrix.shape}, Sparsity: {1 - (len(data) / (len(unique_users) * len(unique_movies))):.4f}\n")
    
    def user_based_cf(self, user_id, n_recommendations=5, n_similar_users=10):
        """User-based CF using sparse matrices"""
        if user_id not in self.user_id_map:
            return pd.DataFrame()
        
        user_idx = self.user_id_map[user_id]
        user_vector = self.interaction_matrix[user_idx].toarray().flatten()
        
        # Compute similarity to all users
        user_similarity = cosine_similarity(self.interaction_matrix)[user_idx]
        similar_user_indices = np.argsort(user_similarity)[-n_similar_users-1:-1][::-1]
        
        # Get recommendations from similar users
        target_movies = set(self.interaction_matrix[user_idx].nonzero()[1])
        recommendations = {}
        
        for similar_idx in similar_user_indices:
            similar_movies = self.interaction_matrix[similar_idx].nonzero()[1]
            for movie_idx in similar_movies:
                if movie_idx not in target_movies:
                    recommendations[movie_idx] = recommendations.get(movie_idx, 0) + \
                        self.interaction_matrix[similar_idx, movie_idx]
        
        recommendations_list = [
            {'movie_id': int(self.reverse_movie_map[mov_idx]), 'score': float(score)}
            for mov_idx, score in recommendations.items()
        ]
        recommendations_df = pd.DataFrame(recommendations_list)
        return recommendations_df.nlargest(n_recommendations, 'score')
    
    def item_based_cf(self, user_id, n_recommendations=5):
        """Item-based CF using sparse matrices"""
        if user_id not in self.user_id_map:
            return pd.DataFrame()
        
        user_idx = self.user_id_map[user_id]
        user_movies = self.interaction_matrix[user_idx].nonzero()[1]
        
        if len(user_movies) == 0:
            return pd.DataFrame()
        
        recommendations = {}
        
        for movie_idx in user_movies:
            movie_vector = self.interaction_matrix[:, movie_idx].toarray().flatten()
            
            # Sample movies for comparison
            n_movies = self.interaction_matrix.shape[1]
            sample_size = min(1000, n_movies)
            sample_movie_indices = np.random.choice(n_movies, sample_size, replace=False)
            
            sample_movies = self.interaction_matrix[:, sample_movie_indices].toarray()
            similarities = cosine_similarity([movie_vector], sample_movies.T)[0]
            
            similar_indices = np.argsort(similarities)[-6:][::-1]
            for sim_idx in similar_indices:
                similar_movie_idx = sample_movie_indices[sim_idx]
                if similar_movie_idx not in user_movies:
                    recommendations[similar_movie_idx] = recommendations.get(similar_movie_idx, 0) + similarities[sim_idx]
        
        recommendations_list = [
            {'movie_id': int(self.reverse_movie_map[mov_idx]), 'score': float(score)}
            for mov_idx, score in recommendations.items()
        ]
        recommendations_df = pd.DataFrame(recommendations_list)
        return recommendations_df.nlargest(n_recommendations, 'score')

# ============================================================================
# CONTENT-BASED FILTERING - OPTIMIZED
# ============================================================================

class OptimizedContentBasedFiltering:
    """Optimized content-based filtering using user profiles"""
    
    def __init__(self, users, movie_watches):
        print("Building user profiles...")
        self.users = users
        self.movie_watches = movie_watches
        self.user_profiles = self._create_user_profiles()
        print(f"✓ Created {len(self.user_profiles)} user profiles\n")
    
    def _create_user_profiles(self):
        user_profiles = self.users.copy()
        
        watch_stats = self.movie_watches.groupby('user_id').agg({
            'watch_count': ['sum', 'mean', 'count']
        }).reset_index()
        watch_stats.columns = ['encoded_user_id', 'total_watches', 'avg_watches', 'num_movies']
        
        user_profiles = user_profiles.merge(watch_stats, on='encoded_user_id', how='left')
        return user_profiles.fillna(0)
    
    def recommend_based_on_profile(self, user_id, n_recommendations=5):
        """Recommend movies based on user profile similarity"""
        if user_id not in self.user_profiles['encoded_user_id'].values:
            return pd.DataFrame()
        
        feature_cols = [col for col in self.user_profiles.columns if col not in ['encoded_user_id', 'original_user_id']]
        features = self.user_profiles[feature_cols].fillna(0).values
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        target_user_idx = np.where(self.user_profiles['encoded_user_id'].values == user_id)[0]
        if len(target_user_idx) == 0:
            return pd.DataFrame()
        
        target_user_idx = target_user_idx[0]
        similarity = cosine_similarity([features_scaled[target_user_idx]], features_scaled)[0]
        similar_indices = np.argsort(similarity)[-11:-1][::-1]
        similar_users = self.user_profiles.iloc[similar_indices]['encoded_user_id'].tolist()
        
        # Get movies watched by similar users
        user_movies = set(self.movie_watches[self.movie_watches['user_id'] == user_id]['movie_id'])
        recommendations = {}
        
        for similar_user in similar_users:
            similar_watches = self.movie_watches[self.movie_watches['user_id'] == similar_user]
            for _, row in similar_watches.iterrows():
                if row['movie_id'] not in user_movies:
                    recommendations[row['movie_id']] = recommendations.get(row['movie_id'], 0) + row['watch_count']
        
        recommendations_df = pd.DataFrame(list(recommendations.items()), columns=['movie_id', 'score'])
        return recommendations_df.nlargest(n_recommendations, 'score')

# ============================================================================
# HYBRID MODEL - OPTIMIZED
# ============================================================================

class OptimizedHybridRecommender:
    """Optimized Hybrid model combining all approaches"""
    
    def __init__(self, movie_watches, users):
        print("Initializing Optimized Hybrid Model...")
        self.cf_model = OptimizedCollaborativeFiltering(movie_watches)
        self.cb_model = OptimizedContentBasedFiltering(users, movie_watches)
        self.movie_watches = movie_watches
        print("✓ Hybrid model initialized\n")
    
    def recommend(self, user_id, n_recommendations=5, weights=None):
        """Generate hybrid recommendations"""
        if weights is None:
            weights = {'cf_user': 0.33, 'cf_item': 0.33, 'cb': 0.34}
        
        # Get recommendations from each model
        cf_user_recs = self.cf_model.user_based_cf(user_id, n_recommendations*3)
        cf_item_recs = self.cf_model.item_based_cf(user_id, n_recommendations*3)
        cb_recs = self.cb_model.recommend_based_on_profile(user_id, n_recommendations*3)
        
        # Combine recommendations
        all_recommendations = {}
        
        for _, rec in cf_user_recs.iterrows():
            all_recommendations[rec['movie_id']] = all_recommendations.get(rec['movie_id'], 0) + \
                rec['score'] * weights['cf_user']
        
        for _, rec in cf_item_recs.iterrows():
            all_recommendations[rec['movie_id']] = all_recommendations.get(rec['movie_id'], 0) + \
                rec['score'] * weights['cf_item']
        
        for _, rec in cb_recs.iterrows():
            all_recommendations[rec['movie_id']] = all_recommendations.get(rec['movie_id'], 0) + \
                rec['score'] * weights['cb']
        
        # Sort and return
        recommendations_df = pd.DataFrame(list(all_recommendations.items()), columns=['movie_id', 'hybrid_score'])
        recommendations_df = recommendations_df.sort_values('hybrid_score', ascending=False).head(n_recommendations)
        
        return recommendations_df
