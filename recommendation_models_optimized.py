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
    """Load the dataset from CSV files - with optional sampling"""
    print("Loading data...")
    
    # Load main data
    play_counts = pd.read_csv('../datasets/data/play_counts.csv')
    users = pd.read_csv('../datasets/data/users.csv')
    
    # Sample data if needed (for testing with large datasets)
    if sample_size:
        print(f"Sampling {sample_size} records for testing...")
        play_counts = play_counts.sample(n=min(sample_size, len(play_counts)), random_state=42)
    
    # Load mappings
    user_id_map = pd.read_csv('../datasets/mapping/user_id_map.csv')
    artist_id_map = pd.read_csv('../datasets/mapping/artist_id_map.csv')
    
    print(f"Play counts shape: {play_counts.shape}")
    print(f"Users shape: {users.shape}")
    print("Data loaded successfully!\n")
    
    return play_counts, users, user_id_map, artist_id_map

# ============================================================================
# 1. COLLABORATIVE FILTERING (OPTIMIZED FOR LARGE DATASETS)
# ============================================================================

class CollaborativeFiltering:
    """
    Optimized Collaborative Filtering using sparse matrices
    Handles large datasets efficiently
    """
    
    def __init__(self, play_counts):
        """Initialize with play counts data"""
        print("Building sparse interaction matrix...")
        
        # Create mappings for user and artist IDs
        unique_users = play_counts['user_id'].unique()
        unique_artists = play_counts['artist_id'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.artist_id_map = {aid: idx for idx, aid in enumerate(unique_artists)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_artist_map = {idx: aid for aid, idx in self.artist_id_map.items()}
        
        # Create sparse matrix
        rows = play_counts['user_id'].map(self.user_id_map).values
        cols = play_counts['artist_id'].map(self.artist_id_map).values
        data = play_counts['play_count'].values
        
        self.interaction_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_users), len(unique_artists))
        )
        
        print(f"Sparse matrix shape: {self.interaction_matrix.shape}")
        print(f"Sparsity: {1 - (self.interaction_matrix.nnz / (self.interaction_matrix.shape[0] * self.interaction_matrix.shape[1])):.4f}\n")
    
    def user_based_cf(self, user_id, n_recommendations=5, n_similar_users=10):
        """
        User-based Collaborative Filtering
        Find similar users and recommend their favorite artists
        """
        if user_id not in self.user_id_map:
            return pd.DataFrame()
        
        user_idx = self.user_id_map[user_id]
        user_vector = self.interaction_matrix[user_idx].toarray().flatten()
        
        # Calculate similarity with other users (sample for speed)
        n_users = self.interaction_matrix.shape[0]
        sample_size = min(1000, n_users)  # Sample 1000 users for similarity calculation
        sample_indices = np.random.choice(n_users, sample_size, replace=False)
        
        sample_matrix = self.interaction_matrix[sample_indices]
        similarities = cosine_similarity([user_vector], sample_matrix.toarray())[0]
        
        # Get top similar users
        similar_user_indices = np.argsort(similarities)[-n_similar_users-1:-1][::-1]
        similar_user_indices = sample_indices[similar_user_indices]
        
        # Get artists liked by similar users
        target_artists = set(self.interaction_matrix[user_idx].nonzero()[1])
        recommendations = {}
        
        for similar_idx in similar_user_indices:
            similar_user_artists = self.interaction_matrix[similar_idx].nonzero()[1]
            for artist_idx in similar_user_artists:
                if artist_idx not in target_artists:
                    recommendations[artist_idx] = recommendations.get(artist_idx, 0) + \
                        self.interaction_matrix[similar_idx, artist_idx]
        
        # Convert indices back to IDs
        recommendations_list = [
            {'artist_id': self.reverse_artist_map[art_idx], 'score': score}
            for art_idx, score in recommendations.items()
        ]
        recommendations_df = pd.DataFrame(recommendations_list)
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(n_recommendations)
        
        return recommendations_df
    
    def item_based_cf(self, user_id, n_recommendations=5):
        """
        Item-based Collaborative Filtering
        Find similar artists to ones the user already likes
        """
        if user_id not in self.user_id_map:
            return pd.DataFrame()
        
        user_idx = self.user_id_map[user_id]
        user_artists = self.interaction_matrix[user_idx].nonzero()[1]
        
        if len(user_artists) == 0:
            return pd.DataFrame()
        
        recommendations = {}
        
        # For each artist the user likes, find similar artists
        for artist_idx in user_artists:
            artist_vector = self.interaction_matrix[:, artist_idx].toarray().flatten()
            
            # Sample columns for similarity (to speed up)
            n_artists = self.interaction_matrix.shape[1]
            sample_size = min(500, n_artists)
            sample_artist_indices = np.random.choice(n_artists, sample_size, replace=False)
            
            sample_artists = self.interaction_matrix[:, sample_artist_indices].toarray()
            similarities = cosine_similarity([artist_vector], sample_artists.T)[0]
            
            # Get top similar artists
            similar_indices = np.argsort(similarities)[-6:][::-1]
            for sim_idx in similar_indices:
                similar_artist_idx = sample_artist_indices[sim_idx]
                if similar_artist_idx not in user_artists:
                    recommendations[similar_artist_idx] = recommendations.get(similar_artist_idx, 0) + similarities[sim_idx]
        
        # Convert to dataframe
        recommendations_list = [
            {'artist_id': self.reverse_artist_map[art_idx], 'score': score}
            for art_idx, score in recommendations.items()
        ]
        recommendations_df = pd.DataFrame(recommendations_list)
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(n_recommendations)
        
        return recommendations_df

# ============================================================================
# 2. CONTENT-BASED FILTERING
# ============================================================================

class ContentBasedFiltering:
    """
    Content-Based Filtering using user demographics and preferences
    """
    
    def __init__(self, users, play_counts):
        """Initialize with user and play count data"""
        self.users = users
        self.play_counts = play_counts
        
        # Create user profiles
        self.user_profiles = self._create_user_profiles()
        print(f"User profiles shape: {self.user_profiles.shape}\n")
    
    def _create_user_profiles(self):
        """Create user profiles based on demographics and listening habits"""
        user_profiles = self.users.copy()
        
        # Add listening statistics per user
        user_stats = self.play_counts.groupby('user_id').agg({
            'play_count': ['sum', 'mean', 'std', 'count']
        }).reset_index()
        user_stats.columns = ['user_id', 'total_plays', 'avg_plays', 'std_plays', 'num_artists']
        
        user_profiles = user_profiles.merge(user_stats, on='user_id', how='left')
        user_profiles = user_profiles.fillna(0)
        
        return user_profiles
    
    def find_similar_users(self, user_id, n_similar=5):
        """Find similar users based on demographics and listening patterns"""
        
        if user_id not in self.user_profiles['user_id'].values:
            return []
        
        # Sample users for similarity calculation (to speed up)
        sample_size = min(5000, len(self.user_profiles))
        sampled_profiles = self.user_profiles.sample(n=sample_size, random_state=42)
        
        # Prepare features
        feature_cols = [col for col in self.user_profiles.columns if col not in ['user_id']]
        features = sampled_profiles[feature_cols].fillna(0).values
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Find target user in sample
        target_user_idx = sampled_profiles[sampled_profiles['user_id'] == user_id].index
        if len(target_user_idx) == 0:
            # If not in sample, find in full profiles
            user_profile = self.user_profiles[self.user_profiles['user_id'] == user_id]
            if len(user_profile) == 0:
                return []
            
            # Calculate similarity with sampled users
            user_features = user_profile[feature_cols].fillna(0).values
            user_features_scaled = scaler.transform(user_features)
            similarity = cosine_similarity(user_features_scaled, features_scaled)[0]
        else:
            target_idx = sampled_profiles.index.get_loc(target_user_idx[0])
            similarity = cosine_similarity([features_scaled[target_idx]], features_scaled)[0]
        
        # Get top similar users
        similar_indices = np.argsort(similarity)[-n_similar-1:-1][::-1]
        similar_users = sampled_profiles.iloc[similar_indices]['user_id'].tolist()
        
        return similar_users
    
    def recommend_based_on_profile(self, user_id, n_recommendations=5):
        """Recommend artists based on user profile similarity"""
        similar_users = self.find_similar_users(user_id, n_similar=10)
        
        if not similar_users:
            return pd.DataFrame()
        
        # Get artists played by similar users
        user_artists = set(self.play_counts[self.play_counts['user_id'] == user_id]['artist_id'])
        
        recommendations = {}
        for similar_user in similar_users:
            similar_user_plays = self.play_counts[self.play_counts['user_id'] == similar_user]
            for _, row in similar_user_plays.iterrows():
                if row['artist_id'] not in user_artists:
                    recommendations[row['artist_id']] = recommendations.get(row['artist_id'], 0) + row['play_count']
        
        recommendations_df = pd.DataFrame(list(recommendations.items()), columns=['artist_id', 'score'])
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
    
    def __init__(self, play_counts, users):
        """Initialize all three models"""
        print("Initializing Hybrid Model...")
        
        self.cf_model = CollaborativeFiltering(play_counts)
        self.cb_model = ContentBasedFiltering(users, play_counts)
        self.play_counts = play_counts
        
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
            Weights for each model
        """
        
        if weights is None:
            weights = {'user_cf': 0.33, 'item_cf': 0.33, 'content': 0.34}
        
        # Get recommendations from each model
        user_cf_recs = self.cf_model.user_based_cf(user_id, n_recommendations=20)
        item_cf_recs = self.cf_model.item_based_cf(user_id, n_recommendations=20)
        content_recs = self.cb_model.recommend_based_on_profile(user_id, n_recommendations=20)
        
        # Combine scores
        all_recommendations = {}
        
        # Normalize scores to 0-1 range
        def normalize_scores(recs):
            if len(recs) == 0:
                return {}
            max_score = recs['score'].max()
            if max_score == 0:
                return dict(zip(recs['artist_id'], [0] * len(recs)))
            return dict(zip(recs['artist_id'], recs['score'] / max_score))
        
        # Add user-based CF scores
        user_cf_normalized = normalize_scores(user_cf_recs)
        for artist, score in user_cf_normalized.items():
            all_recommendations[artist] = all_recommendations.get(artist, 0) + score * weights['user_cf']
        
        # Add item-based CF scores
        item_cf_normalized = normalize_scores(item_cf_recs)
        for artist, score in item_cf_normalized.items():
            all_recommendations[artist] = all_recommendations.get(artist, 0) + score * weights['item_cf']
        
        # Add content-based scores
        content_normalized = normalize_scores(content_recs)
        for artist, score in content_normalized.items():
            all_recommendations[artist] = all_recommendations.get(artist, 0) + score * weights['content']
        
        # Sort and return top N
        final_recommendations = pd.DataFrame(
            list(all_recommendations.items()),
            columns=['artist_id', 'hybrid_score']
        )
        final_recommendations = final_recommendations.sort_values('hybrid_score', ascending=False).head(n_recommendations)
        
        return final_recommendations

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Load data with sampling for faster testing
    print("="*80)
    print("RECOMMENDATION MODELS - OPTIMIZED FOR LARGE DATASETS")
    print("="*80 + "\n")
    
    # For full dataset: comment out sample_size parameter
    # For testing: use sample_size=1000000
    play_counts, users, user_id_map, artist_id_map = load_data(sample_size=2000000)
    
    # Initialize models
    hybrid = HybridRecommender(play_counts, users)
    
    # Get sample users
    sample_users = play_counts['user_id'].unique()[:10]
    
    print("="*80)
    print("TESTING HYBRID MODEL")
    print("="*80 + "\n")
    
    # Test hybrid model
    recommendations_list = []
    for i, user_id in enumerate(sample_users):
        print(f"\n[{i+1}/10] User: {user_id}")
        print("-" * 50)
        
        recs = hybrid.recommend(user_id, n_recommendations=5)
        
        if len(recs) > 0:
            print("Top 5 Recommendations:")
            for idx, (_, row) in enumerate(recs.iterrows(), 1):
                print(f"  {idx}. Artist ID: {int(row['artist_id']):6d} | Score: {row['hybrid_score']:.4f}")
            
            recs['user_id'] = user_id
            recommendations_list.append(recs)
        else:
            print("  No recommendations available")
    
    # Save results
    if recommendations_list:
        hybrid_results = pd.concat(recommendations_list, ignore_index=True)
        output_path = '../datasets/recommendations/hybrid_recommendations_optimized.csv'
        hybrid_results.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
        print(f"  Total recommendations: {len(hybrid_results)}")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
