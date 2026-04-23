"""
Movie Recommendation System - Fast Test Demo
Tests all 3 models on movie data with a smaller sample for quick demonstration
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎬 MOVIE RECOMMENDATION SYSTEM - FAST DEMONSTRATION")
print("="*80 + "\n")

# Load data
print("Loading data...")
movie_watches = pd.read_csv('../datasets/data/movie_watches.csv')
users = pd.read_csv('../datasets/web/user_details.csv')
users.columns = users.columns.str.lower()
print(f"✓ Loaded {len(movie_watches):,} movie watch records from {len(users):,} users\n")

# Sample data for fast testing
sample_records = 100000
movie_watches_sample = movie_watches.sample(n=min(sample_records, len(movie_watches)), random_state=42)
print(f"✓ Using sample of {len(movie_watches_sample):,} records for faster testing\n")

# ============================================================================
# MODEL 1: USER-BASED COLLABORATIVE FILTERING
# ============================================================================

print("="*80)
print("MODEL 1: USER-BASED COLLABORATIVE FILTERING")
print("="*80 + "\n")

print("Building sparse interaction matrix...")
unique_users = movie_watches_sample['user_id'].unique()
unique_movies = movie_watches_sample['movie_id'].unique()

user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
reverse_user_map = {idx: uid for uid, idx in user_id_map.items()}
reverse_movie_map = {idx: mid for mid, idx in movie_id_map.items()}

rows = movie_watches_sample['user_id'].map(user_id_map).values
cols = movie_watches_sample['movie_id'].map(movie_id_map).values
data = movie_watches_sample['watch_count'].values

interaction_matrix = csr_matrix(
    (data, (rows, cols)),
    shape=(len(unique_users), len(unique_movies))
)
print(f"✓ Matrix shape: {interaction_matrix.shape}\n")

# Test with a sample user
test_user_id = movie_watches_sample['user_id'].iloc[0]
test_user_idx = user_id_map[test_user_id]
user_vector = interaction_matrix[test_user_idx].toarray().flatten()

print(f"Computing user similarity for User {test_user_id}...")
user_similarity = cosine_similarity(interaction_matrix)[test_user_idx]

# Find similar users
similar_user_indices = np.argsort(user_similarity)[-11:-1][::-1]

# Get movies watched by similar users
target_movies = set(np.where(user_vector > 0)[0])
recommendations = {}

for similar_idx in similar_user_indices:
    similar_user_movies = interaction_matrix[similar_idx].nonzero()[1]
    for movie_idx in similar_user_movies:
        if movie_idx not in target_movies:
            recommendations[movie_idx] = recommendations.get(movie_idx, 0) + \
                interaction_matrix[similar_idx, movie_idx]

top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"✓ Top 5 Movie Recommendations for User {test_user_id}:\n")
for rank, (movie_idx, score) in enumerate(top_recommendations, 1):
    movie_id = reverse_movie_map[movie_idx]
    print(f"  {rank}. Movie ID: {movie_id}, Score: {score:.2f}")

# ============================================================================
# MODEL 2: ITEM-BASED COLLABORATIVE FILTERING
# ============================================================================

print("\n" + "="*80)
print("MODEL 2: ITEM-BASED COLLABORATIVE FILTERING")
print("="*80 + "\n")

print("Computing movie-to-movie similarity...")
watched_movies = np.where(user_vector > 0)[0]

if len(watched_movies) > 0:
    recommendations = {}
    
    for watched_idx in watched_movies:
        movie_vector = interaction_matrix[:, watched_idx].toarray().flatten()
        
        # Sample movies for comparison (for speed)
        n_movies = len(unique_movies)
        sample_size = min(1000, n_movies)
        sample_movie_indices = np.random.choice(n_movies, sample_size, replace=False)
        
        sample_movies = interaction_matrix[:, sample_movie_indices].toarray()
        similarities = cosine_similarity([movie_vector], sample_movies.T)[0]
        
        similar_indices = np.argsort(similarities)[-6:][::-1]
        for sim_idx in similar_indices:
            similar_movie_idx = sample_movie_indices[sim_idx]
            if similar_movie_idx not in watched_movies:
                recommendations[similar_movie_idx] = recommendations.get(similar_movie_idx, 0) + similarities[sim_idx]
    
    top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"✓ Top 5 Similar Movies for User {test_user_id}:\n")
    for rank, (movie_idx, score) in enumerate(top_recommendations, 1):
        movie_id = reverse_movie_map[movie_idx]
        print(f"  {rank}. Movie ID: {movie_id}, Score: {score:.2f}")
else:
    print("User has no watched movies in sample")

# ============================================================================
# MODEL 3: CONTENT-BASED FILTERING
# ============================================================================

print("\n" + "="*80)
print("MODEL 3: CONTENT-BASED FILTERING")
print("="*80 + "\n")

print("Building user profiles...")

# Get sample users with enough data
user_sample = users[users['encoded_user_id'].isin(unique_users)].copy()

# Add watching statistics
watch_stats = movie_watches_sample.groupby('user_id').agg({
    'watch_count': ['sum', 'mean', 'count']
}).reset_index()
watch_stats.columns = ['encoded_user_id', 'total_watches', 'avg_watches', 'num_movies']

user_sample = user_sample.merge(watch_stats, on='encoded_user_id', how='left')
user_sample = user_sample.fillna(0)

if len(user_sample) > 1:
    # Prepare features
    feature_cols = ['total_watches', 'avg_watches', 'num_movies']
    if 'age' in user_sample.columns:
        feature_cols.insert(0, 'age')
    
    features = user_sample[feature_cols].fillna(0).values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Find similar users
    test_user_idx_in_sample = np.where(user_sample['encoded_user_id'].values == test_user_id)[0]
    
    if len(test_user_idx_in_sample) > 0:
        test_user_idx_in_sample = test_user_idx_in_sample[0]
        similarity = cosine_similarity([features_scaled[test_user_idx_in_sample]], features_scaled)[0]
        similar_indices = np.argsort(similarity)[-11:-1][::-1]
        similar_users = user_sample.iloc[similar_indices]['encoded_user_id'].tolist()
        
        # Get movies watched by similar users
        user_movies = set(movie_watches_sample[movie_watches_sample['user_id'] == test_user_id]['movie_id'])
        recommendations = {}
        
        for similar_user in similar_users:
            similar_watches = movie_watches_sample[movie_watches_sample['user_id'] == similar_user]
            for _, row in similar_watches.iterrows():
                if row['movie_id'] not in user_movies:
                    recommendations[row['movie_id']] = recommendations.get(row['movie_id'], 0) + row['watch_count']
        
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"✓ Top 5 Movies Based on User Profile for User {test_user_id}:\n")
        for rank, (movie_id, score) in enumerate(top_recommendations, 1):
            print(f"  {rank}. Movie ID: {movie_id}, Score: {score:.2f}")
    else:
        print("Test user not found in sample")
else:
    print("Insufficient user data in sample")

print("\n" + "="*80)
print("✓ Demo Complete! All 3 models working with movie data")
print("="*80)
