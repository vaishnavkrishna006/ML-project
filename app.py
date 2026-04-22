"""
Flask Web Application for Movie Recommendation System
Provides a user interface to get recommendations from all 3 models
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store models
cf_model = None
cb_model = None
play_counts = None
users = None

# ============================================================================
# MODEL CLASSES
# ============================================================================

class CollaborativeFiltering:
    def __init__(self, watch_counts):
        print("Building sparse interaction matrix...")
        unique_users = watch_counts['user_id'].unique()
        unique_movies = watch_counts['movie_id'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_movie_map = {idx: mid for mid, idx in self.movie_id_map.items()}
        
        rows = watch_counts['user_id'].map(self.user_id_map).values
        cols = watch_counts['movie_id'].map(self.movie_id_map).values
        data = watch_counts['watch_count'].values
        
        self.interaction_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_users), len(unique_artists))
        )
    
    def user_based_cf(self, user_id, n_recommendations=5):
        if user_id not in self.user_id_map:
            return pd.DataFrame()
        
        user_idx = self.user_id_map[user_id]
        user_vector = self.interaction_matrix[user_idx].toarray().flatten()
        
        n_users = self.interaction_matrix.shape[0]
        sample_size = min(1000, n_users)
        sample_indices = np.random.choice(n_users, sample_size, replace=False)
        
        sample_matrix = self.interaction_matrix[sample_indices]
        similarities = cosine_similarity([user_vector], sample_matrix.toarray())[0]
        
        similar_user_indices = np.argsort(similarities)[-11:-1][::-1]
        similar_user_indices = sample_indices[similar_user_indices]
        
        target_movies = set(self.interaction_matrix[user_idx].nonzero()[1])
        recommendations = {}
        
        for similar_idx in similar_user_indices:
            similar_user_movies = self.interaction_matrix[similar_idx].nonzero()[1]
            for movie_idx in similar_user_movies:
                if movie_idx not in target_movies:
                    recommendations[movie_idx] = recommendations.get(movie_idx, 0) + \
                        self.interaction_matrix[similar_idx, movie_idx]
        
        recommendations_list = [
            {'artist_id': int(self.reverse_artist_map[art_idx]), 'score': float(score)}
            for art_idx, score in recommendations.items()
        ]
        recommendations_df = pd.DataFrame(recommendations_list)
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(n_recommendations)
        
        return recommendations_df
    
    def item_based_cf(self, user_id, n_recommendations=5):
        if user_id not in self.user_id_map:
            return pd.DataFrame()
        
        user_idx = self.user_id_map[user_id]
        user_artists = self.interaction_matrix[user_idx].nonzero()[1]
        
        if len(user_artists) == 0:
            return pd.DataFrame()
        
        recommendations = {}
        
        for artist_idx in user_artists:
            artist_vector = self.interaction_matrix[:, artist_idx].toarray().flatten()
            
            n_artists = self.interaction_matrix.shape[1]
            sample_size = min(500, n_artists)
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
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(n_recommendations)
        
        return recommendations_df

class ContentBasedFiltering:
    def __init__(self, users, watch_counts):
        self.users = users
        self.watch_counts = watch_counts
        self.user_profiles = self._create_user_profiles()
    
    def _create_user_profiles(self):
        user_profiles = self.users.copy()
        
        user_stats = self.watch_counts.groupby('user_id').agg({
            'watch_count': ['sum', 'mean', 'std', 'count']
        }).reset_index()
        user_stats.columns = ['user_id', 'total_watches', 'avg_watches', 'std_watches', 'num_movies']
        
        user_profiles = user_profiles.merge(user_stats, on='user_id', how='left')
        user_profiles = user_profiles.fillna(0)
        
        return user_profiles
    
    def recommend_based_on_profile(self, user_id, n_recommendations=5):
        if user_id not in self.user_profiles['user_id'].values:
            return pd.DataFrame()
        
        sample_size = min(5000, len(self.user_profiles))
        sampled_profiles = self.user_profiles.sample(n=sample_size, random_state=42)
        
        feature_cols = [col for col in self.user_profiles.columns if col not in ['user_id']]
        features = sampled_profiles[feature_cols].fillna(0).values
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        target_user_idx = sampled_profiles[sampled_profiles['user_id'] == user_id].index
        if len(target_user_idx) == 0:
            user_profile = self.user_profiles[self.user_profiles['user_id'] == user_id]
            if len(user_profile) == 0:
                return pd.DataFrame()
            
            user_features = user_profile[feature_cols].fillna(0).values
            user_features_scaled = scaler.transform(user_features)
            similarity = cosine_similarity(user_features_scaled, features_scaled)[0]
        else:
            target_idx = sampled_profiles.index.get_loc(target_user_idx[0])
            similarity = cosine_similarity([features_scaled[target_idx]], features_scaled)[0]
        
        similar_indices = np.argsort(similarity)[-11:-1][::-1]
        similar_users = sampled_profiles.iloc[similar_indices]['user_id'].tolist()
        
        user_movies = set(self.watch_counts[self.watch_counts['user_id'] == user_id]['movie_id'])
        
        recommendations = {}
        for similar_user in similar_users:
            similar_user_watches = self.watch_counts[self.watch_counts['user_id'] == similar_user]
            for _, row in similar_user_watches.iterrows():
                if row['movie_id'] not in user_movies:
                    recommendations[row['movie_id']] = recommendations.get(row['movie_id'], 0) + row['watch_count']
        
        recommendations_df = pd.DataFrame(list(recommendations.items()), columns=['movie_id', 'score'])
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(n_recommendations)
        
        return recommendations_df

class HybridRecommender:
    def __init__(self, watch_counts, users):
        self.cf_model = CollaborativeFiltering(watch_counts)
        self.cb_model = ContentBasedFiltering(users, watch_counts)
        self.watch_counts = watch_counts
    
    def recommend(self, user_id, n_recommendations=5, weights=None):
        if weights is None:
            weights = {'user_cf': 0.33, 'item_cf': 0.33, 'content': 0.34}
        
        user_cf_recs = self.cf_model.user_based_cf(user_id, n_recommendations=20)
        item_cf_recs = self.cf_model.item_based_cf(user_id, n_recommendations=20)
        content_recs = self.cb_model.recommend_based_on_profile(user_id, n_recommendations=20)
        
        all_recommendations = {}
        
        def normalize_scores(recs):
            if len(recs) == 0:
                return {}
            max_score = recs['score'].max()
            if max_score == 0:
                return dict(zip(recs['artist_id'], [0] * len(recs)))
            return dict(zip(recs['artist_id'], recs['score'] / max_score))
        
        user_cf_normalized = normalize_scores(user_cf_recs)
        for artist, score in user_cf_normalized.items():
            all_recommendations[artist] = all_recommendations.get(artist, 0) + score * weights['user_cf']
        
        item_cf_normalized = normalize_scores(item_cf_recs)
        for artist, score in item_cf_normalized.items():
            all_recommendations[artist] = all_recommendations.get(artist, 0) + score * weights['item_cf']
        
        content_normalized = normalize_scores(content_recs)
        for artist, score in content_normalized.items():
            all_recommendations[artist] = all_recommendations.get(artist, 0) + score * weights['content']
        
        final_recommendations = pd.DataFrame(
            list(all_recommendations.items()),
            columns=['artist_id', 'hybrid_score']
        )
        final_recommendations = final_recommendations.sort_values('hybrid_score', ascending=False).head(n_recommendations)
        final_recommendations['artist_id'] = final_recommendations['artist_id'].astype(int)
        
        return final_recommendations

# ============================================================================
# LOAD MODELS AT STARTUP
# ============================================================================

def load_models():
    global cf_model, cb_model, watch_counts, users
    
    print("Loading data...")
    watch_counts = pd.read_csv('../datasets/data/play_counts.csv')
    users = pd.read_csv('../datasets/data/users.csv')
    
    # Rename columns for movie dataset
    watch_counts.columns = ['user_id', 'movie_id', 'watch_count']
    
    # Sample for faster loading
    watch_counts = watch_counts.sample(n=min(1000000, len(watch_counts)), random_state=42)
    
    print("Initializing models...")
    cf_model = CollaborativeFiltering(watch_counts)
    cb_model = ContentBasedFiltering(users, watch_counts)
    
    print("Models loaded successfully!")

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint to get recommendations"""
    try:
        data = request.json
        user_id = int(data.get('user_id'))
        model_type = data.get('model_type', 'hybrid')
        n_recommendations = int(data.get('n_recommendations', 5))
        
        if model_type == 'user_cf':
            recs = cf_model.user_based_cf(user_id, n_recommendations)
            recs = recs.rename(columns={'score': 'recommendation_score'})
        elif model_type == 'item_cf':
            recs = cf_model.item_based_cf(user_id, n_recommendations)
            recs = recs.rename(columns={'score': 'recommendation_score'})
        elif model_type == 'content':
            recs = cb_model.recommend_based_on_profile(user_id, n_recommendations)
            recs = recs.rename(columns={'score': 'recommendation_score'})
        else:  # hybrid
            hybrid = HybridRecommender(watch_counts, users)
            recs = hybrid.recommend(user_id, n_recommendations)
            recs = recs.rename(columns={'hybrid_score': 'recommendation_score'})
        
        if len(recs) == 0:
            return jsonify({
                'success': False,
                'message': f'No recommendations found for User {user_id}'
            })
        
        recs['rank'] = range(1, len(recs) + 1)
        recommendations = recs[['rank', 'movie_id', 'recommendation_score']].to_dict('records')
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'model_type': model_type,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """API endpoint to get dataset statistics"""
    try:
        stats = {
            'total_users': len(users),
            'total_records': len(watch_counts),
            'unique_movies': watch_counts['movie_id'].nunique(),
        }
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'Movie Recommendation System'})

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Starting Movie Recommendation System Interface...")
    load_models()
    app.run(debug=True, host='127.0.0.1', port=5000)
