"""
QUICK REFERENCE GUIDE FOR RECOMMENDATION MODELS
================================================

This guide shows how to use the three implemented models.
"""

# ============================================================================
# SETUP
# ============================================================================

from recommendation_models import (
    load_data,
    CollaborativeFiltering,
    ContentBasedFiltering,
    HybridRecommender
)

# Load data once
play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()

# ============================================================================
# 1. COLLABORATIVE FILTERING
# ============================================================================

print("\n" + "="*80)
print("1. COLLABORATIVE FILTERING")
print("="*80 + "\n")

# Initialize
cf = CollaborativeFiltering(play_counts)

# User-based CF: Find users similar to the target user
user_id = play_counts['user_id'].unique()[0]
user_based_recommendations = cf.user_based_cf(
    user_id=user_id,
    n_recommendations=5,      # Number of recommendations to return
    n_similar_users=10        # Number of similar users to consider
)
print(f"User-based CF recommendations for user {user_id}:")
print(user_based_recommendations)

# Item-based CF: Find artists similar to ones the user likes
item_based_recommendations = cf.item_based_cf(
    user_id=user_id,
    n_recommendations=5       # Number of recommendations to return
)
print(f"\nItem-based CF recommendations for user {user_id}:")
print(item_based_recommendations)

# ============================================================================
# 2. CONTENT-BASED FILTERING
# ============================================================================

print("\n" + "="*80)
print("2. CONTENT-BASED FILTERING")
print("="*80 + "\n")

# Initialize
cb = ContentBasedFiltering(users, play_counts, user_id_map, artist_id_map)

# Find similar users based on demographics
similar_users = cb.find_similar_users(
    user_id=user_id,
    n_similar=5               # Number of similar users
)
print(f"Users similar to {user_id}: {similar_users}")

# Get recommendations based on profile
profile_recommendations = cb.recommend_based_on_profile(
    user_id=user_id,
    n_recommendations=5       # Number of recommendations
)
print(f"\nContent-based recommendations for user {user_id}:")
print(profile_recommendations)

# ============================================================================
# 3. HYBRID MODEL
# ============================================================================

print("\n" + "="*80)
print("3. HYBRID MODEL")
print("="*80 + "\n")

# Initialize hybrid model
hybrid = HybridRecommender(play_counts, users, user_id_map, artist_id_map)

# Default balanced weights
hybrid_recommendations = hybrid.recommend(
    user_id=user_id,
    n_recommendations=5       # Number of recommendations
)
print(f"Hybrid recommendations for user {user_id}:")
print(hybrid_recommendations)

# Custom weights configuration
custom_weights = {
    'user_cf': 0.2,          # Weight for user-based CF (0-1)
    'item_cf': 0.2,          # Weight for item-based CF (0-1)
    'content': 0.6           # Weight for content-based (0-1)
}
weighted_recommendations = hybrid.recommend(
    user_id=user_id,
    n_recommendations=5,
    weights=custom_weights    # Must sum to 1.0
)
print(f"\nHybrid with custom weights (Content-heavy):")
print(weighted_recommendations)

# Get detailed breakdown with contributions from each model
print("\n\nDetailed breakdown:")
hybrid_breakdown = hybrid.recommend_with_breakdown(
    user_id=user_id,
    n_recommendations=5
)

# ============================================================================
# WEIGHT CONFIGURATIONS
# ============================================================================

print("\n" + "="*80)
print("RECOMMENDED WEIGHT CONFIGURATIONS")
print("="*80 + "\n")

configs = {
    'Balanced': {'user_cf': 0.33, 'item_cf': 0.33, 'content': 0.34},
    'CF-Heavy': {'user_cf': 0.4, 'item_cf': 0.4, 'content': 0.2},
    'Content-Heavy': {'user_cf': 0.2, 'item_cf': 0.2, 'content': 0.6},
    'User-CF Focus': {'user_cf': 0.5, 'item_cf': 0.3, 'content': 0.2},
    'Item-CF Focus': {'user_cf': 0.3, 'item_cf': 0.5, 'content': 0.2},
}

print("Use cases for each configuration:")
print("• Balanced: General purpose, good for most scenarios")
print("• CF-Heavy: When collaborative data is strong and reliable")
print("• Content-Heavy: When user demographics matter, cold-start users")
print("• User-CF Focus: When user similarity is most important")
print("• Item-CF Focus: When item similarity is most important")

# ============================================================================
# BATCH PROCESSING
# ============================================================================

print("\n" + "="*80)
print("BATCH PROCESSING - GET RECOMMENDATIONS FOR MULTIPLE USERS")
print("="*80 + "\n")

import pandas as pd

def get_recommendations_for_all_users(model_type='hybrid', n_recs=5):
    """Get recommendations for all users"""
    
    all_recommendations = []
    user_ids = play_counts['user_id'].unique()
    
    print(f"Generating {model_type} recommendations for {len(user_ids)} users...\n")
    
    for i, user_id in enumerate(user_ids[:50]):  # Limit to first 50 for demo
        if model_type == 'user_cf':
            recs = cf.user_based_cf(user_id, n_recommendations=n_recs)
        elif model_type == 'item_cf':
            recs = cf.item_based_cf(user_id, n_recommendations=n_recs)
        elif model_type == 'content':
            recs = cb.recommend_based_on_profile(user_id, n_recommendations=n_recs)
        else:  # hybrid
            recs = hybrid.recommend(user_id, n_recommendations=n_recs)
        
        if len(recs) > 0:
            recs['user_id'] = user_id
            recs['model'] = model_type
            all_recommendations.append(recs)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1} users...")
    
    results_df = pd.concat(all_recommendations, ignore_index=True)
    return results_df

# Generate and save recommendations
hybrid_all = get_recommendations_for_all_users('hybrid', n_recs=5)
hybrid_all.to_csv('../datasets/recommendations/hybrid_recommendations_batch.csv', index=False)
print(f"\n✓ Saved batch recommendations: hybrid_recommendations_batch.csv")
print(f"  Shape: {hybrid_all.shape}")
print(f"  Columns: {list(hybrid_all.columns)}")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON - WHEN TO USE EACH")
print("="*80 + "\n")

comparison = """
┌─────────────────────┬────────────────────┬──────────────────┬─────────────┐
│ Model               │ Best For           │ Pros             │ Cons        │
├─────────────────────┼────────────────────┼──────────────────┼─────────────┤
│ User-Based CF       │ Warm-start users   │ • Intuitive      │ • Cold-start│
│                     │                    │ • Good quality   │ • Scalable  │
│                     │                    │                  │   issues    │
├─────────────────────┼────────────────────┼──────────────────┼─────────────┤
│ Item-Based CF       │ Growing catalogs   │ • Stable         │ • Limited   │
│                     │                    │ • Fast           │   for new   │
│                     │                    │                  │   items     │
├─────────────────────┼────────────────────┼──────────────────┼─────────────┤
│ Content-Based       │ Cold-start users   │ • No cold-start  │ • Limited   │
│                     │ New users          │ • Explainable    │   diversity │
│                     │                    │                  │ • Requires  │
│                     │                    │                  │   features  │
├─────────────────────┼────────────────────┼──────────────────┼─────────────┤
│ Hybrid (All 3)      │ Production systems │ • Best coverage  │ • Complex   │
│                     │ General purpose    │ • Handles cold & │ • Slower    │
│                     │                    │   warm starts    │             │
└─────────────────────┴────────────────────┴──────────────────┴─────────────┘
"""
print(comparison)

print("\n✓ Guide completed. See recommendation_models.py for full implementation.")
