"""
Test script to run the recommendation models and save results
"""

import pandas as pd
import sys
from recommendation_models import (
    load_data,
    CollaborativeFiltering,
    ContentBasedFiltering,
    HybridRecommender
)

def test_collaborative_filtering():
    """Test Collaborative Filtering models"""
    print("\n" + "="*80)
    print("TESTING COLLABORATIVE FILTERING MODELS")
    print("="*80 + "\n")
    
    play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()
    
    cf_model = CollaborativeFiltering(play_counts)
    
    # Test on first 10 users
    test_users = play_counts['user_id'].unique()[:10]
    
    all_user_cf_recs = []
    all_item_cf_recs = []
    
    for user_id in test_users:
        print(f"\nUser ID: {user_id}")
        
        # User-based CF
        user_recs = cf_model.user_based_cf(user_id, n_recommendations=5)
        if len(user_recs) > 0:
            print("  User-based CF Top 3:")
            print("  " + str(user_recs.head(3).to_string()).replace('\n', '\n  '))
            user_recs['user_id'] = user_id
            user_recs['model'] = 'user_cf'
            all_user_cf_recs.append(user_recs)
        
        # Item-based CF
        item_recs = cf_model.item_based_cf(user_id, n_recommendations=5)
        if len(item_recs) > 0:
            print("  Item-based CF Top 3:")
            print("  " + str(item_recs.head(3).to_string()).replace('\n', '\n  '))
            item_recs['user_id'] = user_id
            item_recs['model'] = 'item_cf'
            all_item_cf_recs.append(item_recs)
    
    # Save results
    if all_user_cf_recs:
        user_cf_results = pd.concat(all_user_cf_recs, ignore_index=True)
        user_cf_results.to_csv('../datasets/recommendations/user_cf_recommendations.csv', index=False)
        print(f"\n✓ User-based CF results saved to: user_cf_recommendations.csv")
    
    if all_item_cf_recs:
        item_cf_results = pd.concat(all_item_cf_recs, ignore_index=True)
        item_cf_results.to_csv('../datasets/recommendations/item_cf_recommendations.csv', index=False)
        print(f"✓ Item-based CF results saved to: item_cf_recommendations.csv")

def test_content_based_filtering():
    """Test Content-Based Filtering model"""
    print("\n" + "="*80)
    print("TESTING CONTENT-BASED FILTERING MODEL")
    print("="*80 + "\n")
    
    play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()
    
    cb_model = ContentBasedFiltering(users, play_counts, user_id_map, artist_id_map)
    
    # Test on first 10 users
    test_users = play_counts['user_id'].unique()[:10]
    
    all_content_recs = []
    
    for user_id in test_users:
        print(f"\nUser ID: {user_id}")
        
        recs = cb_model.recommend_based_on_profile(user_id, n_recommendations=5)
        if len(recs) > 0:
            print("  Content-based Recommendations Top 3:")
            print("  " + str(recs.head(3).to_string()).replace('\n', '\n  '))
            recs['user_id'] = user_id
            recs['model'] = 'content_based'
            all_content_recs.append(recs)
        else:
            print("  No recommendations available")
    
    # Save results
    if all_content_recs:
        content_results = pd.concat(all_content_recs, ignore_index=True)
        content_results.to_csv('../datasets/recommendations/content_based_recommendations.csv', index=False)
        print(f"\n✓ Content-based results saved to: content_based_recommendations.csv")

def test_hybrid_model():
    """Test Hybrid Recommendation Model"""
    print("\n" + "="*80)
    print("TESTING HYBRID RECOMMENDATION MODEL")
    print("="*80 + "\n")
    
    play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()
    
    hybrid = HybridRecommender(play_counts, users, user_id_map, artist_id_map)
    
    # Test on first 10 users
    test_users = play_counts['user_id'].unique()[:10]
    
    all_hybrid_recs = []
    
    # Test with different weights
    weights_configs = [
        {'name': 'Balanced', 'weights': {'user_cf': 0.33, 'item_cf': 0.33, 'content': 0.34}},
        {'name': 'CF Heavy', 'weights': {'user_cf': 0.4, 'item_cf': 0.4, 'content': 0.2}},
        {'name': 'Content Heavy', 'weights': {'user_cf': 0.2, 'item_cf': 0.2, 'content': 0.6}},
    ]
    
    for config in weights_configs:
        print(f"\n--- Configuration: {config['name']} ---")
        print(f"Weights: {config['weights']}\n")
        
        config_recs = []
        for user_id in test_users[:5]:  # Test on first 5 users
            recs = hybrid.recommend(user_id, n_recommendations=5, weights=config['weights'])
            if len(recs) > 0:
                print(f"  User {user_id}: Top recommendation is Artist {recs.iloc[0]['artist_id']} (score: {recs.iloc[0]['hybrid_score']:.4f})")
                recs['user_id'] = user_id
                recs['config'] = config['name']
                config_recs.append(recs)
        
        if config_recs:
            all_hybrid_recs.extend(config_recs)
    
    # Save results
    if all_hybrid_recs:
        hybrid_results = pd.concat(all_hybrid_recs, ignore_index=True)
        hybrid_results.to_csv('../datasets/recommendations/hybrid_recommendations_full.csv', index=False)
        print(f"\n✓ Hybrid results saved to: hybrid_recommendations_full.csv")

def compare_models():
    """Compare all three models on sample users"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80 + "\n")
    
    play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()
    
    cf_model = CollaborativeFiltering(play_counts)
    cb_model = ContentBasedFiltering(users, play_counts, user_id_map, artist_id_map)
    hybrid = HybridRecommender(play_counts, users, user_id_map, artist_id_map)
    
    test_user = play_counts['user_id'].unique()[0]
    
    print(f"Comparing recommendations for User: {test_user}\n")
    
    print("1. USER-BASED COLLABORATIVE FILTERING:")
    print("-" * 50)
    ub_cf = cf_model.user_based_cf(test_user, n_recommendations=5)
    print(ub_cf)
    
    print("\n2. ITEM-BASED COLLABORATIVE FILTERING:")
    print("-" * 50)
    ib_cf = cf_model.item_based_cf(test_user, n_recommendations=5)
    print(ib_cf)
    
    print("\n3. CONTENT-BASED FILTERING:")
    print("-" * 50)
    cb = cb_model.recommend_based_on_profile(test_user, n_recommendations=5)
    print(cb)
    
    print("\n4. HYBRID MODEL:")
    print("-" * 50)
    hybrid_recs = hybrid.recommend(test_user, n_recommendations=5)
    print(hybrid_recs)
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("• User-based CF: Recommends based on similar users' preferences")
    print("• Item-based CF: Recommends similar artists to ones user already likes")
    print("• Content-based: Recommends based on user demographic similarity")
    print("• Hybrid: Combines all three for more robust recommendations")

def main():
    """Run all tests"""
    try:
        # Run individual model tests
        test_collaborative_filtering()
        test_content_based_filtering()
        test_hybrid_model()
        
        # Compare models
        compare_models()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  • user_cf_recommendations.csv")
        print("  • item_cf_recommendations.csv")
        print("  • content_based_recommendations.csv")
        print("  • hybrid_recommendations_full.csv")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
