# Recommendation Models Implementation

This project implements **3 state-of-the-art recommendation system models** for your movie dataset.

## 📋 Overview

### Three Models Implemented:

#### 1. **Collaborative Filtering**
- **User-Based CF**: Finds similar users and recommends their favorite movies
- **Item-Based CF**: Finds movies similar to ones the user already watched
- Uses cosine similarity on user-movie interaction matrix

#### 2. **Content-Based Filtering**
- Analyzes user demographics and viewing patterns
- Creates user profiles based on age, total watches, number of movies
- Recommends movies based on similar user profiles

#### 3. **Hybrid Model**
- Combines all three approaches with configurable weights
- Provides balanced recommendations by leveraging strengths of each model
- Handles both cold-start and warm-start scenarios

---

## 📁 Project Structure

```
ml project/
├── recommendation_models.py    # Main implementation
├── test_models.py             # Testing and evaluation
├── quick_reference.py         # Usage examples
├── README.md                  # This file
└── requirements.txt           # Dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

### 2. Run Tests

```bash
python test_models.py
```

This will:
- ✓ Test all three models on sample users
- ✓ Generate recommendations with different configurations
- ✓ Save results to CSV files
- ✓ Compare models performance

### 3. Use the Models

```python
from recommendation_models import HybridRecommender, load_data

# Load data
play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()

# Create hybrid recommender
hybrid = HybridRecommender(play_counts, users, user_id_map, artist_id_map)

# Get recommendations for a user
recommendations = hybrid.recommend(user_id=123, n_recommendations=5)
print(recommendations)
```

---

## 📊 Model Details

### Collaborative Filtering

**How it works:**
- Creates a user-artist interaction matrix (plays count)
- Calculates user similarity using cosine similarity
- For target user, finds similar users and their favorite artists
- Ranks recommendations by aggregated scores from similar users

**Best for:**
- Warm-start users (existing users with listening history)
- Dense interactions
- Finding diverse recommendations

**Parameters:**
```python
cf.user_based_cf(
    user_id=user_id,
    n_recommendations=5,    # Number of results
    n_similar_users=10      # Similar users to consider
)
```

**Item-Based CF:**
```python
cf.item_based_cf(
    user_id=user_id,
    n_recommendations=5
)
```

---

### Content-Based Filtering

**How it works:**
- Extracts user features (age, total plays, num artists)
- Normalizes features using StandardScaler
- Calculates user similarity based on demographics
- Finds similar users and recommends their artists

**Best for:**
- Cold-start users (new users)
- When user demographics matter
- Explainable recommendations

**Parameters:**
```python
cb.recommend_based_on_profile(
    user_id=user_id,
    n_recommendations=5
)
```

---

### Hybrid Model

**How it works:**
- Generates recommendations from all three approaches
- Combines scores using configurable weights (must sum to 1.0)
- Returns ranked list of recommendations

**Weight Configurations:**

| Config | User-CF | Item-CF | Content | Use Case |
|--------|---------|---------|---------|----------|
| Balanced | 0.33 | 0.33 | 0.34 | General purpose |
| CF-Heavy | 0.4 | 0.4 | 0.2 | Strong collaborative data |
| Content-Heavy | 0.2 | 0.2 | 0.6 | Cold-start, demographics matter |
| User-CF Focus | 0.5 | 0.3 | 0.2 | Similar users important |
| Item-CF Focus | 0.3 | 0.5 | 0.2 | Similar items important |

**Usage:**

```python
hybrid = HybridRecommender(play_counts, users, user_id_map, artist_id_map)

# Default balanced weights
recs = hybrid.recommend(user_id=123, n_recommendations=5)

# Custom weights (content-heavy)
custom_weights = {
    'user_cf': 0.2,
    'item_cf': 0.2,
    'content': 0.6
}
recs = hybrid.recommend(
    user_id=123,
    n_recommendations=5,
    weights=custom_weights
)

# Get detailed breakdown
recs_breakdown = hybrid.recommend_with_breakdown(user_id=123)
```

---

## 📈 Output Files

Running `test_models.py` generates:

1. **user_cf_recommendations.csv**
   - Columns: artist_id, score, user_id, model
   - User-based collaborative filtering results

2. **item_cf_recommendations.csv**
   - Columns: artist_id, score, user_id, model
   - Item-based collaborative filtering results

3. **content_based_recommendations.csv**
   - Columns: artist_id, score, user_id, model
   - Content-based filtering results

4. **hybrid_recommendations_full.csv**
   - Columns: artist_id, hybrid_score, user_id, config
   - Hybrid model results with different weight configurations

---

## 🎯 API Reference

### Class: CollaborativeFiltering

```python
class CollaborativeFiltering:
    def __init__(self, play_counts):
        # Initialize with play counts dataframe
        
    def user_based_cf(self, user_id, n_recommendations=5, n_similar_users=10):
        # Returns DataFrame with artist_id and score columns
        
    def item_based_cf(self, user_id, n_recommendations=5):
        # Returns DataFrame with artist_id and score columns
```

### Class: ContentBasedFiltering

```python
class ContentBasedFiltering:
    def __init__(self, users, play_counts, user_id_map, artist_id_map):
        # Initialize with user and play count data
        
    def find_similar_users(self, user_id, n_similar=5):
        # Returns list of similar user IDs
        
    def recommend_based_on_profile(self, user_id, n_recommendations=5):
        # Returns DataFrame with artist_id and score columns
```

### Class: HybridRecommender

```python
class HybridRecommender:
    def __init__(self, play_counts, users, user_id_map, artist_id_map):
        # Initialize all three models
        
    def recommend(self, user_id, n_recommendations=5, weights=None):
        # Returns DataFrame with artist_id and hybrid_score columns
        # weights: dict with keys 'user_cf', 'item_cf', 'content'
        
    def recommend_with_breakdown(self, user_id, n_recommendations=5):
        # Prints detailed breakdown and returns recommendations
```

---

## 💡 Examples

### Example 1: Get Top 10 Recommendations for a User

```python
from recommendation_models import load_data, HybridRecommender

play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()
hybrid = HybridRecommender(play_counts, users, user_id_map, artist_id_map)

recommendations = hybrid.recommend(
    user_id=first_user_id,
    n_recommendations=10
)

print(recommendations)
```

### Example 2: Compare All Models

```python
from recommendation_models import load_data, CollaborativeFiltering, ContentBasedFiltering

play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()

cf = CollaborativeFiltering(play_counts)
cb = ContentBasedFiltering(users, play_counts, user_id_map, artist_id_map)

user_id = 123

print("User-Based CF:")
print(cf.user_based_cf(user_id))

print("\nItem-Based CF:")
print(cf.item_based_cf(user_id))

print("\nContent-Based:")
print(cb.recommend_based_on_profile(user_id))
```

### Example 3: Batch Processing for All Users

```python
from recommendation_models import load_data, HybridRecommender
import pandas as pd

play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()
hybrid = HybridRecommender(play_counts, users, user_id_map, artist_id_map)

all_recommendations = []

for user_id in play_counts['user_id'].unique():
    recs = hybrid.recommend(user_id, n_recommendations=5)
    if len(recs) > 0:
        recs['user_id'] = user_id
        all_recommendations.append(recs)

results = pd.concat(all_recommendations, ignore_index=True)
results.to_csv('all_recommendations.csv', index=False)
```

### Example 4: Find Similar Users for Cold-Start

```python
from recommendation_models import load_data, ContentBasedFiltering

play_counts, users, user_id_map, artist_id_map, users_sim, artists_sim = load_data()
cb = ContentBasedFiltering(users, play_counts, user_id_map, artist_id_map)

# Find users similar to a new/cold-start user
similar_users = cb.find_similar_users(user_id=999, n_similar=10)
print(f"Similar users: {similar_users}")
```

---

## 🔧 Customization

### Adjust Model Parameters

```python
# User-based CF with more similar users
recs = cf.user_based_cf(
    user_id=123,
    n_recommendations=10,
    n_similar_users=20  # Consider more users
)

# Hybrid with custom weights
custom_weights = {
    'user_cf': 0.25,
    'item_cf': 0.25,
    'content': 0.50
}
recs = hybrid.recommend(
    user_id=123,
    n_recommendations=10,
    weights=custom_weights
)
```

### Add New Features to Content-Based

Edit `ContentBasedFiltering._create_user_profiles()` to include:
- User listening frequency
- Genre preferences
- Time-of-day listening patterns
- Device preferences

---

## 📊 Evaluation

To evaluate recommendation quality:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Compare recommendations with actual user preferences
actual_artists = set(play_counts[play_counts['user_id'] == user_id]['artist_id'])
recommended_artists = set(recommendations['artist_id'].values)

# Precision and Recall
precision = len(actual_artists & recommended_artists) / len(recommended_artists)
recall = len(actual_artists & recommended_artists) / len(actual_artists)
f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

---

## ⚠️ Notes & Limitations

1. **Cold-Start Problem**: New users with no history will get poor recommendations from CF models. Use content-based filtering for these.

2. **Scalability**: For large datasets (millions of users/items), consider:
   - Approximation techniques (LSH, clustering)
   - Distributed computing (Spark)
   - Approximate similarity calculations

3. **Sparsity**: If interaction matrix is very sparse, CF models may struggle. Content-based filtering recommended.

4. **Data Quality**: Ensure input data is clean:
   - No missing user_id or artist_id values
   - Plays count > 0
   - Consistent ID formats

---

## 📞 Troubleshooting

### Issue: "No recommendations available"
- **Cause**: User has no listening history or dataset too sparse
- **Solution**: Use content-based filtering; adjust n_similar_users

### Issue: Slow performance
- **Cause**: Large dataset or many similar users to compute
- **Solution**: Reduce n_recommendations or n_similar_users; use sampling

### Issue: Same recommendations for all users
- **Cause**: All weights on collaborative filtering, sparse data
- **Solution**: Increase content-based weight; check data quality

---

## 📝 Dataset Requirements

Your data should have:
- `play_counts.csv`: user_id, artist_id, plays
- `users.csv`: user_id, age, country, gender
- Mapping files for ID conversions (optional)

---

## 🚀 Next Steps

1. ✅ Run `test_models.py` to test all models
2. ✅ Review output CSV files in `datasets/recommendations/`
3. ✅ Adjust weights based on your requirements
4. ✅ Integrate into production system
5. ✅ Monitor and A/B test different configurations

---

**Created**: April 2026  
**Status**: Production Ready
