import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.recommender import MovieRecommender

# --- Configuration & Styling ---
st.set_page_config(page_title="CineMatch - AI Movie Recommendations", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0f0f0f; color: #ffffff; }
    .movie-card {
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 8px;
        border-top: 3px solid #e50914;
        margin-bottom: 10px;
    }
    .similarity-badge {
        background-color: #e50914;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    # Use absolute path relative to this script's directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "src", "models", "artifacts", "recommender.pkl")
    
    if not os.path.exists(model_path):
        # Fallback to local path just in case
        model_path = "src/models/artifacts/recommender.pkl"
        if not os.path.exists(model_path):
            return None
    return MovieRecommender.load(model_path)

def main():
    st.sidebar.markdown("<h1 style='color: #e50914; font-family: sans-serif;'>CineMatch</h1>", unsafe_allow_html=True)
    st.sidebar.title("🎬 Recommender")
    
    recommender = load_recommender()
    
    if recommender is None:
        st.error("Model not found! Please run `python train_model.py` first.")
        return

    st.title("🎬 CineMatch: AI Movie Recommender")
    st.markdown("Search for movies by title or description using TF-IDF Similarity")

    tab1, tab2, tab3 = st.tabs(["🔍 Search by Title", "💬 Natural Language Search", "📊 Dataset Insights"])

    with tab1:
        movie_list = recommender.movies_df['title'].values
        selected_movie = st.selectbox("Type or select a movie you like:", movie_list)
        
        if st.button("Recommend Similar Movies"):
            with st.spinner("Finding matches..."):
                recs = recommender.recommend(selected_movie)
                if recs is not None:
                    st.success(f"Movies similar to **{selected_movie}**:")
                    for _, row in recs.iterrows():
                        st.markdown(f"""
                            <div class="movie-card">
                                <strong>{row['title']}</strong> <span class="similarity-badge">{row['similarity_score']:.2f} score</span>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Movie not found.")

    with tab2:
        st.markdown("### Describe what you want to watch:")
        user_query = st.text_area("Example: 'I want an action thriller movie with sci-fi elements'", "")
        
        if st.button("Search Description"):
            if user_query:
                with st.spinner("Analyzing your request..."):
                    results = recommender.search_by_description(user_query)
                    st.success("Top matches for your description:")
                    for _, row in results.iterrows():
                        st.markdown(f"""
                            <div class="movie-card">
                                <strong>{row['title']}</strong> <span class="similarity-badge">{row['similarity_score']:.2f} score</span>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a description.")

    with tab3:
        st.markdown("### 📊 Dataset Overview")
        df = recommender.movies_df
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Movies", len(df))
            
        # Genre Distribution (Approximate from 'features' column)
        all_features = " ".join(df['features'].tolist()).split()
        feature_counts = pd.Series(all_features).value_counts().head(15)
        
        fig = px.bar(
            x=feature_counts.index, 
            y=feature_counts.values,
            labels={'x': 'Genre/Tag', 'y': 'Count'},
            title="Top Movie Tags/Genres",
            color_discrete_sequence=['#e50914']
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
