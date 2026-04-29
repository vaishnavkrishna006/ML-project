import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast
import os

# --- Configuration & Styling ---
st.set_page_config(page_title="Movie Analytics Dashboard", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0f0f0f; color: #ffffff; }
    .kpi-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e50914;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-title { color: #b3b3b3; font-size: 1.1rem; margin-bottom: 5px; font-weight: 500; }
    .kpi-value { color: #ffffff; font-size: 2.2rem; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

# TMDB Genre Mapping
TMDB_GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

def safe_eval(val):
    """Safely evaluates string representations of lists/arrays without crashing."""
    try:
        if isinstance(val, str):
            val = val.strip()
            if val.startswith('[') and val.endswith(']'):
                return ast.literal_eval(val)
    except:
        pass
    return val

def process_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly processes genre columns ('genre', 'genres', 'genre_ids')
    and creates a unified 'genre_clean' column as a list of strings.
    """
    df_clean = df.copy()
    
    def extract_genres(row):
        # 1. Check 'genre'
        if 'genre' in df_clean.columns and pd.notnull(row['genre']):
            val = row['genre']
            if isinstance(val, list): return val
            if isinstance(val, str): return [g.strip() for g in val.replace('|', ',').split(',')]
            
        # 2. Check 'genres'
        if 'genres' in df_clean.columns and pd.notnull(row['genres']):
            val = row['genres']
            if isinstance(val, list): return val
            if isinstance(val, str): return [g.strip() for g in val.replace('|', ',').split(',')]
            
        # 3. Check 'genre_ids' (TMDB format)
        if 'genre_ids' in df_clean.columns and pd.notnull(row['genre_ids']):
            val = safe_eval(row['genre_ids'])
            if isinstance(val, list):
                return [TMDB_GENRES.get(int(gid), "Unknown") for gid in val if pd.notnull(gid)]
            if isinstance(val, str):
                ids = [int(i.strip()) for i in val.split(',') if i.strip().isdigit()]
                return [TMDB_GENRES.get(i, "Unknown") for i in ids]
                
        return ["Unknown"]

    df_clean['genre_clean'] = df_clean.apply(extract_genres, axis=1)
    return df_clean

def get_unique_genres(df: pd.DataFrame) -> list:
    """Safely extracts all unique genres from the unified genre_clean column."""
    if 'genre_clean' not in df.columns:
        return []
    all_genres = set()
    for genres in df['genre_clean']:
        if isinstance(genres, list):
            for g in genres:
                if g and g != "Unknown":
                    all_genres.add(g)
    return sorted(list(all_genres))

def filter_movies(df: pd.DataFrame, genre: str, min_rating: float, selected_years: list) -> pd.DataFrame:
    """Filters the dataset safely based on multiple interactive criteria."""
    filtered = df.copy()
    
    if genre != "All" and 'genre_clean' in filtered.columns:
        filtered = filtered[filtered['genre_clean'].apply(lambda x: isinstance(x, list) and genre in x)]
        
    if 'rating' in filtered.columns:
        filtered = filtered[filtered['rating'] >= min_rating]
        
    if 'year' in filtered.columns and selected_years:
        filtered = filtered[filtered['year'].isin(selected_years)]
        
    return filtered

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads dataset and runs it through the robust processing pipeline."""
    # Attempt to load from data_loader.py, fallback to raw dataframe if it breaks
    try:
        from data_loader import load_dataset
        df = load_dataset()
    except Exception as e:
        st.warning(f"Could not load data from data_loader.py: {e}")
        # Create ultra-basic fallback if everything is broken
        df = pd.DataFrame({"movie_id": [1, 2, 3], "rating": [4, 5, 3]})

    # Ensure critical columns exist so the UI doesn't crash
    if 'rating' not in df.columns:
        np.random.seed(42)
        df['rating'] = np.random.uniform(1.0, 5.0, len(df))
    if 'year' not in df.columns:
        np.random.seed(42)
        df['year'] = np.random.randint(1990, 2024, len(df))
    if 'title' not in df.columns:
        df['title'] = [f"Movie {i}" for i in df.get('movie_id', range(len(df)))]
        
    return process_genres(df)

def main():
    st.title("🎬 ML Analytics Dashboard")
    st.markdown("Interactive Power BI-style dashboard for robust movie data analysis.")
    
    # Load and process data
    with st.spinner("Loading and processing dataset safely..."):
        df = load_data()
        
    if df.empty:
        st.error("No data available.")
        return
        
    # --- Sidebar Filters ---
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
    st.sidebar.title("Filters")
    
    # 1. Genre Filter (CRASH PROOF)
    available_genres = get_unique_genres(df)
    selected_genre = st.sidebar.selectbox("Filter by Genre", ["All"] + available_genres)
    
    # 2. Rating Filter
    min_rating = 0.0
    if 'rating' in df.columns:
        min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.5)
        
    # 3. Year Filter
    selected_years = []
    if 'year' in df.columns:
        years = sorted(df['year'].dropna().unique().astype(int))
        if years:
            selected_years = st.sidebar.multiselect("Select Year(s)", years, default=[])
            
    # Apply Filters safely
    filtered_df = filter_movies(df, selected_genre, min_rating, selected_years)
    
    if filtered_df.empty:
        st.warning("No movies match the selected filters. Try broadening your criteria.")
        return

    # --- KPI Cards ---
    st.markdown("### 📊 Platform Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Movies</div>
                <div class="kpi-value">{len(filtered_df):,}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        avg_rating = filtered_df['rating'].mean() if 'rating' in filtered_df.columns else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg Rating</div>
                <div class="kpi-value">{avg_rating:.1f} ⭐</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        all_g = [g for sublist in filtered_df['genre_clean'] for g in sublist if isinstance(g, str) and g != "Unknown"]
        most_common = pd.Series(all_g).mode()[0] if all_g else "N/A"
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Most Common Genre</div>
                <div class="kpi-value">{most_common}</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # --- Charts ---
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Genre Distribution
        if all_g:
            genre_counts = pd.Series(all_g).value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']
            fig_bar = px.bar(
                genre_counts.head(10), x='Genre', y='Count',
                title="Top 10 Genres Distribution",
                color_discrete_sequence=['#e50914']
            )
            fig_bar.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No genre data available for charts.")
            
    with chart_col2:
        # Rating Histogram
        if 'rating' in filtered_df.columns:
            fig_hist = px.histogram(
                filtered_df, x='rating', nbins=20,
                title="Rating Distribution",
                labels={'rating': 'Rating Score', 'count': 'Number of Movies'},
                color_discrete_sequence=['#e50914']
            )
            fig_hist.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_hist, use_container_width=True)
            
    st.markdown("---")
    
    # --- Top Movies Table ---
    st.markdown("### 🏆 Top Movies")
    if 'rating' in filtered_df.columns:
        top_movies = filtered_df.sort_values(by='rating', ascending=False).head(50)
    else:
        top_movies = filtered_df.head(50)
        
    # Formatting for display
    display_df = top_movies.copy()
    display_df['genre_clean'] = display_df['genre_clean'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    
    # Select safe columns to show
    cols_to_show = []
    if 'title' in display_df.columns: cols_to_show.append('title')
    if 'genre_clean' in display_df.columns: cols_to_show.append('genre_clean')
    if 'rating' in display_df.columns: cols_to_show.append('rating')
    if 'year' in display_df.columns: cols_to_show.append('year')
    if 'movie_id' in display_df.columns: cols_to_show.append('movie_id')
    
    if cols_to_show:
        st.dataframe(
            display_df[cols_to_show],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write(display_df)

if __name__ == "__main__":
    main()
