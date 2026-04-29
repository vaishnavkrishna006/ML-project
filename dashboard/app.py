import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_dataset, get_model_embeddings, get_recommendation_scores
from metrics import calculate_kpi_metrics, get_confusion_matrix_data, get_pr_curve_data, compute_pca_embeddings

# --- Configuration & Styling ---
st.set_page_config(page_title="ML Recommendations Dashboard", page_icon="🎬", layout="wide")

# Custom CSS for Power BI / Dark Theme aesthetics
st.markdown("""
    <style>
    .stApp {
        background-color: #0f0f0f;
        color: #ffffff;
    }
    .kpi-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e50914;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-title {
        color: #b3b3b3;
        font-size: 1.1rem;
        margin-bottom: 5px;
        font-weight: 500;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_all_data():
    df = load_dataset()
    embeddings = get_model_embeddings(num_items=df['movie_id'].nunique(), embedding_dim=32)
    scores = get_recommendation_scores(num_items=df['movie_id'].nunique())
    return df, embeddings, scores

df, embeddings, scores = load_all_data()

# --- Sidebar Navigation ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# Interactive Filters
selected_genre = st.sidebar.selectbox("Filter by Genre", ["All"] + list(df['genre'].unique()))

if selected_genre != "All":
    filtered_df = df[df['genre'] == selected_genre]
else:
    filtered_df = df

st.sidebar.markdown("---")
st.sidebar.info("Model: **LightFM Hybrid**\n\nLoss: **WARP**")

# --- Main Dashboard ---
st.title("🎬 MovieAI Performance Analytics")
st.markdown("Analyze model performance, user behavior, and recommendation distribution.")

# 1. Overview Metrics (KPI Cards)
st.markdown("### 📊 Platform Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Users</div>
            <div class="kpi-value">{filtered_df['user_id'].nunique():,}</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Movies</div>
            <div class="kpi-value">{filtered_df['movie_id'].nunique():,}</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Interactions</div>
            <div class="kpi-value">{len(filtered_df):,}</div>
        </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Rating</div>
            <div class="kpi-value">{filtered_df['rating'].mean():.2f} ⭐</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Model Performance metrics
st.markdown("### 🎯 Model Performance Evaluation")
kpi_metrics = calculate_kpi_metrics()

perf_col1, perf_col2 = st.columns([1, 2])

with perf_col1:
    # Model metrics KPIs
    st.metric(label="Precision@K", value=f"{kpi_metrics['precision_k']:.3f}", delta="+0.012 vs last run")
    st.metric(label="Recall@K", value=f"{kpi_metrics['recall_k']:.3f}", delta="-0.005 vs last run")
    st.metric(label="F1 Score", value=f"{kpi_metrics['f1_score']:.3f}", delta="+0.008 vs last run")

with perf_col2:
    # Bar chart for KPIs
    fig_metrics = px.bar(
        x=["Precision@10", "Recall@10", "F1 Score"],
        y=[kpi_metrics['precision_k'], kpi_metrics['recall_k'], kpi_metrics['f1_score']],
        labels={'x': 'Metric', 'y': 'Score'},
        title="Offline Evaluation Metrics",
        color_discrete_sequence=['#e50914']
    )
    fig_metrics.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_metrics, use_container_width=True)

st.markdown("---")

# 3 & 4. Confusion Matrix and PR Curve
st.markdown("### 📈 Classification Diagnostics")
diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    # Confusion Matrix
    cm = get_confusion_matrix_data()
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=['Negative (0)', 'Positive (1)'],
        y=['Negative (0)', 'Positive (1)'],
        title="Confusion Matrix (Rating ≥ 4)",
        color_continuous_scale="Reds"
    )
    fig_cm.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_cm, use_container_width=True)

with diag_col2:
    # Precision-Recall Curve
    precision, recall, _ = get_pr_curve_data()
    fig_pr = px.line(
        x=recall, y=precision,
        title="Precision-Recall Curve",
        labels={'x': 'Recall', 'y': 'Precision'}
    )
    fig_pr.update_traces(line_color="#e50914", line_width=3)
    fig_pr.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    # Add optimal point
    fig_pr.add_scatter(x=[0.4], y=[0.6], mode='markers', marker=dict(size=12, color='white'), name='Operating Point')
    st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("---")

# 5. Recommendation Quality & 6. User Behavior
st.markdown("### 👥 Behavior & Recommendations")
beh_col1, beh_col2 = st.columns(2)

with beh_col1:
    # Recommendation Score Distribution
    fig_hist = px.histogram(
        scores, nbins=50,
        title="Recommendation Score Distribution",
        labels={'value': 'Model Score (Probability)', 'count': 'Frequency'},
        color_discrete_sequence=['#e50914']
    )
    fig_hist.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

with beh_col2:
    # Genre Distribution
    genre_counts = filtered_df['genre'].value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    fig_pie = px.pie(
        genre_counts, values='count', names='genre',
        title="Interaction Distribution by Genre",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Reds_r
    )
    fig_pie.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# 7. Embedding Visualization (Advanced)
st.markdown("### 🌌 Item Embedding Visualization (PCA)")
st.info("Visualizing the 32-dimensional LightFM item embeddings reduced to 2D using PCA. Clusters indicate similar movies.")

# Compute PCA
pca_2d = compute_pca_embeddings(embeddings, n_components=2)
# Create a dataframe for visualization mapping movie IDs to coordinates
unique_movies = list(set(filtered_df['movie_id']))[:len(pca_2d)]
movie_genres = filtered_df.drop_duplicates('movie_id').set_index('movie_id')['genre'].to_dict()

emb_df = pd.DataFrame({
    'x': pca_2d[:, 0],
    'y': pca_2d[:, 1],
    'Movie ID': unique_movies,
    'Genre': [movie_genres.get(m, 'Unknown') for m in unique_movies]
})

fig_pca = px.scatter(
    emb_df, x='x', y='y', color='Genre', hover_data=['Movie ID'],
    title="2D Item Embeddings (LightFM Representation)",
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_pca.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=600)
fig_pca.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
st.plotly_chart(fig_pca, use_container_width=True)
