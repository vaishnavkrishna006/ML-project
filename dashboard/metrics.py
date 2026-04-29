import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def calculate_kpi_metrics():
    """
    Simulates evaluation metrics from a LightFM model.
    In production, this would call model.predict() and calculate precision_at_k.
    """
    return {
        "precision_k": 0.245,
        "recall_k": 0.182,
        "f1_score": 0.209
    }

def get_confusion_matrix_data():
    """
    Simulates a confusion matrix for implicit feedback classification
    (e.g., thresholding rating >= 4 as positive).
    """
    # Actual vs Predicted
    # [TN, FP]
    # [FN, TP]
    cm = np.array([
        [15200, 3100],
        [2800, 8400]
    ])
    return cm

def get_pr_curve_data():
    """
    Simulates Precision-Recall curve points.
    """
    y_true = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
    y_scores = np.random.rand(1000)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return precision, recall, thresholds

def compute_pca_embeddings(embeddings, n_components=2):
    """
    Reduces embedding dimensions using PCA for visualization.
    """
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced

def compute_tsne_embeddings(embeddings, n_components=2):
    """
    Reduces embedding dimensions using t-SNE for visualization.
    """
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    return reduced
