import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightFMRecommender:
    def __init__(self, loss='warp', components=30, epochs=10):
        self.model = LightFM(loss=loss, no_components=components)
        self.epochs = epochs
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.interactions = None
        self.is_trained = False

    def prepare_data(self, df):
        """
        Converts pandas DataFrame to scipy sparse matrix and builds index mappings.
        """
        logger.info("Preparing data for LightFM model...")
        unique_users = df['user_id'].unique()
        unique_items = df['movie_id'].unique()

        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}

        row = df['user_id'].map(self.user_mapping).values
        col = df['movie_id'].map(self.item_mapping).values
        data = df['rating'].values

        self.interactions = sp.coo_matrix((data, (row, col)), 
                                          shape=(len(unique_users), len(unique_items)))
        logger.info(f"Interactions matrix shape: {self.interactions.shape}")

    def train(self):
        """
        Trains the LightFM model on the prepared interactions.
        """
        if self.interactions is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
        
        logger.info(f"Training LightFM model with {self.epochs} epochs...")
        self.model.fit(self.interactions, epochs=self.epochs, num_threads=2)
        self.is_trained = True
        logger.info("Training complete.")

    def evaluate(self):
        """
        Evaluates the trained model (Basic Evaluation).
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        logger.info("Evaluating model...")
        train_precision = precision_at_k(self.model, self.interactions, k=5).mean()
        train_recall = recall_at_k(self.model, self.interactions, k=5).mean()
        
        logger.info(f"Train Precision@5: {train_precision:.4f}")
        logger.info(f"Train Recall@5: {train_recall:.4f}")
        return train_precision, train_recall

    def recommend(self, user_id, top_k=5):
        """
        Returns top_k recommended movie_ids for a given user_id.
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        # Ensure user is int/float matched to mapping type. Usually user_id is integer.
        try:
            user_id = int(user_id)
        except ValueError:
            pass
            
        if user_id not in self.user_mapping:
            logger.warning(f"User {user_id} not found in training data.")
            return []

        user_idx = self.user_mapping[user_id]
        n_users, n_items = self.interactions.shape
        
        # Predict scores for all items for this user
        scores = self.model.predict(user_idx, np.arange(n_items))
        
        # Get items already interacted with (to filter them out)
        user_row = self.interactions.tocsr()[user_idx]
        known_items = user_row.indices
        
        # Filter out known items
        scores[known_items] = -np.inf
        
        # Get top K indices
        # If there are fewer than K recommended items with scores > -inf, handle it gracefully
        top_indices = np.argsort(-scores)[:top_k]
        
        # Map indices back to original item IDs
        recommendations = [self.reverse_item_mapping[idx] for idx in top_indices if scores[idx] != -np.inf]
        return recommendations
    
    def save(self, filepath='model.pkl'):
        """Saves the model and mappings."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'reverse_user_mapping': self.reverse_user_mapping,
                'reverse_item_mapping': self.reverse_item_mapping,
                'is_trained': self.is_trained
            }, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath='model.pkl'):
        """Loads a saved model and mappings."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")
            
        instance = cls()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            instance.model = data['model']
            instance.user_mapping = data['user_mapping']
            instance.item_mapping = data['item_mapping']
            instance.reverse_user_mapping = data['reverse_user_mapping']
            instance.reverse_item_mapping = data['reverse_item_mapping']
            instance.is_trained = data['is_trained']
        logger.info(f"Model loaded from {filepath}")
        return instance
