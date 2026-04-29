import logging
from src.utils.data_loader import load_data
from src.models.lightfm_model import LightFMRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting training process...")
    
    # 1. Load data
    logger.info("Step 1: Loading data")
    df = load_data('data/movies.csv')
    logger.info(f"Loaded {len(df)} interaction records.")
    
    # 2. Initialize and prepare model
    logger.info("Step 2: Preparing model")
    recommender = LightFMRecommender(loss='warp', components=30, epochs=20)
    recommender.prepare_data(df)
    
    # 3. Train model
    logger.info("Step 3: Training model")
    recommender.train()
    
    # 4. Evaluate model
    logger.info("Step 4: Evaluating model")
    recommender.evaluate()
    
    # 5. Save model
    logger.info("Step 5: Saving model")
    recommender.save('model.pkl')
    
    logger.info("Training process completed successfully.")

if __name__ == "__main__":
    main()
