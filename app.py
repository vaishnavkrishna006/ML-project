import os
from flask import Flask
from flask_cors import CORS
from src.models.lightfm_model import LightFMRecommender
from src.api.routes import api_bp
import src.api.routes as routes_module

app = Flask(__name__)
# Enable CORS for the React frontend
CORS(app)

def load_model():
    """Load the ML model at startup."""
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        try:
            model = LightFMRecommender.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: {model_path} not found. Run train_model.py first.")
    return None

# Load model and inject into routes
recommender_model = load_model()
routes_module.recommender_model = recommender_model

# Register Blueprint
app.register_blueprint(api_bp)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
