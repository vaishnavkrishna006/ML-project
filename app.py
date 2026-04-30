import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.models.recommender import MovieRecommender

app = Flask(__name__)
CORS(app)

# Global recommender instance
recommender = None

def load_model():
    """Load the Content-Based Recommender at startup."""
    global recommender
    model_path = 'src/models/artifacts/recommender.pkl'
    if os.path.exists(model_path):
        try:
            recommender = MovieRecommender.load(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: {model_path} not found. Run train_model.py first.")

# Initialize model
load_model()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": recommender is not None})

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    Endpoint for recommendations.
    Supports title-based and natural language (bonus) search.
    """
    if recommender is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    title = request.args.get('title')
    query = request.args.get('query') # Bonus: natural language
    top_n = request.args.get('top_n', default=10, type=int)

    try:
        if title:
            results = recommender.recommend(title, top_n=top_n)
            if results is None:
                return jsonify({"error": f"Movie '{title}' not found in dataset."}), 404
        elif query:
            results = recommender.search_by_description(query, top_n=top_n)
        else:
            return jsonify({"error": "Provide 'title' or 'query' parameter."}), 400

        return jsonify({
            "count": len(results),
            "recommendations": results.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
