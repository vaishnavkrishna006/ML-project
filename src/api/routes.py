from flask import Blueprint, jsonify, request
from src.services.tmdb_service import get_movie_details, search_movies
from src.services.omdb_service import get_imdb_ratings

api_bp = Blueprint('api', __name__)

# This will be injected by app.py
recommender_model = None

@api_bp.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = "healthy" if recommender_model is not None and recommender_model.is_trained else "model_not_loaded"
    return jsonify({"status": status, "message": "ML Recommendation API is running."})

def _format_movie_response(tmdb_data):
    """Helper to merge TMDB and OMDb data."""
    if not tmdb_data:
        return None
        
    imdb_id = tmdb_data.get('imdb_id')
    omdb_data = get_imdb_ratings(imdb_id) if imdb_id else {}
    
    poster_path = tmdb_data.get('poster_path')
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
    
    genres = [g.get('name') for g in tmdb_data.get('genres', [])] if 'genres' in tmdb_data else []
    
    # If genres aren't fully expanded (e.g. from search results)
    if not genres and 'genre_ids' in tmdb_data:
        genres = [f"GenreID:{g}" for g in tmdb_data.get('genre_ids', [])]

    return {
        "id": tmdb_data.get('id'),
        "title": tmdb_data.get('title'),
        "overview": tmdb_data.get('overview'),
        "poster": poster_url,
        "genres": genres,
        "release_date": tmdb_data.get('release_date'),
        "tmdb_rating": tmdb_data.get('vote_average'),
        "imdb_rating": omdb_data.get('imdb_rating', 'N/A'),
        "rotten_tomatoes": omdb_data.get('rotten_tomatoes', 'N/A')
    }

@api_bp.route('/recommend/<user_id>', methods=['GET'])
def recommend(user_id):
    """Returns top movie recommendations enriched with TMDB/OMDb data."""
    if recommender_model is None or not recommender_model.is_trained:
        return jsonify({"error": "Model is not loaded or not trained."}), 503

    top_k = request.args.get('top_k', default=10, type=int)

    try:
        recommended_ids = recommender_model.recommend(user_id=user_id, top_k=top_k)
        
        results = []
        for tmdb_id in recommended_ids:
            # Fetch rich details
            movie_data = get_movie_details(tmdb_id)
            if movie_data:
                formatted = _format_movie_response(movie_data)
                if formatted:
                    results.append(formatted)
            else:
                # Fallback if API fails or ID is missing
                results.append({"id": int(tmdb_id), "title": f"Movie {tmdb_id}", "poster": None})
        
        return jsonify({
            "user_id": user_id,
            "recommendations": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    """Fetches details for a single movie."""
    movie_data = get_movie_details(movie_id)
    if not movie_data:
        return jsonify({"error": "Movie not found or API error"}), 404
        
    return jsonify(_format_movie_response(movie_data))

@api_bp.route('/search', methods=['GET'])
def search():
    """Searches for a movie by name."""
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
        
    results = search_movies(query)
    formatted_results = []
    
    # Only format top 5 results to avoid hammering the APIs
    for item in results[:5]:
        # get_movie_details gives us the imdb_id which we need for OMDb
        full_details = get_movie_details(item.get('id'))
        if full_details:
            formatted = _format_movie_response(full_details)
            if formatted:
                formatted_results.append(formatted)
                
    return jsonify({"results": formatted_results})
