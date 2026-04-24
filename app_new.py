"""
Smart Movie Recommendation System - Flask Backend
Complete system with NLP query understanding, hybrid recommendations, and analytics

Architecture:
- TMDb API Integration: Fetch movie data
- NLP Query Processor: Understand natural language queries
- Hybrid Recommendation Engine: Combine multiple signals
- Analytics Module: Genre trends, statistics, insights
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv

# Import custom modules
from tmdb_data_fetcher import get_tmdb_fetcher
from nlp_query_processor import create_query_processor
from recommendation_engine import create_recommendation_engine
from analytics import create_analytics
from external_ratings_fetcher import ExternalRatingsFetcher

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Initialize system components
tmdb_fetcher = None
nlp_processor = None
recommendation_engine = None
analytics_engine = None
movies_db = None
genres_map = None
external_ratings_fetcher = None

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def initialize_system():
    """Initialize all system components at startup"""
    global tmdb_fetcher, nlp_processor, recommendation_engine, analytics_engine, movies_db, genres_map, external_ratings_fetcher
    
    logger.info("=" * 80)
    logger.info("Initializing Smart Movie Recommendation System")
    logger.info("=" * 80)
    
    # Initialize TMDb fetcher
    api_key = os.getenv('TMDB_API_KEY')
    tmdb_fetcher = get_tmdb_fetcher(api_key)
    logger.info("✓ TMDb Fetcher initialized")
    
    # Fetch initial movie data
    try:
        logger.info("Fetching popular movies from TMDb...")
        movies_db = tmdb_fetcher.discover_movies(
            sort_by='popularity.desc',
            max_results=500
        )
        logger.info(f"✓ Loaded {len(movies_db)} movies")
    except Exception as e:
        logger.error(f"Failed to fetch movies: {e}")
        movies_db = None
    
    # Get genres
    try:
        genres_map = tmdb_fetcher.get_genres()
        logger.info(f"✓ Loaded {len(genres_map)} genres")
    except Exception as e:
        logger.warning(f"Failed to load genres: {e}")
        genres_map = {}
    
    # Initialize NLP processor (try with semantic model, fallback to keyword-based)
    try:
        nlp_processor = create_query_processor(use_semantic=True)
        logger.info("✓ NLP Query Processor initialized (semantic mode)")
    except Exception as e:
        logger.warning(f"Semantic mode failed, using keyword-based NLP: {e}")
        nlp_processor = create_query_processor(use_semantic=False)
        logger.info("✓ NLP Query Processor initialized (keyword mode)")
    
    # Initialize recommendation engine
    recommendation_engine = create_recommendation_engine(nlp_processor, tmdb_fetcher)
    logger.info("✓ Hybrid Recommendation Engine initialized")

    # Optional external metadata source: IMDb + Rotten Tomatoes via OMDb
    external_ratings_fetcher = ExternalRatingsFetcher(os.getenv('OMDB_API_KEY'))
    logger.info("✓ External ratings fetcher initialized")
    
    # Initialize analytics
    analytics_engine = create_analytics(movies_db)
    logger.info("✓ Analytics Engine initialized")
    
    logger.info("=" * 80)
    logger.info("System Ready! Starting server...")
    logger.info("=" * 80)

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/system-info', methods=['GET'])
def system_info():
    """Get system information"""
    try:
        return jsonify({
            'success': True,
            'system': 'Smart Movie Recommendation System',
            'version': '1.0.0',
            'status': 'running',
            'components': {
                'tmdb_fetcher': tmdb_fetcher is not None,
                'nlp_processor': nlp_processor is not None,
                'recommendation_engine': recommendation_engine is not None,
                'analytics_engine': analytics_engine is not None,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """
    Get movie recommendations based on natural language query
    
    POST body: {
        "query": "I need action movies with good ratings",
        "n_recommendations": 5
    }
    """
    try:
        data = request.json
        query = data.get('query', '').strip()
        requested_n = int(data.get('n_recommendations', 0))
        
        if not query or len(query) < 3:
            return jsonify({
                'success': False,
                'message': 'Please enter a query with at least 3 characters'
            }), 400
        
        if movies_db is None or movies_db.empty:
            return jsonify({
                'success': False,
                'message': 'Movie database not loaded'
            }), 503
        
        # Parse intent first so natural language count works ("give me 7 ...")
        intent = nlp_processor.process_query(query)
        n_recs = requested_n if requested_n > 0 else int(intent.get('count', 5))

        # Get recommendations
        recommendations = recommendation_engine.recommend(
            query=query,
            movies_df=movies_db,
            n_recommendations=min(n_recs, 100),
            genres_dict=genres_map
        )
        
        if recommendations.empty:
            return jsonify({
                'success': False,
                'message': 'No recommendations found for your query'
            }), 404
        
        # Convert to JSON-serializable format
        recs_json = []
        for _, row in recommendations.iterrows():
            year = str(row.get('date', ''))[:4] if row.get('date') else None
            external_meta = external_ratings_fetcher.get_external_metadata(
                title=str(row['title']),
                year=year
            ) if external_ratings_fetcher else {}
            recs_json.append({
                'rank': int(row['rank']),
                'movie_id': int(row['id']),
                'title': str(row['title']),
                'rating': float(row['rating']),
                'popularity': float(row['popularity']),
                'score': float(row['final_score']),
                'genres': row['genres'] if isinstance(row['genres'], list) else [],
                'overview': str(row['overview'])[:200],  # Truncate overview
                'imdb_title': external_meta.get('imdb_title'),
                'imdb_id': external_meta.get('imdb_id'),
                'imdb_rating': external_meta.get('imdb_rating'),
                'rotten_tomatoes_rating': external_meta.get('rotten_tomatoes_rating'),
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'intent': intent,
            'recommendations': recs_json,
            'count': len(recs_json)
        })
    
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/search', methods=['GET'])
def search_movies():
    """
    Search for movies
    
    Query parameters:
    - query: search string
    - max_results: max number of results (default 20)
    """
    try:
        query = request.args.get('query', '').strip()
        max_results = min(int(request.args.get('max_results', 20)), 50)
        
        if not query or len(query) < 2:
            return jsonify({
                'success': False,
                'message': 'Query too short'
            }), 400
        
        results = tmdb_fetcher.search_movies(query, max_results)
        
        if results.empty:
            return jsonify({
                'success': True,
                'query': query,
                'results': [],
                'count': 0
            })
        
        # Convert to JSON-serializable format
        results_json = []
        for _, row in results.iterrows():
            results_json.append({
                'movie_id': int(row['id']),
                'title': str(row['title']),
                'rating': float(row['rating']) if 'rating' in row else 0,
                'popularity': float(row['popularity']) if 'popularity' in row else 0,
                'overview': str(row['overview'])[:200] if 'overview' in row else '',
                'year': str(row['date'])[:4] if 'date' in row else '',
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results_json,
            'count': len(results_json)
        })
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    """Get detailed information about a specific movie"""
    try:
        details = tmdb_fetcher.get_movie_details(movie_id)
        
        if not details:
            return jsonify({
                'success': False,
                'message': 'Movie not found'
            }), 404
        
        return jsonify({
            'success': True,
            'movie': details
        })
    
    except Exception as e:
        logger.error(f"Movie details error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data"""
    try:
        if analytics_engine is None:
            return jsonify({
                'success': False,
                'message': 'Analytics engine not initialized'
            }), 503
        
        analytics_data = analytics_engine.export_analytics()
        
        return jsonify({
            'success': True,
            'analytics': analytics_data
        })
    
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/analytics/genres', methods=['GET'])
def get_genre_analytics():
    """Get genre analytics"""
    try:
        if analytics_engine is None:
            return jsonify({'success': False}), 503
        
        return jsonify({
            'success': True,
            'genres': analytics_engine.get_popular_genres(),
            'genre_stats': analytics_engine.get_genre_statistics(),
            'genres_per_year': analytics_engine.get_most_popular_genres_per_year(),
        })
    
    except Exception as e:
        logger.error(f"Genre analytics error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/analytics/summary', methods=['GET'])
def get_summary_stats():
    """Get summary statistics"""
    try:
        if analytics_engine is None:
            return jsonify({'success': False}), 503
        
        return jsonify({
            'success': True,
            'summary': analytics_engine.get_summary_statistics(),
            'rating_distribution': analytics_engine.get_rating_distribution(),
            'top_movies': analytics_engine.get_highest_rated_movies().to_dict('records'),
        })
    
    except Exception as e:
        logger.error(f"Summary stats error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Smart Movie Recommendation System'
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Initialize system
    initialize_system()
    
    # Run Flask app
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='127.0.0.1', port=5000, debug=debug)
