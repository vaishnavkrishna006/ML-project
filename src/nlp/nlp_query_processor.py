"""Natural-language intent extraction for movie recommendations."""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Try to import sentence transformers (optional)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_BERT_AVAILABLE = True
except ImportError:
    SENTENCE_BERT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPQueryProcessor:
    """
    Processes natural language queries to extract movie recommendation intent
    """
    
    # Genre mappings (TMDb-compatible names)
    GENRE_KEYWORDS = {
        'action': ['action', 'fight', 'battle', 'explosion', 'adventure'],
        'comedy': ['comedy', 'funny', 'humor', 'laugh', 'hilarious'],
        'drama': ['drama', 'emotional', 'sad', 'touching', 'character-driven'],
        'horror': ['horror', 'scary', 'terror', 'frightening', 'spooky'],
        'romance': ['romance', 'love', 'romantic', 'couple', 'relationship'],
        'thriller': ['thriller', 'suspense', 'mystery', 'crime', 'detective'],
        'science fiction': ['sci-fi', 'science fiction', 'future', 'space', 'alien', 'technology'],
        'fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'magical'],
        'animated': ['animated', 'animation', 'cartoon'],
        'documentary': ['documentary', 'real', 'true story', 'historical'],
        'crime': ['crime', 'gangster', 'mafia', 'heist'],
    }
    
    # Mood mappings
    MOOD_KEYWORDS = {
        'relaxing': ['relax', 'calm', 'peaceful', 'soothing', 'chill'],
        'intense': ['intense', 'exciting', 'adrenaline', 'thrilling'],
        'emotional': ['emotional', 'touching', 'tear', 'heartfelt', 'moving'],
        'funny': ['funny', 'hilarious', 'laugh', 'comedy'],
        'dark': ['dark', 'grim', 'bleak', 'depressing'],
        'uplifting': ['uplifting', 'inspiring', 'hopeful', 'positive'],
    }
    
    def __init__(self, use_semantic=False):
        """
        Initialize NLP processor
        
        Args:
            use_semantic (bool): Use sentence-bert for semantic understanding
        """
        self.use_semantic = use_semantic and SENTENCE_BERT_AVAILABLE
        self.model = None
        
        if self.use_semantic:
            try:
                logger.info("Loading Sentence-BERT model...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ Sentence-BERT loaded successfully")
            except Exception as e:
                logger.warning(f"Sentence-BERT loading failed: {e}")
                self.use_semantic = False
    
    def process_query(self, query: str) -> Dict:
        """
        Process a natural language query and extract intent
        
        Args:
            query (str): User query (e.g., "I need action movies with good ratings")
            
        Returns:
            dict: Extracted intent with genres, mood, count, and filters
        """
        query_lower = query.lower().strip()
        
        result = {
            'original_query': query,
            'genres': self._extract_genres(query_lower),
            'mood': self._extract_mood(query_lower),
            'count': self._extract_count(query_lower),
            'min_rating': self._extract_rating(query_lower),
            'year': self._extract_year(query_lower),
            'keywords': self._extract_keywords(query_lower),
            'reference_title': self._extract_reference_title(query),
            'query_tokens': query_lower.split(),
        }
        
        logger.info(f"Query processed: {result}")
        return result
    
    def _extract_genres(self, query: str) -> List[str]:
        """Extract genre keywords from query"""
        genres = []
        for genre, keywords in self.GENRE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    genres.append(genre)
                    break
        return list(set(genres))  # Remove duplicates
    
    def _extract_mood(self, query: str) -> str:
        """Extract primary mood/tone from query"""
        for mood, keywords in self.MOOD_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    return mood
        return 'general'
    
    def _extract_count(self, query: str) -> int:
        """Extract number of recommendations requested"""
        # Look for number words
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'couple': 2, 'few': 3, 'several': 5, 'handful': 5,
        }
        
        # Check for written numbers
        for word, num in number_words.items():
            if f' {word} ' in f' {query} ':
                return num
        
        # Check for digits
        match = re.search(r'\d+', query)
        if match:
            num = int(match.group())
            if 1 <= num <= 100:
                return num
        
        return 5  # Default to 5
    
    def _extract_rating(self, query: str) -> float:
        """Extract minimum rating filter"""
        rating_keywords = {
            'highly rated': 8.0,
            'best rated': 8.5,
            'top rated': 8.0,
            'excellent': 8.0,
            'good': 7.0,
            'great': 7.5,
            'amazing': 8.5,
            'masterpiece': 9.0,
        }
        
        for keyword, rating in rating_keywords.items():
            if keyword in query:
                return rating
        
        # Check for explicit rating mentions
        match = re.search(r'(?:rating|rate|rated)\s*(?:above|over|>|>=)?\s*(\d\.?\d?)', query)
        if match:
            return float(match.group(1))
        
        return 0.0  # No rating filter by default
    
    def _extract_year(self, query: str) -> Optional[int]:
        """Extract year filter if mentioned"""
        # Look for year patterns like "2023" or "recent", "old"
        year_patterns = [r'20\d{2}', r'19\d{2}']
        
        for pattern in year_patterns:
            match = re.search(pattern, query)
            if match:
                return int(match.group())
        
        # Handle relative time references
        current_year = datetime.now().year
        time_keywords = {
            'recent': current_year,
            'latest': current_year,
            'new': current_year,
            'old': 2000,
            'classic': 1990,
        }
        
        for keyword, year in time_keywords.items():
            if keyword in query:
                return year
        
        return None
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords for semantic search"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'from', 'by', 'i', 'me', 'my', 'you', 'your', 'he', 'she',
            'it', 'we', 'they', 'this', 'that', 'is', 'are', 'was', 'were', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'would',
            'should', 'may', 'might', 'must', 'give', 'me', 'suggest', 'find',
            'get', 'need', 'want', 'like', 'movie', 'movies', 'film', 'films',
        }
        
        words = query.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:10]

    def _extract_reference_title(self, query: str) -> Optional[str]:
        """
        Extract "movie like X" references.
        Examples:
        - "suggest emotional movies like Titanic"
        - "movies similar to Inception"
        """
        patterns = [
            r'like\s+([a-zA-Z0-9:\'\-\s]+)',
            r'similar to\s+([a-zA-Z0-9:\'\-\s]+)',
            r'kind of\s+([a-zA-Z0-9:\'\-\s]+)',
        ]
        cleaned_query = query.strip()
        for pattern in patterns:
            match = re.search(pattern, cleaned_query, flags=re.IGNORECASE)
            if match:
                title = match.group(1).strip(" .,!?:;\"'")
                return title if len(title) >= 2 else None
        return None
    
    def get_semantic_embedding(self, text: str):
        """Get semantic embedding of text using Sentence-BERT"""
        if not self.use_semantic or self.model is None:
            return None
        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def calculate_semantic_similarity(self, query1: str, query2: str) -> float:
        """Calculate semantic similarity between two queries/texts"""
        if not self.use_semantic or self.model is None:
            return 0.0
        
        try:
            embeddings = self.model.encode([query1, query2])
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(sim)
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0


def create_query_processor(use_semantic=False):
    """Factory function to create NLP query processor"""
    return NLPQueryProcessor(use_semantic)
