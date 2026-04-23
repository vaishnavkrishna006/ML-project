// Movie Recommendation System - Frontend Script

// ============================================================================
// CONSTANTS
// ============================================================================

const API_BASE_URL = 'http://127.0.0.1:5000';

const MODEL_DESCRIPTIONS = {
    'hybrid': 'Combines all approaches for best results',
    'user_cf': 'Finds similar users and recommends their favorite movies',
    'item_cf': 'Finds movies similar to ones you already watched',
    'content': 'Analyzes user profiles and viewing patterns'
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Music Recommendation System...');
    
    // Load statistics
    loadStats();
    
    // Setup event listeners
    setupEventListeners();
    
    // Update model description when selection changes
    document.getElementById('modelType').addEventListener('change', updateModelDescription);
});

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    const form = document.getElementById('recommendationForm');
    form.addEventListener('submit', handleFormSubmit);
}

function updateModelDescription() {
    const modelType = document.getElementById('modelType').value;
    const description = document.getElementById('modelDescription');
    description.textContent = MODEL_DESCRIPTIONS[modelType];
}

// ============================================================================
// FORM HANDLING
// ============================================================================

async function handleFormSubmit(e) {
    e.preventDefault();
    
    const userId = document.getElementById('userId').value;
    const modelType = document.getElementById('modelType').value;
    const nRecommendations = document.getElementById('nRecommendations').value;
    
    // Validate input
    if (!userId || userId < 0) {
        showError('Please enter a valid User ID');
        return;
    }
    
    // Show loading
    showLoading(true);
    
    try {
        // Call API
        const response = await fetch(`${API_BASE_URL}/api/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: parseInt(userId),
                model_type: modelType,
                n_recommendations: parseInt(nRecommendations)
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayRecommendations(data);
        } else {
            showError(data.message);
        }
    } catch (error) {
        console.error('Error:', error);
        showError(`Failed to get recommendations: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// ============================================================================
// API CALLS
// ============================================================================

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('stat-users').textContent = formatNumber(data.stats.total_users);
            document.getElementById('stat-records').textContent = formatNumber(data.stats.total_records);
            document.getElementById('stat-movies').textContent = formatNumber(data.stats.unique_movies);
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// ============================================================================
// DISPLAY FUNCTIONS
// ============================================================================

function displayRecommendations(data) {
    // Hide other sections
    hideAllSections();
    
    // Update result info
    document.getElementById('resultUserId').textContent = data.user_id;
    document.getElementById('resultModelType').textContent = formatModelType(data.model_type);
    document.getElementById('resultCount').textContent = data.count;
    
    // Build and display table
    const tableHtml = buildRecommendationsTable(data.recommendations);
    document.getElementById('recommendationsTable').innerHTML = tableHtml;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Scroll to results
    scrollToElement('resultsSection');
}

function buildRecommendationsTable(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        return '<p>No recommendations found.</p>';
    }
    
    let html = `
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Movie ID</th>
                    <th>Recommendation Score</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    // Get max score for normalization
    const maxScore = Math.max(...recommendations.map(r => r.recommendation_score));
    
    recommendations.forEach((rec) => {
        const percentage = (rec.recommendation_score / maxScore * 100).toFixed(1);
        html += `
            <tr>
                <td><span class="rank-badge">${rec.rank}</span></td>
                <td><span class="movie-id">${rec.movie_id}</span></td>
                <td>
                    <div class="score-bar" style="width: ${percentage}%">
                        ${rec.recommendation_score.toFixed(2)}
                    </div>
                </td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
    `;
    
    return html;
}

function showLoading(show) {
    const loadingSection = document.getElementById('loadingSection');
    const submitBtn = document.getElementById('submitBtn');
    
    if (show) {
        loadingSection.style.display = 'block';
        submitBtn.disabled = true;
        submitBtn.innerHTML = '⏳ Loading...';
    } else {
        loadingSection.style.display = 'none';
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>Get Recommendations</span>';
    }
}

function showError(message) {
    hideAllSections();
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorSection').style.display = 'block';
    scrollToElement('errorSection');
}

function hideAllSections() {
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'none';
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatModelType(modelType) {
    const modelNames = {
        'hybrid': '🔀 Hybrid',
        'user_cf': '👥 User-Based CF',
        'item_cf': '� Item-Based CF',
        'content': '👤 Content-Based'
    };
    return modelNames[modelType] || modelType;
}

function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        setTimeout(() => {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
});

// ============================================================================
// LOCAL STORAGE (Optional - for saving preferences)
// ============================================================================

function savePreference(key, value) {
    try {
        localStorage.setItem(`music-rec-${key}`, value);
    } catch (error) {
        console.warn('Could not save preference:', error);
    }
}

function getPreference(key) {
    try {
        return localStorage.getItem(`music-rec-${key}`);
    } catch (error) {
        console.warn('Could not get preference:', error);
        return null;
    }
}

// ============================================================================
// CONSOLE OUTPUT
// ============================================================================

console.log('%cMovie Recommendation System', 'color: #667eea; font-size: 16px; font-weight: bold;');
console.log('%cVersion 1.0.0', 'color: #764ba2;');
console.log('%cReady for recommendations!', 'color: #27ae60; font-weight: bold;');

// ============================================================================
// TMDB MOVIE SEARCH & METADATA (NEW)
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    setupSearchEvent();
});

function setupSearchEvent() {
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', handleSearchSubmit);
    }
}

async function handleSearchSubmit(e) {
    e.preventDefault();
    const query = document.getElementById('searchQuery').value.trim();
    if (!query || query.length < 2) {
        showSearchResults([], 'Please enter at least 2 characters.');
        return;
    }
    try {
        const res = await fetch(`${API_BASE_URL}/api/search?query=${encodeURIComponent(query)}`);
        const data = await res.json();
        if (data.success) {
            showSearchResults(data.results);
        } else {
            showSearchResults([], data.message || 'No results found.');
        }
    } catch (err) {
        showSearchResults([], 'Error searching movies.');
    }
}

function showSearchResults(results, errorMsg = null) {
    const container = document.getElementById('searchResults');
    if (!container) return;
    if (errorMsg) {
        container.innerHTML = `<div class="error-msg">${errorMsg}</div>`;
        container.style.display = 'block';
        return;
    }
    if (!results || results.length === 0) {
        container.innerHTML = '<div class="error-msg">No results found.</div>';
        container.style.display = 'block';
        return;
    }
    let html = '';
    results.forEach(movie => {
        html += `
        <div class="movie-card" onclick="showMovieDetails(${movie.id})">
            <div class="movie-poster">${movie.poster ? `<img src='${movie.poster}' alt='Poster'>` : '🎬'}</div>
            <div class="movie-info">
                <div class="movie-title">${movie.title}</div>
                <div class="movie-year">${movie.year || ''}</div>
                <div class="movie-rating"><span class="rating-star">★</span> <span class="rating-value">${movie.rating || '-'}</span> <span class="rating-votes">(${movie.vote_count || 0})</span></div>
                <div class="movie-overview">${movie.overview || ''}</div>
            </div>
        </div>`;
    });
    container.innerHTML = html;
    container.style.display = 'grid';
}

// Modal for movie details
window.showMovieDetails = async function(movieId) {
    try {
        const res = await fetch(`${API_BASE_URL}/api/movie-metadata/${movieId}`);
        const data = await res.json();
        if (data.success) {
            showMovieModal(data.movie);
        }
    } catch (err) {
        alert('Failed to load movie details.');
    }
};

function showMovieModal(movie) {
    let modal = document.getElementById('movieModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'movieModal';
        modal.className = 'modal';
        document.body.appendChild(modal);
    }
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <span>${movie.title} (${movie.year || ''})</span>
                <button class="modal-close" onclick="closeMovieModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="detail-backdrop">${movie.backdrop ? `<img src='${movie.backdrop}' alt='Backdrop'>` : '🎬'}</div>
                <div class="detail-rating"><span>★</span> ${movie.rating || '-'} <span style="color:#999;font-size:0.8em;">(${movie.vote_count || 0} votes)</span></div>
                <div class="detail-genres">${(movie.genres || []).map(g => `<span class='detail-genre'>${g}</span>`).join(' ')}</div>
                <div class="detail-section"><h4>Overview</h4><p>${movie.overview || 'No description.'}</p></div>
                <div class="detail-section"><h4>Cast</h4><div class="cast-list">${(movie.cast || []).map(c => `<div class='cast-item'><span class='cast-name'>${c.name}</span> <span class='cast-character'>as ${c.character}</span></div>`).join(' ')}</div></div>
                <div class="detail-section"><h4>Production Countries</h4><p>${(movie.production_countries || []).join(', ')}</p></div>
            </div>
        </div>
    `;
    modal.classList.add('show');
}

window.closeMovieModal = function() {
    const modal = document.getElementById('movieModal');
    if (modal) modal.classList.remove('show');
};

// Close modal on outside click
window.addEventListener('click', function(e) {
    const modal = document.getElementById('movieModal');
    if (modal && e.target === modal) {
        closeMovieModal();
    }
});

// ============================================================================
// ENHANCED RECOMMENDATIONS WITH RATINGS (NEW)
// ============================================================================

// Optionally, you can add a toggle to use /api/recommend-with-ratings for boosted recommendations.
// For now, you can swap the endpoint in handleFormSubmit to '/api/recommend-with-ratings' to test.
