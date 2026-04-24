const API_BASE_URL = "http://127.0.0.1:5000";
let latestRecommendations = [];

document.addEventListener("DOMContentLoaded", () => {
    setupEventListeners();
    loadDashboardStats();
});

function setupEventListeners() {
    document.getElementById("recommendationForm").addEventListener("submit", handleQuerySubmit);
    document.getElementById("genreFilterInput").addEventListener("change", applyClientFiltersAndSort);
    document.getElementById("minRatingInput").addEventListener("input", applyClientFiltersAndSort);
    document.getElementById("sortByInput").addEventListener("change", applyClientFiltersAndSort);
}

async function loadDashboardStats() {
    try {
        const [systemRes, analyticsRes] = await Promise.all([
            fetch(`${API_BASE_URL}/api/system-info`),
            fetch(`${API_BASE_URL}/api/analytics/summary`),
        ]);

        const systemData = await systemRes.json();
        const analyticsData = await analyticsRes.json();

        document.getElementById("stat-status").textContent = systemData.success ? "Online" : "Offline";

        if (analyticsData.success && analyticsData.summary) {
            document.getElementById("stat-total-movies").textContent = formatNumber(analyticsData.summary.total_movies || 0);
            document.getElementById("stat-avg-rating").textContent = Number(analyticsData.summary.avg_rating || 0).toFixed(2);
        } else {
            document.getElementById("stat-total-movies").textContent = "-";
            document.getElementById("stat-avg-rating").textContent = "-";
        }
    } catch (error) {
        document.getElementById("stat-status").textContent = "Offline";
    }
}

async function handleQuerySubmit(e) {
    e.preventDefault();

    const query = document.getElementById("naturalQuery").value.trim();
    const rawCount = parseInt(document.getElementById("nRecommendations").value || "10", 10);
    const nRecommendations = Math.min(Math.max(rawCount, 1), 100);

    if (query.length < 3) {
        showError("Please enter at least 3 characters in your query.");
        return;
    }

    showLoading(true);
    hideError();

    try {
        const response = await fetch(`${API_BASE_URL}/api/recommend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query,
                n_recommendations: nRecommendations,
            }),
        });

        const data = await response.json();
        if (!data.success) {
            showError(data.message || "No recommendations found.");
            return;
        }

        latestRecommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
        document.getElementById("resultQuery").textContent = data.query || query;

        applyClientFiltersAndSort();
        document.getElementById("resultsSection").style.display = "block";
        scrollToElement("resultsSection");
    } catch (error) {
        showError(`Unable to fetch recommendations: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function applyClientFiltersAndSort() {
    const genreFilter = document.getElementById("genreFilterInput").value.trim().toLowerCase();
    const minRating = parseFloat(document.getElementById("minRatingInput").value || "0");
    const sortBy = document.getElementById("sortByInput").value;

    let filtered = [...latestRecommendations].filter((movie) => {
        const rating = Number(movie.rating || 0);
        const genres = Array.isArray(movie.genres) ? movie.genres : [];
        const genreMatch = !genreFilter || genres.some((g) => String(g).toLowerCase() === genreFilter);
        return rating >= minRating && genreMatch;
    });

    filtered.sort((a, b) => {
        if (sortBy === "rating_desc") return Number(b.rating || 0) - Number(a.rating || 0);
        if (sortBy === "popularity_desc") return Number(b.popularity || 0) - Number(a.popularity || 0);
        if (sortBy === "title_asc") return String(a.title || "").localeCompare(String(b.title || ""));
        return Number(b.score || 0) - Number(a.score || 0);
    });

    document.getElementById("resultCount").textContent = filtered.length;
    document.getElementById("recommendationsGrid").innerHTML = buildRecommendationCards(filtered);
}

function buildRecommendationCards(recommendations) {
    if (!recommendations.length) {
        return `<p>No results for selected filters. Try reducing filters.</p>`;
    }

    return recommendations
        .map((movie, idx) => {
            const genres = Array.isArray(movie.genres) ? movie.genres.join(", ") : "-";
            return `
                <article class="rec-card">
                    <div class="rec-top">
                        <span class="rank-badge">#${idx + 1}</span>
                        <span class="info-badge">Score ${(Number(movie.score || 0)).toFixed(3)}</span>
                    </div>
                    <h3>Movie: ${escapeHtml(movie.title || "Untitled")}</h3>
                    <p class="rec-meta">Rating: ${Number(movie.rating || 0).toFixed(1)} | Popularity: ${Number(movie.popularity || 0).toFixed(1)}</p>
                    <p class="rec-meta">
                        IMDb: ${escapeHtml(movie.imdb_title || "N/A")}
                        ${movie.imdb_rating ? ` | Rating: ${escapeHtml(String(movie.imdb_rating))}` : ""}
                        ${movie.rotten_tomatoes_rating ? ` | Rotten Tomatoes: ${escapeHtml(String(movie.rotten_tomatoes_rating))}` : ""}
                    </p>
                    <p class="rec-genres">${escapeHtml(genres)}</p>
                    <p class="rec-overview">${escapeHtml(movie.overview || "No overview available.")}</p>
                </article>
            `;
        })
        .join("");
}

function showLoading(show) {
    const loadingSection = document.getElementById("loadingSection");
    const submitBtn = document.getElementById("submitBtn");

    if (show) {
        loadingSection.style.display = "block";
        submitBtn.disabled = true;
        submitBtn.innerHTML = "<span>Loading...</span>";
    } else {
        loadingSection.style.display = "none";
        submitBtn.disabled = false;
        submitBtn.innerHTML = "<span>Search Recommendations</span>";
    }
}

function showError(message) {
    document.getElementById("errorMessage").textContent = message;
    document.getElementById("errorSection").style.display = "block";
    document.getElementById("resultsSection").style.display = "none";
}

function hideError() {
    document.getElementById("errorSection").style.display = "none";
}

function hideAllSections() {
    document.getElementById("resultsSection").style.display = "none";
    document.getElementById("errorSection").style.display = "none";
    document.getElementById("loadingSection").style.display = "none";
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + "M";
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + "K";
    }
    return num.toString();
}

function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        setTimeout(() => {
            element.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 100);
    }
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}
