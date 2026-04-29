import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import Navbar from "../components/Navbar.jsx";
import MovieCard from "../components/MovieCard.jsx";
import { SkeletonCard } from "../components/SkeletonCard.jsx";
import api from "../services/api.js";

// Demo data for search when backend is offline
const DEMO_SEARCH_RESULTS = [
  { title: "Batman Begins", genre: "Action", imdb_rating: 8.2, poster: "https://image.tmdb.org/t/p/w300/8RW2runSEc34ZlMnyTrmyoA2Ea9.jpg" },
  { title: "The Dark Knight Rises", genre: "Action", imdb_rating: 8.4, poster: "https://image.tmdb.org/t/p/w300/hr0L2aueqlP2BYUblTTjmtn0hw4.jpg" },
  { title: "Batman", genre: "Action", imdb_rating: 7.5, poster: "https://image.tmdb.org/t/p/w300/tDekno1z8xGEQk2R5T0m2A5hQ4P.jpg" },
];

export default function Search() {
  const [searchParams] = useSearchParams();
  const query = searchParams.get("q") || "";
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!query) {
      setResults([]);
      return;
    }
    setLoading(true);
    setError(false);

    api
      .get(`/search?q=${encodeURIComponent(query)}`)
      .then((res) => {
        const data = Array.isArray(res.data) ? res.data : res.data?.results || [];
        setResults(data);
      })
      .catch((err) => {
        console.error(err);
        setError(true);
        // Fallback demo results for "batman" search for visual testing
        if (query.toLowerCase().includes("batman")) {
          setResults(DEMO_SEARCH_RESULTS);
        } else {
          setResults([]);
        }
      })
      .finally(() => setLoading(false));
  }, [query]);

  return (
    <div>
      <Navbar />
      <div style={{ padding: "40px 32px" }}>
        <h2 style={{ fontSize: "1.8rem", marginBottom: "24px", fontWeight: 700 }}>
          Results for "{query}"
        </h2>

        {error && (
          <div style={{ marginBottom: "20px", padding: "12px", background: "rgba(229,9,20,0.1)", borderLeft: "4px solid #e50914", color: "#e50914" }}>
            Warning: Could not connect to backend API. Showing offline/demo data if available.
          </div>
        )}

        {loading ? (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
              gap: "24px",
            }}
          >
            {Array.from({ length: 10 }).map((_, i) => (
              <SkeletonCard key={i} />
            ))}
          </div>
        ) : results.length > 0 ? (
          <div
            className="animate-fadeIn"
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
              gap: "24px",
            }}
          >
            {results.map((m, idx) => (
              <MovieCard key={m.title + idx} {...m} />
            ))}
          </div>
        ) : (
          <div style={{ textAlign: "center", padding: "60px 0", color: "#b3b3b3" }}>
            <span style={{ fontSize: "3rem", display: "block", marginBottom: "16px" }}>😕</span>
            <p style={{ fontSize: "1.2rem" }}>No movies found matching your search.</p>
          </div>
        )}
      </div>
    </div>
  );
}
