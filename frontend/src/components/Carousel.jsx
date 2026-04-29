import { useEffect, useRef, useState } from "react";
import api from "../services/api.js";
import MovieCard from "./MovieCard.jsx";
import { SkeletonCard } from "./SkeletonCard.jsx";

/* Fallback demo movies when backend is offline */
const DEMO_MOVIES = [
  { title: "Inception", genre: "Sci-Fi / Thriller", imdb_rating: 8.8, poster: "https://image.tmdb.org/t/p/w300/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg" },
  { title: "Interstellar", genre: "Sci-Fi / Drama", imdb_rating: 8.7, poster: "https://image.tmdb.org/t/p/w300/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg" },
  { title: "The Dark Knight", genre: "Action / Crime", imdb_rating: 9.0, poster: "https://image.tmdb.org/t/p/w300/qJ2tW6WMUDux911r6m7haRef0WH.jpg" },
  { title: "Parasite", genre: "Thriller / Drama", imdb_rating: 8.5, poster: "https://image.tmdb.org/t/p/w300/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg" },
  { title: "The Godfather", genre: "Crime / Drama", imdb_rating: 9.2, poster: "https://image.tmdb.org/t/p/w300/3bhkrj58Vtu7enYsLegiokantuP.jpg" },
  { title: "Pulp Fiction", genre: "Crime / Drama", imdb_rating: 8.9, poster: "https://image.tmdb.org/t/p/w300/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg" },
  { title: "The Matrix", genre: "Sci-Fi / Action", imdb_rating: 8.7, poster: "https://image.tmdb.org/t/p/w300/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg" },
  { title: "Fight Club", genre: "Drama / Thriller", imdb_rating: 8.8, poster: "https://image.tmdb.org/t/p/w300/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg" },
];

export default function Carousel({ title = "Recommended for You", userId = 1 }) {
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    setLoading(true);
    setError(false);
    api
      .get(`/recommend/${userId}`)
      .then((res) => {
        const data = Array.isArray(res.data) ? res.data : res.data?.recommendations || [];
        setMovies(data.length > 0 ? data : DEMO_MOVIES);
      })
      .catch(() => {
        setMovies(DEMO_MOVIES); // fallback to demo
        setError(true);
      })
      .finally(() => setLoading(false));
  }, [userId]);

  const scroll = (dir) => {
    if (scrollRef.current) {
      scrollRef.current.scrollBy({ left: dir * 300, behavior: "smooth" });
    }
  };

  return (
    <section style={{ marginBottom: "40px" }}>
      {/* Section header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px", padding: "0 32px" }}>
        <h2 style={{ color: "#fff", fontSize: "1.3rem", fontWeight: 700, margin: 0 }}>
          {title}
        </h2>
        {error && (
          <span style={{ fontSize: "0.75rem", color: "#b3b3b3", background: "#1a1a1a", padding: "4px 10px", borderRadius: "4px" }}>
            📡 Showing demo data (backend offline)
          </span>
        )}
        <div style={{ display: "flex", gap: "8px" }}>
          <button onClick={() => scroll(-1)} style={arrowBtn}>‹</button>
          <button onClick={() => scroll(1)}  style={arrowBtn}>›</button>
        </div>
      </div>

      {/* Scrollable row */}
      <div
        ref={scrollRef}
        className="no-scrollbar"
        style={{
          display: "flex",
          gap: "14px",
          overflowX: "auto",
          padding: "8px 32px 16px",
          scrollSnapType: "x mandatory",
        }}
      >
        {loading
          ? Array.from({ length: 8 }).map((_, i) => (
              <div key={i} style={{ flexShrink: 0, width: "160px", scrollSnapAlign: "start" }}>
                <SkeletonCard />
              </div>
            ))
          : movies.map((m) => (
              <div key={m.title + Math.random()} style={{ flexShrink: 0, width: "160px", scrollSnapAlign: "start" }}>
                <MovieCard {...m} />
              </div>
            ))}
      </div>
    </section>
  );
}

const arrowBtn = {
  background: "rgba(255,255,255,0.08)",
  border: "1px solid rgba(255,255,255,0.12)",
  color: "#fff",
  width: "34px",
  height: "34px",
  borderRadius: "50%",
  cursor: "pointer",
  fontSize: "1.2rem",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  transition: "background 0.2s",
};
