import { useEffect, useState } from "react";
import api from "../services/api.js";

const FALLBACK_HERO = {
  title: "Stranger Things",
  description: "When a young boy vanishes, a small town uncovers a mystery involving secret experiments, terrifying supernatural forces and one strange little girl.",
  poster: "https://image.tmdb.org/t/p/original/56v2KjBlU4XaOv9rVYEQypROD7P.jpg",
};

export default function Hero() {
  const [featured, setFeatured] = useState(FALLBACK_HERO);
  const [loading, setLoading] = useState(false);

  // You can fetch a real featured movie from your API later
  // useEffect(() => {
  //   setLoading(true);
  //   api.get("/recommend/featured").then(res => setFeatured(res.data)).catch(() => {}).finally(() => setLoading(false));
  // }, []);

  if (loading) {
    return (
      <div style={{ height: "70vh", background: "#1a1a1a" }} className="skeleton" />
    );
  }

  return (
    <div
      className="animate-fadeIn"
      style={{
        position: "relative",
        height: "75vh",
        minHeight: "500px",
        maxHeight: "800px",
        width: "100%",
        marginBottom: "32px",
      }}
    >
      {/* Background Image */}
      <img
        src={featured.poster}
        alt={featured.title}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          objectPosition: "top",
        }}
      />

      {/* Gradients for blending */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "linear-gradient(90deg, rgba(15,15,15,1) 0%, rgba(15,15,15,0.6) 50%, rgba(15,15,15,0) 100%)",
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "linear-gradient(0deg, rgba(15,15,15,1) 0%, rgba(15,15,15,0) 40%)",
        }}
      />

      {/* Content */}
      <div
        style={{
          position: "absolute",
          bottom: "15%",
          left: "32px",
          maxWidth: "600px",
          zIndex: 10,
        }}
        className="animate-slideUp"
      >
        <span
          style={{
            color: "#e50914",
            fontWeight: 800,
            letterSpacing: "2px",
            fontSize: "0.8rem",
            textTransform: "uppercase",
            marginBottom: "8px",
            display: "block",
          }}
        >
          N O W   S T R E A M I N G
        </span>
        <h1
          style={{
            fontSize: "clamp(2.5rem, 5vw, 4.5rem)",
            fontWeight: 900,
            margin: "0 0 16px 0",
            lineHeight: 1.1,
            textShadow: "0 2px 10px rgba(0,0,0,0.5)",
          }}
        >
          {featured.title}
        </h1>
        <p
          style={{
            fontSize: "1.1rem",
            color: "#b3b3b3",
            margin: "0 0 32px 0",
            lineHeight: 1.5,
            maxWidth: "90%",
            textShadow: "0 1px 4px rgba(0,0,0,0.8)",
          }}
        >
          {featured.description}
        </p>

        <div style={{ display: "flex", gap: "16px" }}>
          <button
            style={{
              padding: "12px 32px",
              background: "#e50914",
              color: "#fff",
              border: "none",
              borderRadius: "6px",
              fontSize: "1.1rem",
              fontWeight: 700,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              transition: "background 0.2s, transform 0.2s",
            }}
            onMouseOver={(e) => (e.currentTarget.style.background = "#f40612")}
            onMouseOut={(e) => (e.currentTarget.style.background = "#e50914")}
            onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.95)")}
            onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
          >
            <span style={{ fontSize: "1.4rem" }}>▶</span> Play Now
          </button>
          <button
            style={{
              padding: "12px 32px",
              background: "rgba(109, 109, 110, 0.7)",
              color: "#fff",
              border: "none",
              borderRadius: "6px",
              fontSize: "1.1rem",
              fontWeight: 700,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              transition: "background 0.2s, transform 0.2s",
            }}
            onMouseOver={(e) => (e.currentTarget.style.background = "rgba(109, 109, 110, 0.9)")}
            onMouseOut={(e) => (e.currentTarget.style.background = "rgba(109, 109, 110, 0.7)")}
            onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.95)")}
            onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
          >
            <span style={{ fontSize: "1.2rem", fontWeight: "bold" }}>ℹ</span> More Info
          </button>
        </div>
      </div>
    </div>
  );
}
