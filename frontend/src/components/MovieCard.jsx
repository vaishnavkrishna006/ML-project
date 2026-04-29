import { useState } from "react";

export default function MovieCard({ title, poster, genre, imdb_rating }) {
  const [hovered, setHovered] = useState(false);
  const [imgError, setImgError] = useState(false);

  const fallbackPoster = `https://via.placeholder.com/300x450/1a1a1a/e50914?text=${encodeURIComponent(title || "Movie")}`;

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: "relative",
        borderRadius: "10px",
        overflow: "hidden",
        backgroundColor: "#1a1a1a",
        cursor: "pointer",
        transform: hovered ? "scale(1.06)" : "scale(1)",
        transition: "transform 0.3s ease, box-shadow 0.3s ease",
        boxShadow: hovered
          ? "0 0 20px rgba(229,9,20,0.55), 0 8px 32px rgba(0,0,0,0.6)"
          : "0 4px 16px rgba(0,0,0,0.4)",
        minWidth: "160px",
      }}
    >
      {/* Poster */}
      <img
        src={imgError ? fallbackPoster : (poster || fallbackPoster)}
        alt={title}
        onError={() => setImgError(true)}
        style={{
          width: "100%",
          aspectRatio: "2/3",
          objectFit: "cover",
          display: "block",
          transition: "opacity 0.3s",
          opacity: hovered ? 0.7 : 1,
        }}
      />

      {/* Rating Badge */}
      <div
        style={{
          position: "absolute",
          top: "8px",
          right: "8px",
          backgroundColor: "rgba(0,0,0,0.75)",
          color: "#ffd700",
          fontSize: "0.72rem",
          fontWeight: 700,
          padding: "3px 7px",
          borderRadius: "5px",
          backdropFilter: "blur(4px)",
        }}
      >
        ★ {imdb_rating ? Number(imdb_rating).toFixed(1) : "N/A"}
      </div>

      {/* Hover overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "linear-gradient(to top, rgba(0,0,0,0.95) 40%, transparent 100%)",
          opacity: hovered ? 1 : 0,
          transition: "opacity 0.3s ease",
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-end",
          padding: "14px",
          gap: "4px",
        }}
      >
        <h3
          style={{
            color: "#fff",
            fontSize: "0.9rem",
            fontWeight: 700,
            margin: 0,
            lineHeight: 1.3,
            textShadow: "0 1px 4px rgba(0,0,0,0.8)",
          }}
        >
          {title}
        </h3>
        <p
          style={{
            color: "#b3b3b3",
            fontSize: "0.75rem",
            margin: 0,
          }}
        >
          {genre}
        </p>
        <p
          style={{
            color: "#e50914",
            fontSize: "0.75rem",
            fontWeight: 600,
            margin: 0,
          }}
        >
          IMDb ★ {imdb_rating ? Number(imdb_rating).toFixed(1) : "N/A"}
        </p>
        <button
          style={{
            marginTop: "8px",
            padding: "6px 12px",
            backgroundColor: "#e50914",
            border: "none",
            borderRadius: "5px",
            color: "#fff",
            fontSize: "0.78rem",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          ▶ Watch Now
        </button>
      </div>
    </div>
  );
}
