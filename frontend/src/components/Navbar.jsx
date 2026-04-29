import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function Navbar() {
  const navigate = useNavigate();
  const [term, setTerm] = useState("");
  const [focused, setFocused] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (term.trim()) navigate(`/search?q=${encodeURIComponent(term.trim())}`);
  };

  return (
    <nav
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 32px",
        backgroundColor: "rgba(15,15,15,0.95)",
        backdropFilter: "blur(12px)",
        position: "sticky",
        top: 0,
        zIndex: 100,
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {/* Logo */}
      <Link
        to="/"
        style={{
          fontSize: "1.6rem",
          fontWeight: 900,
          color: "#e50914",
          textDecoration: "none",
          letterSpacing: "-0.5px",
          flexShrink: 0,
        }}
      >
        Movie<span style={{ color: "#fff" }}>AI</span>
      </Link>

      {/* Search Bar */}
      <form
        onSubmit={handleSubmit}
        style={{
          flex: 1,
          maxWidth: "520px",
          margin: "0 24px",
          display: "flex",
          alignItems: "center",
          borderRadius: "8px",
          border: `1px solid ${focused ? "#e50914" : "rgba(255,255,255,0.12)"}`,
          background: "rgba(255,255,255,0.05)",
          transition: "border 0.2s",
          overflow: "hidden",
        }}
      >
        <span style={{ padding: "0 12px", color: "#b3b3b3", fontSize: "1rem" }}>🔍</span>
        <input
          type="text"
          placeholder="Search movies, titles, genres…"
          value={term}
          onChange={(e) => setTerm(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          style={{
            flex: 1,
            background: "transparent",
            border: "none",
            outline: "none",
            color: "#fff",
            fontSize: "0.9rem",
            padding: "10px 0",
          }}
        />
        {term && (
          <button
            type="button"
            onClick={() => setTerm("")}
            style={{
              padding: "0 12px",
              background: "none",
              border: "none",
              color: "#b3b3b3",
              cursor: "pointer",
              fontSize: "1rem",
            }}
          >
            ✕
          </button>
        )}
        <button
          type="submit"
          style={{
            padding: "10px 16px",
            background: "#e50914",
            border: "none",
            color: "#fff",
            cursor: "pointer",
            fontWeight: 600,
            fontSize: "0.85rem",
            transition: "background 0.2s",
          }}
        >
          Search
        </button>
      </form>

      {/* Profile */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px", flexShrink: 0 }}>
        <img
          src="https://api.dicebear.com/7.x/avataaars/svg?seed=MovieAI"
          alt="Profile"
          style={{
            width: "38px",
            height: "38px",
            borderRadius: "50%",
            border: "2px solid #e50914",
            background: "#1a1a1a",
          }}
        />
      </div>
    </nav>
  );
}
