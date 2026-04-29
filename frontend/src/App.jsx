import { Routes, Route, Navigate } from "react-router-dom";
import Home from "./pages/Home.jsx";
import Search from "./pages/Search.jsx";

export default function App() {
  return (
    <div style={{ backgroundColor: "#0f0f0f", minHeight: "100vh" }}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/search" element={<Search />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
