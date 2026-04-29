import Navbar from "../components/Navbar.jsx";
import Hero from "../components/Hero.jsx";
import Carousel from "../components/Carousel.jsx";

export default function Home() {
  return (
    <div style={{ paddingBottom: "60px" }}>
      <Navbar />
      <Hero />
      <div style={{ marginTop: "-80px", position: "relative", zIndex: 20 }}>
        <Carousel title="Trending Now" userId="trending" />
        <Carousel title="Top Picks for You" userId="1" />
        <Carousel title="Action & Adventure" userId="action" />
      </div>
    </div>
  );
}
