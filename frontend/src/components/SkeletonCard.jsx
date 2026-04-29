/* --------------------------------------------------------------------------
   Skeleton loader – used while data is fetching
-------------------------------------------------------------------------- */
export function SkeletonCard() {
  return (
    <div
      style={{
        minWidth: "160px",
        borderRadius: "10px",
        overflow: "hidden",
        backgroundColor: "#1a1a1a",
        aspectRatio: "2/3",
        background:
          "linear-gradient(90deg, #1a1a1a 25%, #2a2a2a 37%, #1a1a1a 63%)",
        backgroundSize: "800px 100%",
        animation: "shimmer 1.4s infinite linear",
      }}
    />
  );
}
