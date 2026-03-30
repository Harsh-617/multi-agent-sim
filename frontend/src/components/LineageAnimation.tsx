"use client";

const NODES = [
  { id: "n1", x: 300, y: 60,  label: "Cooperative", short: "Coop",  color: "#22c55e", delay: 0,   rating: 929 },
  { id: "n2", x: 160, y: 160, label: "Robust",      short: "Rob",   color: "#14b8a6", delay: 0.8, rating: 997 },
  { id: "n3", x: 440, y: 160, label: "Aggressive",   short: "Aggr",  color: "#ef4444", delay: 0.8, rating: 1025 },
  { id: "n4", x: 80,  y: 260, label: "Unstable",     short: "Unst",  color: "#6b7280", delay: 1.6, rating: 855 },
  { id: "n5", x: 240, y: 260, label: "Cooperative",  short: "Coop",  color: "#22c55e", delay: 1.6, rating: 1071 },
  { id: "n6", x: 380, y: 260, label: "Robust",       short: "Rob",   color: "#14b8a6", delay: 2.4, rating: 1147 },
  { id: "n7", x: 300, y: 360, label: "Champion",     short: "Chmp",  color: "#f59e0b", delay: 3.2, rating: 1315 },
];

const EDGES = [
  { from: "n1", to: "n2" },
  { from: "n1", to: "n3" },
  { from: "n2", to: "n4" },
  { from: "n2", to: "n5" },
  { from: "n3", to: "n6" },
  { from: "n5", to: "n7" },
  { from: "n6", to: "n7" },
];

const nodeMap = Object.fromEntries(NODES.map((n) => [n.id, n]));

const cssKeyframes = `
@keyframes fadeIn {
  from { opacity: 0; transform: scale(0.8); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes drawLine {
  from { stroke-dashoffset: 200; opacity: 0; }
  to   { stroke-dashoffset: 0;   opacity: 0.3; }
}
`;

export default function LineageAnimation() {
  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        opacity: 0.25,
        pointerEvents: "none",
        zIndex: 0,
      }}
    >
      <svg
        viewBox="0 0 600 420"
        width="100%"
        preserveAspectRatio="xMidYMid meet"
        style={{ position: "absolute", top: "50%", transform: "translateY(-50%)" }}
      >
        <style>{cssKeyframes}</style>

        {/* Edges */}
        {EDGES.map((edge) => {
          const p = nodeMap[edge.from];
          const c = nodeMap[edge.to];
          const d = `M ${p.x} ${p.y} C ${p.x} ${p.y + 40} ${c.x} ${c.y - 40} ${c.x} ${c.y}`;
          return (
            <path
              key={`${edge.from}-${edge.to}`}
              d={d}
              stroke={p.color}
              strokeWidth={1}
              strokeOpacity={0.25}
              fill="none"
              strokeDasharray={200}
              strokeDashoffset={200}
              style={{
                animation: `drawLine 0.6s ease forwards`,
                animationDelay: `${c.delay}s`,
              }}
            />
          );
        })}

        {/* Nodes */}
        {NODES.map((node) => (
          <g
            key={node.id}
            style={{
              animation: `fadeIn 5s ease-in-out infinite alternate`,
              animationDelay: `${node.delay}s`,
              opacity: 0,
              transformOrigin: `${node.x}px ${node.y}px`,
            }}
          >
            <circle
              cx={node.x}
              cy={node.y}
              r={18}
              fill={node.color}
              fillOpacity={0.15}
              stroke={node.color}
              strokeWidth={1}
              strokeOpacity={0.4}
            />
            <text
              x={node.x}
              y={node.y}
              dy={4}
              fontSize={9}
              fill={node.color}
              textAnchor="middle"
            >
              {node.short}
            </text>
            <text
              x={node.x}
              y={node.y}
              dy={18}
              fontSize={8}
              fill="#444444"
              textAnchor="middle"
            >
              {node.rating}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
