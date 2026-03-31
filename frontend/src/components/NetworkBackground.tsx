"use client";

import { useMemo } from "react";

interface Node {
  id: number;
  cx: number;
  cy: number;
  dur: number;
  dx: number;
  dy: number;
}

function generateNodes(count: number): Node[] {
  // Deterministic pseudo-random using simple seed
  let seed = 42;
  function rand() {
    seed = (seed * 16807 + 0) % 2147483647;
    return (seed - 1) / 2147483646;
  }

  const nodes: Node[] = [];
  for (let i = 0; i < count; i++) {
    nodes.push({
      id: i,
      cx: 5 + rand() * 90,
      cy: 5 + rand() * 90,
      dur: 8 + rand() * 7,
      dx: (rand() - 0.5) * 2, // ±1 in viewBox units ≈ ±20px at 1000px wide
      dy: (rand() - 0.5) * 2,
    });
  }
  return nodes;
}

function nearestConnections(nodes: Node[]): [number, number][] {
  const edges: [number, number][] = [];
  const added = new Set<string>();

  for (const node of nodes) {
    const sorted = [...nodes]
      .filter((n) => n.id !== node.id)
      .sort((a, b) => {
        const da = (a.cx - node.cx) ** 2 + (a.cy - node.cy) ** 2;
        const db = (b.cx - node.cx) ** 2 + (b.cy - node.cy) ** 2;
        return da - db;
      });

    const count = 2 + (node.id % 2); // 2 or 3
    for (let i = 0; i < count && i < sorted.length; i++) {
      const key = [Math.min(node.id, sorted[i].id), Math.max(node.id, sorted[i].id)].join("-");
      if (!added.has(key)) {
        added.add(key);
        edges.push([node.id, sorted[i].id]);
      }
    }
  }
  return edges;
}

export default function NetworkBackground() {
  const nodes = useMemo(() => generateNodes(20), []);
  const edges = useMemo(() => nearestConnections(nodes), [nodes]);

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 0,
      }}
    >
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        width="100%"
        height="100%"
        style={{ display: "block" }}
      >
        <style>{`
          @keyframes float {
            0%, 100% { translate: 0 0; }
            50% { translate: var(--dx) var(--dy); }
          }
        `}</style>

        {edges.map(([a, b]) => (
          <line
            key={`${a}-${b}`}
            x1={nodes[a].cx}
            y1={nodes[a].cy}
            x2={nodes[b].cx}
            y2={nodes[b].cy}
            stroke="#14b8a6"
            strokeWidth={0.15}
            opacity={0.08}
          />
        ))}

        {nodes.map((n) => (
          <circle
            key={n.id}
            cx={n.cx}
            cy={n.cy}
            r={0.3}
            fill="#14b8a6"
            opacity={0.3}
            style={{
              "--dx": `${n.dx}px`,
              "--dy": `${n.dy}px`,
              animation: `float ${n.dur}s ease-in-out infinite alternate`,
            } as React.CSSProperties}
          />
        ))}
      </svg>
    </div>
  );
}
