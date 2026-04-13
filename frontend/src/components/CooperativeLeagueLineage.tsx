"use client";

import { CooperativeLineageMember } from "@/lib/api";

interface Props {
  members: CooperativeLineageMember[];
  onSelect?: (memberId: string) => void;
  selectedId?: string | null;
}

const LABEL_COLORS: Record<string, string> = {
  "Dedicated Specialist": "#14b8a6",
  "Adaptive Generalist": "#3b82f6",
  "Free Rider": "#ef4444",
  "Overcontributor": "#f59e0b",
  "Opportunist": "#8b5cf6",
  Developing: "#6b7280",
};

function labelColor(label: string | undefined): string {
  return LABEL_COLORS[label ?? "Developing"] ?? "#6b7280";
}

export default function CooperativeLeagueLineage({ members, onSelect, selectedId }: Props) {
  if (members.length === 0) {
    return (
      <div style={{ color: "#555555", fontSize: 13, textAlign: "center", padding: "32px 0" }}>
        No league members yet.
      </div>
    );
  }

  const NODE_W = 140;
  const NODE_H = 52;
  const H_GAP = 20;
  const V_GAP = 70;

  // Build tree layout
  const idIndex: Record<string, number> = {};
  members.forEach((m, i) => { idIndex[m.member_id] = i; });

  // Assign rows (depth from root)
  const depth: Record<string, number> = {};
  function getDepth(mid: string): number {
    if (mid in depth) return depth[mid];
    const m = members.find((x) => x.member_id === mid);
    if (!m || !m.parent_id) {
      depth[mid] = 0;
      return 0;
    }
    depth[mid] = getDepth(m.parent_id) + 1;
    return depth[mid];
  }
  members.forEach((m) => getDepth(m.member_id));

  const maxDepth = Math.max(0, ...Object.values(depth));
  const rowGroups: string[][] = Array.from({ length: maxDepth + 1 }, () => []);
  members.forEach((m) => rowGroups[depth[m.member_id]].push(m.member_id));

  // Compute x position per node
  const posX: Record<string, number> = {};
  const posY: Record<string, number> = {};
  let totalW = 0;

  rowGroups.forEach((row, rowIdx) => {
    const rowW = row.length * (NODE_W + H_GAP) - H_GAP;
    totalW = Math.max(totalW, rowW);
    row.forEach((mid, col) => {
      posX[mid] = col * (NODE_W + H_GAP);
      posY[mid] = rowIdx * (NODE_H + V_GAP);
    });
  });

  const svgH = (maxDepth + 1) * (NODE_H + V_GAP) + 20;

  return (
    <div style={{ overflowX: "auto" }}>
      <svg
        width={Math.max(totalW + 20, 300)}
        height={svgH}
        style={{ display: "block" }}
      >
        {/* Draw edges */}
        {members.map((m) => {
          if (!m.parent_id || !(m.parent_id in posX)) return null;
          const cx1 = posX[m.parent_id] + NODE_W / 2;
          const cy1 = posY[m.parent_id] + NODE_H;
          const cx2 = posX[m.member_id] + NODE_W / 2;
          const cy2 = posY[m.member_id];
          const midY = (cy1 + cy2) / 2;
          return (
            <path
              key={`edge-${m.member_id}`}
              d={`M${cx1},${cy1} C${cx1},${midY} ${cx2},${midY} ${cx2},${cy2}`}
              fill="none"
              stroke="#2a2a2a"
              strokeWidth={2}
            />
          );
        })}

        {/* Draw nodes */}
        {members.map((m) => {
          const x = posX[m.member_id];
          const y = posY[m.member_id];
          const color = labelColor(m.label);
          const isSelected = m.member_id === selectedId;
          return (
            <g
              key={m.member_id}
              onClick={() => onSelect?.(m.member_id)}
              style={{ cursor: "pointer" }}
            >
              {/* Card background */}
              <rect
                x={x}
                y={y}
                width={NODE_W}
                height={NODE_H}
                rx={5}
                fill={isSelected ? "#1a2a2a" : "#111111"}
                stroke={isSelected ? color : "#222222"}
                strokeWidth={isSelected ? 2 : 1}
              />
              {/* Left accent border */}
              <rect x={x} y={y} width={3} height={NODE_H} rx={2} fill={color} />

              {/* Member ID */}
              <text
                x={x + 10}
                y={y + 16}
                fontSize={9}
                fill="#666666"
                fontFamily="monospace"
              >
                {m.member_id.length > 16
                  ? `…${m.member_id.slice(-14)}`
                  : m.member_id}
              </text>

              {/* Rating */}
              <text x={x + 10} y={y + 30} fontSize={13} fontWeight={600} fill="#ededed">
                {m.rating.toFixed(1)}
              </text>

              {/* Label */}
              <text
                x={x + NODE_W - 6}
                y={y + 44}
                fontSize={9}
                fill={color}
                textAnchor="end"
              >
                {m.label ?? "Developing"}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
