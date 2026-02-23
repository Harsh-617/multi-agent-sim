"use client";

import { useState } from "react";
import { LineageMember } from "@/lib/api";

interface Props {
  members: LineageMember[];
}

interface TreeNode {
  member: LineageMember;
  children: TreeNode[];
  depth: number;
  x: number;
  y: number;
}

function buildForest(members: LineageMember[]): TreeNode[] {
  const byId = new Map<string, LineageMember>();
  for (const m of members) byId.set(m.member_id, m);

  const childrenMap = new Map<string, string[]>();
  const roots: string[] = [];

  for (const m of members) {
    if (m.parent_id && byId.has(m.parent_id)) {
      const siblings = childrenMap.get(m.parent_id) || [];
      siblings.push(m.member_id);
      childrenMap.set(m.parent_id, siblings);
    } else {
      roots.push(m.member_id);
    }
  }

  function build(id: string, depth: number): TreeNode {
    const member = byId.get(id)!;
    const kids = (childrenMap.get(id) || []).map((cid) =>
      build(cid, depth + 1)
    );
    return { member, children: kids, depth, x: 0, y: 0 };
  }

  return roots.map((id) => build(id, 0));
}

function flattenNodes(forest: TreeNode[]): TreeNode[] {
  const all: TreeNode[] = [];
  function walk(n: TreeNode) {
    all.push(n);
    n.children.forEach(walk);
  }
  forest.forEach(walk);
  return all;
}

function layoutTree(forest: TreeNode[]): {
  nodes: TreeNode[];
  width: number;
  height: number;
} {
  const nodes = flattenNodes(forest);
  if (nodes.length === 0) return { nodes: [], width: 0, height: 0 };

  // Group by depth
  const byDepth = new Map<number, TreeNode[]>();
  for (const n of nodes) {
    const group = byDepth.get(n.depth) || [];
    group.push(n);
    byDepth.set(n.depth, group);
  }

  const maxDepth = Math.max(...nodes.map((n) => n.depth));
  const nodeSpacingX = 120;
  const nodeSpacingY = 80;
  const padX = 60;
  const padY = 40;

  let maxWidth = 0;
  for (let d = 0; d <= maxDepth; d++) {
    const group = byDepth.get(d) || [];
    for (let i = 0; i < group.length; i++) {
      group[i].x = padX + i * nodeSpacingX;
      group[i].y = padY + d * nodeSpacingY;
    }
    maxWidth = Math.max(maxWidth, group.length * nodeSpacingX);
  }

  return {
    nodes,
    width: maxWidth + padX * 2,
    height: (maxDepth + 1) * nodeSpacingY + padY * 2,
  };
}

export default function LeagueLineage({ members }: Props) {
  const [selected, setSelected] = useState<LineageMember | null>(null);

  if (members.length === 0) {
    return (
      <p className="text-gray-500">
        No members to display. Save snapshots to build a lineage.
      </p>
    );
  }

  const forest = buildForest(members);
  const { nodes, width, height } = layoutTree(forest);

  // Rating range for sizing
  const ratings = members.map((m) => m.rating);
  const minR = Math.min(...ratings);
  const maxR = Math.max(...ratings);
  const rangeR = maxR - minR || 1;

  function nodeRadius(rating: number) {
    return 10 + ((rating - minR) / rangeR) * 12;
  }

  function nodeColor(rating: number) {
    const t = (rating - minR) / rangeR;
    // Lerp from blue (#3b82f6) to green (#22c55e)
    const r = Math.round(59 + t * (34 - 59));
    const g = Math.round(130 + t * (197 - 130));
    const b = Math.round(246 + t * (94 - 246));
    return `rgb(${r},${g},${b})`;
  }

  // Build parent lookup for edges
  const nodeById = new Map<string, TreeNode>();
  for (const n of nodes) nodeById.set(n.member.member_id, n);

  const edges: { x1: number; y1: number; x2: number; y2: number }[] = [];
  for (const n of nodes) {
    if (n.member.parent_id && nodeById.has(n.member.parent_id)) {
      const parent = nodeById.get(n.member.parent_id)!;
      edges.push({ x1: parent.x, y1: parent.y, x2: n.x, y2: n.y });
    }
  }

  const suffix = (id: string) => id.slice(-6);

  return (
    <div className="flex gap-4">
      <div className="overflow-auto border border-gray-200 rounded" style={{ maxWidth: "100%" }}>
        <svg
          width={Math.max(width, 200)}
          height={Math.max(height, 120)}
          className="block"
        >
          {/* Edges */}
          {edges.map((e, i) => (
            <line
              key={i}
              x1={e.x1}
              y1={e.y1}
              x2={e.x2}
              y2={e.y2}
              stroke="#9ca3af"
              strokeWidth={1.5}
            />
          ))}
          {/* Nodes */}
          {nodes.map((n) => {
            const r = nodeRadius(n.member.rating);
            const isSelected =
              selected?.member_id === n.member.member_id;
            return (
              <g
                key={n.member.member_id}
                onClick={() => setSelected(n.member)}
                className="cursor-pointer"
              >
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={r}
                  fill={nodeColor(n.member.rating)}
                  stroke={isSelected ? "#000" : "#6b7280"}
                  strokeWidth={isSelected ? 2.5 : 1}
                  opacity={0.9}
                />
                <text
                  x={n.x}
                  y={n.y + r + 14}
                  textAnchor="middle"
                  fontSize={10}
                  fill="currentColor"
                >
                  {suffix(n.member.member_id)}
                </text>
                <text
                  x={n.x}
                  y={n.y + 4}
                  textAnchor="middle"
                  fontSize={9}
                  fill="#fff"
                  fontWeight="bold"
                >
                  {n.member.rating.toFixed(0)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Detail panel */}
      {selected && (
        <div className="border border-gray-200 rounded p-3 text-sm min-w-[200px]">
          <h4 className="font-bold mb-2">Details</h4>
          <dl className="space-y-1">
            <dt className="text-gray-500">Member ID</dt>
            <dd className="font-mono text-xs">{selected.member_id}</dd>
            <dt className="text-gray-500">Rating</dt>
            <dd>{selected.rating.toFixed(1)}</dd>
            <dt className="text-gray-500">Parent</dt>
            <dd className="font-mono text-xs">
              {selected.parent_id ?? "none"}
            </dd>
            <dt className="text-gray-500">Created</dt>
            <dd className="text-xs">
              {selected.created_at
                ? new Date(selected.created_at).toLocaleString()
                : "—"}
            </dd>
            <dt className="text-gray-500">Notes</dt>
            <dd className="text-xs">{selected.notes ?? "—"}</dd>
          </dl>
          <button
            onClick={() => setSelected(null)}
            className="mt-2 text-xs text-blue-500 hover:underline"
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}
