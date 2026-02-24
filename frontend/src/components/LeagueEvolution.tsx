"use client";

import { useState } from "react";
import {
  LeagueEvolutionMember,
  ChampionHistoryEntry,
  LeagueEvolutionResponse,
} from "@/lib/api";

interface Props {
  data: LeagueEvolutionResponse;
}

const LABEL_COLOR: Record<string, string> = {
  Champion: "#f59e0b",
  Competitive: "#3b82f6",
  Developing: "#9ca3af",
};

function labelColor(label: string): string {
  return LABEL_COLOR[label] ?? "#9ca3af";
}

interface TreeNode {
  member: LeagueEvolutionMember;
  children: TreeNode[];
  depth: number;
  x: number;
  y: number;
}

function buildForest(members: LeagueEvolutionMember[]): TreeNode[] {
  const byId = new Map<string, LeagueEvolutionMember>();
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

function TimelineEntry({
  entry,
  idx,
}: {
  entry: ChampionHistoryEntry;
  idx: number;
}) {
  return (
    <li className="border border-gray-200 rounded p-2 text-xs">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-gray-400 font-mono">#{idx + 1}</span>
        <span
          className="font-medium"
          style={{ color: labelColor(entry.label) }}
        >
          {entry.label}
        </span>
        {entry.cluster_id != null && (
          <span className="text-gray-400">cluster {entry.cluster_id}</span>
        )}
      </div>
      <div
        className="font-mono text-gray-700 mb-1 truncate"
        title={entry.member_id}
      >
        {entry.member_id}
      </div>
      <div className="flex gap-3 text-gray-600">
        <span>
          Rating:{" "}
          <span className="font-medium">{entry.rating.toFixed(1)}</span>
        </span>
        {entry.robustness_score != null && (
          <span>
            Robust:{" "}
            <span className="font-medium">
              {entry.robustness_score.toFixed(3)}
            </span>
          </span>
        )}
      </div>
      {entry.created_at && (
        <div className="text-gray-400 mt-1">
          {new Date(entry.created_at).toLocaleString()}
        </div>
      )}
    </li>
  );
}

export default function LeagueEvolution({ data }: Props) {
  const [selected, setSelected] = useState<LeagueEvolutionMember | null>(null);
  const { members, champion_history } = data;

  if (members.length === 0 && champion_history.length === 0) {
    return (
      <p className="text-gray-500">
        No evolution data yet. Train and save snapshots to build history.
      </p>
    );
  }

  const forest = buildForest(members);
  const { nodes, width, height } = layoutTree(forest);

  const ratings = members.map((m) => m.rating);
  const minR = ratings.length > 0 ? Math.min(...ratings) : 0;
  const maxR = ratings.length > 0 ? Math.max(...ratings) : 1;
  const rangeR = maxR - minR || 1;

  function nodeRadius(rating: number) {
    return 10 + ((rating - minR) / rangeR) * 10;
  }

  function strokeThickness(rating: number) {
    return 1 + ((rating - minR) / rangeR) * 2;
  }

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
    <div className="flex gap-6">
      {/* Left: SVG lineage graph */}
      <div className="flex-1 min-w-0">
        <h3 className="text-sm font-semibold mb-2">Lineage Graph</h3>
        {members.length === 0 ? (
          <p className="text-gray-500 text-sm">No members to display.</p>
        ) : (
          <>
            {/* Legend */}
            <div className="flex gap-4 mb-2 text-xs">
              {Object.entries(LABEL_COLOR).map(([label, color]) => (
                <span key={label} className="flex items-center gap-1">
                  <span
                    style={{
                      background: color,
                      display: "inline-block",
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                    }}
                  />
                  {label}
                </span>
              ))}
            </div>

            <div className="overflow-auto border border-gray-200 rounded">
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
                  const sw = strokeThickness(n.member.rating);
                  const isSelected =
                    selected?.member_id === n.member.member_id;
                  const fill = labelColor(n.member.label);
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
                        fill={fill}
                        stroke={isSelected ? "#000" : "#6b7280"}
                        strokeWidth={isSelected ? 2.5 : sw}
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
          </>
        )}

        {/* Detail panel */}
        {selected && (
          <div className="border border-gray-200 rounded p-3 text-sm mt-3">
            <h4 className="font-bold mb-2">Details</h4>
            <dl className="space-y-1">
              <dt className="text-gray-500">Member ID</dt>
              <dd className="font-mono text-xs">{selected.member_id}</dd>
              <dt className="text-gray-500">Label</dt>
              <dd>
                <span
                  className="font-medium"
                  style={{ color: labelColor(selected.label) }}
                >
                  {selected.label}
                </span>
              </dd>
              <dt className="text-gray-500">Rating</dt>
              <dd>{selected.rating.toFixed(1)}</dd>
              <dt className="text-gray-500">Parent</dt>
              <dd className="font-mono text-xs">
                {selected.parent_id ?? "none"}
              </dd>
              <dt className="text-gray-500">Cluster</dt>
              <dd>{selected.cluster_id ?? "—"}</dd>
              <dt className="text-gray-500">Robustness</dt>
              <dd>
                {selected.robustness_score != null
                  ? selected.robustness_score.toFixed(3)
                  : "—"}
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

      {/* Right: Champion history timeline */}
      <div className="w-72 flex-shrink-0">
        <h3 className="text-sm font-semibold mb-2">Champion History</h3>
        {champion_history.length === 0 ? (
          <p className="text-gray-500 text-sm">No champion history yet.</p>
        ) : (
          <ol className="space-y-2">
            {champion_history.map((entry, idx) => (
              <TimelineEntry key={entry.member_id} entry={entry} idx={idx} />
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}
