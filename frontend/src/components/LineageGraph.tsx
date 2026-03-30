"use client";

import { useState } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LineageNode {
  id: string;
  parent_id: string | null;
  rating: number;
  label?: string;
  cluster?: number | null;
  robustness?: number | null;
  created_at?: string | null;
  notes?: string | null;
}

export interface LineageGraphProps {
  nodes: LineageNode[];
  emptyMessage?: string;
}

// ---------------------------------------------------------------------------
// Label → accent-bar colour
// ---------------------------------------------------------------------------

const LABEL_COLORS: Record<string, string> = {
  Champion: "#14b8a6",
  Dominant: "#f59e0b",
  Aggressive: "#ef4444",
  Consistent: "#22c55e",
  Weak: "#6b7280",
  Competitive: "#8b5cf6",
  Developing: "#6b7280",
};

function accentColor(label?: string): string {
  if (!label) return "#333333";
  return LABEL_COLORS[label] ?? "#333333";
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

interface LayoutNode {
  node: LineageNode;
  children: LayoutNode[];
  depth: number;
  x: number;
  y: number;
}

const NODE_W = 150;
const NODE_H = 56;
const SPACING_X = 180;
const SPACING_Y = 100;
const PAD_TOP = 40;
const PAD_LR = 30;

function buildForest(nodes: LineageNode[]): LayoutNode[] {
  const byId = new Map<string, LineageNode>();
  for (const n of nodes) byId.set(n.id, n);

  const childrenMap = new Map<string, string[]>();
  const roots: string[] = [];

  for (const n of nodes) {
    if (n.parent_id && byId.has(n.parent_id)) {
      const siblings = childrenMap.get(n.parent_id) || [];
      siblings.push(n.id);
      childrenMap.set(n.parent_id, siblings);
    } else {
      roots.push(n.id);
    }
  }

  function build(id: string, depth: number): LayoutNode {
    const node = byId.get(id)!;
    const kids = (childrenMap.get(id) || []).map((cid) => build(cid, depth + 1));
    return { node, children: kids, depth, x: 0, y: 0 };
  }

  return roots.map((id) => build(id, 0));
}

function flattenNodes(forest: LayoutNode[]): LayoutNode[] {
  const all: LayoutNode[] = [];
  function walk(n: LayoutNode) {
    all.push(n);
    n.children.forEach(walk);
  }
  forest.forEach(walk);
  return all;
}

function layoutTree(forest: LayoutNode[]): {
  nodes: LayoutNode[];
  width: number;
  height: number;
} {
  const nodes = flattenNodes(forest);
  if (nodes.length === 0) return { nodes: [], width: 0, height: 0 };

  const byDepth = new Map<number, LayoutNode[]>();
  for (const n of nodes) {
    const group = byDepth.get(n.depth) || [];
    group.push(n);
    byDepth.set(n.depth, group);
  }

  const maxDepth = Math.max(...nodes.map((n) => n.depth));

  let maxWidth = 0;
  for (let d = 0; d <= maxDepth; d++) {
    const group = byDepth.get(d) || [];
    for (let i = 0; i < group.length; i++) {
      group[i].x = PAD_LR + i * SPACING_X;
      group[i].y = PAD_TOP + d * SPACING_Y;
    }
    maxWidth = Math.max(maxWidth, group.length * SPACING_X);
  }

  return {
    nodes,
    width: maxWidth + PAD_LR * 2,
    height: (maxDepth + 1) * SPACING_Y + PAD_TOP * 2,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function LineageGraph({ nodes: inputNodes, emptyMessage }: LineageGraphProps) {
  const [selectedNode, setSelectedNode] = useState<LineageNode | null>(null);

  if (inputNodes.length === 0) {
    return (
      <p style={{ color: "#666666", fontSize: 13 }}>
        {emptyMessage || "No members yet."}
      </p>
    );
  }

  const forest = buildForest(inputNodes);
  const { nodes, width, height } = layoutTree(forest);

  // Build lookup for edges
  const nodeById = new Map<string, LayoutNode>();
  for (const n of nodes) nodeById.set(n.node.id, n);

  // Collect edges
  const edges: { parent: LayoutNode; child: LayoutNode }[] = [];
  for (const n of nodes) {
    if (n.node.parent_id && nodeById.has(n.node.parent_id)) {
      edges.push({ parent: nodeById.get(n.node.parent_id)!, child: n });
    }
  }

  // Legend: only labels present in data
  const presentLabels = new Map<string, string>();
  for (const n of inputNodes) {
    if (n.label && !presentLabels.has(n.label)) {
      presentLabels.set(n.label, accentColor(n.label));
    }
  }

  const svgW = width + 20;
  const svgH = height + 20;

  function handleNodeClick(node: LineageNode) {
    setSelectedNode((prev) => (prev?.id === node.id ? null : node));
  }

  return (
    <div>
      {/* Legend */}
      {presentLabels.size > 0 && (
        <div style={{ marginBottom: 12, display: "flex", flexWrap: "wrap", gap: "8px 16px" }}>
          {Array.from(presentLabels.entries()).map(([label, color]) => (
            <span key={label} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: color,
                  display: "inline-block",
                }}
              />
              <span style={{ fontSize: 11, color: "#666666" }}>{label}</span>
            </span>
          ))}
        </div>
      )}

      {/* SVG + sidebar layout */}
      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        {/* Left: scrollable SVG */}
        <div style={{ flex: 1, minWidth: 0, overflowX: "auto" }}>
          <svg width={svgW} height={svgH} style={{ display: "block" }}>
            {/* Connectors */}
            {edges.map((e, i) => {
              const px = e.parent.x + NODE_W / 2;
              const py = e.parent.y + NODE_H;
              const cx = e.child.x + NODE_W / 2;
              const cy = e.child.y;
              return (
                <path
                  key={i}
                  d={`M ${px} ${py} C ${px} ${py + 40}, ${cx} ${cy - 40}, ${cx} ${cy}`}
                  stroke="#2a2a2a"
                  strokeWidth={1.5}
                  fill="none"
                />
              );
            })}

            {/* Nodes */}
            {nodes.map((n) => {
              const isSelected = selectedNode?.id === n.node.id;
              const accent = accentColor(n.node.label);
              return (
                <g
                  key={n.node.id}
                  onClick={() => handleNodeClick(n.node)}
                  style={{ cursor: "pointer" }}
                >
                  {/* Background rect */}
                  <rect
                    x={n.x}
                    y={n.y}
                    width={NODE_W}
                    height={NODE_H}
                    rx={6}
                    fill="#1a1a1a"
                    stroke={isSelected ? "#14b8a6" : "#2a2a2a"}
                    strokeWidth={isSelected ? 1.5 : 1}
                  />
                  {/* Left accent bar */}
                  <rect
                    x={n.x}
                    y={n.y}
                    width={4}
                    height={NODE_H}
                    rx={0}
                    fill={accent}
                    opacity={isSelected ? 1.0 : 0.8}
                  />
                  {/* Member ID (truncated) */}
                  <text
                    x={n.x + 14}
                    y={n.y + 18}
                    fontSize={10}
                    fill="#888888"
                  >
                    {n.node.id.length > 14 ? n.node.id.slice(0, 14) : n.node.id}
                  </text>
                  {/* Rating */}
                  <text
                    x={n.x + 14}
                    y={n.y + 36}
                    fontSize={14}
                    fontWeight={600}
                    fill="#ededed"
                  >
                    {Math.round(n.node.rating)}
                  </text>
                  {/* Label */}
                  {n.node.label && (
                    <text
                      x={n.x + 14}
                      y={n.y + 50}
                      fontSize={10}
                      fill="#666666"
                    >
                      {n.node.label}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>

        {/* Right: detail panel — always rendered, 220px wide */}
        <div style={{ width: 220, flexShrink: 0, position: "sticky", top: 80 }}>
          {selectedNode ? (
            <div
              style={{
                width: 220,
                background: "#111111",
                border: "1px solid #2a2a2a",
                borderRadius: 8,
                padding: 16,
                fontSize: 13,
              }}
            >
              {/* Header */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: 12, color: "#888888" }}>Details</span>
                <button
                  onClick={() => setSelectedNode(null)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "#666666",
                    cursor: "pointer",
                    fontSize: 16,
                    lineHeight: 1,
                    padding: 0,
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = "#ededed"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = "#666666"; }}
                >
                  &times;
                </button>
              </div>
              {/* Divider */}
              <div style={{ borderTop: "1px solid #1e1e1e", margin: "8px 0" }} />
              {/* Rows */}
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <DetailRow label="Member ID" value={selectedNode.id} mono />
                <DetailRow label="Rating" value={Math.round(selectedNode.rating).toString()} />
                <DetailRow label="Parent" value={selectedNode.parent_id ?? "—"} mono />
                {(selectedNode.cluster != null || selectedNode.robustness != null) && (
                  <>
                    <DetailRow label="Cluster" value={selectedNode.cluster != null ? String(selectedNode.cluster) : "—"} />
                    <DetailRow
                      label="Robustness"
                      value={selectedNode.robustness != null ? selectedNode.robustness.toFixed(3) : "—"}
                    />
                  </>
                )}
                <DetailRow
                  label="Created"
                  value={
                    selectedNode.created_at
                      ? new Date(selectedNode.created_at).toLocaleString()
                      : "—"
                  }
                />
                <DetailRow label="Notes" value={selectedNode.notes ?? "—"} />
              </div>
            </div>
          ) : (
            <div
              style={{
                background: "#111111",
                border: "1px solid #1e1e1e",
                borderRadius: 8,
                padding: "24px 16px",
                textAlign: "center",
              }}
            >
              <div
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: "50%",
                  background: "#1a1a1a",
                  border: "1px solid #2a2a2a",
                  margin: "0 auto 10px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <div
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "#333333",
                  }}
                />
              </div>
              <p style={{ fontSize: 12, color: "#555555", lineHeight: 1.5, margin: 0 }}>
                Click a node to inspect
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function DetailRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "#666666" }}>{label}</div>
      <div
        style={{
          fontSize: 12,
          color: "#ededed",
          fontFamily: mono ? "'JetBrains Mono', 'Fira Code', monospace" : undefined,
        }}
      >
        {value}
      </div>
    </div>
  );
}
