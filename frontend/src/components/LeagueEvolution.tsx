"use client";

import {
  ChampionHistoryEntry,
  LeagueEvolutionResponse,
} from "@/lib/api";
import LineageGraph from "./LineageGraph";

interface Props {
  data: LeagueEvolutionResponse;
}

const LABEL_COLOR: Record<string, string> = {
  Champion: "#f59e0b",
  Dominant: "#f59e0b",
  Aggressive: "#ef4444",
  Consistent: "#22c55e",
  Weak: "#6b7280",
  Competitive: "#8b5cf6",
  Developing: "#6b7280",
  Unstable: "#6b7280",
  Exploitative: "#ef4444",
  Cooperative: "#22c55e",
  Robust: "#14b8a6",
};

function labelColor(label: string): string {
  return LABEL_COLOR[label] ?? "#9ca3af";
}

function ChampionHistoryCard({
  entry,
  idx,
}: {
  entry: ChampionHistoryEntry;
  idx: number;
}) {
  const color = labelColor(entry.label);
  return (
    <div
      style={{
        background: "#111111",
        border: "1px solid #1e1e1e",
        borderLeft: `3px solid ${color}`,
        borderRadius: 6,
        padding: "10px 12px",
      }}
    >
      {/* Top row: index + label + cluster on left, rating on right */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 4,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              fontSize: 11,
              color: "#555555",
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            #{idx + 1}
          </span>
          <span
            style={{ fontSize: 12, fontWeight: 500, color }}
          >
            {entry.label}
          </span>
          {entry.cluster_id != null && (
            <span style={{ fontSize: 11, color: "#444444" }}>
              cluster {entry.cluster_id}
            </span>
          )}
        </div>
        <span
          style={{ fontSize: 13, fontWeight: 600, color: "#ededed" }}
        >
          {entry.rating.toFixed(1)}
        </span>
      </div>

      {/* Member ID */}
      <div
        style={{
          fontSize: 11,
          color: "#666666",
          fontFamily: "'JetBrains Mono', monospace",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          marginBottom: 2,
        }}
        title={entry.member_id}
      >
        {entry.member_id}
      </div>

      {/* Robustness + date row */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        {entry.robustness_score != null ? (
          <span style={{ fontSize: 11, color: "#555555" }}>
            Robust: {entry.robustness_score.toFixed(3)}
          </span>
        ) : (
          <span />
        )}
        {entry.created_at && (
          <span style={{ fontSize: 10, color: "#444444" }}>
            {new Date(entry.created_at).toLocaleDateString()}
          </span>
        )}
      </div>
    </div>
  );
}

export default function LeagueEvolution({ data }: Props) {
  const { members, champion_history } = data;

  if (members.length === 0 && champion_history.length === 0) {
    return (
      <p style={{ color: "#666666", fontSize: 13 }}>
        No evolution data yet. Train and save snapshots to build history.
      </p>
    );
  }

  const nodes = members.map((m) => ({
    id: m.member_id,
    parent_id: m.parent_id,
    rating: m.rating,
    label: m.label ?? m.strategy?.label,
    cluster: m.cluster_id ?? m.strategy?.cluster_id,
    robustness: m.robustness_score,
    created_at: m.created_at,
    notes: m.notes,
  }));

  return (
    <div>
      {/* Top: LineageGraph — full width, has its own built-in sidebar */}
      <div style={{ marginBottom: 32 }}>
        <LineageGraph nodes={nodes} emptyMessage="No members to display." />
      </div>

      {/* Bottom: Champion History — vertical scrollable list */}
      <div>
        <h3
          style={{
            fontSize: 13,
            fontWeight: 500,
            color: "#888888",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            marginBottom: 16,
          }}
        >
          Champion History
        </h3>
        {champion_history.length === 0 ? (
          <p style={{ color: "#666666", fontSize: 13 }}>
            No champion history yet.
          </p>
        ) : (
          <div
            style={{
              maxHeight: 400,
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              gap: 8,
              paddingRight: 4,
              scrollbarWidth: "thin",
              scrollbarColor: "#2a2a2a transparent",
            }}
          >
            {champion_history.map((entry, idx) => (
              <ChampionHistoryCard
                key={entry.member_id}
                entry={entry}
                idx={idx}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
