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
  Competitive: "#3b82f6",
  Developing: "#9ca3af",
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
  return (
    <div
      style={{
        flexShrink: 0,
        width: 180,
        background: "#111111",
        border: "1px solid #1e1e1e",
        borderRadius: 8,
        padding: 14,
        borderTop: `2px solid ${labelColor(entry.label)}`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 8,
        }}
      >
        <span
          style={{ fontSize: 11, color: "#555555", fontFamily: "monospace" }}
        >
          #{idx + 1}
        </span>
        <span
          style={{
            fontSize: 12,
            fontWeight: 500,
            color: labelColor(entry.label),
          }}
        >
          {entry.label}
        </span>
      </div>
      <div
        style={{
          fontSize: 11,
          color: "#888888",
          fontFamily: "monospace",
          marginBottom: 6,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
        title={entry.member_id}
      >
        {entry.member_id}
      </div>
      <div
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: "#ededed",
          marginBottom: 4,
        }}
      >
        {entry.rating.toFixed(1)}
      </div>
      {entry.robustness_score != null && (
        <div style={{ fontSize: 11, color: "#666666" }}>
          Robust: {entry.robustness_score.toFixed(3)}
        </div>
      )}
      {entry.created_at && (
        <div style={{ fontSize: 10, color: "#444444", marginTop: 6 }}>
          {new Date(entry.created_at).toLocaleDateString()}
        </div>
      )}
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
    label: m.label,
    cluster: m.cluster_id,
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

      {/* Bottom: Champion History — full width horizontal scroll */}
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
              display: "flex",
              gap: 12,
              overflowX: "auto",
              paddingBottom: 8,
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
