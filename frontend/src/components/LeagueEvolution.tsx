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
  const { members, champion_history } = data;

  if (members.length === 0 && champion_history.length === 0) {
    return (
      <p className="text-gray-500">
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
    <div className="flex gap-6">
      {/* Left: Lineage graph */}
      <div className="flex-1 min-w-0">
        <h3 className="text-sm font-semibold mb-2">Lineage Graph</h3>
        <LineageGraph
          nodes={nodes}
          emptyMessage="No members to display."
        />
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
