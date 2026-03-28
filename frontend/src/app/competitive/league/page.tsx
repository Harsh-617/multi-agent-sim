"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  CompetitiveLeagueMember,
  CompetitiveLeagueRating,
  CompetitiveLineageMember,
  CompetitiveEvolutionMember,
  CompetitiveEvolutionResponse,
  CompetitiveChampionBenchmarkResponse,
  CompetitiveChampionBenchmarkResult,
  CompetitiveChampionRobustnessRequest,
  CompetitiveChampionInfo,
  ChampionHistoryEntry,
  ConfigListItem,
  getCompetitiveLeagueMembers,
  getCompetitiveLeagueRatings,
  getCompetitiveLeagueLineage,
  getCompetitiveLeagueEvolution,
  getCompetitiveChampion,
  recomputeCompetitiveLeagueRatings,
  runCompetitiveChampionBenchmark,
  runCompetitiveChampionRobustness,
  startCompetitiveLeagueMemberRun,
  listConfigs,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Constants & helpers
// ---------------------------------------------------------------------------

type Tab = "ratings" | "lineage" | "champion" | "evolution";

const COMP_LABEL_COLOR: Record<string, string> = {
  Dominant: "#f59e0b",
  Aggressive: "#ef4444",
  Consistent: "#22c55e",
  Weak: "#9ca3af",
  Competitive: "#3b82f6",
};

function compLabelColor(label: string): string {
  return COMP_LABEL_COLOR[label] ?? "#9ca3af";
}

// ---------------------------------------------------------------------------
// Lineage SVG helpers (follows LeagueLineage / LeagueEvolution pattern)
// ---------------------------------------------------------------------------

interface LineageNode {
  id: string;
  parent_id: string | null;
  rating: number;
  label: string;
  created_at: string | null;
  notes: string | null;
  children: LineageNode[];
  depth: number;
  x: number;
  y: number;
}

function buildLineageForest(
  members: CompetitiveEvolutionMember[],
): LineageNode[] {
  const byId = new Map<string, CompetitiveEvolutionMember>();
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

  function build(id: string, depth: number): LineageNode {
    const m = byId.get(id)!;
    const kids = (childrenMap.get(id) || []).map((cid) =>
      build(cid, depth + 1),
    );
    return {
      id: m.member_id,
      parent_id: m.parent_id,
      rating: m.rating,
      label: m.strategy.label,
      created_at: m.created_at,
      notes: m.notes,
      children: kids,
      depth,
      x: 0,
      y: 0,
    };
  }

  return roots.map((id) => build(id, 0));
}

function flattenLineageNodes(forest: LineageNode[]): LineageNode[] {
  const all: LineageNode[] = [];
  function walk(n: LineageNode) {
    all.push(n);
    n.children.forEach(walk);
  }
  forest.forEach(walk);
  return all;
}

function layoutLineageTree(forest: LineageNode[]): {
  nodes: LineageNode[];
  width: number;
  height: number;
} {
  const nodes = flattenLineageNodes(forest);
  if (nodes.length === 0) return { nodes: [], width: 0, height: 0 };

  const byDepth = new Map<number, LineageNode[]>();
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

// ---------------------------------------------------------------------------
// Bar chart (follows ChampionBenchmark BarChart pattern)
// ---------------------------------------------------------------------------

const BAR_COLORS: Record<string, string> = {
  league_champion: "#8b5cf6",
  random: "#6b7280",
  always_attack: "#ef4444",
  always_build: "#22c55e",
  always_defend: "#3b82f6",
  competitive_ppo: "#f59e0b",
};

function CompetitiveBarChart({
  results,
}: {
  results: CompetitiveChampionBenchmarkResult[];
}) {
  if (results.length === 0) return null;

  const maxVal = Math.max(
    ...results.map((r) => Math.abs(r.mean_total_reward)),
    0.01,
  );
  const barW = 50;
  const barGap = 12;
  const chartH = 140;
  const chartW = results.length * (barW + barGap);
  const labelH = 48;

  return (
    <svg width={chartW} height={chartH + labelH} className="block">
      {results.map((r, i) => {
        const h = (Math.abs(r.mean_total_reward) / maxVal) * chartH;
        const x = i * (barW + barGap);
        const color = BAR_COLORS[r.policy] || "#6b7280";
        return (
          <g key={r.policy}>
            <rect
              x={x}
              y={chartH - h}
              width={barW}
              height={h}
              fill={color}
              rx={3}
            />
            <text
              x={x + barW / 2}
              y={chartH - h - 4}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
            >
              {r.mean_total_reward.toFixed(2)}
            </text>
            <text
              x={x + barW / 2}
              y={chartH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
            >
              {r.policy.replace("_", " ")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// SVG graph component (shared by Lineage and Evolution tabs)
// ---------------------------------------------------------------------------

function LineageSVG({
  members,
  onSelect,
  selectedId,
}: {
  members: CompetitiveEvolutionMember[];
  onSelect: (m: CompetitiveEvolutionMember | null) => void;
  selectedId: string | null;
}) {
  if (members.length === 0) {
    return <p className="text-gray-500 text-sm">No members to display.</p>;
  }

  const forest = buildLineageForest(members);
  const { nodes, width, height } = layoutLineageTree(forest);

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

  const nodeById = new Map<string, LineageNode>();
  for (const n of nodes) nodeById.set(n.id, n);

  const edges: { x1: number; y1: number; x2: number; y2: number }[] = [];
  for (const n of nodes) {
    if (n.parent_id && nodeById.has(n.parent_id)) {
      const parent = nodeById.get(n.parent_id)!;
      edges.push({ x1: parent.x, y1: parent.y, x2: n.x, y2: n.y });
    }
  }

  const memberById = new Map<string, CompetitiveEvolutionMember>();
  for (const m of members) memberById.set(m.member_id, m);

  const suffix = (id: string) => id.slice(-6);

  return (
    <>
      {/* Legend */}
      <div className="flex gap-4 mb-2 text-xs">
        {Object.entries(COMP_LABEL_COLOR).map(([label, color]) => (
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
          {nodes.map((n) => {
            const r = nodeRadius(n.rating);
            const sw = strokeThickness(n.rating);
            const isSelected = selectedId === n.id;
            const fill = compLabelColor(n.label);
            return (
              <g
                key={n.id}
                onClick={() => onSelect(memberById.get(n.id) ?? null)}
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
                  {suffix(n.id)}
                </text>
                <text
                  x={n.x}
                  y={n.y + 4}
                  textAnchor="middle"
                  fontSize={9}
                  fill="#fff"
                  fontWeight="bold"
                >
                  {n.rating.toFixed(0)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Timeline entry (follows LeagueEvolution TimelineEntry pattern)
// ---------------------------------------------------------------------------

function CompetitiveTimelineEntry({
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
          style={{ color: compLabelColor(entry.label) }}
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

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function CompetitiveLeaguePage() {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("ratings");
  const [members, setMembers] = useState<CompetitiveLeagueMember[]>([]);
  const [ratings, setRatings] = useState<Map<string, number>>(new Map());
  const [configs, setConfigs] = useState<ConfigListItem[]>([]);
  const [evolutionData, setEvolutionData] =
    useState<CompetitiveEvolutionResponse>({
      members: [],
      champion_history: [],
    });
  const [loading, setLoading] = useState(true);
  const [recomputing, setRecomputing] = useState(false);
  const [startingId, setStartingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Champion tab state
  const [benchConfigId, setBenchConfigId] = useState("");
  const [benchEpisodes, setBenchEpisodes] = useState(5);
  const [benchRunning, setBenchRunning] = useState(false);
  const [benchData, setBenchData] =
    useState<CompetitiveChampionBenchmarkResponse | null>(null);
  const [robConfigId, setRobConfigId] = useState("default");
  const [robSeeds, setRobSeeds] = useState(3);
  const [robEpisodesPerSeed, setRobEpisodesPerSeed] = useState(2);
  const [robLimitSweeps, setRobLimitSweeps] = useState<string>("");
  const [robSeed, setRobSeed] = useState(42);
  const [robRunning, setRobRunning] = useState(false);

  // Lineage / evolution selection
  const [selectedEvolution, setSelectedEvolution] =
    useState<CompetitiveEvolutionMember | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [m, r, c, evo] = await Promise.all([
        getCompetitiveLeagueMembers(),
        getCompetitiveLeagueRatings(),
        listConfigs(),
        getCompetitiveLeagueEvolution(),
      ]);
      setMembers(m);
      setRatings(new Map(r.map((x) => [x.member_id, x.rating])));
      setConfigs(c);
      setEvolutionData(evo);
      if (c.length > 0 && !benchConfigId) {
        setBenchConfigId(c[0].config_id);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleRecompute() {
    setRecomputing(true);
    setError(null);
    try {
      const r = await recomputeCompetitiveLeagueRatings();
      setRatings(new Map(r.map((x) => [x.member_id, x.rating])));
      const evo = await getCompetitiveLeagueEvolution();
      setEvolutionData(evo);
    } catch (e) {
      setError(String(e));
    } finally {
      setRecomputing(false);
    }
  }

  async function handleRun(memberId: string) {
    if (configs.length === 0) {
      setError("No configs available. Create one on the home page first.");
      return;
    }
    setStartingId(memberId);
    setError(null);
    try {
      const { run_id } = await startCompetitiveLeagueMemberRun(
        configs[0].config_id,
        memberId,
      );
      router.push(`/run/${run_id}`);
    } catch (e) {
      setError(String(e));
      setStartingId(null);
    }
  }

  async function handleBenchmark() {
    if (!benchConfigId) {
      setError("Select a config first.");
      return;
    }
    setBenchRunning(true);
    setError(null);
    setBenchData(null);
    try {
      const resp = await runCompetitiveChampionBenchmark(
        benchConfigId,
        benchEpisodes,
      );
      setBenchData(resp);
    } catch (e) {
      setError(String(e));
    } finally {
      setBenchRunning(false);
    }
  }

  async function handleRobustness() {
    setRobRunning(true);
    setError(null);
    try {
      const payload: CompetitiveChampionRobustnessRequest = {
        config_id: robConfigId,
        seeds: robSeeds,
        episodes_per_seed: robEpisodesPerSeed,
        seed: robSeed,
        ...(robLimitSweeps !== ""
          ? { limit_sweeps: Number(robLimitSweeps) }
          : {}),
      };
      const resp = await runCompetitiveChampionRobustness(payload);
      router.push(`/competitive/reports/${resp.report_id}`);
    } catch (e) {
      setError(String(e));
      setRobRunning(false);
    }
  }

  // Sort members by rating descending
  const sorted = [...members].sort((a, b) => {
    const ra = ratings.get(a.member_id) ?? 0;
    const rb = ratings.get(b.member_id) ?? 0;
    return rb - ra || a.member_id.localeCompare(b.member_id);
  });

  // Find champion (highest rated)
  const champion: CompetitiveLeagueMember | null =
    sorted.length > 0 ? sorted[0] : null;
  const championRating = champion ? ratings.get(champion.member_id) : null;

  const tabClass = (t: Tab) =>
    `px-4 py-2 text-sm font-medium rounded-t ${
      tab === t
        ? "bg-white border border-b-0 border-gray-300"
        : "text-gray-500 hover:text-gray-700"
    }`;

  return (
    <main className="max-w-5xl mx-auto p-8">
      <div className="flex items-center gap-4 mb-6">
        <Link href="/" className="text-blue-500 hover:underline text-sm">
          &larr; Home
        </Link>
        <h1 className="text-2xl font-bold">Competitive League</h1>
        <button
          onClick={handleRecompute}
          disabled={recomputing || members.length === 0}
          className="ml-auto px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm disabled:opacity-50"
        >
          {recomputing ? "Recomputing..." : "Recompute Ratings"}
        </button>
      </div>

      {error && <p className="text-red-500 mb-2 text-sm">{error}</p>}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-300 mb-4">
        <button className={tabClass("ratings")} onClick={() => setTab("ratings")}>
          Ratings
        </button>
        <button className={tabClass("lineage")} onClick={() => setTab("lineage")}>
          Lineage
        </button>
        <button className={tabClass("champion")} onClick={() => setTab("champion")}>
          Champion
        </button>
        <button className={tabClass("evolution")} onClick={() => setTab("evolution")}>
          Evolution
        </button>
      </div>

      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : (
        <>
          {/* ---- Ratings tab ---- */}
          {tab === "ratings" &&
            (sorted.length === 0 ? (
              <p className="text-gray-500">
                No league members yet &mdash; run the pipeline first.
              </p>
            ) : (
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-gray-300">
                    <th className="py-2 pr-4">#</th>
                    <th className="py-2 pr-4">Member ID</th>
                    <th className="py-2 pr-4">Rating</th>
                    <th className="py-2 pr-4">Parent</th>
                    <th className="py-2 pr-4">Created</th>
                    <th className="py-2" />
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((m, idx) => (
                    <tr key={m.member_id} className="border-b border-gray-200">
                      <td className="py-2 pr-4 text-gray-400">{idx + 1}</td>
                      <td className="py-2 pr-4 font-mono">{m.member_id}</td>
                      <td className="py-2 pr-4 font-mono">
                        {ratings.has(m.member_id)
                          ? ratings.get(m.member_id)!.toFixed(1)
                          : "—"}
                      </td>
                      <td className="py-2 pr-4 font-mono text-xs">
                        {m.parent_id ?? "—"}
                      </td>
                      <td className="py-2 pr-4 text-xs text-gray-500">
                        {m.created_at
                          ? new Date(m.created_at).toLocaleString()
                          : "—"}
                      </td>
                      <td className="py-2">
                        <button
                          onClick={() => handleRun(m.member_id)}
                          disabled={startingId !== null}
                          className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm disabled:opacity-50"
                        >
                          {startingId === m.member_id
                            ? "Starting..."
                            : "Run"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ))}

          {/* ---- Lineage tab ---- */}
          {tab === "lineage" && (
            <div className="flex gap-4">
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-semibold mb-2">Lineage Graph</h3>
                <LineageSVG
                  members={evolutionData.members}
                  onSelect={setSelectedEvolution}
                  selectedId={selectedEvolution?.member_id ?? null}
                />

                {selectedEvolution && (
                  <div className="border border-gray-200 rounded p-3 text-sm mt-3">
                    <h4 className="font-bold mb-2">Details</h4>
                    <dl className="space-y-1">
                      <dt className="text-gray-500">Member ID</dt>
                      <dd className="font-mono text-xs">
                        {selectedEvolution.member_id}
                      </dd>
                      <dt className="text-gray-500">Label</dt>
                      <dd>
                        <span
                          className="font-medium"
                          style={{
                            color: compLabelColor(
                              selectedEvolution.strategy.label,
                            ),
                          }}
                        >
                          {selectedEvolution.strategy.label}
                        </span>
                      </dd>
                      <dt className="text-gray-500">Rating</dt>
                      <dd>{selectedEvolution.rating.toFixed(1)}</dd>
                      <dt className="text-gray-500">Parent</dt>
                      <dd className="font-mono text-xs">
                        {selectedEvolution.parent_id ?? "none"}
                      </dd>
                      <dt className="text-gray-500">Cluster</dt>
                      <dd>
                        {selectedEvolution.strategy.cluster_id ?? "—"}
                      </dd>
                      <dt className="text-gray-500">Robustness</dt>
                      <dd>
                        {selectedEvolution.robustness_score != null
                          ? selectedEvolution.robustness_score.toFixed(3)
                          : "—"}
                      </dd>
                      <dt className="text-gray-500">Created</dt>
                      <dd className="text-xs">
                        {selectedEvolution.created_at
                          ? new Date(
                              selectedEvolution.created_at,
                            ).toLocaleString()
                          : "—"}
                      </dd>
                    </dl>
                    <button
                      onClick={() => setSelectedEvolution(null)}
                      className="mt-2 text-xs text-blue-500 hover:underline"
                    >
                      Close
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ---- Champion tab ---- */}
          {tab === "champion" && (
            <div className="space-y-6">
              {/* Champion info */}
              {champion ? (
                <div className="border border-gray-200 rounded p-4 text-sm">
                  <h3 className="font-semibold mb-2">Current Champion</h3>
                  <dl className="grid grid-cols-2 gap-x-6 gap-y-1">
                    <dt className="text-gray-500">Member ID</dt>
                    <dd className="font-mono text-xs">{champion.member_id}</dd>
                    <dt className="text-gray-500">Rating</dt>
                    <dd className="font-mono">
                      {championRating != null
                        ? championRating.toFixed(1)
                        : "—"}
                    </dd>
                    <dt className="text-gray-500">Parent</dt>
                    <dd className="font-mono text-xs">
                      {champion.parent_id ?? "none"}
                    </dd>
                  </dl>
                </div>
              ) : (
                <p className="text-gray-500">
                  No league members yet &mdash; run the pipeline first.
                </p>
              )}

              {/* Benchmark section */}
              <div>
                <h3 className="text-sm font-semibold mb-2">
                  Champion Benchmark
                </h3>
                <div className="flex items-end gap-3">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">
                      Config
                    </label>
                    <select
                      value={benchConfigId}
                      onChange={(e) => setBenchConfigId(e.target.value)}
                      className="border rounded px-2 py-1 text-sm"
                    >
                      {configs.map((c) => (
                        <option key={c.config_id} value={c.config_id}>
                          {c.config_id} (agents={c.num_agents}, steps=
                          {c.max_steps})
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">
                      Episodes
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={50}
                      value={benchEpisodes}
                      onChange={(e) =>
                        setBenchEpisodes(Number(e.target.value))
                      }
                      className="border rounded px-2 py-1 text-sm w-16"
                    />
                  </div>
                  <button
                    onClick={handleBenchmark}
                    disabled={
                      benchRunning ||
                      configs.length === 0 ||
                      members.length === 0
                    }
                    className="px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700 text-sm disabled:opacity-50"
                  >
                    {benchRunning ? "Running..." : "Run Champion Benchmark"}
                  </button>
                </div>

                {benchData && (
                  <div className="mt-4 space-y-3">
                    <p className="text-sm text-gray-600">
                      Champion:{" "}
                      <span className="font-mono">
                        {benchData.champion.member_id}
                      </span>{" "}
                      (rating{" "}
                      {benchData.champion.rating != null
                        ? benchData.champion.rating.toFixed(1)
                        : "—"}
                      )
                    </p>

                    <h4 className="text-sm font-semibold">
                      Mean Total Reward
                    </h4>
                    <CompetitiveBarChart results={benchData.results} />

                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-gray-300">
                          <th className="py-1 pr-3">Policy</th>
                          <th className="py-1 pr-3">Mean Reward</th>
                          <th className="py-1 pr-3">Mean Score</th>
                          <th className="py-1 pr-3">Win Rate</th>
                          <th className="py-1 pr-3">Mean Length</th>
                        </tr>
                      </thead>
                      <tbody>
                        {benchData.results.map((r) => (
                          <tr
                            key={r.policy}
                            className="border-b border-gray-200"
                          >
                            <td className="py-1 pr-3 font-mono">
                              {r.policy}
                            </td>
                            <td className="py-1 pr-3">
                              {r.mean_total_reward.toFixed(4)}
                            </td>
                            <td className="py-1 pr-3">
                              {r.mean_final_score.toFixed(2)}
                            </td>
                            <td className="py-1 pr-3">
                              {(r.win_rate * 100).toFixed(0)}%
                            </td>
                            <td className="py-1 pr-3">
                              {r.mean_episode_length}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Robustness section */}
              <div>
                <h3 className="text-sm font-semibold mb-2">
                  Run Robustness on Champion
                </h3>
                <p className="text-sm text-gray-600 mb-3">
                  Evaluates the competitive league champion against all baseline
                  policies across multiple environment variants and saves a
                  robustness report.
                </p>

                <div className="flex flex-wrap items-end gap-3">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">
                      Config
                    </label>
                    <select
                      value={robConfigId}
                      onChange={(e) => setRobConfigId(e.target.value)}
                      className="border rounded px-2 py-1 text-sm"
                    >
                      <option value="default">default</option>
                      {configs.map((c) => (
                        <option key={c.config_id} value={c.config_id}>
                          {c.config_id} (agents={c.num_agents}, steps=
                          {c.max_steps})
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">
                      Seeds
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={robSeeds}
                      onChange={(e) => setRobSeeds(Number(e.target.value))}
                      className="border rounded px-2 py-1 text-sm w-16"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">
                      Episodes/seed
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={10}
                      value={robEpisodesPerSeed}
                      onChange={(e) =>
                        setRobEpisodesPerSeed(Number(e.target.value))
                      }
                      className="border rounded px-2 py-1 text-sm w-16"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">
                      Limit sweeps (opt.)
                    </label>
                    <input
                      type="number"
                      min={1}
                      placeholder="—"
                      value={robLimitSweeps}
                      onChange={(e) => setRobLimitSweeps(e.target.value)}
                      className="border rounded px-2 py-1 text-sm w-20"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">
                      Seed
                    </label>
                    <input
                      type="number"
                      value={robSeed}
                      onChange={(e) => setRobSeed(Number(e.target.value))}
                      className="border rounded px-2 py-1 text-sm w-20"
                    />
                  </div>
                  <button
                    onClick={handleRobustness}
                    disabled={robRunning || members.length === 0}
                    className="px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 text-sm disabled:opacity-50"
                  >
                    {robRunning ? "Running..." : "Run Robustness"}
                  </button>
                </div>

                {robRunning && (
                  <p className="text-sm text-gray-500 mt-2">
                    Running robustness sweep &mdash; this may take a moment...
                  </p>
                )}
              </div>
            </div>
          )}

          {/* ---- Evolution tab ---- */}
          {tab === "evolution" && (
            <div className="flex gap-6">
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-semibold mb-2">Lineage Graph</h3>
                {evolutionData.members.length === 0 &&
                evolutionData.champion_history.length === 0 ? (
                  <p className="text-gray-500">
                    No evolution data yet. Train and save snapshots to build
                    history.
                  </p>
                ) : (
                  <>
                    <LineageSVG
                      members={evolutionData.members}
                      onSelect={setSelectedEvolution}
                      selectedId={selectedEvolution?.member_id ?? null}
                    />

                    {selectedEvolution && (
                      <div className="border border-gray-200 rounded p-3 text-sm mt-3">
                        <h4 className="font-bold mb-2">Details</h4>
                        <dl className="space-y-1">
                          <dt className="text-gray-500">Member ID</dt>
                          <dd className="font-mono text-xs">
                            {selectedEvolution.member_id}
                          </dd>
                          <dt className="text-gray-500">Label</dt>
                          <dd>
                            <span
                              className="font-medium"
                              style={{
                                color: compLabelColor(
                                  selectedEvolution.strategy.label,
                                ),
                              }}
                            >
                              {selectedEvolution.strategy.label}
                            </span>
                          </dd>
                          <dt className="text-gray-500">Rating</dt>
                          <dd>{selectedEvolution.rating.toFixed(1)}</dd>
                          <dt className="text-gray-500">Parent</dt>
                          <dd className="font-mono text-xs">
                            {selectedEvolution.parent_id ?? "none"}
                          </dd>
                          <dt className="text-gray-500">Cluster</dt>
                          <dd>
                            {selectedEvolution.strategy.cluster_id ?? "—"}
                          </dd>
                          <dt className="text-gray-500">Robustness</dt>
                          <dd>
                            {selectedEvolution.robustness_score != null
                              ? selectedEvolution.robustness_score.toFixed(3)
                              : "—"}
                          </dd>
                          <dt className="text-gray-500">Created</dt>
                          <dd className="text-xs">
                            {selectedEvolution.created_at
                              ? new Date(
                                  selectedEvolution.created_at,
                                ).toLocaleString()
                              : "—"}
                          </dd>
                          <dt className="text-gray-500">Notes</dt>
                          <dd className="text-xs">
                            {selectedEvolution.notes ?? "—"}
                          </dd>
                        </dl>
                        <button
                          onClick={() => setSelectedEvolution(null)}
                          className="mt-2 text-xs text-blue-500 hover:underline"
                        >
                          Close
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Champion history timeline */}
              <div className="w-72 flex-shrink-0">
                <h3 className="text-sm font-semibold mb-2">
                  Champion History
                </h3>
                {evolutionData.champion_history.length === 0 ? (
                  <p className="text-gray-500 text-sm">
                    No champion history yet.
                  </p>
                ) : (
                  <ol className="space-y-2">
                    {evolutionData.champion_history.map((entry, idx) => (
                      <CompetitiveTimelineEntry
                        key={entry.member_id}
                        entry={entry}
                        idx={idx}
                      />
                    ))}
                  </ol>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </main>
  );
}
