"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  LeagueMember,
  LeagueRating,
  LineageMember,
  LeagueEvolutionResponse,
  listLeagueMembers,
  getLeagueRatings,
  getLeagueLineage,
  getLeagueEvolution,
  recomputeLeagueRatings,
  listConfigs,
  startRun,
  ConfigListItem,
  CompetitiveLeagueMember,
  CompetitiveLeagueRating,
  CompetitiveEvolutionMember,
  CompetitiveEvolutionResponse,
  CompetitiveChampionBenchmarkResponse,
  CompetitiveChampionBenchmarkResult,
  CompetitiveChampionRobustnessRequest,
  ChampionHistoryEntry,
  getCompetitiveLeagueMembers,
  getCompetitiveLeagueRatings,
  getCompetitiveLeagueEvolution,
  recomputeCompetitiveLeagueRatings,
  runCompetitiveChampionBenchmark,
  runCompetitiveChampionRobustness,
  startCompetitiveLeagueMemberRun,
} from "@/lib/api";
import LeagueLineage from "@/components/LeagueLineage";
import ChampionBenchmark from "@/components/ChampionBenchmark";
import ChampionRobustness from "@/components/ChampionRobustness";
import LeagueEvolution from "@/components/LeagueEvolution";
import LineageGraph from "@/components/LineageGraph";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Archetype = "resource-sharing" | "head-to-head";
type Tab = "ratings" | "lineage" | "champion" | "evolution";

// ---------------------------------------------------------------------------
// Competitive helpers (copied from competitive/league/page.tsx)
// ---------------------------------------------------------------------------

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


// Bar chart for competitive benchmark

const COMP_BAR_COLORS: Record<string, string> = {
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
        const color = COMP_BAR_COLORS[r.policy] || "#6b7280";
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

// Competitive history card (inline styles, matches ChampionHistoryCard layout)

function CompetitiveHistoryCard({
  entry,
  idx,
}: {
  entry: ChampionHistoryEntry;
  idx: number;
}) {
  const color = compLabelColor(entry.label);
  return (
    <div style={{
      background: "#111111",
      border: "1px solid #1e1e1e",
      borderLeft: `3px solid ${color}`,
      borderRadius: 6,
      padding: "10px 12px",
    }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 4,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 11, color: "#555555",
            fontFamily: "monospace" }}>
            #{idx + 1}
          </span>
          <span style={{ fontSize: 12, fontWeight: 500, color }}>
            {entry.label}
          </span>
          {entry.cluster_id != null && (
            <span style={{ fontSize: 11, color: "#444444" }}>
              cluster {entry.cluster_id}
            </span>
          )}
        </div>
        <span style={{ fontSize: 13, fontWeight: 600,
          color: "#ededed" }}>
          {entry.rating.toFixed(1)}
        </span>
      </div>
      <div style={{
        fontSize: 11,
        color: "#666666",
        fontFamily: "monospace",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        marginBottom: 2,
      }} title={entry.member_id}>
        {entry.member_id}
      </div>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        {entry.robustness_score != null ? (
          <span style={{ fontSize: 11, color: "#555555" }}>
            Robust: {entry.robustness_score.toFixed(3)}
          </span>
        ) : <span />}
        {entry.created_at && (
          <span style={{ fontSize: 10, color: "#444444" }}>
            {new Date(entry.created_at).toLocaleDateString()}
          </span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function LeaguePage() {
  const router = useRouter();
  const [archetype, setArchetype] = useState<Archetype>("resource-sharing");
  const [tab, setTab] = useState<Tab>("ratings");

  // --- Resource Sharing (Mixed) state ---
  const [rsMembers, setRsMembers] = useState<LeagueMember[]>([]);
  const [rsLineageMembers, setRsLineageMembers] = useState<LineageMember[]>([]);
  const [rsRatings, setRsRatings] = useState<Map<string, number>>(new Map());
  const [rsConfigs, setRsConfigs] = useState<ConfigListItem[]>([]);
  const [rsEvolutionData, setRsEvolutionData] = useState<LeagueEvolutionResponse>({
    members: [],
    champion_history: [],
  });
  const [rsLoading, setRsLoading] = useState(true);
  const [rsRecomputing, setRsRecomputing] = useState(false);
  const [rsStartingId, setRsStartingId] = useState<string | null>(null);
  const [rsError, setRsError] = useState<string | null>(null);

  // --- Head-to-Head (Competitive) state ---
  const [hhMembers, setHhMembers] = useState<CompetitiveLeagueMember[]>([]);
  const [hhRatings, setHhRatings] = useState<Map<string, number>>(new Map());
  const [hhConfigs, setHhConfigs] = useState<ConfigListItem[]>([]);
  const [hhEvolutionData, setHhEvolutionData] =
    useState<CompetitiveEvolutionResponse>({
      members: [],
      champion_history: [],
    });
  const [hhLoading, setHhLoading] = useState(true);
  const [hhRecomputing, setHhRecomputing] = useState(false);
  const [hhStartingId, setHhStartingId] = useState<string | null>(null);
  const [hhError, setHhError] = useState<string | null>(null);

  // Competitive champion tab state
  const [hhBenchConfigId, setHhBenchConfigId] = useState("");
  const [hhBenchEpisodes, setHhBenchEpisodes] = useState(5);
  const [hhBenchRunning, setHhBenchRunning] = useState(false);
  const [hhBenchData, setHhBenchData] =
    useState<CompetitiveChampionBenchmarkResponse | null>(null);
  const [hhRobConfigId, setHhRobConfigId] = useState("default");
  const [hhRobSeeds, setHhRobSeeds] = useState(3);
  const [hhRobEpisodesPerSeed, setHhRobEpisodesPerSeed] = useState(2);
  const [hhRobLimitSweeps, setHhRobLimitSweeps] = useState<string>("");
  const [hhRobSeed, setHhRobSeed] = useState(42);
  const [hhRobRunning, setHhRobRunning] = useState(false);

  // Recompute feedback state (per archetype so only one message shows)
  const [rsRecomputeStatus, setRsRecomputeStatus] = useState<"idle" | "running" | "success" | "error">("idle");
  const [hhRecomputeStatus, setHhRecomputeStatus] = useState<"idle" | "running" | "success" | "error">("idle");

  // --- Load Resource Sharing data ---
  async function loadResourceSharing() {
    setRsLoading(true);
    setRsError(null);
    try {
      const [m, r, c, lin, evo] = await Promise.all([
        listLeagueMembers(),
        getLeagueRatings(),
        listConfigs(),
        getLeagueLineage(),
        getLeagueEvolution(),
      ]);
      setRsMembers(m);
      setRsRatings(new Map(r.map((x) => [x.member_id, x.rating])));
      setRsConfigs(c);
      setRsLineageMembers(lin.members);
      setRsEvolutionData(evo);
    } catch (e) {
      setRsError(String(e));
    } finally {
      setRsLoading(false);
    }
  }

  // --- Load Head-to-Head data ---
  async function loadHeadToHead() {
    setHhLoading(true);
    setHhError(null);
    try {
      const [m, r, c, evo] = await Promise.all([
        getCompetitiveLeagueMembers(),
        getCompetitiveLeagueRatings(),
        listConfigs(),
        getCompetitiveLeagueEvolution(),
      ]);
      setHhMembers(m);
      setHhRatings(new Map(r.map((x) => [x.member_id, x.rating])));
      setHhConfigs(c);
      setHhEvolutionData(evo);
      if (c.length > 0 && !hhBenchConfigId) {
        setHhBenchConfigId(c[0].config_id);
      }
    } catch (e) {
      setHhError(String(e));
    } finally {
      setHhLoading(false);
    }
  }

  useEffect(() => {
    loadResourceSharing();
    loadHeadToHead();
  }, []);

  // --- Resource Sharing handlers ---
  async function handleRsRecompute() {
    setRsRecomputing(true);
    setRsError(null);
    setRsRecomputeStatus("running");
    try {
      const r = await recomputeLeagueRatings();
      setRsRatings(new Map(r.map((x) => [x.member_id, x.rating])));
      const lin = await getLeagueLineage();
      setRsLineageMembers(lin.members);
      setRsRecomputeStatus("success");
      setTimeout(() => setRsRecomputeStatus("idle"), 3000);
    } catch (e) {
      setRsError(String(e));
      setRsRecomputeStatus("error");
    } finally {
      setRsRecomputing(false);
    }
  }

  async function handleRsRun(memberId: string) {
    if (rsConfigs.length === 0) {
      setRsError("No configs available. Create one on the home page first.");
      return;
    }
    setRsStartingId(memberId);
    setRsError(null);
    try {
      const { run_id } = await startRun(rsConfigs[0].config_id, "league_snapshot", memberId);
      router.push(`/simulate/resource-sharing/run/${run_id}`);
    } catch (e) {
      setRsError(String(e));
      setRsStartingId(null);
    }
  }

  // --- Head-to-Head handlers ---
  async function handleHhRecompute() {
    setHhRecomputing(true);
    setHhError(null);
    setHhRecomputeStatus("running");
    try {
      const r = await recomputeCompetitiveLeagueRatings();
      setHhRatings(new Map(r.map((x) => [x.member_id, x.rating])));
      const evo = await getCompetitiveLeagueEvolution();
      setHhEvolutionData(evo);
      setHhRecomputeStatus("success");
      setTimeout(() => setHhRecomputeStatus("idle"), 3000);
    } catch (e) {
      setHhError(String(e));
      setHhRecomputeStatus("error");
    } finally {
      setHhRecomputing(false);
    }
  }

  async function handleHhRun(memberId: string) {
    if (hhConfigs.length === 0) {
      setHhError("No configs available. Create one on the home page first.");
      return;
    }
    setHhStartingId(memberId);
    setHhError(null);
    try {
      const { run_id } = await startCompetitiveLeagueMemberRun(
        hhConfigs[0].config_id,
        memberId,
      );
      router.push(`/simulate/head-to-head/run/${run_id}`);
    } catch (e) {
      setHhError(String(e));
      setHhStartingId(null);
    }
  }

  async function handleHhBenchmark() {
    if (!hhBenchConfigId) {
      setHhError("Select a config first.");
      return;
    }
    setHhBenchRunning(true);
    setHhError(null);
    setHhBenchData(null);
    try {
      const resp = await runCompetitiveChampionBenchmark(
        hhBenchConfigId,
        hhBenchEpisodes,
      );
      setHhBenchData(resp);
    } catch (e) {
      setHhError(String(e));
    } finally {
      setHhBenchRunning(false);
    }
  }

  async function handleHhRobustness() {
    setHhRobRunning(true);
    setHhError(null);
    try {
      const payload: CompetitiveChampionRobustnessRequest = {
        config_id: hhRobConfigId,
        seeds: [hhRobSeeds],
        episodes_per_seed: hhRobEpisodesPerSeed,
        seed: hhRobSeed,
        ...(hhRobLimitSweeps !== ""
          ? { limit_sweeps: Number(hhRobLimitSweeps) }
          : {}),
      };
      const resp = await runCompetitiveChampionRobustness(payload);
      router.push(`/research/${encodeURIComponent(resp.report_id)}`);
    } catch (e) {
      setHhError(String(e));
      setHhRobRunning(false);
    }
  }

  // --- Derived values ---
  const isRS = archetype === "resource-sharing";
  const loading = isRS ? rsLoading : hhLoading;
  const error = isRS ? rsError : hhError;
  const recomputing = isRS ? rsRecomputing : hhRecomputing;
  const members = isRS ? rsMembers : hhMembers;
  const ratings = isRS ? rsRatings : hhRatings;

  // Sorted members for ratings tab
  const rsSorted = [...rsMembers].sort((a, b) => {
    const ra = rsRatings.get(a.member_id) ?? 0;
    const rb = rsRatings.get(b.member_id) ?? 0;
    return rb - ra || a.member_id.localeCompare(b.member_id);
  });

  const hhSorted = [...hhMembers].sort((a, b) => {
    const ra = hhRatings.get(a.member_id) ?? 0;
    const rb = hhRatings.get(b.member_id) ?? 0;
    return rb - ra || a.member_id.localeCompare(b.member_id);
  });

  // Head-to-Head champion (highest rated)
  const hhChampion: CompetitiveLeagueMember | null =
    hhSorted.length > 0 ? hhSorted[0] : null;
  const hhChampionRating = hhChampion ? hhRatings.get(hhChampion.member_id) : null;

  function handleArchetypeSwitch(a: Archetype) {
    if (a !== archetype) {
      setArchetype(a);
      setTab("ratings");
    }
  }

  // --- Styles ---
  const pillBase: React.CSSProperties = {
    borderRadius: "9999px",
    padding: "6px 16px",
    fontSize: "13px",
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 150ms",
  };

  const pillActive: React.CSSProperties = {
    ...pillBase,
    background: "var(--accent)",
    color: "white",
    border: "1px solid transparent",
  };

  const pillInactive: React.CSSProperties = {
    ...pillBase,
    background: "transparent",
    color: "var(--text-secondary)",
    border: "1px solid var(--bg-border)",
  };

  const subTabBase: React.CSSProperties = {
    fontSize: "13px",
    fontWeight: 500,
    padding: "8px 0",
    marginRight: "24px",
    cursor: "pointer",
    background: "none",
    border: "none",
    borderBottom: "2px solid transparent",
    transition: "all 150ms",
  };

  const subTabActive: React.CSSProperties = {
    ...subTabBase,
    color: "var(--text-primary)",
    borderBottomColor: "var(--accent)",
  };

  const subTabInactive: React.CSSProperties = {
    ...subTabBase,
    color: "var(--text-secondary)",
  };

  return (
    <main style={{ maxWidth: "1100px", margin: "0 auto", padding: "48px 24px" }}>
      {/* Header */}
      <div style={{ marginBottom: "8px" }}>
        <h1 style={{ fontSize: "24px", fontWeight: 500, color: "var(--text-primary)", margin: 0 }}>
          League
        </h1>
        <p style={{ fontSize: "14px", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
          Elo-rated agent leagues across all environments
        </p>
      </div>

      {/* Archetype switcher */}
      <div style={{ display: "flex", gap: "8px", marginBottom: "24px", marginTop: "16px" }}>
        <button
          style={archetype === "resource-sharing" ? pillActive : pillInactive}
          onClick={() => handleArchetypeSwitch("resource-sharing")}
          onMouseEnter={(e) => {
            if (archetype !== "resource-sharing") {
              e.currentTarget.style.color = "var(--text-primary)";
              e.currentTarget.style.borderColor = "var(--accent)";
            }
          }}
          onMouseLeave={(e) => {
            if (archetype !== "resource-sharing") {
              e.currentTarget.style.color = "var(--text-secondary)";
              e.currentTarget.style.borderColor = "var(--bg-border)";
            }
          }}
        >
          Resource Sharing
        </button>
        <button
          style={archetype === "head-to-head" ? pillActive : pillInactive}
          onClick={() => handleArchetypeSwitch("head-to-head")}
          onMouseEnter={(e) => {
            if (archetype !== "head-to-head") {
              e.currentTarget.style.color = "var(--text-primary)";
              e.currentTarget.style.borderColor = "var(--accent)";
            }
          }}
          onMouseLeave={(e) => {
            if (archetype !== "head-to-head") {
              e.currentTarget.style.color = "var(--text-secondary)";
              e.currentTarget.style.borderColor = "var(--bg-border)";
            }
          }}
        >
          Head-to-Head
        </button>

        {/* Recompute Ratings button */}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <button
            onClick={isRS ? handleRsRecompute : handleHhRecompute}
            disabled={recomputing || members.length === 0}
            style={{
              padding: "6px 12px",
              background: recomputing ? "var(--bg-elevated)" : "var(--accent)",
              color: recomputing ? "var(--text-tertiary)" : "#fff",
              borderRadius: 6,
              border: "none",
              fontSize: 13,
              fontWeight: 500,
              cursor: recomputing || members.length === 0 ? "default" : "pointer",
              opacity: recomputing || members.length === 0 ? 0.5 : 1,
            }}
          >
            {recomputing ? "Recomputing..." : "Recompute Ratings"}
          </button>
          {(isRS ? rsRecomputeStatus : hhRecomputeStatus) === "running" && (
            <span style={{ fontSize: 12, color: "var(--text-tertiary)" }}>Recomputing...</span>
          )}
          {(isRS ? rsRecomputeStatus : hhRecomputeStatus) === "success" && (
            <span style={{ fontSize: 12, color: "var(--accent)" }}>&#10003; Ratings updated</span>
          )}
          {(isRS ? rsRecomputeStatus : hhRecomputeStatus) === "error" && (
            <span style={{ fontSize: 12, color: "#f87171" }}>Failed to recompute</span>
          )}
        </div>
      </div>

      {/* Sub-tabs */}
      <div style={{ display: "flex", borderBottom: "1px solid var(--bg-border)", marginBottom: "32px" }}>
        {(["ratings", "lineage", "champion", "evolution"] as Tab[]).map((t) => (
          <button
            key={t}
            style={tab === t ? subTabActive : subTabInactive}
            onClick={() => setTab(t)}
            onMouseEnter={(e) => {
              if (tab !== t) e.currentTarget.style.color = "var(--text-primary)";
            }}
            onMouseLeave={(e) => {
              if (tab !== t) e.currentTarget.style.color = "var(--text-secondary)";
            }}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {error && <p className="text-red-500 mb-2 text-sm">{error}</p>}

      {loading ? (
        <p style={{ color: "var(--text-secondary)" }}>Loading...</p>
      ) : (
        <>
          {/* ============================================================ */}
          {/* RESOURCE SHARING CONTENT                                      */}
          {/* ============================================================ */}
          {isRS && (
            <>
              {/* Ratings tab */}
              {tab === "ratings" && (
                rsSorted.length === 0 ? (
                  <p className="text-gray-500">
                    No league members yet. Train a policy and save a snapshot to get started.
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
                        <th className="py-2 pr-4">Notes</th>
                        <th className="py-2" />
                      </tr>
                    </thead>
                    <tbody>
                      {rsSorted.map((m, idx) => (
                        <tr key={m.member_id} className="border-b border-gray-200">
                          <td className="py-2 pr-4 text-gray-400">{idx + 1}</td>
                          <td className="py-2 pr-4 font-mono">{m.member_id}</td>
                          <td className="py-2 pr-4 font-mono">
                            {rsRatings.has(m.member_id)
                              ? rsRatings.get(m.member_id)!.toFixed(1)
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
                          <td className="py-2 pr-4 text-xs">{m.notes ?? "—"}</td>
                          <td className="py-2">
                            <button
                              onClick={() => handleRsRun(m.member_id)}
                              disabled={rsStartingId !== null}
                              className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm disabled:opacity-50"
                            >
                              {rsStartingId === m.member_id ? "Starting..." : "Run"}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )
              )}

              {/* Lineage tab */}
              {tab === "lineage" && (
                <LeagueLineage members={rsLineageMembers} />
              )}

              {/* Champion tab */}
              {tab === "champion" && (
                rsConfigs.length === 0 ? (
                  <p className="text-gray-500">
                    No configs available. Create one on the home page first.
                  </p>
                ) : (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-sm font-semibold mb-2">Champion Benchmark</h3>
                      <ChampionBenchmark configs={rsConfigs} />
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold mb-2">Run Robustness on Champion</h3>
                      {rsMembers.length === 0 ? (
                        <p className="text-gray-500">
                          No league members yet. Train a policy and save a snapshot to get started.
                        </p>
                      ) : (
                        <ChampionRobustness configs={rsConfigs} />
                      )}
                    </div>
                  </div>
                )
              )}

              {/* Evolution tab */}
              {tab === "evolution" && (
                <LeagueEvolution data={rsEvolutionData} />
              )}
            </>
          )}

          {/* ============================================================ */}
          {/* HEAD-TO-HEAD CONTENT                                          */}
          {/* ============================================================ */}
          {!isRS && (
            <>
              {/* Ratings tab */}
              {tab === "ratings" &&
                (hhSorted.length === 0 ? (
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
                      {hhSorted.map((m, idx) => (
                        <tr key={m.member_id} className="border-b border-gray-200">
                          <td className="py-2 pr-4 text-gray-400">{idx + 1}</td>
                          <td className="py-2 pr-4 font-mono">{m.member_id}</td>
                          <td className="py-2 pr-4 font-mono">
                            {hhRatings.has(m.member_id)
                              ? hhRatings.get(m.member_id)!.toFixed(1)
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
                              onClick={() => handleHhRun(m.member_id)}
                              disabled={hhStartingId !== null}
                              className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm disabled:opacity-50"
                            >
                              {hhStartingId === m.member_id
                                ? "Starting..."
                                : "Run"}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ))}

              {/* Lineage tab */}
              {tab === "lineage" && (
                <div>
                  <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: "var(--text-primary)" }}>Lineage Graph</h3>
                  <LineageGraph nodes={hhEvolutionData.members.map(m => ({
                    id: m.member_id,
                    parent_id: m.parent_id,
                    rating: m.rating,
                    label: m.strategy?.label,
                    cluster: m.strategy?.cluster_id,
                    robustness: m.robustness_score,
                    created_at: m.created_at,
                    notes: m.notes,
                  }))} />
                </div>
              )}

              {/* Champion tab */}
              {tab === "champion" && (
                <div className="space-y-6">
                  {/* Champion info */}
                  {hhChampion ? (
                    <div className="border border-gray-200 rounded p-4 text-sm">
                      <h3 className="font-semibold mb-2">Current Champion</h3>
                      <dl className="grid grid-cols-2 gap-x-6 gap-y-1">
                        <dt className="text-gray-500">Member ID</dt>
                        <dd className="font-mono text-xs">{hhChampion.member_id}</dd>
                        <dt className="text-gray-500">Rating</dt>
                        <dd className="font-mono">
                          {hhChampionRating != null
                            ? hhChampionRating.toFixed(1)
                            : "—"}
                        </dd>
                        <dt className="text-gray-500">Parent</dt>
                        <dd className="font-mono text-xs">
                          {hhChampion.parent_id ?? "none"}
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
                          value={hhBenchConfigId}
                          onChange={(e) => setHhBenchConfigId(e.target.value)}
                          className="border rounded px-2 py-1 text-sm"
                        >
                          {hhConfigs.map((c) => (
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
                          value={hhBenchEpisodes}
                          onChange={(e) =>
                            setHhBenchEpisodes(Number(e.target.value))
                          }
                          className="border rounded px-2 py-1 text-sm w-16"
                        />
                      </div>
                      <button
                        onClick={handleHhBenchmark}
                        disabled={
                          hhBenchRunning ||
                          hhConfigs.length === 0 ||
                          hhMembers.length === 0
                        }
                        className="px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700 text-sm disabled:opacity-50"
                      >
                        {hhBenchRunning ? "Running..." : "Run Champion Benchmark"}
                      </button>
                    </div>

                    {hhBenchData && (
                      <div className="mt-4 space-y-3">
                        <p className="text-sm text-gray-600">
                          Champion:{" "}
                          <span className="font-mono">
                            {hhBenchData.champion.member_id}
                          </span>{" "}
                          (rating{" "}
                          {hhBenchData.champion.rating != null
                            ? hhBenchData.champion.rating.toFixed(1)
                            : "—"}
                          )
                        </p>

                        <h4 className="text-sm font-semibold">
                          Mean Total Reward
                        </h4>
                        <CompetitiveBarChart results={hhBenchData.results} />

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
                            {hhBenchData.results.map((r) => (
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
                          value={hhRobConfigId}
                          onChange={(e) => setHhRobConfigId(e.target.value)}
                          className="border rounded px-2 py-1 text-sm"
                        >
                          <option value="default">default</option>
                          {hhConfigs.map((c) => (
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
                          value={hhRobSeeds}
                          onChange={(e) => setHhRobSeeds(Number(e.target.value))}
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
                          value={hhRobEpisodesPerSeed}
                          onChange={(e) =>
                            setHhRobEpisodesPerSeed(Number(e.target.value))
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
                          value={hhRobLimitSweeps}
                          onChange={(e) => setHhRobLimitSweeps(e.target.value)}
                          className="border rounded px-2 py-1 text-sm w-20"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">
                          Seed
                        </label>
                        <input
                          type="number"
                          value={hhRobSeed}
                          onChange={(e) => setHhRobSeed(Number(e.target.value))}
                          className="border rounded px-2 py-1 text-sm w-20"
                        />
                      </div>
                      <button
                        onClick={handleHhRobustness}
                        disabled={hhRobRunning || hhMembers.length === 0}
                        className="px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 text-sm disabled:opacity-50"
                      >
                        {hhRobRunning ? "Running..." : "Run Robustness"}
                      </button>
                    </div>

                    {hhRobRunning && (
                      <p className="text-sm text-gray-500 mt-2">
                        Running robustness sweep &mdash; this may take a moment...
                      </p>
                    )}
                  </div>
                </div>
              )}

              {/* Evolution tab */}
              {tab === "evolution" && (
                <div>
                  {/* LineageGraph full width on top */}
                  <div style={{ marginBottom: 32 }}>
                    {hhEvolutionData.members.length === 0 &&
                    hhEvolutionData.champion_history.length === 0 ? (
                      <p style={{ color: "var(--text-tertiary)" }}>
                        No evolution data yet. Train and save snapshots to build
                        history.
                      </p>
                    ) : (
                      <LineageGraph nodes={hhEvolutionData.members.map(m => ({
                        id: m.member_id,
                        parent_id: m.parent_id,
                        rating: m.rating,
                        label: m.strategy?.label,
                        cluster: m.strategy?.cluster_id,
                        robustness: m.robustness_score,
                        created_at: m.created_at,
                        notes: m.notes,
                      }))} />
                    )}
                  </div>

                  {/* Champion History scrollable list below */}
                  <div>
                    <h3 style={{
                      fontSize: 13,
                      fontWeight: 500,
                      color: "#888888",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                      marginBottom: 16,
                    }}>
                      Champion History
                    </h3>
                    {hhEvolutionData?.champion_history?.length === 0 ? (
                      <p style={{ color: "#666666", fontSize: 13 }}>
                        No champion history yet.
                      </p>
                    ) : (
                      <div style={{
                        maxHeight: 400,
                        overflowY: "auto",
                        display: "flex",
                        flexDirection: "column",
                        gap: 8,
                        paddingRight: 4,
                        scrollbarWidth: "thin",
                        scrollbarColor: "#2a2a2a transparent",
                      }}>
                        {hhEvolutionData?.champion_history?.map((entry, idx) => (
                          <CompetitiveHistoryCard
                            key={entry.member_id}
                            entry={entry}
                            idx={idx}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </main>
  );
}
