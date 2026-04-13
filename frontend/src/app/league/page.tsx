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
  getCompetitiveRobustnessStatus,
  startCompetitiveLeagueMemberRun,
  startMixedPipeline,
  getMixedPipelineStatus,
  startCompetitivePipeline,
  getCompetitivePipelineStatus,
  getConfigDetail,
  PipelineStatusResponse,
  CooperativeLeagueMember,
  CooperativeLineageMember,
  CooperativeEvolutionResponse,
  CooperativeChampionInfo,
  CooperativeRobustnessHeatmapResponse,
  getCooperativeLeagueMembers,
  getCooperativeLeagueLineage,
  getCooperativeLeagueEvolution,
  getCooperativeChampion,
  runCooperativeChampionRobustness,
  getCooperativeRobustnessStatus,
} from "@/lib/api";
import LeagueLineage from "@/components/LeagueLineage";
import ChampionBenchmark from "@/components/ChampionBenchmark";
import ChampionRobustness from "@/components/ChampionRobustness";
import LeagueEvolution from "@/components/LeagueEvolution";
import LineageGraph from "@/components/LineageGraph";
import CooperativeLeagueLineage from "@/components/CooperativeLeagueLineage";
import CooperativeChampionBenchmark from "@/components/CooperativeChampionBenchmark";
import CooperativeChampionRobustness from "@/components/CooperativeChampionRobustness";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Archetype = "resource-sharing" | "head-to-head" | "cooperative";
type Tab = "ratings" | "champion" | "evolution";

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
    <svg width={chartW} height={chartH + labelH} style={{display: "block"}}>
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
// Pipeline stage helpers
// ---------------------------------------------------------------------------

const PIPELINE_STAGES = ["loading_config", "training", "rating", "evaluating", "done"] as const;
const STAGE_LABELS: Record<string, string> = {
  loading_config: "Loading",
  training: "Training",
  rating: "Rating",
  evaluating: "Evaluating",
  done: "Done",
};
const STAGE_DESCRIPTIONS: Record<string, string> = {
  loading_config: "Loading configuration...",
  training: "Training PPO agent with self-play...",
  snapshotting: "Saving league snapshots...",
  rating: "Computing Elo ratings...",
  evaluating: "Running robustness sweeps...",
  reporting: "Generating report...",
  done: "Pipeline complete!",
  error: "Pipeline failed — check logs",
};

function stageIndex(stage: string): number {
  const idx = PIPELINE_STAGES.indexOf(stage as typeof PIPELINE_STAGES[number]);
  // snapshotting/reporting map to their parent stage
  if (stage === "snapshotting") return 1; // training
  if (stage === "reporting") return 3; // evaluating
  return idx >= 0 ? idx : -1;
}

// ---------------------------------------------------------------------------
// Pipeline panel components
// ---------------------------------------------------------------------------

function stageLabel(stage: string): string {
  const labels: Record<string, string> = {
    loading_config: "Loading config...",
    training: "Training PPO agents...",
    snapshotting: "Saving snapshots...",
    rating: "Computing Elo ratings...",
    evaluating: "Running robustness sweeps...",
    reporting: "Generating report...",
  };
  return labels[stage] ?? stage;
}

interface PipelineConfig {
  totalTimesteps: number;
  snapshotEvery: number;
  seed: number;
}

interface PipelinePanelProps {
  label: string;
  accentColor: string;
  stage: string | null;
  error: string | null;
  reportId: string | null;
  onRun: () => void;
  running: boolean;
  config: PipelineConfig;
  onConfigChange: (config: PipelineConfig) => void;
}

function PipelinePanel({
  label, accentColor, stage, error, reportId, onRun, running,
  config, onConfigChange,
}: PipelinePanelProps) {
  const [showSettings, setShowSettings] = useState(false);

  const inputStyle: React.CSSProperties = {
    background: "var(--bg-base)",
    border: "1px solid var(--bg-border)",
    borderRadius: 4,
    padding: "5px 8px",
    color: "var(--text-primary)",
    fontSize: 12,
    width: "100%",
    fontFamily: "var(--font-mono)",
    boxSizing: "border-box",
  };

  const labelStyle: React.CSSProperties = {
    fontSize: 10,
    color: "var(--text-tertiary)",
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    marginBottom: 3,
  };

  return (
    <div style={{
      background: "var(--bg-elevated)",
      border: "1px solid var(--bg-border)",
      borderRadius: 6,
      padding: 16,
    }}>
      {/* Archetype label */}
      <p style={{
        fontSize: 11,
        fontWeight: 500,
        color: accentColor,
        textTransform: "uppercase",
        letterSpacing: "0.06em",
        marginBottom: 12,
      }}>{label}</p>

      {/* Run button */}
      <button
        onClick={onRun}
        disabled={running}
        style={{
          width: "100%",
          padding: "8px 16px",
          background: running ? "var(--bg-surface)" : accentColor,
          color: running ? "var(--text-tertiary)" : "#0a0a0a",
          border: "none",
          borderRadius: 6,
          fontSize: 13,
          fontWeight: 500,
          cursor: running ? "not-allowed" : "pointer",
          marginBottom: 0,
        }}
      >
        {running ? "Running..." : "Run Pipeline →"}
      </button>

      {/* Settings toggle */}
      <button
        onClick={() => setShowSettings((v) => !v)}
        style={{
          fontSize: 11,
          color: "var(--text-tertiary)",
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: 0,
          marginTop: 8,
        }}
        onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text-secondary)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-tertiary)"; }}
      >
        Settings {showSettings ? "▴" : "▾"}
      </button>

      {/* Settings inputs */}
      {showSettings && (
        <div style={{
          display: "flex",
          flexDirection: "column",
          gap: 8,
          marginTop: 12,
          paddingTop: 12,
          borderTop: "1px solid var(--bg-border)",
        }}>
          <div>
            <div style={labelStyle}>Training steps</div>
            <input
              type="number"
              min={1000}
              step={1000}
              value={config.totalTimesteps}
              onChange={(e) => onConfigChange({ ...config, totalTimesteps: Number(e.target.value) })}
              style={inputStyle}
            />
          </div>
          <div>
            <div style={labelStyle}>Snapshot every N steps</div>
            <input
              type="number"
              min={500}
              step={500}
              value={config.snapshotEvery}
              onChange={(e) => onConfigChange({ ...config, snapshotEvery: Number(e.target.value) })}
              style={inputStyle}
            />
          </div>
          <div>
            <div style={labelStyle}>Seed</div>
            <input
              type="number"
              min={0}
              value={config.seed}
              onChange={(e) => onConfigChange({ ...config, seed: Number(e.target.value) })}
              style={inputStyle}
            />
          </div>
        </div>
      )}

      {/* Stage display */}
      {stage && (
        <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 12 }}>
          {stage === "done" ? (
            <span style={{ color: "var(--accent)" }}>
              ✓ Complete
              {reportId && (
                <a
                  href={`/research/${reportId}`}
                  style={{
                    color: accentColor,
                    marginLeft: 8,
                    textDecoration: "none",
                  }}
                >
                  View report →
                </a>
              )}
            </span>
          ) : stage === "error" ? (
            <span style={{ color: "#ef4444" }}>
              Failed: {error}
            </span>
          ) : (
            <span>{stageLabel(stage)}</span>
          )}
        </div>
      )}
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
  const [hhRobId, setHhRobId] = useState<string | null>(null);
  const [hhRobStage, setHhRobStage] = useState<string | null>(null);
  const [hhRobReportId, setHhRobReportId] = useState<string | null>(null);

  // Filtered configs for HH inline dropdowns (competitive only)
  const [hhFilteredConfigs, setHhFilteredConfigs] = useState<ConfigListItem[]>([]);
  const [hhFilteredLoading, setHhFilteredLoading] = useState(false);

  // Recompute feedback state (per archetype so only one message shows)
  const [rsRecomputeStatus, setRsRecomputeStatus] = useState<"idle" | "running" | "success" | "error">("idle");
  const [hhRecomputeStatus, setHhRecomputeStatus] = useState<"idle" | "running" | "success" | "error">("idle");

  // --- Pipeline state (per archetype) ---
  const [rsPipelineId, setRsPipelineId] = useState<string | null>(null);
  const [rsPipelineStage, setRsPipelineStage] = useState<string | null>(null);
  const [rsPipelineError, setRsPipelineError] = useState<string | null>(null);
  const [rsPipelineReportId, setRsPipelineReportId] = useState<string | null>(null);

  const [hhPipelineId, setHhPipelineId] = useState<string | null>(null);
  const [hhPipelineStage, setHhPipelineStage] = useState<string | null>(null);
  const [hhPipelineError, setHhPipelineError] = useState<string | null>(null);
  const [hhPipelineReportId, setHhPipelineReportId] = useState<string | null>(null);

  // --- Cooperative state ---
  const [coopMembers, setCoopMembers] = useState<CooperativeLeagueMember[]>([]);
  const [coopLineage, setCoopLineage] = useState<CooperativeLineageMember[]>([]);
  const [coopEvolution, setCoopEvolution] = useState<CooperativeEvolutionResponse>({ members: [], champion_history: [] });
  const [coopChampion, setCoopChampion] = useState<CooperativeChampionInfo | null>(null);
  const [coopRobData, setCoopRobData] = useState<CooperativeRobustnessHeatmapResponse | null>(null);
  const [coopLoading, setCoopLoading] = useState(true);
  const [coopError, setCoopError] = useState<string | null>(null);
  const [coopRobRunning, setCoopRobRunning] = useState(false);
  const [coopRobId, setCoopRobId] = useState<string | null>(null);
  const [coopRobStage, setCoopRobStage] = useState<string | null>(null);
  const [coopRobReportId, setCoopRobReportId] = useState<string | null>(null);
  const [coopRobSeeds, setCoopRobSeeds] = useState(3);
  const [coopRobEpisodesPerSeed, setCoopRobEpisodesPerSeed] = useState(2);
  const [coopRobSeed, setCoopRobSeed] = useState(42);

  // --- Pipeline config state (per archetype) ---
  const [rsPipelineConfig, setRsPipelineConfig] = useState<PipelineConfig>({
    totalTimesteps: 50000,
    snapshotEvery: 10000,
    seed: 42,
  });

  const [hhPipelineConfig, setHhPipelineConfig] = useState<PipelineConfig>({
    totalTimesteps: 50000,
    snapshotEvery: 10000,
    seed: 42,
  });

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

  // --- Load Cooperative data ---
  async function loadCooperative() {
    setCoopLoading(true);
    setCoopError(null);
    try {
      const [m, lin, evo, champ] = await Promise.all([
        getCooperativeLeagueMembers(),
        getCooperativeLeagueLineage(),
        getCooperativeLeagueEvolution(),
        getCooperativeChampion().catch(() => null),
      ]);
      setCoopMembers(m);
      setCoopLineage(lin.members);
      setCoopEvolution(evo);
      setCoopChampion(champ);
    } catch (e) {
      setCoopError(String(e));
    } finally {
      setCoopLoading(false);
    }
  }

  useEffect(() => {
    loadResourceSharing();
    loadHeadToHead();
    loadCooperative();
  }, []);

  // Filter hhConfigs to only competitive configs for inline HH dropdowns
  useEffect(() => {
    if (hhConfigs.length === 0) {
      setHhFilteredConfigs([]);
      return;
    }
    let cancelled = false;
    async function filter() {
      setHhFilteredLoading(true);
      const results: ConfigListItem[] = [];
      for (const c of hhConfigs) {
        try {
          const detail = await getConfigDetail(c.config_id);
          const identity = detail.identity as Record<string, unknown> | undefined;
          if (identity?.environment_type === "competitive") {
            results.push(c);
          }
        } catch { /* skip */ }
      }
      if (!cancelled) {
        setHhFilteredConfigs(results);
        if (results.length > 0 && !hhBenchConfigId) {
          setHhBenchConfigId(results[0].config_id);
        }
        if (results.length > 0 && hhRobConfigId === "default") {
          setHhRobConfigId(results[0].config_id);
        }
        setHhFilteredLoading(false);
      }
    }
    filter();
    return () => { cancelled = true; };
  }, [hhConfigs]);

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
    setHhRobId(null);
    setHhRobStage(null);
    setHhRobReportId(null);
    try {
      const payload: CompetitiveChampionRobustnessRequest = {
        config_id: hhRobConfigId,
        seeds: hhRobSeeds,
        episodes_per_seed: hhRobEpisodesPerSeed,
        seed: hhRobSeed,
        ...(hhRobLimitSweeps !== ""
          ? { limit_sweeps: Number(hhRobLimitSweeps) }
          : {}),
      };
      const resp = await runCompetitiveChampionRobustness(payload);
      setHhRobId(resp.robustness_id);
      setHhRobStage("loading_config");
    } catch (e) {
      setHhError(String(e));
      setHhRobRunning(false);
    }
  }

  async function handleCoopRobustness() {
    setCoopRobRunning(true);
    setCoopError(null);
    setCoopRobId(null);
    setCoopRobStage(null);
    setCoopRobReportId(null);
    setCoopRobData(null);
    try {
      const resp = await runCooperativeChampionRobustness({
        seeds: coopRobSeeds,
        episodes_per_seed: coopRobEpisodesPerSeed,
        seed: coopRobSeed,
      });
      setCoopRobId(resp.robustness_id);
      setCoopRobStage("loading_config");
    } catch (e) {
      setCoopError(String(e));
      setCoopRobRunning(false);
    }
  }

  // --- Pipeline handlers ---
  async function handleStartRsPipeline() {
    try {
      const { pipeline_id } = await startMixedPipeline({
        total_timesteps: rsPipelineConfig.totalTimesteps,
        snapshot_every_timesteps: rsPipelineConfig.snapshotEvery,
        seed: rsPipelineConfig.seed,
        episodes_per_seed: 2,
        seeds: 3,
      });
      setRsPipelineId(pipeline_id);
      setRsPipelineStage("loading_config");
      setRsPipelineError(null);
      setRsPipelineReportId(null);
    } catch (e) {
      setRsPipelineError(String(e));
    }
  }

  async function handleStartHhPipeline() {
    try {
      const seedList = Array.from({ length: 3 }, (_, i) => hhPipelineConfig.seed + i);
      const { pipeline_id } = await startCompetitivePipeline({
        total_timesteps: hhPipelineConfig.totalTimesteps,
        snapshot_every_timesteps: hhPipelineConfig.snapshotEvery,
        seed: hhPipelineConfig.seed,
        episodes_per_seed: 2,
        seeds: seedList,
      });
      setHhPipelineId(pipeline_id);
      setHhPipelineStage("loading_config");
      setHhPipelineError(null);
      setHhPipelineReportId(null);
    } catch (e) {
      setHhPipelineError(String(e));
    }
  }

  // Pipeline polling
  useEffect(() => {
    if (!rsPipelineId || rsPipelineStage === "done" || rsPipelineStage === "error") return;
    const interval = setInterval(async () => {
      try {
        const status = await getMixedPipelineStatus(rsPipelineId);
        setRsPipelineStage(status.stage);
        if (status.error) setRsPipelineError(status.error);
        if (status.report_id) setRsPipelineReportId(status.report_id);
        if (status.stage === "done" || status.stage === "error") {
          clearInterval(interval);
        }
      } catch {
        setRsPipelineStage("error");
        setRsPipelineError("Lost connection to server — check if backend is running");
        clearInterval(interval);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [rsPipelineId, rsPipelineStage]);

  useEffect(() => {
    if (!hhPipelineId || hhPipelineStage === "done" || hhPipelineStage === "error") return;
    const interval = setInterval(async () => {
      try {
        const status = await getCompetitivePipelineStatus(hhPipelineId);
        setHhPipelineStage(status.stage);
        if (status.error) setHhPipelineError(status.error);
        if (status.report_id) setHhPipelineReportId(status.report_id);
        if (status.stage === "done" || status.stage === "error") {
          clearInterval(interval);
        }
      } catch {
        setHhPipelineStage("error");
        setHhPipelineError("Lost connection to server — check if backend is running");
        clearInterval(interval);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [hhPipelineId, hhPipelineStage]);

  // Competitive robustness polling
  useEffect(() => {
    if (!hhRobId || hhRobStage === "done" || hhRobStage === "error") return;
    const interval = setInterval(async () => {
      try {
        const status = await getCompetitiveRobustnessStatus(hhRobId);
        setHhRobStage(status.stage);
        if (status.error) {
          setHhError(status.error);
          setHhRobRunning(false);
        }
        if (status.report_id) setHhRobReportId(status.report_id);
        if (status.stage === "done" || status.stage === "error") {
          setHhRobRunning(false);
          clearInterval(interval);
        }
      } catch {
        clearInterval(interval);
        setHhRobRunning(false);
        setHhRobStage("error");
        setHhError("Lost connection to server — check if backend is running");
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [hhRobId, hhRobStage]);

  // Cooperative robustness polling
  useEffect(() => {
    if (!coopRobId || coopRobStage === "done" || coopRobStage === "error") return;
    const interval = setInterval(async () => {
      try {
        const status = await getCooperativeRobustnessStatus(coopRobId);
        setCoopRobStage(status.stage);
        if (status.error) {
          setCoopError(status.error);
          setCoopRobRunning(false);
        }
        if (status.report_id) setCoopRobReportId(status.report_id);
        if (status.stage === "done" || status.stage === "error") {
          setCoopRobRunning(false);
          clearInterval(interval);
        }
      } catch {
        clearInterval(interval);
        setCoopRobRunning(false);
        setCoopRobStage("error");
        setCoopError("Lost connection to server — check if backend is running");
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [coopRobId, coopRobStage]);

  // --- Derived values ---
  const isRS = archetype === "resource-sharing";
  const isCoop = archetype === "cooperative";
  const loading = isCoop ? coopLoading : (isRS ? rsLoading : hhLoading);
  const error = isCoop ? coopError : (isRS ? rsError : hhError);
  const recomputing = isRS ? rsRecomputing : hhRecomputing;
  const members = isCoop ? coopMembers : (isRS ? rsMembers : hhMembers);

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

  // Resource Sharing champion (highest rated)
  const rsChampion: LeagueMember | null =
    rsSorted.length > 0 ? rsSorted[0] : null;
  const rsChampionRating = rsChampion ? rsRatings.get(rsChampion.member_id) : null;

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

      {/* Pipeline panel — always visible */}
      <div style={{
        background: "var(--bg-surface)",
        border: "1px solid var(--bg-border)",
        borderRadius: 8,
        padding: "20px 24px",
        marginBottom: 24,
      }}>
        {/* Header row */}
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}>
          <div>
            <p style={{
              fontSize: 13,
              fontWeight: 500,
              color: "var(--text-primary)",
              marginBottom: 4,
            }}>Training Pipeline</p>
            <p style={{
              fontSize: 12,
              color: "var(--text-secondary)",
            }}>
              Train PPO agents and grow the league automatically.
              Runs for both archetypes independently.
            </p>
          </div>
        </div>

        {/* Two archetype pipeline sections side by side */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}>
          {/* Resource Sharing pipeline */}
          <PipelinePanel
            label="Resource Sharing"
            accentColor="var(--accent)"
            stage={rsPipelineStage}
            error={rsPipelineError}
            reportId={rsPipelineReportId}
            onRun={handleStartRsPipeline}
            running={rsPipelineStage !== null &&
              rsPipelineStage !== "done" &&
              rsPipelineStage !== "error"}
            config={rsPipelineConfig}
            onConfigChange={setRsPipelineConfig}
          />

          {/* Head-to-Head pipeline */}
          <PipelinePanel
            label="Head-to-Head"
            accentColor="#f97316"
            stage={hhPipelineStage}
            error={hhPipelineError}
            reportId={hhPipelineReportId}
            onRun={handleStartHhPipeline}
            running={hhPipelineStage !== null &&
              hhPipelineStage !== "done" &&
              hhPipelineStage !== "error"}
            config={hhPipelineConfig}
            onConfigChange={setHhPipelineConfig}
          />
        </div>
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
        <button
          style={archetype === "cooperative" ? pillActive : pillInactive}
          onClick={() => handleArchetypeSwitch("cooperative")}
          onMouseEnter={(e) => {
            if (archetype !== "cooperative") {
              e.currentTarget.style.color = "var(--text-primary)";
              e.currentTarget.style.borderColor = "var(--accent)";
            }
          }}
          onMouseLeave={(e) => {
            if (archetype !== "cooperative") {
              e.currentTarget.style.color = "var(--text-secondary)";
              e.currentTarget.style.borderColor = "var(--bg-border)";
            }
          }}
        >
          Cooperative
        </button>

        {/* Recompute Ratings button — only for RS and HH */}
        {!isCoop && (
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
        )}
      </div>

      {/* Sub-tabs */}
      <div style={{ display: "flex", borderBottom: "1px solid var(--bg-border)", marginBottom: "32px" }}>
        {(["ratings", "champion", "evolution"] as Tab[]).map((t) => {
          const label = isCoop
            ? t === "ratings" ? "Lineage" : t === "champion" ? "Champion" : "Evolution"
            : t.charAt(0).toUpperCase() + t.slice(1);
          return (
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
              {label}
            </button>
          );
        })}
      </div>

      {error && <p style={{color: "#f87171", marginBottom: 8, fontSize: 13}}>{error}</p>}

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
                  <p style={{color: "var(--text-secondary)"}}>
                    No league members yet. Train a policy and save a snapshot to get started.
                  </p>
                ) : (
                  <table style={{width: "100%", textAlign: "left", fontSize: 13, borderCollapse: "collapse"}}>
                    <thead>
                      <tr style={{borderBottom: "1px solid var(--bg-border)"}}>
                        <th style={{padding: "8px 16px 8px 0"}}>#</th>
                        <th style={{padding: "8px 16px 8px 0"}}>Member ID</th>
                        <th style={{padding: "8px 16px 8px 0"}}>Rating</th>
                        <th style={{padding: "8px 16px 8px 0"}}>Parent</th>
                        <th style={{padding: "8px 16px 8px 0"}}>Created</th>
                        <th style={{padding: "8px 16px 8px 0"}}>Notes</th>
                        <th style={{padding: "8px 0"}} />
                      </tr>
                    </thead>
                    <tbody>
                      {rsSorted.map((m, idx) => (
                        <tr key={m.member_id} style={{borderBottom: "1px solid var(--bg-border)"}}>
                          <td style={{padding: "8px 16px 8px 0", color: "var(--text-tertiary)"}}>{idx + 1}</td>
                          <td style={{padding: "8px 16px 8px 0", fontFamily: "var(--font-mono)"}}>{m.member_id}</td>
                          <td style={{padding: "8px 16px 8px 0", fontFamily: "var(--font-mono)"}}>
                            {rsRatings.has(m.member_id)
                              ? rsRatings.get(m.member_id)!.toFixed(1)
                              : "—"}
                          </td>
                          <td style={{padding: "8px 16px 8px 0", fontFamily: "var(--font-mono)", fontSize: 12}}>
                            {m.parent_id ?? "—"}
                          </td>
                          <td style={{padding: "8px 16px 8px 0", fontSize: 12, color: "var(--text-secondary)"}}>
                            {m.created_at
                              ? new Date(m.created_at).toLocaleString()
                              : "—"}
                          </td>
                          <td style={{padding: "8px 16px 8px 0", fontSize: 12}}>{m.notes ?? "—"}</td>
                          <td style={{padding: "8px 0"}}>
                            <button
                              onClick={() => handleRsRun(m.member_id)}
                              disabled={rsStartingId !== null}
                              style={{padding: "4px 12px", background: "var(--accent)", color: "#fff", borderRadius: 6, fontSize: 13, border: "none", cursor: "pointer", opacity: rsStartingId !== null ? 0.5 : 1}}
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


              {/* Champion tab */}
              {tab === "champion" && (
                rsConfigs.length === 0 ? (
                  <p style={{color: "var(--text-secondary)"}}>
                    No configs available. Create one on the home page first.
                  </p>
                ) : (
                  <div style={{display: "flex", flexDirection: "column", gap: 24}}>
                    {/* Champion info */}
                    {rsChampion ? (
                      <div style={{border: "1px solid var(--bg-border)", borderRadius: 6, padding: 16, fontSize: 13}}>
                        <h3 style={{fontWeight: 600, marginBottom: 8}}>Current Champion</h3>
                        <dl style={{display: "grid", gridTemplateColumns: "1fr 1fr", columnGap: 24, rowGap: 4}}>
                          <dt style={{color: "var(--text-secondary)"}}>Member ID</dt>
                          <dd style={{fontFamily: "var(--font-mono)", fontSize: 12}}>{rsChampion.member_id}</dd>
                          <dt style={{color: "var(--text-secondary)"}}>Rating</dt>
                          <dd style={{fontFamily: "var(--font-mono)"}}>
                            {rsChampionRating != null
                              ? rsChampionRating.toFixed(1)
                              : "—"}
                          </dd>
                          <dt style={{color: "var(--text-secondary)"}}>Parent</dt>
                          <dd style={{fontFamily: "var(--font-mono)", fontSize: 12}}>
                            {rsChampion.parent_id ?? "none"}
                          </dd>
                        </dl>
                      </div>
                    ) : (
                      <p style={{color: "var(--text-secondary)"}}>
                        No league members yet. Train a policy and save a snapshot to get started.
                      </p>
                    )}

                    <div>
                      <h3 style={{fontSize: 13, fontWeight: 600, marginBottom: 8}}>Champion Benchmark</h3>
                      <ChampionBenchmark configs={rsConfigs} archetypeFilter="mixed" />
                    </div>
                    <div>
                      <h3 style={{fontSize: 13, fontWeight: 600, marginBottom: 8}}>Run Robustness on Champion</h3>
                      {rsMembers.length === 0 ? (
                        <p style={{color: "var(--text-secondary)"}}>
                          No league members yet. Train a policy and save a snapshot to get started.
                        </p>
                      ) : (
                        <ChampionRobustness configs={rsConfigs} archetypeFilter="mixed" />
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
                  <p style={{color: "var(--text-secondary)"}}>
                    No league members yet &mdash; run the pipeline first.
                  </p>
                ) : (
                  <table style={{width: "100%", textAlign: "left", fontSize: 13, borderCollapse: "collapse"}}>
                    <thead>
                      <tr style={{borderBottom: "1px solid var(--bg-border)"}}>
                        <th style={{padding: "8px 16px 8px 0"}}>#</th>
                        <th style={{padding: "8px 16px 8px 0"}}>Member ID</th>
                        <th style={{padding: "8px 16px 8px 0"}}>Rating</th>
                        <th style={{padding: "8px 16px 8px 0"}}>Parent</th>
                        <th style={{padding: "8px 16px 8px 0"}}>Created</th>
                        <th style={{padding: "8px 0"}} />
                      </tr>
                    </thead>
                    <tbody>
                      {hhSorted.map((m, idx) => (
                        <tr key={m.member_id} style={{borderBottom: "1px solid var(--bg-border)"}}>
                          <td style={{padding: "8px 16px 8px 0", color: "var(--text-tertiary)"}}>{idx + 1}</td>
                          <td style={{padding: "8px 16px 8px 0", fontFamily: "var(--font-mono)"}}>{m.member_id}</td>
                          <td style={{padding: "8px 16px 8px 0", fontFamily: "var(--font-mono)"}}>
                            {hhRatings.has(m.member_id)
                              ? hhRatings.get(m.member_id)!.toFixed(1)
                              : "—"}
                          </td>
                          <td style={{padding: "8px 16px 8px 0", fontFamily: "var(--font-mono)", fontSize: 12}}>
                            {m.parent_id ?? "—"}
                          </td>
                          <td style={{padding: "8px 16px 8px 0", fontSize: 12, color: "var(--text-secondary)"}}>
                            {m.created_at
                              ? new Date(m.created_at).toLocaleString()
                              : "—"}
                          </td>
                          <td style={{padding: "8px 0"}}>
                            <button
                              onClick={() => handleHhRun(m.member_id)}
                              disabled={hhStartingId !== null}
                              style={{padding: "4px 12px", background: "var(--accent)", color: "#fff", borderRadius: 6, fontSize: 13, border: "none", cursor: "pointer", opacity: hhStartingId !== null ? 0.5 : 1}}
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


              {/* Champion tab */}
              {tab === "champion" && (
                <div style={{display: "flex", flexDirection: "column", gap: 24}}>
                  {/* Champion info */}
                  {hhChampion ? (
                    <div style={{border: "1px solid var(--bg-border)", borderRadius: 6, padding: 16, fontSize: 13}}>
                      <h3 style={{fontWeight: 600, marginBottom: 8}}>Current Champion</h3>
                      <dl style={{display: "grid", gridTemplateColumns: "1fr 1fr", columnGap: 24, rowGap: 4}}>
                        <dt style={{color: "var(--text-secondary)"}}>Member ID</dt>
                        <dd style={{fontFamily: "var(--font-mono)", fontSize: 12}}>{hhChampion.member_id}</dd>
                        <dt style={{color: "var(--text-secondary)"}}>Rating</dt>
                        <dd style={{fontFamily: "var(--font-mono)"}}>
                          {hhChampionRating != null
                            ? hhChampionRating.toFixed(1)
                            : "—"}
                        </dd>
                        <dt style={{color: "var(--text-secondary)"}}>Parent</dt>
                        <dd style={{fontFamily: "var(--font-mono)", fontSize: 12}}>
                          {hhChampion.parent_id ?? "none"}
                        </dd>
                      </dl>
                    </div>
                  ) : (
                    <p style={{color: "var(--text-secondary)"}}>
                      No league members yet &mdash; run the pipeline first.
                    </p>
                  )}

                  {/* Benchmark section */}
                  <div>
                    <h3 style={{fontSize: 13, fontWeight: 600, marginBottom: 8}}>
                      Champion Benchmark
                    </h3>
                    <div style={{display: "flex", alignItems: "flex-end", gap: 12}}>
                      <div>
                        <label style={{display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4}}>
                          Config
                        </label>
                        {hhFilteredLoading ? (
                          <span style={{fontSize: 12, color: "var(--text-tertiary)"}}>Loading configs...</span>
                        ) : (
                          <select
                            value={hhBenchConfigId}
                            onChange={(e) => setHhBenchConfigId(e.target.value)}
                            style={{border: "1px solid var(--bg-border)", borderRadius: 4, padding: "4px 8px", fontSize: 13, background: "var(--bg-base)", color: "var(--text-primary)"}}
                          >
                            {hhFilteredConfigs.map((c) => (
                              <option key={c.config_id} value={c.config_id}>
                                {c.config_id} (agents={c.num_agents}, steps=
                                {c.max_steps})
                              </option>
                            ))}
                          </select>
                        )}
                      </div>
                      <div>
                        <label style={{display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4}}>
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
                          style={{border: "1px solid var(--bg-border)", borderRadius: 4, padding: "4px 8px", fontSize: 13, width: 64, background: "var(--bg-base)", color: "var(--text-primary)"}}
                        />
                      </div>
                      <button
                        onClick={handleHhBenchmark}
                        disabled={
                          hhBenchRunning ||
                          hhFilteredConfigs.length === 0 ||
                          hhMembers.length === 0 ||
                          hhFilteredLoading
                        }
                        style={{padding: "4px 12px", background: "#9333ea", color: "#fff", borderRadius: 6, fontSize: 13, border: "none", cursor: "pointer", opacity: (hhBenchRunning || hhFilteredConfigs.length === 0 || hhMembers.length === 0 || hhFilteredLoading) ? 0.5 : 1}}
                      >
                        {hhBenchRunning ? "Running..." : "Run Champion Benchmark"}
                      </button>
                    </div>

                    {hhBenchData && (
                      <div style={{marginTop: 16, display: "flex", flexDirection: "column" as const, gap: 12}}>
                        <p style={{fontSize: 13, color: "var(--text-secondary)"}}>
                          Champion:{" "}
                          <span style={{fontFamily: "var(--font-mono)"}}>
                            {hhBenchData.champion.member_id}
                          </span>{" "}
                          (rating{" "}
                          {hhBenchData.champion.rating != null
                            ? hhBenchData.champion.rating.toFixed(1)
                            : "—"}
                          )
                        </p>

                        <h4 style={{fontSize: 13, fontWeight: 600}}>
                          Mean Total Reward
                        </h4>
                        <CompetitiveBarChart results={hhBenchData.results} />

                        <table style={{width: "100%", textAlign: "left", fontSize: 12, borderCollapse: "collapse"}}>
                          <thead>
                            <tr style={{borderBottom: "1px solid var(--bg-border)"}}>
                              <th style={{padding: "4px 12px 4px 0"}}>Policy</th>
                              <th style={{padding: "4px 12px 4px 0"}}>Mean Reward</th>
                              <th style={{padding: "4px 12px 4px 0"}}>Mean Score</th>
                              <th style={{padding: "4px 12px 4px 0"}}>Win Rate</th>
                              <th style={{padding: "4px 12px 4px 0"}}>Mean Length</th>
                            </tr>
                          </thead>
                          <tbody>
                            {hhBenchData.results.map((r) => (
                              <tr
                                key={r.policy}
                                style={{borderBottom: "1px solid var(--bg-border)"}}
                              >
                                <td style={{padding: "4px 12px 4px 0", fontFamily: "var(--font-mono)"}}>
                                  {r.policy}
                                </td>
                                <td style={{padding: "4px 12px 4px 0"}}>
                                  {r.mean_total_reward.toFixed(4)}
                                </td>
                                <td style={{padding: "4px 12px 4px 0"}}>
                                  {r.mean_final_score.toFixed(2)}
                                </td>
                                <td style={{padding: "4px 12px 4px 0"}}>
                                  {(r.win_rate * 100).toFixed(0)}%
                                </td>
                                <td style={{padding: "4px 12px 4px 0"}}>
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
                    <h3 style={{fontSize: 13, fontWeight: 600, marginBottom: 8}}>
                      Run Robustness on Champion
                    </h3>
                    <p style={{fontSize: 13, color: "var(--text-secondary)", marginBottom: 12}}>
                      Evaluates the competitive league champion against all baseline
                      policies across multiple environment variants and saves a
                      robustness report.
                    </p>

                    <div style={{display: "flex", flexWrap: "wrap", alignItems: "flex-end", gap: 12}}>
                      <div>
                        <label style={{display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4}}>
                          Config
                        </label>
                        {hhFilteredLoading ? (
                          <span style={{fontSize: 12, color: "var(--text-tertiary)"}}>Loading configs...</span>
                        ) : (
                          <select
                            value={hhRobConfigId}
                            onChange={(e) => setHhRobConfigId(e.target.value)}
                            style={{border: "1px solid var(--bg-border)", borderRadius: 4, padding: "4px 8px", fontSize: 13, background: "var(--bg-base)", color: "var(--text-primary)"}}
                          >
                            {hhFilteredConfigs.map((c) => (
                              <option key={c.config_id} value={c.config_id}>
                                {c.config_id} (agents={c.num_agents}, steps=
                                {c.max_steps})
                              </option>
                            ))}
                          </select>
                        )}
                      </div>
                      <div>
                        <label style={{display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4}}>
                          Seeds
                        </label>
                        <input
                          type="number"
                          min={1}
                          max={20}
                          value={hhRobSeeds}
                          onChange={(e) => setHhRobSeeds(Number(e.target.value))}
                          style={{border: "1px solid var(--bg-border)", borderRadius: 4, padding: "4px 8px", fontSize: 13, width: 64, background: "var(--bg-base)", color: "var(--text-primary)"}}
                        />
                      </div>
                      <div>
                        <label style={{display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4}}>
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
                          style={{border: "1px solid var(--bg-border)", borderRadius: 4, padding: "4px 8px", fontSize: 13, width: 64, background: "var(--bg-base)", color: "var(--text-primary)"}}
                        />
                      </div>
                      <div>
                        <label style={{display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4}}>
                          Limit sweeps (opt.)
                        </label>
                        <input
                          type="number"
                          min={1}
                          placeholder="—"
                          value={hhRobLimitSweeps}
                          onChange={(e) => setHhRobLimitSweeps(e.target.value)}
                          style={{border: "1px solid var(--bg-border)", borderRadius: 4, padding: "4px 8px", fontSize: 13, width: 80, background: "var(--bg-base)", color: "var(--text-primary)"}}
                        />
                      </div>
                      <div>
                        <label style={{display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4}}>
                          Seed
                        </label>
                        <input
                          type="number"
                          value={hhRobSeed}
                          onChange={(e) => setHhRobSeed(Number(e.target.value))}
                          style={{border: "1px solid var(--bg-border)", borderRadius: 4, padding: "4px 8px", fontSize: 13, width: 80, background: "var(--bg-base)", color: "var(--text-primary)"}}
                        />
                      </div>
                      <button
                        onClick={handleHhRobustness}
                        disabled={hhRobRunning || hhMembers.length === 0}
                        style={{padding: "4px 12px", background: "#4f46e5", color: "#fff", borderRadius: 6, fontSize: 13, border: "none", cursor: "pointer", opacity: (hhRobRunning || hhMembers.length === 0) ? 0.5 : 1}}
                      >
                        {hhRobRunning ? "Running..." : "Run Robustness"}
                      </button>
                    </div>

                    {hhRobRunning && hhRobStage && (
                      <p style={{fontSize: 13, color: "var(--text-secondary)", marginTop: 8}}>
                        {hhRobStage === "loading_config" && "Loading configuration..."}
                        {hhRobStage === "evaluating" && "Running robustness sweeps..."}
                        {hhRobStage === "writing_report" && "Generating report..."}
                      </p>
                    )}

                    {hhRobStage === "done" && hhRobReportId && (
                      <p style={{fontSize: 13, color: "var(--accent)", marginTop: 8}}>
                        Complete!{" "}
                        <a
                          href={`/research/${encodeURIComponent(hhRobReportId)}`}
                          style={{textDecoration: "underline", fontWeight: 500}}
                        >
                          View report &rarr;
                        </a>
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

          {/* ============================================================ */}
          {/* COOPERATIVE CONTENT                                           */}
          {/* ============================================================ */}
          {isCoop && (
            <>
              {/* Lineage tab (= ratings) */}
              {tab === "ratings" && (
                coopLineage.length === 0 ? (
                  <p style={{ color: "var(--text-secondary)" }}>
                    No league members yet — run the cooperative pipeline first.
                  </p>
                ) : (
                  <CooperativeLeagueLineage members={coopLineage} />
                )
              )}

              {/* Champion tab */}
              {tab === "champion" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                  {/* Champion info */}
                  {coopChampion && coopChampion.member_id ? (
                    <div style={{ border: "1px solid var(--bg-border)", borderRadius: 6, padding: 16, fontSize: 13 }}>
                      <h3 style={{ fontWeight: 600, marginBottom: 8 }}>Current Champion</h3>
                      <dl style={{ display: "grid", gridTemplateColumns: "1fr 1fr", columnGap: 24, rowGap: 4 }}>
                        <dt style={{ color: "var(--text-secondary)" }}>Member ID</dt>
                        <dd style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{coopChampion.member_id}</dd>
                        <dt style={{ color: "var(--text-secondary)" }}>Rating</dt>
                        <dd style={{ fontFamily: "var(--font-mono)" }}>
                          {coopChampion.rating != null ? coopChampion.rating.toFixed(1) : "—"}
                        </dd>
                        <dt style={{ color: "var(--text-secondary)" }}>Parent</dt>
                        <dd style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
                          {coopChampion.parent_id ?? "none"}
                        </dd>
                      </dl>
                    </div>
                  ) : (
                    <p style={{ color: "var(--text-secondary)" }}>
                      No league members yet — run the cooperative pipeline first.
                    </p>
                  )}

                  {/* Benchmark results */}
                  <div>
                    <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Champion Benchmark</h3>
                    <CooperativeChampionBenchmark
                      champion={coopChampion && coopChampion.member_id
                        ? { member_id: coopChampion.member_id, rating: coopChampion.rating ?? 0 }
                        : null}
                      results={[]}
                    />
                  </div>

                  {/* Robustness */}
                  <div>
                    <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Run Robustness on Champion</h3>
                    {coopMembers.length === 0 ? (
                      <p style={{ color: "var(--text-secondary)" }}>
                        No league members yet — run the cooperative pipeline first.
                      </p>
                    ) : (
                      <>
                        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", gap: 12 }}>
                          <div>
                            <label style={{ display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>Seeds</label>
                            <input
                              type="number" min={1} max={20} value={coopRobSeeds}
                              onChange={(e) => setCoopRobSeeds(Number(e.target.value))}
                              style={{ border: "1px solid var(--bg-border)", borderRadius: 4, padding: "4px 8px", fontSize: 13, width: 64, background: "var(--bg-base)", color: "var(--text-primary)" }}
                            />
                          </div>
                          <div>
                            <label style={{ display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>Episodes/seed</label>
                            <input
                              type="number" min={1} max={10} value={coopRobEpisodesPerSeed}
                              onChange={(e) => setCoopRobEpisodesPerSeed(Number(e.target.value))}
                              style={{ border: "1px solid var(--bg-border)", borderRadius: 4, padding: "4px 8px", fontSize: 13, width: 64, background: "var(--bg-base)", color: "var(--text-primary)" }}
                            />
                          </div>
                          <div>
                            <label style={{ display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>Seed</label>
                            <input
                              type="number" value={coopRobSeed}
                              onChange={(e) => setCoopRobSeed(Number(e.target.value))}
                              style={{ border: "1px solid var(--bg-border)", borderRadius: 4, padding: "4px 8px", fontSize: 13, width: 80, background: "var(--bg-base)", color: "var(--text-primary)" }}
                            />
                          </div>
                          <button
                            onClick={handleCoopRobustness}
                            disabled={coopRobRunning || coopMembers.length === 0}
                            style={{ padding: "4px 12px", background: "#8b5cf6", color: "#fff", borderRadius: 6, fontSize: 13, border: "none", cursor: "pointer", opacity: (coopRobRunning || coopMembers.length === 0) ? 0.5 : 1 }}
                          >
                            {coopRobRunning ? "Running..." : "Run Robustness"}
                          </button>
                        </div>
                        {coopRobRunning && coopRobStage && (
                          <p style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 8 }}>
                            {coopRobStage === "loading_config" && "Loading configuration..."}
                            {coopRobStage === "evaluating" && "Running robustness sweeps..."}
                            {coopRobStage === "writing_report" && "Generating report..."}
                          </p>
                        )}
                        {coopRobStage === "done" && coopRobReportId && (
                          <p style={{ fontSize: 13, color: "var(--accent)", marginTop: 8 }}>
                            Complete!{" "}
                            <a href={`/research/${encodeURIComponent(coopRobReportId)}`} style={{ textDecoration: "underline", fontWeight: 500 }}>
                              View report &rarr;
                            </a>
                          </p>
                        )}
                        {coopRobData && <CooperativeChampionRobustness data={coopRobData} />}
                      </>
                    )}
                  </div>
                </div>
              )}

              {/* Evolution tab */}
              {tab === "evolution" && (
                <div>
                  {coopEvolution.members.length === 0 && coopEvolution.champion_history.length === 0 ? (
                    <p style={{ color: "var(--text-tertiary)" }}>
                      No evolution data yet. Train and save snapshots to build history.
                    </p>
                  ) : (
                    <LineageGraph nodes={coopEvolution.members.map((m) => ({
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
                  {coopEvolution.champion_history.length > 0 && (
                    <div style={{ marginTop: 32 }}>
                      <h3 style={{ fontSize: 13, fontWeight: 500, color: "#888888", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 16 }}>
                        Champion History
                      </h3>
                      <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 400, overflowY: "auto" }}>
                        {coopEvolution.champion_history.map((entry, idx) => (
                          <div key={entry.member_id} style={{ border: "1px solid var(--bg-border)", borderRadius: 6, padding: "10px 14px", fontSize: 13 }}>
                            <span style={{ color: "var(--text-tertiary)", marginRight: 8 }}>#{idx + 1}</span>
                            <span style={{ fontFamily: "var(--font-mono)" }}>{entry.member_id}</span>
                            <span style={{ marginLeft: 12, color: "var(--text-secondary)" }}>
                              rating {entry.rating.toFixed(1)}
                            </span>
                            {entry.label && (
                              <span style={{ marginLeft: 12, fontSize: 11, color: "var(--text-tertiary)" }}>{entry.label}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </>
      )}
    </main>
  );
}
