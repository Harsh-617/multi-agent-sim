/** REST + WebSocket helpers for the simulation backend. */

const BASE = "/api";

// ---------------------------------------------------------------------------
// Types (mirroring backend schemas)
// ---------------------------------------------------------------------------

export interface ConfigListItem {
  config_id: string;
  seed: number;
  num_agents: number;
  max_steps: number;
}

export interface RunStatusResponse {
  running: boolean;
  run_id: string | null;
  step: number | null;
  max_steps: number | null;
  termination_reason: string | null;
}

export interface StepMetric {
  step: number;
  agent_id: string;
  reward: number;
  action_type: string;
  action_amount: number;
  shared_pool: number;
  agent_resources: number;
  coop_ratio: number;
  extraction_ratio: number;
}

export interface StepEvent {
  event: string;
  step: number;
  shared_pool?: number;
  agent_id?: string;
  resources?: number;
}

export interface EpisodeSummary {
  episode_length: number;
  termination_reason: string;
  final_shared_pool: number;
  total_reward_per_agent: Record<string, number>;
}

export interface WsStepMessage {
  type: "step";
  run_id: string;
  t: number;
  metrics: StepMetric[];
  events: StepEvent[] | null;
}

export interface WsDoneMessage {
  type: "done";
  run_id: string;
  termination_reason: string;
  episode_summary: EpisodeSummary;
}

export type WsMessage = WsStepMessage | WsDoneMessage;

// ---------------------------------------------------------------------------
// REST helpers
// ---------------------------------------------------------------------------

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function listConfigs(): Promise<ConfigListItem[]> {
  return json(`${BASE}/configs`);
}

export function createDefaultConfig(): Promise<{ config_id: string }> {
  return json(`${BASE}/configs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      identity: {
        environment_type: "mixed",
        environment_version: "0.1.0",
        seed: 42,
      },
      population: {
        num_agents: 5,
        max_steps: 200,
        initial_shared_pool: 100.0,
        initial_agent_resources: 10.0,
        collapse_threshold: 5.0,
      },
      layers: {
        information_asymmetry: 0.3,
        temporal_memory_depth: 10,
        reputation_sensitivity: 0.5,
        incentive_softness: 0.5,
        uncertainty_intensity: 0.1,
      },
      rewards: {
        individual_weight: 1.0,
        group_weight: 0.5,
        relational_weight: 0.3,
        penalty_scaling: 1.0,
      },
      agents: {
        observation_memory_steps: 5,
      },
      instrumentation: {
        enable_step_metrics: true,
        enable_episode_metrics: true,
        enable_event_log: true,
        step_log_frequency: 1,
      },
    }),
  });
}

export type AgentPolicy =
  | "random"
  | "always_cooperate"
  | "always_extract"
  | "tit_for_tat"
  | "ppo_shared"
  | "league_snapshot";

export function startRun(
  configId: string,
  agentPolicy: AgentPolicy = "random",
  leagueMemberId?: string,
): Promise<{ run_id: string }> {
  return json(`${BASE}/runs/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      config_id: configId,
      agent_policy: agentPolicy,
      ...(leagueMemberId ? { league_member_id: leagueMemberId } : {}),
    }),
  });
}

export function stopRun(): Promise<{ detail: string; run_id: string }> {
  return json(`${BASE}/runs/stop`, { method: "POST" });
}

export function getRunStatus(): Promise<RunStatusResponse> {
  return json(`${BASE}/runs/status`);
}

// ---------------------------------------------------------------------------
// Run history
// ---------------------------------------------------------------------------

export interface RunListItem {
  run_id: string;
  config_id: string | null;
  agent_policy: string | null;
  termination_reason: string | null;
  episode_length: number | null;
  timestamp: string | null;
}

export interface RunDetail extends RunListItem {
  episode_summary: EpisodeSummary | null;
}

export function listRuns(): Promise<RunListItem[]> {
  return json(`${BASE}/runs/history`);
}

export function getRunDetail(runId: string): Promise<RunDetail> {
  return json(`${BASE}/runs/${runId}/detail`);
}

export interface BenchmarkResult {
  agent_policy: string;
  mean_reward: number | null;
  final_shared_pool: number | null;
  termination_reason: string | null;
  episode_length: number | null;
}

export interface BenchmarkResponse {
  config_id: string;
  results: BenchmarkResult[];
}

export function runBenchmark(
  configId: string,
  agentPolicies: string[],
): Promise<BenchmarkResponse> {
  return json(`${BASE}/benchmark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_id: configId, agent_policies: agentPolicies }),
  });
}

// ---------------------------------------------------------------------------
// Replay (SSE)
// ---------------------------------------------------------------------------

export function connectReplay(
  runId: string,
  onMessage: (msg: WsMessage) => void,
  onDone?: () => void,
): EventSource {
  const es = new EventSource(`${BASE}/runs/${runId}/replay`);
  es.onmessage = (ev) => {
    const msg = JSON.parse(ev.data) as WsMessage;
    onMessage(msg);
    if (msg.type === "done") {
      es.close();
      onDone?.();
    }
  };
  es.onerror = () => {
    es.close();
    onDone?.();
  };
  return es;
}

// ---------------------------------------------------------------------------
// League
// ---------------------------------------------------------------------------

export interface LeagueMember {
  member_id: string;
  parent_id: string | null;
  created_at: string | null;
  notes: string | null;
  [key: string]: unknown;
}

export interface LeagueRating {
  member_id: string;
  rating: number;
}

export function listLeagueMembers(): Promise<LeagueMember[]> {
  return json(`${BASE}/league/members`);
}

export function getLeagueRatings(): Promise<LeagueRating[]> {
  return json(`${BASE}/league/ratings`);
}

export function recomputeLeagueRatings(
  numMatches: number = 10,
  seed: number = 42,
): Promise<LeagueRating[]> {
  return json(`${BASE}/league/ratings/recompute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_matches: numMatches, seed }),
  });
}

// Lineage

export interface LineageMember {
  member_id: string;
  parent_id: string | null;
  created_at: string | null;
  notes: string | null;
  rating: number;
}

export interface LineageResponse {
  members: LineageMember[];
}

export function getLeagueLineage(): Promise<LineageResponse> {
  return json(`${BASE}/league/lineage`);
}

// Champion

export interface ChampionInfo {
  member_id: string;
  rating: number;
  parent_id: string | null;
  created_at: string | null;
  notes: string | null;
}

export function getChampion(): Promise<ChampionInfo> {
  return json(`${BASE}/league/champion`);
}

// Champion benchmark

export interface ChampionBenchmarkResult {
  policy: string;
  mean_total_reward: number;
  mean_final_shared_pool: number;
  collapse_rate: number;
  mean_episode_length: number;
}

export interface ChampionBenchmarkResponse {
  champion: ChampionInfo;
  results: ChampionBenchmarkResult[];
}

export function runChampionBenchmark(
  configId: string,
  episodes: number = 10,
  seed: number = 42,
): Promise<ChampionBenchmarkResponse> {
  return json(`${BASE}/league/champion/benchmark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_id: configId, episodes, seed }),
  });
}

// Champion robustness

export interface ChampionRobustnessRequest {
  config_id: string;
  seeds?: number;
  episodes_per_seed?: number;
  max_steps?: number | null;
  limit_sweeps?: number | null;
  seed?: number;
}

export interface ChampionRobustnessResponse {
  report_id: string;
  report_path: string;
}

export function runChampionRobustness(
  payload: ChampionRobustnessRequest,
): Promise<ChampionRobustnessResponse> {
  return json(`${BASE}/league/champion/robustness`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

export interface ReportListItem {
  report_id: string;
  kind: "eval" | "robust";
  config_hash: string;
  timestamp: string;
  path_name: string;
}

export function listReports(): Promise<ReportListItem[]> {
  return json(`${BASE}/reports`);
}

export function getReport(reportId: string): Promise<Record<string, unknown>> {
  return json(`${BASE}/reports/${encodeURIComponent(reportId)}`);
}

// ---------------------------------------------------------------------------
// Strategy Groups
// ---------------------------------------------------------------------------

export interface StrategyFeatureRow {
  mean_return?: number | null;
  worst_case_return?: number | null;
  collapse_rate?: number | null;
  mean_final_pool?: number | null;
  robustness_score?: number | null;
}

export type StrategyFeatures = Record<string, StrategyFeatureRow>;

export interface StrategyResponse {
  features: StrategyFeatures;
  clusters: Record<string, number>;
  labels: Record<string, string>;
  summaries: Record<string, string>;
}

export function getReportStrategies(reportId: string): Promise<StrategyResponse> {
  return json(`${BASE}/reports/${encodeURIComponent(reportId)}/strategies`);
}

// ---------------------------------------------------------------------------
// League Evolution
// ---------------------------------------------------------------------------

export interface LeagueEvolutionMember {
  member_id: string;
  parent_id: string | null;
  created_at: string | null;
  notes: string | null;
  rating: number;
  label: string;
  cluster_id: number | null;
  robustness_score: number | null;
}

export interface ChampionHistoryEntry {
  member_id: string;
  created_at: string | null;
  rating: number;
  label: string;
  cluster_id: number | null;
  robustness_score: number | null;
}

export interface LeagueEvolutionResponse {
  members: LeagueEvolutionMember[];
  champion_history: ChampionHistoryEntry[];
}

export function getLeagueEvolution(): Promise<LeagueEvolutionResponse> {
  return json(`${BASE}/league/evolution`);
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

export function connectMetrics(
  runId: string,
  onMessage: (msg: WsMessage) => void,
  onError?: (err: Event) => void,
  onClose?: () => void
): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.hostname}:8000/api/ws/metrics/${runId}`);

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data) as WsMessage;
    onMessage(msg);
  };

  ws.onerror = (ev) => onError?.(ev);
  ws.onclose = () => onClose?.();

  return ws;
}
