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

export interface CompetitiveEpisodeSummary {
  episode_length: number;
  termination_reason: string;
  winner_id: string | null;
  final_rankings: string[];
  final_scores: Record<string, number>;
  score_spread: number;
  num_eliminations: number;
  total_reward_per_agent: Record<string, number>;
}

/** Type guard: returns true when the summary looks like a competitive episode. */
export function isCompetitiveSummary(
  s: EpisodeSummary | CompetitiveEpisodeSummary,
): s is CompetitiveEpisodeSummary {
  return "winner_id" in s || "final_scores" in s;
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

export function getConfigDetail(configId: string): Promise<Record<string, unknown>> {
  return json(`${BASE}/configs/${encodeURIComponent(configId)}`);
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
  episode_summary: EpisodeSummary | CompetitiveEpisodeSummary | null;
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
    let msg: WsMessage;
    try {
      msg = JSON.parse(ev.data) as WsMessage;
    } catch {
      es.close();
      onDone?.();
      return;
    }
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
  label?: string;
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
  robustness_id: string;
}

export interface RobustnessStatusResponse {
  robustness_id: string;
  running: boolean;
  stage: string;
  error?: string;
  report_id?: string;
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

export function getMixedRobustnessStatus(
  robustnessId: string,
): Promise<RobustnessStatusResponse> {
  return json(
    `${BASE}/league/champion/robustness/${encodeURIComponent(robustnessId)}/status`,
  );
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
  strategy?: { label: string; cluster_id: number };
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
// Competitive League
// ---------------------------------------------------------------------------

export type CompetitiveAgentPolicy =
  | "random"
  | "always_attack"
  | "always_build"
  | "always_defend"
  | "competitive_ppo";

export interface CompetitiveConfigParams {
  num_agents: number;
  max_steps: number;
  seed: number;
  /* population */
  initial_score?: number;
  initial_resources?: number;
  resource_regeneration_rate?: number;
  elimination_threshold?: number;
  dominance_margin?: number;
  /* layers */
  information_asymmetry?: number;
  opponent_history_depth?: number;
  opponent_obs_window?: number;
  history_sensitivity?: number;
  incentive_softness?: number;
  uncertainty_intensity?: number;
  gamble_variance?: number;
  /* rewards */
  absolute_gain_weight?: number;
  relative_gain_weight?: number;
  efficiency_weight?: number;
  terminal_bonus_scale?: number;
  penalty_scaling?: number;
  /* agents */
  observation_memory_steps?: number;
}

export function createCompetitiveConfig(config: CompetitiveConfigParams): Promise<{ config_id: string }> {
  return json(`${BASE}/configs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      identity: {
        environment_type: "competitive",
        environment_version: "0.1.0",
        seed: config.seed,
      },
      population: {
        num_agents: config.num_agents,
        max_steps: config.max_steps,
        initial_score: config.initial_score ?? 0.0,
        initial_resources: config.initial_resources ?? 20.0,
        resource_regeneration_rate: config.resource_regeneration_rate ?? 1.0,
        elimination_threshold: config.elimination_threshold ?? 0.0,
        dominance_margin: config.dominance_margin ?? 0.0,
      },
      layers: {
        information_asymmetry: config.information_asymmetry ?? 0.3,
        opponent_history_depth: config.opponent_history_depth ?? 10,
        opponent_obs_window: config.opponent_obs_window ?? 5,
        history_sensitivity: config.history_sensitivity ?? 0.5,
        incentive_softness: config.incentive_softness ?? 0.8,
        uncertainty_intensity: config.uncertainty_intensity ?? 0.1,
        gamble_variance: config.gamble_variance ?? 0.5,
      },
      rewards: {
        absolute_gain_weight: config.absolute_gain_weight ?? 1.0,
        relative_gain_weight: config.relative_gain_weight ?? 0.5,
        efficiency_weight: config.efficiency_weight ?? 0.3,
        terminal_bonus_scale: config.terminal_bonus_scale ?? 2.0,
        penalty_scaling: config.penalty_scaling ?? 1.0,
      },
      agents: {
        observation_memory_steps: config.observation_memory_steps ?? 5,
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

export function startCompetitiveRun(
  configId: string,
  agentPolicy: CompetitiveAgentPolicy = "random",
): Promise<{ run_id: string }> {
  return json(`${BASE}/runs/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      config_id: configId,
      agent_policy: agentPolicy,
    }),
  });
}

export interface CompetitiveLeagueMember {
  member_id: string;
  parent_id: string | null;
  created_at: string | null;
  notes: string | null;
  [key: string]: unknown;
}

export interface CompetitiveLeagueRating {
  member_id: string;
  rating: number;
}

export function getCompetitiveLeagueMembers(): Promise<CompetitiveLeagueMember[]> {
  return json(`${BASE}/competitive/league/members`);
}

export function getCompetitiveLeagueRatings(): Promise<CompetitiveLeagueRating[]> {
  return json(`${BASE}/competitive/league/ratings`);
}

export function recomputeCompetitiveLeagueRatings(
  numMatches: number = 10,
  seed: number = 42,
): Promise<CompetitiveLeagueRating[]> {
  return json(`${BASE}/competitive/league/ratings/recompute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_matches: numMatches, seed }),
  });
}

export interface CompetitiveLineageMember {
  member_id: string;
  parent_id: string | null;
  created_at: string | null;
  notes: string | null;
  rating: number;
}

export interface CompetitiveLineageResponse {
  members: CompetitiveLineageMember[];
}

export function getCompetitiveLeagueLineage(): Promise<CompetitiveLineageResponse> {
  return json(`${BASE}/competitive/league/lineage`);
}

export interface CompetitiveChampionInfo {
  member_id: string | null;
  rating?: number;
  parent_id?: string | null;
  created_at?: string | null;
  notes?: string | null;
}

export function getCompetitiveChampion(): Promise<CompetitiveChampionInfo> {
  return json(`${BASE}/competitive/league/champion`);
}

// Competitive Champion Benchmark

export interface CompetitiveChampionBenchmarkResult {
  policy: string;
  mean_total_reward: number;
  mean_final_score: number;
  win_rate: number;
  mean_episode_length: number;
}

export interface CompetitiveChampionBenchmarkResponse {
  champion: CompetitiveChampionInfo;
  results: CompetitiveChampionBenchmarkResult[];
}

export function runCompetitiveChampionBenchmark(
  configId: string,
  episodes: number = 10,
  seed: number = 42,
): Promise<CompetitiveChampionBenchmarkResponse> {
  return json(`${BASE}/competitive/league/champion/benchmark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_id: configId, episodes, seed }),
  });
}

// Competitive Champion Robustness

export interface CompetitiveChampionRobustnessRequest {
  config_id: string;
  seeds?: number;
  episodes_per_seed?: number;
  limit_sweeps?: number | null;
  seed?: number;
}

export interface CompetitiveChampionRobustnessResponse {
  robustness_id: string;
}

export function runCompetitiveChampionRobustness(
  payload: CompetitiveChampionRobustnessRequest,
): Promise<CompetitiveChampionRobustnessResponse> {
  return json(`${BASE}/competitive/league/champion/robustness`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function getCompetitiveRobustnessStatus(
  robustnessId: string,
): Promise<RobustnessStatusResponse> {
  return json(
    `${BASE}/competitive/league/champion/robustness/${encodeURIComponent(robustnessId)}/status`,
  );
}

// Start competitive league member run

export function startCompetitiveLeagueMemberRun(
  configId: string,
  memberId: string,
): Promise<{ run_id: string }> {
  return json(`${BASE}/runs/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      config_id: configId,
      agent_policy: "competitive_ppo",
      league_member_id: memberId,
    }),
  });
}

// Competitive Evolution

export interface CompetitiveEvolutionMember {
  member_id: string;
  parent_id: string | null;
  created_at: string | null;
  notes: string | null;
  rating: number;
  strategy: {
    cluster_id: number | null;
    label: string;
    features: Record<string, unknown>;
  };
  robustness_score: number | null;
}

export interface CompetitiveEvolutionResponse {
  members: CompetitiveEvolutionMember[];
  champion_history: ChampionHistoryEntry[];
}

export function getCompetitiveLeagueEvolution(): Promise<CompetitiveEvolutionResponse> {
  return json(`${BASE}/competitive/league/evolution`);
}

// ---------------------------------------------------------------------------
// Competitive Reports
// ---------------------------------------------------------------------------

export interface CompetitiveReportListItem {
  report_id: string;
  kind: string;
  config_hash: string;
  timestamp: string;
  path_name: string;
}

export function getCompetitiveReports(): Promise<CompetitiveReportListItem[]> {
  return json(`${BASE}/competitive/reports`);
}

export function getCompetitiveReport(reportId: string): Promise<Record<string, unknown>> {
  return json(`${BASE}/competitive/reports/${encodeURIComponent(reportId)}`);
}

export function getCompetitiveReportStrategies(reportId: string): Promise<StrategyResponse> {
  return json(`${BASE}/competitive/reports/${encodeURIComponent(reportId)}/strategies`);
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

export interface PipelineRunParams {
  config_id?: string;
  seed?: number;
  seeds?: number;
  episodes_per_seed?: number;
  max_steps?: number | null;
  total_timesteps?: number;
  snapshot_every_timesteps?: number;
  max_league_members?: number;
  num_matches?: number;
  limit_sweeps?: number | null;
}

export interface PipelineStatusResponse {
  pipeline_id: string;
  running: boolean;
  stage: string;
  error?: string;
  report_id?: string;
  summary_path?: string;
}

export function startMixedPipeline(
  params: PipelineRunParams,
): Promise<{ pipeline_id: string }> {
  return json(`${BASE}/pipeline/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export function getMixedPipelineStatus(
  pipelineId: string,
): Promise<PipelineStatusResponse> {
  return json(`${BASE}/pipeline/${encodeURIComponent(pipelineId)}/status`);
}

export interface CompetitivePipelineRunParams {
  config_id?: string;
  seed?: number;
  seeds?: number[] | null;
  episodes_per_seed?: number;
  max_steps?: number | null;
  total_timesteps?: number;
  snapshot_every_timesteps?: number;
  max_league_members?: number;
  num_matches?: number;
  limit_sweeps?: number | null;
}

export function startCompetitivePipeline(
  params: CompetitivePipelineRunParams,
): Promise<{ pipeline_id: string }> {
  return json(`${BASE}/pipeline/competitive/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export function getCompetitivePipelineStatus(
  pipelineId: string,
): Promise<PipelineStatusResponse> {
  return json(
    `${BASE}/pipeline/competitive/${encodeURIComponent(pipelineId)}/status`,
  );
}

// ---------------------------------------------------------------------------
// Cooperative archetype
// ---------------------------------------------------------------------------

/** Per-agent step metrics for a cooperative run (mirrors STEP_METRIC_KEYS). */
export interface CooperativeStepMetric {
  step: number;
  agent_id: string;
  reward: number;
  task_type: number | null;
  effort_amount: number;
  effective_contribution: number;
  r_group: number;
  r_individual: number;
  r_efficiency: number;
  system_stress: number;
  backlog_level: number;
  completion_rate: number;
}

/** Per-agent episode-level metrics included in CooperativeEpisodeSummary. */
export interface CooperativeAgentMetrics {
  cumulative_reward: number;
  mean_reward_per_step: number;
  effort_utilization: number;
  idle_rate: number;
  dominant_task_type: number | null;
  dominant_type_fraction: number;
  final_specialization_score: number;
  peak_specialization_score: number;
  role_stability: number;
  strategy_label: string;
}

/** Full episode summary for a cooperative run. */
export interface CooperativeEpisodeSummary {
  episode_length: number;
  termination_reason: string | null;
  total_tasks_arrived: number;
  total_tasks_completed: number;
  completion_ratio: number;
  final_backlog_level: number;
  final_system_stress: number;
  mean_system_stress: number;
  peak_system_stress: number;
  collapse_occurred: boolean;
  total_reward_per_agent: Record<string, number>;
  mean_reward_per_step_per_agent: Record<string, number>;
  group_efficiency_ratio: number;
  contribution_variance: number;
  specialization_divergence: number;
  mean_role_stability: number;
  free_rider_count: number;
  free_rider_fraction: number;
  effort_gini_coefficient: number;
  agent_metrics: Record<string, CooperativeAgentMetrics>;
}

/** Lightweight cooperative run list item. */
export interface CooperativeRunListItem {
  run_id: string;
  seed: number | null;
  num_agents: number | null;
  max_steps: number | null;
  num_task_types: number | null;
  agent_policy: string | null;
  written_at: string | null;
  termination_reason: string | null;
  episode_length: number | null;
  completion_ratio: number | null;
}

/** Full cooperative run detail (metadata + episode summary). */
export interface CooperativeRunDetail extends CooperativeRunListItem {
  episode_summary: CooperativeEpisodeSummary | null;
}

/** WsStepMessage variant carrying CooperativeStepMetric records. */
export interface CooperativeWsStepMessage {
  type: "step";
  run_id: string;
  t: number;
  metrics: CooperativeStepMetric[];
  events: Array<Record<string, unknown>> | null;
}

export interface CooperativeWsDoneMessage {
  type: "done";
  run_id: string;
  termination_reason: string | null;
  episode_summary: CooperativeEpisodeSummary | null;
}

export type CooperativeWsMessage =
  | CooperativeWsStepMessage
  | CooperativeWsDoneMessage;

/** List all cooperative runs. */
export function getCooperativeRuns(): Promise<CooperativeRunListItem[]> {
  return json(`${BASE}/cooperative/runs`);
}

/** Get full detail for a single cooperative run. */
export function getCooperativeRunDetail(
  runId: string,
): Promise<CooperativeRunDetail> {
  return json(`${BASE}/cooperative/runs/${encodeURIComponent(runId)}`);
}

/** Get only the episode summary for a cooperative run. */
export function getCooperativeRunSummary(
  runId: string,
): Promise<CooperativeEpisodeSummary> {
  return json(`${BASE}/cooperative/runs/${encodeURIComponent(runId)}/summary`);
}

/**
 * Stream a completed cooperative run's metrics.jsonl via SSE.
 * Returns the EventSource so the caller can close it.
 */
export function streamCooperativeReplay(
  runId: string,
  onMessage: (msg: CooperativeWsMessage) => void,
  onDone?: () => void,
  onError?: (detail: string) => void,
): EventSource {
  const es = new EventSource(
    `${BASE}/cooperative/runs/${encodeURIComponent(runId)}/replay`,
  );
  es.onmessage = (ev) => {
    let msg: CooperativeWsMessage;
    try {
      msg = JSON.parse(ev.data) as CooperativeWsMessage;
    } catch {
      es.close();
      onDone?.();
      return;
    }
    onMessage(msg);
    if (msg.type === "done") {
      es.close();
      onDone?.();
    }
  };
  es.onerror = () => {
    es.close();
    if (onError) {
      onError(`Could not load replay for run '${runId}' — run may not exist or has no recorded metrics.`);
    } else {
      onDone?.();
    }
  };
  return es;
}

// ---------------------------------------------------------------------------
// Cooperative League
// ---------------------------------------------------------------------------

export interface CooperativeLeagueMember {
  member_id: string;
  parent_id: string | null;
  created_at: string | null;
  notes: string | null;
  rating: number;
  [key: string]: unknown;
}

export interface CooperativeLeagueRating {
  member_id: string;
  rating: number;
}

export interface CooperativeLineageMember {
  member_id: string;
  parent_id: string | null;
  created_at: string | null;
  notes: string | null;
  rating: number;
  label?: string;
}

export interface CooperativeLineageResponse {
  members: CooperativeLineageMember[];
}

export interface CooperativeChampionInfo {
  member_id: string | null;
  rating?: number;
  parent_id?: string | null;
  created_at?: string | null;
  notes?: string | null;
}

export interface CooperativeEvolutionMember {
  member_id: string;
  parent_id: string | null;
  created_at: string | null;
  notes: string | null;
  rating: number;
  strategy: {
    cluster_id: number | null;
    label: string;
    features: Record<string, unknown>;
  };
  robustness_score: number | null;
}

export interface CooperativeChampionHistoryEntry {
  member_id: string;
  created_at: string | null;
  rating: number;
  label: string;
  cluster_id: number | null;
  robustness_score: number | null;
}

export interface CooperativeEvolutionResponse {
  members: CooperativeEvolutionMember[];
  champion_history: CooperativeChampionHistoryEntry[];
}

export interface CooperativeRobustnessRequest {
  config_id?: string;
  seeds?: number;
  episodes_per_seed?: number;
  limit_sweeps?: number | null;
  seed?: number;
}

export interface CooperativeRobustnessResponse {
  robustness_id: string;
}

export interface CooperativeRobustnessStatusResponse {
  robustness_id: string;
  running: boolean;
  stage: string;
  error?: string;
  report_id?: string;
}

export function getCooperativeLeagueMembers(): Promise<CooperativeLeagueMember[]> {
  return json(`${BASE}/cooperative/league/members`);
}

export function getCooperativeLeagueMember(
  memberId: string,
): Promise<CooperativeLeagueMember> {
  return json(`${BASE}/cooperative/league/members/${encodeURIComponent(memberId)}`);
}

export function startCooperativeLeagueMemberRun(
  memberId: string,
  configId: string,
): Promise<{ run_id: string }> {
  return json(
    `${BASE}/cooperative/league/members/${encodeURIComponent(memberId)}/run`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config_id: configId }),
    },
  );
}

export function getCooperativeChampion(): Promise<CooperativeChampionInfo> {
  return json(`${BASE}/cooperative/league/champion`);
}

export function getCooperativeLeagueLineage(): Promise<CooperativeLineageResponse> {
  return json(`${BASE}/cooperative/league/lineage`);
}

export function getCooperativeLeagueEvolution(): Promise<CooperativeEvolutionResponse> {
  return json(`${BASE}/cooperative/league/evolution`);
}

export function recomputeCooperativeLeagueRatings(
  numMatches: number = 10,
  seed: number = 42,
): Promise<{ status: string }> {
  return json(`${BASE}/cooperative/league/ratings/recompute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_matches: numMatches, seed }),
  });
}

export function runCooperativeChampionRobustness(
  payload: CooperativeRobustnessRequest,
): Promise<CooperativeRobustnessResponse> {
  return json(`${BASE}/cooperative/league/champion/robustness`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function getCooperativeRobustnessStatus(
  robustnessId: string,
): Promise<CooperativeRobustnessStatusResponse> {
  return json(
    `${BASE}/cooperative/league/champion/robustness/${encodeURIComponent(robustnessId)}/status`,
  );
}

export interface CooperativeBenchmarkResult {
  policy: string;
  mean_completion_ratio: number;
  mean_return: number;
  mean_episode_length: number;
  [key: string]: unknown;
}

export interface CooperativeBenchmarkResponse {
  champion: CooperativeChampionInfo;
  results: CooperativeBenchmarkResult[];
}

export function runCooperativeChampionBenchmark(
  configId: string,
  episodes: number,
  seed = 42,
): Promise<CooperativeBenchmarkResponse> {
  return json(`${BASE}/cooperative/league/champion/benchmark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_id: configId, episodes, seed }),
  });
}

// ---------------------------------------------------------------------------
// Cooperative Pipeline
// ---------------------------------------------------------------------------

export interface CooperativePipelineRunParams {
  config_id?: string;
  seed?: number;
  seeds?: number;
  episodes_per_seed?: number;
  max_steps?: number | null;
  total_timesteps?: number;
  snapshot_every_timesteps?: number;
  max_league_members?: number;
  num_matches?: number;
  limit_sweeps?: number | null;
}

export interface CooperativePipelineStatusResponse {
  pipeline_id: string;
  running: boolean;
  stage: string;
  error?: string;
  report_id?: string;
  summary_path?: string;
}

export function startCooperativePipeline(
  params: CooperativePipelineRunParams,
): Promise<{ pipeline_id: string }> {
  return json(`${BASE}/cooperative/pipeline/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export function getCooperativePipelineStatus(
  pipelineId: string,
): Promise<CooperativePipelineStatusResponse> {
  return json(
    `${BASE}/cooperative/pipeline/status/${encodeURIComponent(pipelineId)}`,
  );
}

export function listCooperativePipelineRuns(): Promise<
  Array<{
    pipeline_id: string;
    timestamp: string;
    config_id: string;
    config_hash: string;
    report_id: string | null;
    archetype: string;
  }>
> {
  return json(`${BASE}/cooperative/pipeline/runs`);
}

// ---------------------------------------------------------------------------
// Cooperative Reports
// ---------------------------------------------------------------------------

export interface CooperativeReportListItem {
  report_id: string;
  kind: string;
  config_hash: string;
  timestamp: string;
  path_name: string;
  mean_completion_ratio?: number | null;
  robustness_score?: number | null;
}

export interface CooperativeRobustnessHeatmapResponse {
  sweep_names: string[];
  policies: string[];
  heatmap: Record<string, Record<string, number | null>>;
  per_policy_robustness: Record<string, unknown>;
}

export function getCooperativeReports(): Promise<CooperativeReportListItem[]> {
  return json(`${BASE}/cooperative/reports`);
}

export function getCooperativeReport(
  reportId: string,
): Promise<Record<string, unknown>> {
  return json(`${BASE}/cooperative/reports/${encodeURIComponent(reportId)}`);
}

export function getCooperativeReportRobustness(
  reportId: string,
): Promise<CooperativeRobustnessHeatmapResponse> {
  return json(
    `${BASE}/cooperative/reports/${encodeURIComponent(reportId)}/robustness`,
  );
}

export function getCooperativeReportStrategies(
  reportId: string,
): Promise<StrategyResponse> {
  return json(
    `${BASE}/cooperative/reports/${encodeURIComponent(reportId)}/strategies`,
  );
}

// ---------------------------------------------------------------------------
// Transfer Experiment
// ---------------------------------------------------------------------------

export interface TransferRequest {
  source_member_id: string;
  source_archetype: "mixed" | "competitive" | "cooperative";
  target_archetype: "mixed" | "competitive" | "cooperative";
  target_config_id: string;
  episodes: number;
  seed: number;
}

export interface TransferStatus {
  transfer_id: string;
  status: "pending" | "running_transfer" | "running_baseline" | "saving" | "done" | "error";
  error?: string;
  report_id?: string;
}

export interface TransferEpisodeResult {
  episode: number;
  primary_metric: number;
  [key: string]: unknown;
}

export interface TransferReport {
  report_id: string;
  report_type: "transfer";
  source_archetype: "mixed" | "competitive" | "cooperative";
  source_member_id: string;
  source_obs_dim: number;
  source_strategy_label: string | null;
  source_elo: number;
  target_archetype: "mixed" | "competitive" | "cooperative";
  target_config_hash: string;
  target_obs_dim: number;
  obs_mismatch_strategy: "truncate" | "pad" | "none";
  episodes: number;
  seed: number;
  transferred_results: TransferEpisodeResult[];
  baseline_results: TransferEpisodeResult[];
  transferred_mean: number;
  baseline_mean: number;
  vs_baseline_delta: number;
  vs_baseline_pct: number;
  timestamp?: string;
}

export function startTransferExperiment(
  req: TransferRequest,
): Promise<{ transfer_id: string; status: string }> {
  return json(`${BASE}/transfer/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function getTransferStatus(transferId: string): Promise<TransferStatus> {
  return json(`${BASE}/transfer/status/${encodeURIComponent(transferId)}`);
}

export function getTransferReports(): Promise<TransferReport[]> {
  return json(`${BASE}/transfer/reports`);
}

export function getTransferReport(reportId: string): Promise<TransferReport> {
  return json(`${BASE}/transfer/reports/${encodeURIComponent(reportId)}`);
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
  // NEXT_PUBLIC_WS_BASE lets local dev point directly at the backend
  // (e.g. "localhost:8000"). In production, omit it and the current origin
  // is used, which works correctly behind any reverse proxy.
  const base = process.env.NEXT_PUBLIC_WS_BASE ?? window.location.host;
  const ws = new WebSocket(`${proto}://${base}/api/ws/metrics/${runId}`);

  ws.onmessage = (ev) => {
    let msg: WsMessage;
    try {
      msg = JSON.parse(ev.data) as WsMessage;
    } catch {
      onError?.(ev);
      return;
    }
    onMessage(msg);
  };

  ws.onerror = (ev) => onError?.(ev);
  ws.onclose = () => onClose?.();

  return ws;
}
