"use client";

/**
 * Cooperative simulation page — /simulate/cooperative
 *
 * Left panel: config form (mirrors head-to-head pattern).
 * Right panel: run history for this template.
 *
 * During a live run: CooperativeMetricsChart receives WebSocket step data.
 * After a run: CooperativeRunSummary is shown.
 * Each past run row links to its replay page.
 *
 * ?mode=advanced — reveals all advanced cooperative schema parameters.
 */

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState, useCallback } from "react";

import {
  getCooperativeRuns,
  CooperativeRunListItem,
  CooperativeStepMetric,
  CooperativeEpisodeSummary,
  CooperativeWsMessage,
  connectMetrics,
} from "@/lib/api";

import CooperativeMetricsChart from "@/components/CooperativeMetricsChart";
import CooperativeRunSummary from "@/components/CooperativeRunSummary";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type CoopPolicy =
  | "random"
  | "always_work"
  | "always_idle"
  | "specialist"
  | "balancer"
  | "cooperative_ppo";

const COOP_POLICIES: CoopPolicy[] = [
  "random",
  "always_work",
  "always_idle",
  "specialist",
  "balancer",
  "cooperative_ppo",
];

// ---------------------------------------------------------------------------
// Inline styles (mirrors head-to-head pattern)
// ---------------------------------------------------------------------------

const panelStyle: React.CSSProperties = {
  background: "var(--bg-surface)",
  border: "1px solid var(--bg-border)",
  borderRadius: 8,
  padding: 28,
};

const labelStyle: React.CSSProperties = {
  fontSize: 12,
  color: "var(--text-secondary)",
  marginBottom: 6,
  display: "block",
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  background: "var(--bg-elevated)",
  border: "1px solid var(--bg-border)",
  borderRadius: 6,
  padding: "8px 12px",
  color: "var(--text-primary)",
  fontSize: 13,
  height: 36,
  outline: "none",
  boxSizing: "border-box",
};

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  cursor: "pointer",
  appearance: "none" as const,
};

// ---------------------------------------------------------------------------
// Create a cooperative config via the generic /api/configs endpoint
// ---------------------------------------------------------------------------

interface CoopConfigParams {
  numAgents: number;
  maxSteps: number;
  seed: number;
  numTaskTypes: number;
  // advanced
  agentEffortCapacity: number;
  collapseSustainWindow: number;
  enableEarlySuccess: boolean;
  clearanceSustainWindow: number;
  observationNoise: number;
  historyWindow: number;
  specializationScale: number;
  specializationDecay: number;
  taskArrivalNoise: number;
  taskDifficultyVariance: number;
  freeRiderPressureScale: number;
  taskArrivalRate: number;
  taskDifficulty: number;
  collapseThreshold: number;
  initialBacklog: number;
  wGroup: number;
  wIndividual: number;
  wEfficiency: number;
}

async function createCoopConfig(
  params: CoopConfigParams,
): Promise<{ config_id: string }> {
  const body = {
    identity: {
      environment_type: "cooperative",
      environment_version: "1.0.0",
      archetype: "shared_goal_collective",
      seed: params.seed,
    },
    population: {
      num_agents: params.numAgents,
      max_steps: params.maxSteps,
      num_task_types: params.numTaskTypes,
      agent_effort_capacity: params.agentEffortCapacity,
      collapse_sustain_window: params.collapseSustainWindow,
      enable_early_success: params.enableEarlySuccess,
      clearance_sustain_window: params.clearanceSustainWindow,
    },
    layers: {
      observation_noise: params.observationNoise,
      history_window: params.historyWindow,
      specialization_scale: params.specializationScale,
      specialization_decay: params.specializationDecay,
      task_arrival_noise: params.taskArrivalNoise,
      task_difficulty_variance: params.taskDifficultyVariance,
      free_rider_pressure_scale: params.freeRiderPressureScale,
    },
    task: {
      task_arrival_rate: params.taskArrivalRate,
      task_difficulty: params.taskDifficulty,
      collapse_threshold: params.collapseThreshold,
      initial_backlog: params.initialBacklog,
    },
    rewards: {
      w_group: params.wGroup,
      w_individual: params.wIndividual,
      w_efficiency: params.wEfficiency,
    },
    instrumentation: {
      enable_step_metrics: true,
      enable_episode_metrics: true,
      step_log_frequency: 1,
    },
  };
  const res = await fetch("/api/configs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`${res.status}: ${t}`);
  }
  return res.json();
}

async function startCoopRun(
  configId: string,
  agentPolicy: string,
): Promise<{ run_id: string }> {
  const res = await fetch("/api/runs/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_id: configId, agent_policy: agentPolicy }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`${res.status}: ${t}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Run history panel
// ---------------------------------------------------------------------------

function RunHistoryPanel({
  runs,
  loading,
}: {
  runs: CooperativeRunListItem[];
  loading: boolean;
}) {
  if (loading) {
    return (
      <div style={{ color: "var(--text-tertiary)", fontSize: 13 }}>
        Loading runs…
      </div>
    );
  }
  if (runs.length === 0) {
    return (
      <div
        style={{
          color: "var(--text-tertiary)",
          fontSize: 13,
          textAlign: "center",
          padding: "32px 0",
        }}
      >
        No runs yet — start your first simulation.
      </div>
    );
  }
  return (
    <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
      <thead>
        <tr
          style={{
            color: "var(--text-tertiary)",
            borderBottom: "1px solid var(--bg-border)",
          }}
        >
          {["Run ID", "Policy", "Steps", "Result", "Replay"].map((h) => (
            <th
              key={h}
              style={{
                textAlign: "left",
                padding: "6px 8px",
                fontWeight: 500,
              }}
            >
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {runs.map((r) => (
          <tr
            key={r.run_id}
            style={{ borderBottom: "1px solid var(--bg-border-subtle)" }}
          >
            <td
              style={{
                padding: "7px 8px",
                fontFamily: "var(--font-mono)",
                color: "var(--accent)",
                fontSize: 11,
              }}
            >
              {r.run_id.slice(0, 8)}
            </td>
            <td
              style={{
                padding: "7px 8px",
                fontFamily: "var(--font-mono)",
                color: "var(--text-secondary)",
              }}
            >
              {r.agent_policy ?? "—"}
            </td>
            <td style={{ padding: "7px 8px", color: "var(--text-secondary)" }}>
              {r.episode_length ?? "—"}
            </td>
            <td style={{ padding: "7px 8px", color: "var(--text-secondary)" }}>
              {r.termination_reason ?? "—"}
            </td>
            <td style={{ padding: "7px 8px" }}>
              <Link
                href={`/simulate/cooperative/replay/${r.run_id}`}
                style={{
                  fontSize: 11,
                  color: "var(--accent)",
                  textDecoration: "none",
                }}
              >
                replay →
              </Link>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// Main page — Suspense wrapper required for useSearchParams
// ---------------------------------------------------------------------------

export default function CooperativePage() {
  return (
    <Suspense>
      <CooperativePageInner />
    </Suspense>
  );
}

function CooperativePageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const advanced = searchParams.get("mode") === "advanced";

  // Basic config form state
  const [numAgents, setNumAgents] = useState(4);
  const [maxSteps, setMaxSteps] = useState(200);
  const [seed, setSeed] = useState(42);
  const [numTaskTypes, setNumTaskTypes] = useState(3);
  const [agentPolicy, setAgentPolicy] = useState<CoopPolicy>("random");

  // Advanced config state — defaults match createCoopConfig previous hardcoded values
  const [agentEffortCapacity, setAgentEffortCapacity] = useState(1.0);
  const [collapseSustainWindow, setCollapseSustainWindow] = useState(10);
  const [enableEarlySuccess, setEnableEarlySuccess] = useState(false);
  const [clearanceSustainWindow, setClearanceSustainWindow] = useState(15);
  const [observationNoise, setObservationNoise] = useState(0.0);
  const [historyWindow, setHistoryWindow] = useState(5);
  const [specializationScale, setSpecializationScale] = useState(0.3);
  const [specializationDecay, setSpecializationDecay] = useState(0.1);
  const [taskArrivalNoise, setTaskArrivalNoise] = useState(0.1);
  const [taskDifficultyVariance, setTaskDifficultyVariance] = useState(0.0);
  const [freeRiderPressureScale, setFreeRiderPressureScale] = useState(1.0);
  const [taskArrivalRate, setTaskArrivalRate] = useState(1.0);
  const [taskDifficulty, setTaskDifficulty] = useState(1.0);
  const [collapseThreshold, setCollapseThreshold] = useState(50);
  const [initialBacklog, setInitialBacklog] = useState(0);
  const [wGroup, setWGroup] = useState(0.7);
  const [wIndividual, setWIndividual] = useState(0.2);
  const [wEfficiency, setWEfficiency] = useState(0.1);

  // Run state
  const [runId, setRunId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stepHistory, setStepHistory] = useState<CooperativeStepMetric[]>([]);
  const [episodeSummary, setEpisodeSummary] =
    useState<CooperativeEpisodeSummary | null>(null);
  const [wsStatus, setWsStatus] = useState<"idle" | "connected" | "closed">(
    "idle",
  );

  // Run history
  const [runs, setRuns] = useState<CooperativeRunListItem[]>([]);
  const [runsLoading, setRunsLoading] = useState(true);

  const wsRef = useRef<WebSocket | null>(null);

  const fetchRuns = useCallback(async () => {
    try {
      const data = await getCooperativeRuns();
      setRuns(data);
    } catch {
      // ignore
    } finally {
      setRunsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  const handleStart = async () => {
    setError(null);
    setEpisodeSummary(null);
    setStepHistory([]);
    setRunning(true);

    try {
      const { config_id } = await createCoopConfig({
        numAgents,
        maxSteps,
        seed,
        numTaskTypes,
        agentEffortCapacity,
        collapseSustainWindow,
        enableEarlySuccess,
        clearanceSustainWindow,
        observationNoise,
        historyWindow,
        specializationScale,
        specializationDecay,
        taskArrivalNoise,
        taskDifficultyVariance,
        freeRiderPressureScale,
        taskArrivalRate,
        taskDifficulty,
        collapseThreshold,
        initialBacklog,
        wGroup,
        wIndividual,
        wEfficiency,
      });
      const { run_id } = await startCoopRun(config_id, agentPolicy);
      setRunId(run_id);
      router.push(`/simulate/cooperative/run/${run_id}`);
      return;

      // Connect WebSocket
      const ws = connectMetrics(
        run_id,
        (msg) => {
          const m = msg as unknown as CooperativeWsMessage;
          if (m.type === "step") {
            setStepHistory((prev) => [...prev, ...m.metrics]);
          } else if (m.type === "done") {
            if (m.episode_summary) {
              setEpisodeSummary(m.episode_summary as CooperativeEpisodeSummary);
            }
            setRunning(false);
            setWsStatus("closed");
            fetchRuns();
          }
        },
        () => {
          setWsStatus("closed");
          setRunning(false);
        },
        () => {
          setWsStatus("closed");
          setRunning(false);
        },
      );
      wsRef.current = ws;
      setWsStatus("connected");
    } catch (e) {
      setError(String(e));
      setRunning(false);
    }
  };

  return (
    <main
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        padding: "48px 24px",
        paddingTop: 96,
      }}
    >
      {/* Back link */}
      <Link
        href="/simulate"
        style={{
          fontSize: 13,
          color: "var(--text-tertiary)",
          textDecoration: "none",
        }}
      >
        ← Simulate
      </Link>

      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1
          style={{
            fontSize: 22,
            fontWeight: 500,
            color: "var(--text-primary)",
            margin: 0,
          }}
        >
          Cooperative Task Arena
        </h1>
        <p
          style={{
            fontSize: 13,
            color: "var(--text-secondary)",
            margin: "8px 0 0",
          }}
        >
          Agents share a task queue and coordinate effort to prevent system collapse.
        </p>
      </div>

      <div style={{ display: "flex", gap: 24 }}>
        {/* ── Config panel ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16, flex: 1, minWidth: 0 }}>
          <div style={{ ...panelStyle, borderTop: "2px solid var(--accent)" }}>
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                color: "var(--text-tertiary)",
                marginBottom: 20,
              }}
            >
              Configuration
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {/* ── Basic parameters ── */}
              <div>
                <label style={labelStyle}>num_agents</label>
                <input
                  type="number"
                  min={2}
                  max={10}
                  value={numAgents}
                  onChange={(e) => setNumAgents(Number(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div>
                <label style={labelStyle}>max_steps</label>
                <input
                  type="number"
                  min={10}
                  max={2000}
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(Number(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div>
                <label style={labelStyle}>num_task_types</label>
                <input
                  type="number"
                  min={1}
                  max={5}
                  value={numTaskTypes}
                  onChange={(e) => setNumTaskTypes(Number(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div>
                <label style={labelStyle}>seed</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div>
                <label style={labelStyle}>agent_policy</label>
                <select
                  value={agentPolicy}
                  onChange={(e) =>
                    setAgentPolicy(e.target.value as CoopPolicy)
                  }
                  style={selectStyle}
                >
                  {COOP_POLICIES.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>

              {/* ── Advanced parameters ── */}
              {advanced && (
                <>
                  <hr
                    style={{
                      border: "none",
                      borderTop: "1px solid var(--bg-border)",
                      margin: "4px 0",
                    }}
                  />
                  <p
                    style={{
                      fontSize: 11,
                      textTransform: "uppercase" as const,
                      letterSpacing: "0.05em",
                      color: "var(--text-tertiary)",
                      marginBottom: 0,
                    }}
                  >
                    Advanced parameters
                  </p>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "12px 16px",
                    }}
                  >
                    {/* Population */}
                    <div>
                      <label style={labelStyle}>agent_effort_capacity</label>
                      <input
                        type="number"
                        step={0.1}
                        min={0.1}
                        value={agentEffortCapacity}
                        onChange={(e) =>
                          setAgentEffortCapacity(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>collapse_sustain_window</label>
                      <input
                        type="number"
                        min={1}
                        value={collapseSustainWindow}
                        onChange={(e) =>
                          setCollapseSustainWindow(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div style={{ gridColumn: "1 / -1" }}>
                      <label style={labelStyle}>enable_early_success</label>
                      <select
                        value={enableEarlySuccess ? "true" : "false"}
                        onChange={(e) =>
                          setEnableEarlySuccess(e.target.value === "true")
                        }
                        style={selectStyle}
                      >
                        <option value="false">false</option>
                        <option value="true">true</option>
                      </select>
                    </div>
                    <div>
                      <label style={labelStyle}>clearance_sustain_window</label>
                      <input
                        type="number"
                        min={1}
                        value={clearanceSustainWindow}
                        onChange={(e) =>
                          setClearanceSustainWindow(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>

                    {/* Layers */}
                    <div>
                      <label style={labelStyle}>observation_noise</label>
                      <input
                        type="number"
                        step={0.01}
                        min={0}
                        max={0.2}
                        value={observationNoise}
                        onChange={(e) =>
                          setObservationNoise(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>history_window</label>
                      <input
                        type="number"
                        min={1}
                        value={historyWindow}
                        onChange={(e) =>
                          setHistoryWindow(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>specialization_scale</label>
                      <input
                        type="number"
                        step={0.05}
                        min={0}
                        max={0.5}
                        value={specializationScale}
                        onChange={(e) =>
                          setSpecializationScale(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>specialization_decay</label>
                      <input
                        type="number"
                        step={0.05}
                        min={0.01}
                        max={0.99}
                        value={specializationDecay}
                        onChange={(e) =>
                          setSpecializationDecay(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>task_arrival_noise</label>
                      <input
                        type="number"
                        step={0.05}
                        min={0}
                        max={0.3}
                        value={taskArrivalNoise}
                        onChange={(e) =>
                          setTaskArrivalNoise(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>task_difficulty_variance</label>
                      <input
                        type="number"
                        step={0.05}
                        min={0}
                        max={0.3}
                        value={taskDifficultyVariance}
                        onChange={(e) =>
                          setTaskDifficultyVariance(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>free_rider_pressure_scale</label>
                      <input
                        type="number"
                        step={0.1}
                        min={0}
                        max={1}
                        value={freeRiderPressureScale}
                        onChange={(e) =>
                          setFreeRiderPressureScale(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>

                    {/* Task */}
                    <div>
                      <label style={labelStyle}>task_arrival_rate</label>
                      <input
                        type="number"
                        step={0.1}
                        min={0.1}
                        value={taskArrivalRate}
                        onChange={(e) =>
                          setTaskArrivalRate(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>task_difficulty</label>
                      <input
                        type="number"
                        step={0.1}
                        min={0.1}
                        value={taskDifficulty}
                        onChange={(e) =>
                          setTaskDifficulty(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>collapse_threshold</label>
                      <input
                        type="number"
                        min={1}
                        value={collapseThreshold}
                        onChange={(e) =>
                          setCollapseThreshold(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>initial_backlog</label>
                      <input
                        type="number"
                        min={0}
                        value={initialBacklog}
                        onChange={(e) =>
                          setInitialBacklog(Number(e.target.value))
                        }
                        style={inputStyle}
                      />
                    </div>

                    {/* Rewards */}
                    <div>
                      <label style={labelStyle}>w_group</label>
                      <input
                        type="number"
                        step={0.05}
                        min={0.5}
                        max={0.95}
                        value={wGroup}
                        onChange={(e) => setWGroup(Number(e.target.value))}
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>w_individual</label>
                      <input
                        type="number"
                        step={0.05}
                        min={0.01}
                        max={0.49}
                        value={wIndividual}
                        onChange={(e) => setWIndividual(Number(e.target.value))}
                        style={inputStyle}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>w_efficiency</label>
                      <input
                        type="number"
                        step={0.05}
                        min={0.01}
                        max={0.49}
                        value={wEfficiency}
                        onChange={(e) => setWEfficiency(Number(e.target.value))}
                        style={inputStyle}
                      />
                    </div>
                  </div>
                </>
              )}

              <button
                onClick={handleStart}
                disabled={running}
                style={{
                  width: "100%",
                  height: 38,
                  background: running ? "var(--bg-elevated)" : "var(--accent)",
                  color: running ? "var(--text-tertiary)" : "white",
                  fontSize: 13,
                  fontWeight: 500,
                  border: "none",
                  borderRadius: 6,
                  cursor: running ? "not-allowed" : "pointer",
                  transition: "background 150ms",
                  marginTop: 4,
                }}
              >
                {running ? "Running…" : "Start Run"}
              </button>

              {error && (
                <div style={{ fontSize: 12, color: "#ef4444" }}>{error}</div>
              )}
            </div>
          </div>
        </div>

        {/* ── Right panel ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20, flex: 1, minWidth: 0 }}>
          {/* Live chart */}
          {(running || stepHistory.length > 0) && (
            <div style={panelStyle}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  marginBottom: 16,
                }}
              >
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 500,
                    color: "var(--text-secondary)",
                  }}
                >
                  Live Run
                </span>
                {runId && (
                  <span
                    style={{
                      fontSize: 11,
                      fontFamily: "var(--font-mono)",
                      color: "var(--text-tertiary)",
                    }}
                  >
                    {runId.slice(0, 8)}
                  </span>
                )}
                {wsStatus === "connected" && (
                  <span
                    style={{
                      fontSize: 10,
                      padding: "2px 6px",
                      borderRadius: 4,
                      background: "rgba(20,184,166,0.15)",
                      color: "var(--accent)",
                    }}
                  >
                    live
                  </span>
                )}
              </div>
              <CooperativeMetricsChart history={stepHistory} />
            </div>
          )}

          {/* Episode summary */}
          {episodeSummary && (
            <div>
              <div
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  color: "var(--text-tertiary)",
                  marginBottom: 12,
                }}
              >
                Episode Summary
              </div>
              <CooperativeRunSummary summary={episodeSummary} />
            </div>
          )}

          {/* Run history */}
          <div style={panelStyle}>
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                color: "var(--text-tertiary)",
                marginBottom: 12,
              }}
            >
              Past Runs
            </div>
            <RunHistoryPanel runs={runs} loading={runsLoading} />
          </div>
        </div>
      </div>
    </main>
  );
}
