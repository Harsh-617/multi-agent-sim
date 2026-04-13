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
 */

import Link from "next/link";
import { useEffect, useRef, useState, useCallback } from "react";

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

async function createCoopConfig(params: {
  numAgents: number;
  maxSteps: number;
  seed: number;
  numTaskTypes: number;
}): Promise<{ config_id: string }> {
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
      agent_effort_capacity: 1.0,
      collapse_sustain_window: 10,
      enable_early_success: false,
      clearance_sustain_window: 15,
    },
    layers: {
      observation_noise: 0.0,
      history_window: 5,
      specialization_scale: 0.3,
      specialization_decay: 0.1,
      task_arrival_noise: 0.1,
      task_difficulty_variance: 0.0,
      free_rider_pressure_scale: 1.0,
    },
    task: {
      task_arrival_rate: 1.0,
      task_difficulty: 1.0,
      collapse_threshold: 50,
      initial_backlog: 0,
    },
    rewards: {
      w_group: 0.7,
      w_individual: 0.2,
      w_efficiency: 0.1,
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
// Main page
// ---------------------------------------------------------------------------

export default function CooperativePage() {
  // Config form state
  const [numAgents, setNumAgents] = useState(4);
  const [maxSteps, setMaxSteps] = useState(200);
  const [seed, setSeed] = useState(42);
  const [numTaskTypes, setNumTaskTypes] = useState(3);
  const [agentPolicy, setAgentPolicy] = useState<CoopPolicy>("random");

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
      });
      const { run_id } = await startCoopRun(config_id, agentPolicy);
      setRunId(run_id);

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
          Cooperative Task Simulation
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

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 24 }}>
        {/* ── Config panel ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
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
              <div>
                <label style={labelStyle}>Agents</label>
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
                <label style={labelStyle}>Max Steps</label>
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
                <label style={labelStyle}>Task Types</label>
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
                <label style={labelStyle}>Seed</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div>
                <label style={labelStyle}>Agent Policy</label>
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
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
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
