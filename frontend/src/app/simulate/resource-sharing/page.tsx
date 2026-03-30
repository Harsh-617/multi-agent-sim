"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";

import {
  listRuns,
  getRunDetail,
  startRun,
  RunListItem,
  AgentPolicy,
} from "@/lib/api";

const BASE = "/api";

const MIXED_POLICIES: AgentPolicy[] = [
  "random",
  "always_cooperate",
  "always_extract",
  "tit_for_tat",
  "ppo_shared",
  "league_snapshot",
];

/** Policies that are unambiguously Mixed — never appear in Head-to-Head */
const DEFINITE_MIXED_POLICIES = new Set([
  "always_cooperate",
  "always_extract",
  "tit_for_tat",
  "ppo_shared",
  "league_snapshot",
]);

/** Returns 'mixed' | 'competitive' | 'unknown' by inspecting episode_summary */
async function resolveRunArchetype(
  runId: string,
): Promise<"mixed" | "competitive" | "unknown"> {
  try {
    const detail = await getRunDetail(runId);
    if (!detail.episode_summary) return "unknown";
    if ("winner_id" in detail.episode_summary) return "competitive";
    if ("final_shared_pool" in detail.episode_summary) return "mixed";
    return "unknown";
  } catch {
    return "unknown";
  }
}

/* ── shared inline styles ── */

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

export default function ResourceSharingPage() {
  return (
    <Suspense>
      <ResourceSharingInner />
    </Suspense>
  );
}

function ResourceSharingInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const advanced = searchParams.get("mode") === "advanced";

  /* ── Config form state ── */
  const [numAgents, setNumAgents] = useState(4);
  const [maxSteps, setMaxSteps] = useState(200);
  const [seed, setSeed] = useState(42);
  const [agentPolicy, setAgentPolicy] = useState<AgentPolicy>("random");
  const [leagueMemberId, setLeagueMemberId] = useState("");

  /* Advanced-only fields (Mixed schema) */
  const [initialSharedPool, setInitialSharedPool] = useState(100);
  const [initialAgentResources, setInitialAgentResources] = useState(10);
  const [collapseThreshold, setCollapseThreshold] = useState(5);
  const [informationAsymmetry, setInformationAsymmetry] = useState(0.3);
  const [temporalMemoryDepth, setTemporalMemoryDepth] = useState(10);
  const [reputationSensitivity, setReputationSensitivity] = useState(0.5);
  const [incentiveSoftness, setIncentiveSoftness] = useState(0.5);
  const [uncertaintyIntensity, setUncertaintyIntensity] = useState(0.1);
  const [individualWeight, setIndividualWeight] = useState(1.0);
  const [groupWeight, setGroupWeight] = useState(0.5);
  const [relationalWeight, setRelationalWeight] = useState(0.3);
  const [penaltyScaling, setPenaltyScaling] = useState(1.0);
  const [observationMemorySteps, setObservationMemorySteps] = useState(5);

  const [starting, setStarting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  /* ── Run history state ── */
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [runsLoading, setRunsLoading] = useState(true);

  const fetchRuns = useCallback(async () => {
    setRunsLoading(true);
    try {
      const all = await listRuns();

      const definite: RunListItem[] = [];
      const ambiguous: RunListItem[] = [];

      for (const r of all) {
        if (r.agent_policy && DEFINITE_MIXED_POLICIES.has(r.agent_policy)) {
          definite.push(r);
        } else if (
          r.agent_policy === "random" ||
          r.agent_policy === null ||
          r.agent_policy === ""
        ) {
          ambiguous.push(r);
        }
        // competitive-only policies are skipped entirely
      }

      // Resolve up to 20 ambiguous runs via detail endpoint
      const toResolve = ambiguous.slice(0, 20);
      const resolved = await Promise.all(
        toResolve.map(async (r) => {
          const archetype = await resolveRunArchetype(r.run_id);
          return { run: r, archetype };
        }),
      );

      const mixedFromAmbiguous = resolved
        .filter((x) => x.archetype === "mixed")
        .map((x) => x.run);

      setRuns([...definite, ...mixedFromAmbiguous]);
    } catch {
      /* silently ignore — empty table is fine */
    } finally {
      setRunsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  /* ── Handlers ── */

  async function handleStartRun() {
    setStarting(true);
    setFormError(null);
    try {
      /* Create a Mixed config via POST /api/configs */
      const res = await fetch(`${BASE}/configs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          identity: {
            environment_type: "mixed",
            environment_version: "0.1.0",
            seed,
          },
          population: {
            num_agents: numAgents,
            max_steps: maxSteps,
            initial_shared_pool: initialSharedPool,
            initial_agent_resources: initialAgentResources,
            collapse_threshold: collapseThreshold,
          },
          layers: {
            information_asymmetry: informationAsymmetry,
            temporal_memory_depth: temporalMemoryDepth,
            reputation_sensitivity: reputationSensitivity,
            incentive_softness: incentiveSoftness,
            uncertainty_intensity: uncertaintyIntensity,
          },
          rewards: {
            individual_weight: individualWeight,
            group_weight: groupWeight,
            relational_weight: relationalWeight,
            penalty_scaling: penaltyScaling,
          },
          agents: {
            observation_memory_steps: observationMemorySteps,
          },
          instrumentation: {
            enable_step_metrics: true,
            enable_episode_metrics: true,
            enable_event_log: true,
            step_log_frequency: 1,
          },
        }),
      });
      if (!res.ok) throw new Error(`Config creation failed: ${res.status}`);
      const { config_id } = (await res.json()) as { config_id: string };

      const { run_id } = await startRun(
        config_id,
        agentPolicy,
        agentPolicy === "league_snapshot" ? leagueMemberId || undefined : undefined,
      );
      router.push(`/simulate/resource-sharing/run/${run_id}`);
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  }

  function formatTime(ts: string | null) {
    if (!ts) return "—";
    try {
      return new Date(ts).toLocaleString();
    } catch {
      return ts;
    }
  }

  return (
    <main
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        padding: "48px 24px",
      }}
    >
      <div style={{ display: "flex", gap: 24 }}>
        {/* ── LEFT PANEL — Config ── */}
        <div style={{ ...panelStyle, flex: 1, minWidth: 0 }}>
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

          <h1
            style={{
              fontSize: 20,
              fontWeight: 500,
              margin: "12px 0 4px",
              color: "var(--text-primary)",
            }}
          >
            Resource Sharing Arena
          </h1>
          <p
            style={{
              fontSize: 13,
              color: "var(--text-secondary)",
              marginBottom: 24,
              lineHeight: 1.5,
            }}
          >
            Agents share a common resource pool and decide when to cooperate or
            compete.
          </p>

          <hr
            style={{
              border: "none",
              borderTop: "1px solid var(--bg-border)",
              marginBottom: 20,
            }}
          />

          {/* ── Form fields ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
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
                min={50}
                max={1000}
                value={maxSteps}
                onChange={(e) => setMaxSteps(Number(e.target.value))}
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
                onChange={(e) => setAgentPolicy(e.target.value as AgentPolicy)}
                style={inputStyle}
              >
                {MIXED_POLICIES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>

            {agentPolicy === "league_snapshot" && (
              <div>
                <label style={labelStyle}>league_member_id</label>
                <input
                  type="text"
                  value={leagueMemberId}
                  onChange={(e) => setLeagueMemberId(e.target.value)}
                  placeholder="Enter league member ID"
                  style={inputStyle}
                />
              </div>
            )}

            {/* ── Advanced fields (Mixed schema) ── */}
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

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px 16px" }}>
                {/* Population */}
                <div>
                  <label style={labelStyle}>initial_shared_pool</label>
                  <input
                    type="number"
                    value={initialSharedPool}
                    onChange={(e) =>
                      setInitialSharedPool(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>initial_agent_resources</label>
                  <input
                    type="number"
                    value={initialAgentResources}
                    onChange={(e) =>
                      setInitialAgentResources(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>collapse_threshold</label>
                  <input
                    type="number"
                    step={0.1}
                    value={collapseThreshold}
                    onChange={(e) =>
                      setCollapseThreshold(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>

                {/* Layers */}
                <div>
                  <label style={labelStyle}>information_asymmetry</label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={1}
                    value={informationAsymmetry}
                    onChange={(e) =>
                      setInformationAsymmetry(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>temporal_memory_depth</label>
                  <input
                    type="number"
                    min={1}
                    value={temporalMemoryDepth}
                    onChange={(e) =>
                      setTemporalMemoryDepth(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>reputation_sensitivity</label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={1}
                    value={reputationSensitivity}
                    onChange={(e) =>
                      setReputationSensitivity(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>incentive_softness</label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={1}
                    value={incentiveSoftness}
                    onChange={(e) =>
                      setIncentiveSoftness(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>uncertainty_intensity</label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={1}
                    value={uncertaintyIntensity}
                    onChange={(e) =>
                      setUncertaintyIntensity(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>

                {/* Rewards */}
                <div>
                  <label style={labelStyle}>individual_weight</label>
                  <input
                    type="number"
                    step={0.1}
                    value={individualWeight}
                    onChange={(e) =>
                      setIndividualWeight(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>group_weight</label>
                  <input
                    type="number"
                    step={0.1}
                    value={groupWeight}
                    onChange={(e) => setGroupWeight(Number(e.target.value))}
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>relational_weight</label>
                  <input
                    type="number"
                    step={0.1}
                    value={relationalWeight}
                    onChange={(e) =>
                      setRelationalWeight(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>penalty_scaling</label>
                  <input
                    type="number"
                    step={0.1}
                    value={penaltyScaling}
                    onChange={(e) => setPenaltyScaling(Number(e.target.value))}
                    style={inputStyle}
                  />
                </div>

                {/* Agents */}
                <div>
                  <label style={labelStyle}>observation_memory_steps</label>
                  <input
                    type="number"
                    min={1}
                    value={observationMemorySteps}
                    onChange={(e) =>
                      setObservationMemorySteps(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                </div>
              </>
            )}
          </div>

          {formError && (
            <p style={{ color: "#ef4444", fontSize: 13, marginTop: 12 }}>
              {formError}
            </p>
          )}

          <button
            onClick={handleStartRun}
            disabled={starting}
            style={{
              width: "100%",
              marginTop: 20,
              background: "var(--accent)",
              color: "#fff",
              height: 40,
              borderRadius: 6,
              border: "none",
              fontSize: 13,
              fontWeight: 500,
              cursor: starting ? "default" : "pointer",
              opacity: starting ? 0.6 : 1,
            }}
            onMouseEnter={(e) => {
              if (!starting)
                (e.target as HTMLButtonElement).style.background =
                  "var(--accent-hover)";
            }}
            onMouseLeave={(e) => {
              (e.target as HTMLButtonElement).style.background =
                "var(--accent)";
            }}
          >
            {starting ? "Starting…" : "Start Run"}
          </button>
        </div>

        {/* ── RIGHT PANEL — Run History ── */}
        <div style={{ ...panelStyle, flex: 1, minWidth: 0 }}>
          <p
            style={{
              fontSize: 11,
              textTransform: "uppercase" as const,
              letterSpacing: "0.05em",
              color: "var(--text-tertiary)",
              marginBottom: 16,
            }}
          >
            Recent runs
          </p>

          <div style={{ maxHeight: "calc(100vh - 220px)", overflowY: "auto", scrollbarWidth: "thin", scrollbarColor: "var(--bg-elevated) transparent" }}>
          {runsLoading && (
            <p style={{ fontSize: 13, color: "var(--text-tertiary)" }}>
              Loading...
            </p>
          )}

          {!runsLoading && runs.length === 0 && (
            <p
              style={{
                fontSize: 13,
                color: "var(--text-tertiary)",
                textAlign: "center",
                marginTop: 32,
              }}
            >
              No runs yet — start your first simulation above
            </p>
          )}

          {!runsLoading && runs.length > 0 && (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr
                  style={{
                    borderBottom: "1px solid var(--bg-border)",
                    fontSize: 11,
                    textTransform: "uppercase" as const,
                    letterSpacing: "0.05em",
                    color: "var(--text-tertiary)",
                  }}
                >
                  <th style={{ textAlign: "left", padding: "10px 8px 10px 0", fontWeight: 500 }}>
                    Policy
                  </th>
                  <th style={{ textAlign: "left", padding: "10px 8px", fontWeight: 500 }}>
                    Steps
                  </th>
                  <th style={{ textAlign: "left", padding: "10px 8px", fontWeight: 500 }}>
                    Termination
                  </th>
                  <th style={{ textAlign: "left", padding: "10px 8px", fontWeight: 500 }}>
                    Time
                  </th>
                  <th style={{ textAlign: "right", padding: "10px 0 10px 8px", fontWeight: 500 }}>
                    Replay
                  </th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr
                    key={r.run_id}
                    style={{
                      borderBottom: "1px solid var(--bg-border)",
                      fontSize: 13,
                      color: "var(--text-primary)",
                    }}
                  >
                    <td style={{ padding: "10px 8px 10px 0" }}>
                      {r.agent_policy ?? "—"}
                    </td>
                    <td style={{ padding: "10px 8px" }}>
                      {r.episode_length ?? "—"}
                    </td>
                    <td style={{ padding: "10px 8px" }}>
                      {r.termination_reason ?? "—"}
                    </td>
                    <td style={{ padding: "10px 8px" }}>
                      {formatTime(r.timestamp)}
                    </td>
                    <td style={{ padding: "10px 0 10px 8px", textAlign: "right" }}>
                      <Link
                        href={`/simulate/resource-sharing/replay/${r.run_id}`}
                        style={{
                          fontSize: 12,
                          color: "var(--accent)",
                          textDecoration: "none",
                        }}
                      >
                        Replay
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          </div>
        </div>
      </div>
    </main>
  );
}
