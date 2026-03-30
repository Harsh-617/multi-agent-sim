"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";

import {
  createCompetitiveConfig,
  startCompetitiveRun,
  listRuns,
  getRunDetail,
  RunListItem,
  CompetitiveAgentPolicy,
} from "@/lib/api";

const COMPETITIVE_POLICIES: CompetitiveAgentPolicy[] = [
  "random",
  "always_attack",
  "always_build",
  "always_defend",
  "competitive_ppo",
];

/** Policies that are unambiguously Competitive — never appear in Resource Sharing */
const DEFINITE_COMPETITIVE_POLICIES = new Set<string>([
  "always_attack",
  "always_build",
  "always_defend",
  "competitive_ppo",
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

export default function HeadToHeadPage() {
  return (
    <Suspense>
      <HeadToHeadInner />
    </Suspense>
  );
}

function HeadToHeadInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const advanced = searchParams.get("mode") === "advanced";

  /* ── Config form state ── */
  const [numAgents, setNumAgents] = useState(4);
  const [maxSteps, setMaxSteps] = useState(200);
  const [seed, setSeed] = useState(42);
  const [agentPolicy, setAgentPolicy] =
    useState<CompetitiveAgentPolicy>("random");

  /* Advanced-only fields (Competitive schema) */
  const [initialScore, setInitialScore] = useState(0);
  const [initialResources, setInitialResources] = useState(20);
  const [resourceRegenerationRate, setResourceRegenerationRate] = useState(1);
  const [eliminationThreshold, setEliminationThreshold] = useState(0);
  const [dominanceMargin, setDominanceMargin] = useState(0);
  const [informationAsymmetry, setInformationAsymmetry] = useState(0.3);
  const [opponentHistoryDepth, setOpponentHistoryDepth] = useState(10);
  const [opponentObsWindow, setOpponentObsWindow] = useState(5);
  const [historySensitivity, setHistorySensitivity] = useState(0.5);
  const [incentiveSoftness, setIncentiveSoftness] = useState(0.8);
  const [uncertaintyIntensity, setUncertaintyIntensity] = useState(0.1);
  const [gambleVariance, setGambleVariance] = useState(0.5);
  const [absoluteGainWeight, setAbsoluteGainWeight] = useState(1.0);
  const [relativeGainWeight, setRelativeGainWeight] = useState(0.5);
  const [efficiencyWeight, setEfficiencyWeight] = useState(0.3);
  const [terminalBonusScale, setTerminalBonusScale] = useState(2.0);
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
        if (r.agent_policy && DEFINITE_COMPETITIVE_POLICIES.has(r.agent_policy)) {
          definite.push(r);
        } else if (
          r.agent_policy === "random" ||
          r.agent_policy === null ||
          r.agent_policy === ""
        ) {
          ambiguous.push(r);
        }
        // mixed-only policies are skipped entirely
      }

      // Resolve up to 20 ambiguous runs via detail endpoint
      const toResolve = ambiguous.slice(0, 20);
      const overflow = ambiguous.slice(20);

      const resolved = await Promise.all(
        toResolve.map(async (r) => {
          const archetype = await resolveRunArchetype(r.run_id);
          return { run: r, archetype };
        }),
      );

      const competitiveFromAmbiguous = resolved
        .filter((x) => x.archetype === "competitive" || x.archetype === "unknown")
        .map((x) => x.run);

      // Overflow ambiguous runs (beyond 20) default to Head-to-Head
      setRuns([...definite, ...competitiveFromAmbiguous, ...overflow]);
    } catch {
      /* silently ignore */
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
      const { config_id } = await createCompetitiveConfig({
        num_agents: numAgents,
        max_steps: maxSteps,
        seed,
      });
      const { run_id } = await startCompetitiveRun(config_id, agentPolicy);
      router.push(`/simulate/head-to-head/run/${run_id}`);
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
            Head-to-Head Strategy
          </h1>
          <p
            style={{
              fontSize: 13,
              color: "var(--text-secondary)",
              marginBottom: 24,
              lineHeight: 1.5,
            }}
          >
            Pure zero-sum competition. Agents fight for score dominance. One
            winner per episode.
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
                max={8}
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
                onChange={(e) =>
                  setAgentPolicy(e.target.value as CompetitiveAgentPolicy)
                }
                style={inputStyle}
              >
                {COMPETITIVE_POLICIES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>

            {/* ── Advanced fields (Competitive schema) ── */}
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
                  <label style={labelStyle}>initial_score</label>
                  <input
                    type="number"
                    step={0.1}
                    value={initialScore}
                    onChange={(e) => setInitialScore(Number(e.target.value))}
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>initial_resources</label>
                  <input
                    type="number"
                    step={0.1}
                    value={initialResources}
                    onChange={(e) =>
                      setInitialResources(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>resource_regeneration_rate</label>
                  <input
                    type="number"
                    step={0.1}
                    value={resourceRegenerationRate}
                    onChange={(e) =>
                      setResourceRegenerationRate(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>elimination_threshold</label>
                  <input
                    type="number"
                    step={0.1}
                    value={eliminationThreshold}
                    onChange={(e) =>
                      setEliminationThreshold(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>dominance_margin</label>
                  <input
                    type="number"
                    step={0.1}
                    value={dominanceMargin}
                    onChange={(e) =>
                      setDominanceMargin(Number(e.target.value))
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
                  <label style={labelStyle}>opponent_history_depth</label>
                  <input
                    type="number"
                    min={1}
                    value={opponentHistoryDepth}
                    onChange={(e) =>
                      setOpponentHistoryDepth(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>opponent_obs_window</label>
                  <input
                    type="number"
                    min={1}
                    value={opponentObsWindow}
                    onChange={(e) =>
                      setOpponentObsWindow(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>history_sensitivity</label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={1}
                    value={historySensitivity}
                    onChange={(e) =>
                      setHistorySensitivity(Number(e.target.value))
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
                <div>
                  <label style={labelStyle}>gamble_variance</label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    value={gambleVariance}
                    onChange={(e) =>
                      setGambleVariance(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>

                {/* Rewards */}
                <div>
                  <label style={labelStyle}>absolute_gain_weight</label>
                  <input
                    type="number"
                    step={0.1}
                    value={absoluteGainWeight}
                    onChange={(e) =>
                      setAbsoluteGainWeight(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>relative_gain_weight</label>
                  <input
                    type="number"
                    step={0.1}
                    value={relativeGainWeight}
                    onChange={(e) =>
                      setRelativeGainWeight(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>efficiency_weight</label>
                  <input
                    type="number"
                    step={0.1}
                    value={efficiencyWeight}
                    onChange={(e) =>
                      setEfficiencyWeight(Number(e.target.value))
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>terminal_bonus_scale</label>
                  <input
                    type="number"
                    step={0.1}
                    value={terminalBonusScale}
                    onChange={(e) =>
                      setTerminalBonusScale(Number(e.target.value))
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
                    onChange={(e) =>
                      setPenaltyScaling(Number(e.target.value))
                    }
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
              background: "#f97316",
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
                (e.target as HTMLButtonElement).style.background = "#ea580c";
            }}
            onMouseLeave={(e) => {
              (e.target as HTMLButtonElement).style.background = "#f97316";
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
                        href={`/simulate/head-to-head/replay/${r.run_id}`}
                        style={{
                          fontSize: 12,
                          color: "#f97316",
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
