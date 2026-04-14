"use client";

/**
 * CooperativeRunSummary
 *
 * Episode summary card shown after a cooperative run completes.
 * Displays outcome metrics, social health indicators, and a per-agent table.
 * Dark theme, teal accent.
 */

import { CooperativeEpisodeSummary, CooperativeAgentMetrics } from "@/lib/api";

interface Props {
  summary: CooperativeEpisodeSummary;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pct(v: number | undefined | null): string {
  if (v == null || isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function num(v: number | undefined | null, decimals = 3): string {
  if (v == null || isNaN(v)) return "—";
  return v.toFixed(decimals);
}

function terminationBadgeColor(reason: string | null): string {
  if (!reason) return "var(--bg-elevated)";
  if (reason === "perfect_clearance") return "rgba(34,197,94,0.15)";
  if (reason === "system_collapse") return "rgba(239,68,68,0.15)";
  return "rgba(20,184,166,0.1)";
}

function terminationTextColor(reason: string | null): string {
  if (!reason) return "var(--text-tertiary)";
  if (reason === "perfect_clearance") return "#22c55e";
  if (reason === "system_collapse") return "#ef4444";
  return "var(--accent)";
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const panelStyle: React.CSSProperties = {
  background: "var(--bg-surface)",
  border: "1px solid var(--bg-border)",
  borderRadius: 6,
  padding: "20px 24px",
};

const sectionTitle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: "var(--text-tertiary)",
  marginBottom: 12,
};

const dlStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: "6px 16px",
  fontSize: 12,
};

const dtStyle: React.CSSProperties = { color: "var(--text-secondary)" };

const ddStyle: React.CSSProperties = {
  color: "var(--text-primary)",
  fontFamily: "var(--font-mono)",
  fontWeight: 500,
};

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt style={dtStyle}>{label}</dt>
      <dd style={ddStyle}>{value}</dd>
    </>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function CooperativeRunSummary({ summary }: Props) {
  const agentIds = Object.keys(summary.agent_metrics ?? {});

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* ── Termination banner ── */}
      <div
        style={{
          padding: "10px 16px",
          borderRadius: 6,
          background: terminationBadgeColor(summary.termination_reason),
          border: `1px solid ${terminationTextColor(summary.termination_reason)}33`,
          fontSize: 13,
          fontWeight: 500,
          color: terminationTextColor(summary.termination_reason),
          fontFamily: "var(--font-mono)",
          textAlign: "center",
        }}
      >
        {summary.termination_reason ?? "unknown"}
      </div>

      {/* ── Outcome metrics ── */}
      <div style={panelStyle}>
        <div style={{ ...sectionTitle, display: "flex", alignItems: "center", gap: 8 }}>
          <span>Outcome</span>
          {summary.termination_reason === "system_collapse" && (
            <span
              style={{
                fontSize: 11,
                color: "#ef4444",
                fontWeight: 400,
                textTransform: "none",
                letterSpacing: 0,
              }}
            >
              Group failed to keep up with task demand
            </span>
          )}
          {summary.termination_reason === "perfect_clearance" && (
            <span
              style={{
                fontSize: 11,
                color: "#22c55e",
                fontWeight: 400,
                textTransform: "none",
                letterSpacing: 0,
              }}
            >
              Group cleared the entire backlog
            </span>
          )}
        </div>
        <dl style={dlStyle}>
          <MetricRow
            label="Completion ratio"
            value={
              summary.termination_reason === "system_collapse" &&
              summary.completion_ratio === 1.0
                ? "N/A"
                : pct(summary.completion_ratio)
            }
          />
          <MetricRow label="Group efficiency" value={pct(summary.group_efficiency_ratio)} />
          <MetricRow label="Mean system stress" value={num(summary.mean_system_stress)} />
          <MetricRow label="Peak system stress" value={num(summary.peak_system_stress)} />
          <MetricRow label="Episode length" value={`${summary.episode_length} steps`} />
          <MetricRow
            label="Tasks completed"
            value={`${summary.total_tasks_completed} / ${summary.total_tasks_arrived}`}
          />
        </dl>
      </div>

      {/* ── Social health ── */}
      <div style={panelStyle}>
        <div style={sectionTitle}>Social Health</div>
        <dl style={dlStyle}>
          <MetricRow
            label="Effort Gini coefficient"
            value={num(summary.effort_gini_coefficient)}
          />
          <MetricRow
            label="Free rider fraction"
            value={pct(summary.free_rider_fraction)}
          />
          <MetricRow
            label="Free rider count"
            value={String(summary.free_rider_count ?? 0)}
          />
          <MetricRow
            label="Specialization divergence"
            value={pct(summary.specialization_divergence)}
          />
          <MetricRow
            label="Mean role stability"
            value={pct(summary.mean_role_stability)}
          />
        </dl>
      </div>

      {/* ── Per-agent table ── */}
      {agentIds.length > 0 && (
        <div style={panelStyle}>
          <div style={sectionTitle}>Per-Agent Metrics</div>
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                fontSize: 11,
                borderCollapse: "collapse",
              }}
            >
              <thead>
                <tr
                  style={{
                    color: "var(--text-tertiary)",
                    borderBottom: "1px solid var(--bg-border)",
                  }}
                >
                  {[
                    "Agent",
                    "Effort util.",
                    "Idle rate",
                    "Dominant type",
                    "Final spec.",
                    "Role stability",
                    "Strategy",
                  ].map((h) => (
                    <th
                      key={h}
                      style={{
                        textAlign: "left",
                        padding: "6px 8px",
                        fontWeight: 500,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {agentIds.map((aid) => {
                  const m: CooperativeAgentMetrics = summary.agent_metrics[aid];
                  return (
                    <tr
                      key={aid}
                      style={{ borderBottom: "1px solid var(--bg-border-subtle)" }}
                    >
                      <td
                        style={{
                          padding: "6px 8px",
                          fontFamily: "var(--font-mono)",
                          color: "var(--accent)",
                        }}
                      >
                        {aid}
                      </td>
                      <td style={{ padding: "6px 8px", fontFamily: "var(--font-mono)" }}>
                        {pct(m.effort_utilization)}
                      </td>
                      <td style={{ padding: "6px 8px", fontFamily: "var(--font-mono)" }}>
                        {pct(m.idle_rate)}
                      </td>
                      <td style={{ padding: "6px 8px", fontFamily: "var(--font-mono)" }}>
                        {m.dominant_task_type != null
                          ? `type_${m.dominant_task_type}`
                          : "—"}
                      </td>
                      <td style={{ padding: "6px 8px", fontFamily: "var(--font-mono)" }}>
                        {num(m.final_specialization_score)}
                      </td>
                      <td style={{ padding: "6px 8px", fontFamily: "var(--font-mono)" }}>
                        {pct(m.role_stability)}
                      </td>
                      <td
                        style={{
                          padding: "6px 8px",
                          color: "var(--text-secondary)",
                        }}
                      >
                        {m.strategy_label || "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
