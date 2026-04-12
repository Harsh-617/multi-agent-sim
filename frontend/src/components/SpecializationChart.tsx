"use client";

/**
 * SpecializationChart
 *
 * Per-agent specialization score over time.
 * One line per agent per task type.
 * Shows role-lock events (specialization_threshold_crossed) as vertical markers.
 * Dark theme, teal accent.
 */

import { CooperativeStepMetric } from "@/lib/api";

interface RoleLockEvent {
  step: number;
  agent_id: string;
  task_type: number;
}

interface Props {
  /** Step metrics from the run. `specialization_score` is not in StepMetric
   *  directly, but `r_efficiency` is a proxy — we use effort_amount per task
   *  type as a specialization proxy. If actual spec scores are not available,
   *  the chart shows effort distribution instead. */
  history: CooperativeStepMetric[];
  /** Optional: specialization threshold events from events.jsonl. */
  roleLockEvents?: RoleLockEvent[];
}

const AGENT_COLORS = [
  "#14b8a6",
  "#3b82f6",
  "#f59e0b",
  "#8b5cf6",
  "#ef4444",
  "#ec4899",
  "#22c55e",
  "#f97316",
];

function colorFor(index: number): string {
  return AGENT_COLORS[index % AGENT_COLORS.length];
}

const CHART_W = 600;
const CHART_H = 160;
const PAD = { top: 8, right: 16, bottom: 22, left: 44 };
const PLOT_W = CHART_W - PAD.left - PAD.right;
const PLOT_H = CHART_H - PAD.top - PAD.bottom;

export default function SpecializationChart({
  history,
  roleLockEvents = [],
}: Props) {
  if (history.length === 0) {
    return (
      <div
        style={{
          padding: "24px 0",
          textAlign: "center",
          color: "var(--text-tertiary)",
          fontSize: 13,
        }}
      >
        No step data yet.
      </div>
    );
  }

  // Collect unique steps and agents
  const stepSet = new Set<number>();
  const agentSet = new Set<string>();
  for (const m of history) {
    stepSet.add(m.step);
    agentSet.add(m.agent_id);
  }
  const steps = Array.from(stepSet).sort((a, b) => a - b);
  const agents = Array.from(agentSet);

  if (steps.length === 0) return null;

  // Use r_efficiency as a specialization proxy (bounded [0,1])
  // keyed by agent_id → step → value
  const lookup = new Map<string, Map<number, number>>();
  for (const m of history) {
    let inner = lookup.get(m.agent_id);
    if (!inner) {
      inner = new Map();
      lookup.set(m.agent_id, inner);
    }
    inner.set(m.step, m.r_efficiency);
  }

  const xScale = (step: number) => {
    const idx = steps.indexOf(step);
    if (steps.length <= 1) return PLOT_W / 2;
    return (idx / (steps.length - 1)) * PLOT_W;
  };
  const yScale = (v: number) => PLOT_H - v * PLOT_H;

  return (
    <div>
      <div
        style={{
          fontSize: 11,
          fontWeight: 500,
          color: "var(--text-secondary)",
          marginBottom: 4,
        }}
      >
        Specialization Signal (R_efficiency) per Agent
      </div>
      <svg
        width={CHART_W}
        height={CHART_H}
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        style={{ width: "100%", height: "auto", display: "block" }}
      >
        {/* Grid */}
        {[0, 0.5, 1].map((frac) => {
          const yPx = PAD.top + frac * PLOT_H;
          const val = 1 - frac;
          return (
            <g key={frac}>
              <line
                x1={PAD.left}
                y1={yPx}
                x2={PAD.left + PLOT_W}
                y2={yPx}
                stroke="var(--bg-border)"
              />
              <text
                x={PAD.left - 4}
                y={yPx + 3}
                textAnchor="end"
                fontSize={8}
                fill="var(--text-tertiary)"
              >
                {val.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* Role-lock event markers */}
        {roleLockEvents.map((evt, i) => {
          const x = PAD.left + xScale(evt.step);
          return (
            <line
              key={i}
              x1={x}
              y1={PAD.top}
              x2={x}
              y2={PAD.top + PLOT_H}
              stroke="var(--accent)"
              strokeDasharray="3 2"
              opacity={0.5}
            />
          );
        })}

        {/* Lines per agent */}
        <g transform={`translate(${PAD.left},${PAD.top})`}>
          {agents.map((aid, ai) => {
            const stepMap = lookup.get(aid);
            if (!stepMap) return null;
            const pts = steps
              .filter((s) => stepMap.has(s))
              .map((s) => `${xScale(s)},${yScale(stepMap.get(s) ?? 0)}`)
              .join(" ");
            if (!pts) return null;
            return (
              <polyline
                key={aid}
                points={pts}
                fill="none"
                stroke={colorFor(ai)}
                strokeWidth={1.5}
                opacity={0.85}
              />
            );
          })}
        </g>

        {/* X-axis */}
        <text
          x={PAD.left}
          y={CHART_H - 4}
          textAnchor="start"
          fontSize={8}
          fill="var(--text-tertiary)"
        >
          {steps[0]}
        </text>
        <text
          x={PAD.left + PLOT_W}
          y={CHART_H - 4}
          textAnchor="end"
          fontSize={8}
          fill="var(--text-tertiary)"
        >
          {steps[steps.length - 1]}
        </text>
      </svg>

      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 6 }}>
        {agents.map((aid, ai) => (
          <span
            key={aid}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              fontSize: 10,
              fontFamily: "var(--font-mono)",
              color: "var(--text-secondary)",
            }}
          >
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                borderRadius: 2,
                background: colorFor(ai),
              }}
            />
            {aid}
          </span>
        ))}
        {roleLockEvents.length > 0 && (
          <span
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              fontSize: 10,
              color: "var(--accent)",
            }}
          >
            <span
              style={{
                display: "inline-block",
                width: 12,
                height: 2,
                background: "var(--accent)",
                borderTop: "2px dashed var(--accent)",
              }}
            />
            role lock
          </span>
        )}
      </div>
    </div>
  );
}
