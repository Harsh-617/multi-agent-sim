"use client";

/**
 * CompetitiveMetricsChart
 *
 * Four-panel SVG chart for competitive-archetype step metrics:
 *   1. Cumulative score per agent over time (one line per agent)
 *   2. Resources per agent over time (one line per agent)
 *   3. Live rank per agent over time (derived from cumulative score; 1 = best at top)
 *   4. Action amount per agent over time (one line per agent)
 *
 * Accepts `history: StepMetric[]` — the HH live run page feeds data in via
 * WebSocket. Reads competitive-specific fields: reward (score increments),
 * agent_resources, action_amount. Does NOT read shared_pool.
 */

import { StepMetric } from "@/lib/api";

interface Props {
  history: StepMetric[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AGENT_COLORS = [
  "#14b8a6", // teal-500
  "#3b82f6", // blue-500
  "#f59e0b", // amber-500
  "#8b5cf6", // violet-500
  "#ef4444", // red-500
  "#ec4899", // pink-500
  "#22c55e", // green-500
  "#f97316", // orange-500
];

function colorFor(index: number): string {
  return AGENT_COLORS[index % AGENT_COLORS.length];
}

const CHART_W = 500;
const CHART_H = 200;
const PAD = { top: 12, right: 16, bottom: 28, left: 52 };
const PLOT_W = CHART_W - PAD.left - PAD.right;
const PLOT_H = CHART_H - PAD.top - PAD.bottom;

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------

/** Unique sorted steps. */
function uniqueSteps(history: StepMetric[]): number[] {
  const set = new Set<number>();
  for (const m of history) set.add(m.step);
  return Array.from(set).sort((a, b) => a - b);
}

/** Unique agent ids in first-seen order. */
function uniqueAgents(history: StepMetric[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const m of history) {
    if (!seen.has(m.agent_id)) {
      seen.add(m.agent_id);
      out.push(m.agent_id);
    }
  }
  return out;
}

type StepMap = Map<string, Map<number, StepMetric>>;

/** Build agent → step → metric lookup. */
function buildLookup(history: StepMetric[]): StepMap {
  const map: StepMap = new Map();
  for (const m of history) {
    let inner = map.get(m.agent_id);
    if (!inner) {
      inner = new Map();
      map.set(m.agent_id, inner);
    }
    inner.set(m.step, m);
  }
  return map;
}

/**
 * Cumulative reward (score) per agent at each step.
 * Returns agent → step → cumulative score.
 */
function buildCumulativeScores(
  steps: number[],
  agents: string[],
  lookup: StepMap,
): Map<string, Map<number, number>> {
  const result = new Map<string, Map<number, number>>();
  for (const aid of agents) {
    const agentMap = new Map<number, number>();
    let running = 0;
    for (const step of steps) {
      const m = lookup.get(aid)?.get(step);
      if (m !== undefined) running += m.reward;
      agentMap.set(step, running);
    }
    result.set(aid, agentMap);
  }
  return result;
}

/**
 * Live rank per agent at each step (1 = best score).
 * Returns agent → step → rank. Stored as (numAgents - rank + 1) so that
 * rank-1 agents plot at the top of the chart naturally.
 */
function buildRankSeries(
  steps: number[],
  agents: string[],
  cumulScores: Map<string, Map<number, number>>,
): Map<string, Map<number, number>> {
  const result = new Map<string, Map<number, number>>();
  for (const aid of agents) result.set(aid, new Map());

  for (const step of steps) {
    const scored = agents.map((aid) => ({
      aid,
      score: cumulScores.get(aid)?.get(step) ?? 0,
    }));
    // Sort descending so index 0 = rank 1 (best)
    scored.sort((a, b) => b.score - a.score);
    scored.forEach(({ aid }, idx) => {
      // Invert so rank 1 → numAgents (top of chart), rank N → 1 (bottom)
      result.get(aid)!.set(step, agents.length - idx);
    });
  }
  return result;
}

// ---------------------------------------------------------------------------
// SVG helpers
// ---------------------------------------------------------------------------

function yRange(values: number[], minVal = 0, maxVal?: number): [number, number] {
  const lo = Math.min(minVal, ...values);
  const hi = maxVal ?? Math.max(...values, 0.01);
  return [lo, hi === lo ? lo + 1 : hi];
}

function mkXScale(steps: number[]) {
  return (step: number) => {
    const idx = steps.indexOf(step);
    if (steps.length <= 1) return PLOT_W / 2;
    return (idx / (steps.length - 1)) * PLOT_W;
  };
}

function mkYScale(lo: number, hi: number) {
  return (v: number) => PLOT_H - ((v - lo) / (hi - lo)) * PLOT_H;
}

// ---------------------------------------------------------------------------
// Shared axis components
// ---------------------------------------------------------------------------

function YAxis({ lo, hi, label }: { lo: number; hi: number; label: string }) {
  return (
    <g>
      <text
        x={4}
        y={PAD.top + PLOT_H / 2}
        textAnchor="middle"
        fontSize={10}
        fill="var(--text-tertiary)"
        transform={`rotate(-90, 4, ${PAD.top + PLOT_H / 2})`}
      >
        {label}
      </text>
      {[0, 0.5, 1].map((frac) => {
        const yPx = PAD.top + frac * PLOT_H;
        const val = hi - frac * (hi - lo);
        return (
          <g key={frac}>
            <line
              x1={PAD.left}
              y1={yPx}
              x2={PAD.left + PLOT_W}
              y2={yPx}
              stroke="var(--bg-border)"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 4}
              y={yPx + 4}
              textAnchor="end"
              fontSize={10}
              fill="var(--text-tertiary)"
            >
              {Number.isInteger(val) ? val : val.toFixed(2)}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function XAxisLabels({ steps }: { steps: number[] }) {
  if (steps.length === 0) return null;
  return (
    <g>
      <text
        x={PAD.left}
        y={CHART_H - 6}
        textAnchor="start"
        fontSize={10}
        fill="var(--text-tertiary)"
      >
        {steps[0]}
      </text>
      <text
        x={PAD.left + PLOT_W}
        y={CHART_H - 6}
        textAnchor="end"
        fontSize={10}
        fill="var(--text-tertiary)"
      >
        {steps[steps.length - 1]}
      </text>
    </g>
  );
}

// ---------------------------------------------------------------------------
// Panel: per-agent multi-line chart
// ---------------------------------------------------------------------------

function PerAgentPanel({
  steps,
  agents,
  label,
  valueFn,
  yMin = 0,
  yMax,
  yAxisLabel = "",
}: {
  steps: number[];
  agents: string[];
  label: string;
  valueFn: (aid: string, step: number) => number | undefined;
  yMin?: number;
  yMax?: number;
  yAxisLabel?: string;
}) {
  // Collect all values to determine y range when yMax is not fixed
  const allVals: number[] = [];
  for (const aid of agents) {
    for (const step of steps) {
      const v = valueFn(aid, step);
      if (v !== undefined) allVals.push(v);
    }
  }
  const [lo, hi] = yRange(allVals, yMin, yMax);
  const xScale = mkXScale(steps);
  const yScale = mkYScale(lo, hi);

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
        {label}
      </div>
      <svg
        width={CHART_W}
        height={CHART_H}
        className="block"
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        style={{ width: "100%", height: "auto" }}
      >
        <YAxis lo={lo} hi={hi} label={yAxisLabel} />
        <g transform={`translate(${PAD.left},${PAD.top})`}>
          {agents.map((aid, ai) => {
            const pts = steps
              .map((s) => {
                const v = valueFn(aid, s);
                if (v === undefined) return null;
                return `${xScale(s)},${yScale(v)}`;
              })
              .filter(Boolean)
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
        <XAxisLabels steps={steps} />
      </svg>
      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 4 }}>
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
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function CompetitiveMetricsChart({ history }: Props) {
  if (history.length === 0) {
    return (
      <div
        style={{
          padding: "40px 0",
          textAlign: "center",
          color: "var(--text-tertiary)",
          fontSize: 13,
        }}
      >
        Waiting for step data…
      </div>
    );
  }

  const steps = uniqueSteps(history);
  const agents = uniqueAgents(history);
  const lookup = buildLookup(history);
  const cumulScores = buildCumulativeScores(steps, agents, lookup);
  const rankMap = buildRankSeries(steps, agents, cumulScores);

  const panelStyle: React.CSSProperties = {
    background: "var(--bg-surface)",
    border: "1px solid var(--bg-border)",
    borderRadius: 6,
    padding: "16px 20px",
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {/* Panel 1 — Cumulative score per agent */}
      <div style={panelStyle}>
        <PerAgentPanel
          steps={steps}
          agents={agents}
          label="Cumulative Score (per agent)"
          yAxisLabel="score"
          valueFn={(aid, step) => cumulScores.get(aid)?.get(step)}
        />
      </div>

      {/* Panel 2 — Resources per agent */}
      <div style={panelStyle}>
        <PerAgentPanel
          steps={steps}
          agents={agents}
          label="Resources (per agent)"
          yAxisLabel="res"
          yMin={0}
          valueFn={(aid, step) => lookup.get(aid)?.get(step)?.agent_resources}
        />
      </div>

      {/* Panel 3 — Live rank per agent (rank 1 at top) */}
      <div style={panelStyle}>
        <PerAgentPanel
          steps={steps}
          agents={agents}
          label="Live Rank (higher = better position)"
          yAxisLabel="rank"
          yMin={1}
          yMax={agents.length}
          valueFn={(aid, step) => rankMap.get(aid)?.get(step)}
        />
      </div>

      {/* Panel 4 — Action amount per agent */}
      <div style={panelStyle}>
        <PerAgentPanel
          steps={steps}
          agents={agents}
          label="Action Amount (per agent)"
          yAxisLabel="amt"
          yMin={0}
          valueFn={(aid, step) => lookup.get(aid)?.get(step)?.action_amount}
        />
      </div>
    </div>
  );
}
