"use client";

/**
 * CooperativeMetricsChart
 *
 * Four-panel SVG chart for cooperative-archetype step metrics:
 *   1. Backlog level over time
 *   2. System stress over time
 *   3. Group completion rate (rolling) over time
 *   4. Per-agent effort utilization over time (one line per agent)
 *
 * Accepts `history: CooperativeStepMetric[]` — the page component feeds
 * data in via WebSocket (live run) or SSE replay.
 * Handles empty / loading state gracefully.
 */

import { CooperativeStepMetric } from "@/lib/api";

interface Props {
  history: CooperativeStepMetric[];
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
function uniqueSteps(history: CooperativeStepMetric[]): number[] {
  const set = new Set<number>();
  for (const m of history) set.add(m.step);
  return Array.from(set).sort((a, b) => a - b);
}

/** Unique agent ids in first-seen order. */
function uniqueAgents(history: CooperativeStepMetric[]): string[] {
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

type StepMap = Map<string, Map<number, CooperativeStepMetric>>;

/** Build agent → step → metric lookup. */
function buildLookup(history: CooperativeStepMetric[]): StepMap {
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

/** One global-level value per step (take first agent's reading). */
function globalSeries(
  history: CooperativeStepMetric[],
  steps: number[],
  key: keyof CooperativeStepMetric,
): [number, number][] {
  const seenSteps = new Map<number, number>();
  for (const m of history) {
    if (!seenSteps.has(m.step)) {
      seenSteps.set(m.step, m[key] as number);
    }
  }
  return steps.map((s) => [s, seenSteps.get(s) ?? 0]);
}

// ---------------------------------------------------------------------------
// SVG helpers
// ---------------------------------------------------------------------------

function yRange(
  series: [number, number][],
  minVal = 0,
  maxVal?: number,
): [number, number] {
  const vals = series.map(([, v]) => v);
  const lo = Math.min(minVal, ...vals);
  const hi = maxVal ?? Math.max(...vals, 0.01);
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

function polylinePoints(
  series: [number, number][],
  xScale: (s: number) => number,
  yScale: (v: number) => number,
): string {
  return series.map(([s, v]) => `${xScale(s)},${yScale(v)}`).join(" ");
}

// ---------------------------------------------------------------------------
// Shared axis labels & grid lines
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
// Panel: single-line sparkline (for global signals)
// ---------------------------------------------------------------------------

function SparklinePanel({
  series,
  steps,
  label,
  color,
  yMin = 0,
  yMax,
}: {
  series: [number, number][];
  steps: number[];
  label: string;
  color: string;
  yMin?: number;
  yMax?: number;
}) {
  const [lo, hi] = yRange(series, yMin, yMax);
  const xScale = mkXScale(steps);
  const yScale = mkYScale(lo, hi);
  const pts = polylinePoints(series, xScale, yScale);
  const latest = series[series.length - 1]?.[1];

  return (
    <div>
      <div
        style={{
          fontSize: 11,
          fontWeight: 500,
          color: "var(--text-secondary)",
          marginBottom: 4,
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>{label}</span>
        {latest !== undefined && (
          <span style={{ color, fontFamily: "var(--font-mono)" }}>
            {typeof latest === "number" && !Number.isInteger(latest)
              ? latest.toFixed(3)
              : latest}
          </span>
        )}
      </div>
      <svg
        width={CHART_W}
        height={CHART_H}
        className="block"
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        style={{ width: "100%", height: "auto" }}
      >
        <YAxis lo={lo} hi={hi} label={label} />
        <g transform={`translate(${PAD.left},${PAD.top})`}>
          {series.length >= 2 && (
            <polyline
              points={pts}
              fill="none"
              stroke={color}
              strokeWidth={1.5}
            />
          )}
        </g>
        <XAxisLabels steps={steps} />
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel: per-agent multi-line (for effort utilization)
// ---------------------------------------------------------------------------

function PerAgentPanel({
  steps,
  agents,
  lookup,
  label,
  valueFn,
  yMin = 0,
  yMax = 1,
}: {
  steps: number[];
  agents: string[];
  lookup: StepMap;
  label: string;
  valueFn: (m: CooperativeStepMetric) => number;
  yMin?: number;
  yMax?: number;
}) {
  const [lo, hi] = [yMin, yMax];
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
        <YAxis lo={lo} hi={hi} label="" />
        <g transform={`translate(${PAD.left},${PAD.top})`}>
          {agents.map((aid, ai) => {
            const stepMap = lookup.get(aid);
            if (!stepMap) return null;
            const pts = steps
              .filter((s) => stepMap.has(s))
              .map((s) => {
                const m = stepMap.get(s)!;
                return `${xScale(s)},${yScale(valueFn(m))}`;
              })
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

export default function CooperativeMetricsChart({ history }: Props) {
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

  const backlogSeries = globalSeries(history, steps, "backlog_level");
  const stressSeries = globalSeries(history, steps, "system_stress");
  const completionSeries = globalSeries(history, steps, "completion_rate");

  const panelStyle: React.CSSProperties = {
    background: "var(--bg-surface)",
    border: "1px solid var(--bg-border)",
    borderRadius: 6,
    padding: "16px 20px",
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {/* Panel 1 — Backlog level */}
      <div style={panelStyle}>
        <SparklinePanel
          series={backlogSeries}
          steps={steps}
          label="Backlog Level"
          color="var(--accent)"
          yMin={0}
        />
      </div>

      {/* Panel 2 — System stress */}
      <div style={panelStyle}>
        <SparklinePanel
          series={stressSeries}
          steps={steps}
          label="System Stress"
          color="#ef4444"
          yMin={0}
          yMax={1}
        />
      </div>

      {/* Panel 3 — Group completion rate */}
      <div style={panelStyle}>
        <SparklinePanel
          series={completionSeries}
          steps={steps}
          label="Group Completion Rate"
          color="#22c55e"
          yMin={0}
          yMax={1}
        />
      </div>

      {/* Panel 4 — Per-agent effort utilization */}
      <div style={panelStyle}>
        <PerAgentPanel
          steps={steps}
          agents={agents}
          lookup={lookup}
          label="Effort Utilization (per agent)"
          valueFn={(m) => m.effort_amount}
          yMin={0}
          yMax={1}
        />
      </div>
    </div>
  );
}
