"use client";

/**
 * ContributionVarianceChart
 *
 * Two-line chart showing:
 *   1. Effort Gini coefficient proxy over time (using r_individual variance
 *      across agents as a rolling proxy — exact Gini is in the episode summary)
 *   2. Free rider pressure signal over time (from completion_rate inversion)
 *
 * Dark theme, teal accent.
 */

import { CooperativeStepMetric } from "@/lib/api";

interface Props {
  history: CooperativeStepMetric[];
}

const CHART_W = 600;
const CHART_H = 140;
const PAD = { top: 8, right: 16, bottom: 22, left: 44 };
const PLOT_W = CHART_W - PAD.left - PAD.right;
const PLOT_H = CHART_H - PAD.top - PAD.bottom;

function mkXScale(steps: number[]) {
  return (step: number) => {
    const idx = steps.indexOf(step);
    if (steps.length <= 1) return PLOT_W / 2;
    return (idx / (steps.length - 1)) * PLOT_W;
  };
}

function mkYScale(lo: number, hi: number) {
  const range = hi === lo ? 1 : hi - lo;
  return (v: number) => PLOT_H - ((v - lo) / range) * PLOT_H;
}

/** Rolling Gini proxy: std-dev of effort_amount across agents at a step. */
function rollingGiniProxy(
  history: CooperativeStepMetric[],
  steps: number[],
): [number, number][] {
  const byStep = new Map<number, number[]>();
  for (const m of history) {
    let arr = byStep.get(m.step);
    if (!arr) {
      arr = [];
      byStep.set(m.step, arr);
    }
    arr.push(m.effort_amount);
  }
  return steps.map((s) => {
    const vals = byStep.get(s) ?? [];
    if (vals.length < 2) return [s, 0];
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance =
      vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
    return [s, Math.sqrt(variance)];
  });
}

/** Free-rider pressure: 1 - rolling completion_rate. */
function freeRiderPressure(
  history: CooperativeStepMetric[],
  steps: number[],
): [number, number][] {
  const seen = new Map<number, number>();
  for (const m of history) {
    if (!seen.has(m.step)) seen.set(m.step, m.completion_rate);
  }
  return steps.map((s) => [s, 1 - (seen.get(s) ?? 1)]);
}

export default function ContributionVarianceChart({ history }: Props) {
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

  const stepSet = new Set<number>();
  for (const m of history) stepSet.add(m.step);
  const steps = Array.from(stepSet).sort((a, b) => a - b);

  const giniSeries = rollingGiniProxy(history, steps);
  const pressureSeries = freeRiderPressure(history, steps);

  const allVals = [...giniSeries.map(([, v]) => v), ...pressureSeries.map(([, v]) => v)];
  const lo = 0;
  const hi = Math.max(...allVals, 0.01);

  const xScale = mkXScale(steps);
  const yScale = mkYScale(lo, hi);

  const SERIES = [
    { data: giniSeries, color: "#f97316", label: "Effort spread (std-dev proxy)" },
    { data: pressureSeries, color: "#ef4444", label: "Free-rider pressure (1 − completion)" },
  ];

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
        Contribution Variance & Free-Rider Pressure
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
          const val = hi - frac * (hi - lo);
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
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Lines */}
        <g transform={`translate(${PAD.left},${PAD.top})`}>
          {SERIES.map(({ data, color, label }) => {
            const pts = data.map(([s, v]) => `${xScale(s)},${yScale(v)}`).join(" ");
            return (
              <polyline
                key={label}
                points={pts}
                fill="none"
                stroke={color}
                strokeWidth={1.5}
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
      <div style={{ display: "flex", gap: 12, marginTop: 6 }}>
        {SERIES.map(({ color, label }) => (
          <span
            key={label}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              fontSize: 10,
              color: "var(--text-secondary)",
            }}
          >
            <span
              style={{
                display: "inline-block",
                width: 12,
                height: 3,
                borderRadius: 1,
                background: color,
              }}
            />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
