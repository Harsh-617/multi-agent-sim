"use client";

import { StepMetric } from "@/lib/api";

interface Props {
  /** All accumulated metrics, across all steps. */
  history: StepMetric[];
}

/**
 * Minimal SVG bar chart showing the latest per-agent reward.
 * Also renders a small line-sparkline of shared_pool over time.
 */
export default function MetricsChart({ history }: Props) {
  if (history.length === 0) return null;

  // --- Per-agent reward bars (latest step, deduplicated) ---
  const latestStep = history[history.length - 1].step;
  const latestMap = new Map<string, StepMetric>();
  for (const m of history) {
    if (m.step === latestStep) latestMap.set(m.agent_id, m);
  }
  const latest = Array.from(latestMap.values());

  const maxReward = Math.max(...latest.map((m) => Math.abs(m.reward)), 0.01);
  const barW = 40;
  const barGap = 8;
  const chartH = 120;
  const chartW = latest.length * (barW + barGap);

  // --- Shared pool sparkline ---
  // One value per step (take from first agent's metric each step)
  const seenSteps = new Map<number, number>();
  for (const m of history) {
    if (!seenSteps.has(m.step)) seenSteps.set(m.step, m.shared_pool);
  }
  const poolSeries = Array.from(seenSteps.entries()).sort((a, b) => a[0] - b[0]);
  const sparkW = 300;
  const sparkH = 60;
  const maxPool = Math.max(...poolSeries.map((p) => p[1]), 0.01);

  function sparkPoints(): string {
    if (poolSeries.length < 2) return "";
    return poolSeries
      .map(([, v], i) => {
        const x = (i / (poolSeries.length - 1)) * sparkW;
        const y = sparkH - (v / maxPool) * sparkH;
        return `${x},${y}`;
      })
      .join(" ");
  }

  return (
    <div className="space-y-4">
      {/* Reward bars */}
      <div>
        <h3 className="text-sm font-semibold mb-1">
          Per-Agent Reward (step {latestStep})
        </h3>
        <svg width={chartW} height={chartH + 24} className="block">
          {latest.map((m, i) => {
            const h = (Math.abs(m.reward) / maxReward) * chartH;
            const x = i * (barW + barGap);
            const color = m.reward >= 0 ? "#22c55e" : "#ef4444";
            return (
              <g key={m.agent_id}>
                <rect x={x} y={chartH - h} width={barW} height={h} fill={color} rx={2} />
                <text
                  x={x + barW / 2}
                  y={chartH + 14}
                  textAnchor="middle"
                  fontSize={10}
                  fill="currentColor"
                >
                  {m.agent_id}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Shared pool sparkline */}
      <div>
        <h3 className="text-sm font-semibold mb-1">
          Shared Pool ({poolSeries[poolSeries.length - 1]?.[1].toFixed(1)})
        </h3>
        <svg width={sparkW} height={sparkH} className="block">
          {poolSeries.length >= 2 && (
            <polyline
              points={sparkPoints()}
              fill="none"
              stroke="#3b82f6"
              strokeWidth={1.5}
            />
          )}
        </svg>
      </div>
    </div>
  );
}
