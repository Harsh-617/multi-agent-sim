"use client";

/**
 * Competitive replay visualisation.
 *
 * Renders three SVG charts for competitive-archetype replay data:
 *   1. Score over time (per agent, line chart)
 *   2. Rank over time (per agent, line chart — inverted so rank 1 is top)
 *   3. Action distribution over time (BUILD / ATTACK / DEFEND / GAMBLE)
 *
 * Follows the same minimal-SVG patterns used in MetricsChart.tsx.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CompetitiveStepMetric {
  step: number;
  agent_id: string;
  reward: number;
  action_type: string;
  action_amount: number;
  own_score: number;
  own_resources: number;
  own_rank: number;
  num_active_agents: number;
  attack_ratio: number;
  defend_ratio: number;
  build_ratio: number;
  gamble_ratio: number;
}

interface Props {
  history: CompetitiveStepMetric[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const AGENT_COLORS = [
  "#3b82f6", // blue
  "#ef4444", // red
  "#22c55e", // green
  "#f59e0b", // amber
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#14b8a6", // teal
  "#f97316", // orange
];

function colorFor(index: number): string {
  return AGENT_COLORS[index % AGENT_COLORS.length];
}

/** Collect unique sorted steps from the history. */
function uniqueSteps(history: CompetitiveStepMetric[]): number[] {
  const set = new Set<number>();
  for (const m of history) set.add(m.step);
  return Array.from(set).sort((a, b) => a - b);
}

/** Collect unique agent ids preserving first-seen order. */
function uniqueAgents(history: CompetitiveStepMetric[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const m of history) {
    if (!seen.has(m.agent_id)) {
      seen.add(m.agent_id);
      result.push(m.agent_id);
    }
  }
  return result;
}

/** Build a lookup: agentId → step → metric */
function buildLookup(
  history: CompetitiveStepMetric[],
): Map<string, Map<number, CompetitiveStepMetric>> {
  const map = new Map<string, Map<number, CompetitiveStepMetric>>();
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

// ---------------------------------------------------------------------------
// Sub-components (SVG charts)
// ---------------------------------------------------------------------------

const CHART_W = 480;
const CHART_H = 160;
const PAD = { top: 8, right: 12, bottom: 24, left: 40 };
const PLOT_W = CHART_W - PAD.left - PAD.right;
const PLOT_H = CHART_H - PAD.top - PAD.bottom;

/** Generic per-agent line chart. */
function LineChart({
  steps,
  agents,
  lookup,
  valueFn,
  invertY = false,
  yLabel,
}: {
  steps: number[];
  agents: string[];
  lookup: Map<string, Map<number, CompetitiveStepMetric>>;
  valueFn: (m: CompetitiveStepMetric) => number;
  invertY?: boolean;
  yLabel: string;
}) {
  if (steps.length === 0) return null;

  // Compute y range across all agents.
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const [, stepMap] of lookup) {
    for (const [, m] of stepMap) {
      const v = valueFn(m);
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  if (!isFinite(yMin)) {
    yMin = 0;
    yMax = 1;
  }
  if (yMax === yMin) {
    yMax = yMin + 1;
  }

  const xScale = (step: number) => {
    if (steps.length === 1) return PLOT_W / 2;
    const idx = steps.indexOf(step);
    return (idx / (steps.length - 1)) * PLOT_W;
  };

  const yScale = (v: number) => {
    const norm = (v - yMin) / (yMax - yMin);
    return invertY ? norm * PLOT_H : (1 - norm) * PLOT_H;
  };

  return (
    <svg
      width={CHART_W}
      height={CHART_H}
      className="block"
      viewBox={`0 0 ${CHART_W} ${CHART_H}`}
    >
      {/* Y-axis label */}
      <text
        x={4}
        y={PAD.top + PLOT_H / 2}
        textAnchor="middle"
        fontSize={9}
        fill="currentColor"
        transform={`rotate(-90, 4, ${PAD.top + PLOT_H / 2})`}
      >
        {yLabel}
      </text>

      {/* Y-axis ticks */}
      {[0, 0.5, 1].map((frac) => {
        const yPx = PAD.top + frac * PLOT_H;
        const val = invertY ? yMin + frac * (yMax - yMin) : yMax - frac * (yMax - yMin);
        return (
          <g key={frac}>
            <line
              x1={PAD.left}
              y1={yPx}
              x2={PAD.left + PLOT_W}
              y2={yPx}
              stroke="currentColor"
              opacity={0.1}
            />
            <text x={PAD.left - 4} y={yPx + 3} textAnchor="end" fontSize={8} fill="currentColor">
              {Number.isInteger(val) ? val : val.toFixed(1)}
            </text>
          </g>
        );
      })}

      {/* Lines per agent */}
      <g transform={`translate(${PAD.left},${PAD.top})`}>
        {agents.map((agentId, ai) => {
          const stepMap = lookup.get(agentId);
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
              key={agentId}
              points={pts}
              fill="none"
              stroke={colorFor(ai)}
              strokeWidth={1.5}
            />
          );
        })}
      </g>

      {/* X-axis labels (first & last step) */}
      <text
        x={PAD.left}
        y={CHART_H - 4}
        textAnchor="start"
        fontSize={8}
        fill="currentColor"
      >
        {steps[0]}
      </text>
      <text
        x={PAD.left + PLOT_W}
        y={CHART_H - 4}
        textAnchor="end"
        fontSize={8}
        fill="currentColor"
      >
        {steps[steps.length - 1]}
      </text>
    </svg>
  );
}

/** Stacked-area style action distribution chart. */
function ActionDistributionChart({
  steps,
  history,
}: {
  steps: number[];
  history: CompetitiveStepMetric[];
}) {
  if (steps.length === 0) return null;

  const ACTION_KEYS = [
    { key: "build_ratio" as const, color: "#22c55e", label: "BUILD" },
    { key: "attack_ratio" as const, color: "#ef4444", label: "ATTACK" },
    { key: "defend_ratio" as const, color: "#3b82f6", label: "DEFEND" },
    { key: "gamble_ratio" as const, color: "#f59e0b", label: "GAMBLE" },
  ];

  // Average ratios across all agents per step.
  const stepAvg = new Map<
    number,
    { build_ratio: number; attack_ratio: number; defend_ratio: number; gamble_ratio: number }
  >();
  for (const step of steps) {
    const metricsAtStep = history.filter((m) => m.step === step);
    const n = metricsAtStep.length || 1;
    stepAvg.set(step, {
      build_ratio: metricsAtStep.reduce((s, m) => s + (m.build_ratio ?? 0), 0) / n,
      attack_ratio: metricsAtStep.reduce((s, m) => s + (m.attack_ratio ?? 0), 0) / n,
      defend_ratio: metricsAtStep.reduce((s, m) => s + (m.defend_ratio ?? 0), 0) / n,
      gamble_ratio: metricsAtStep.reduce((s, m) => s + (m.gamble_ratio ?? 0), 0) / n,
    });
  }

  const xScale = (idx: number) => {
    if (steps.length === 1) return PLOT_W / 2;
    return (idx / (steps.length - 1)) * PLOT_W;
  };

  // Build stacked area paths (bottom-up).
  const areas: { path: string; color: string; label: string }[] = [];
  // For each step index, accumulate the baseline.
  const baselines = new Array(steps.length).fill(0);

  for (const { key, color, label } of ACTION_KEYS) {
    const topPoints: string[] = [];
    const bottomPoints: string[] = [];

    for (let i = 0; i < steps.length; i++) {
      const avg = stepAvg.get(steps[i]);
      const val = avg ? avg[key] : 0;
      const x = xScale(i);
      const yBottom = PLOT_H - baselines[i] * PLOT_H;
      const yTop = PLOT_H - (baselines[i] + val) * PLOT_H;
      topPoints.push(`${x},${yTop}`);
      bottomPoints.unshift(`${x},${yBottom}`);
      baselines[i] += val;
    }

    const path = `M${topPoints.join(" L")} L${bottomPoints.join(" L")}Z`;
    areas.push({ path, color, label });
  }

  return (
    <div>
      <svg
        width={CHART_W}
        height={CHART_H}
        className="block"
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
      >
        <g transform={`translate(${PAD.left},${PAD.top})`}>
          {areas.map(({ path, color, label }) => (
            <path key={label} d={path} fill={color} opacity={0.6} />
          ))}
        </g>

        {/* X-axis labels */}
        <text
          x={PAD.left}
          y={CHART_H - 4}
          textAnchor="start"
          fontSize={8}
          fill="currentColor"
        >
          {steps[0]}
        </text>
        <text
          x={PAD.left + PLOT_W}
          y={CHART_H - 4}
          textAnchor="end"
          fontSize={8}
          fill="currentColor"
        >
          {steps[steps.length - 1]}
        </text>
      </svg>

      {/* Legend */}
      <div className="flex gap-3 mt-1">
        {areas.map(({ color, label }) => (
          <span key={label} className="flex items-center gap-1 text-xs">
            <span
              className="inline-block w-3 h-3 rounded-sm"
              style={{ backgroundColor: color, opacity: 0.6 }}
            />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

function AgentLegend({ agents }: { agents: string[] }) {
  return (
    <div className="flex flex-wrap gap-3 mb-2">
      {agents.map((id, i) => (
        <span key={id} className="flex items-center gap-1 text-xs font-mono">
          <span
            className="inline-block w-3 h-3 rounded-sm"
            style={{ backgroundColor: colorFor(i) }}
          />
          {id.length > 8 ? id.slice(0, 8) : id}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function CompetitiveReplayView({ history }: Props) {
  if (history.length === 0) return null;

  const steps = uniqueSteps(history);
  const agents = uniqueAgents(history);
  const lookup = buildLookup(history);

  return (
    <div className="space-y-6">
      <AgentLegend agents={agents} />

      {/* Score over time */}
      <div>
        <h3 className="text-sm font-semibold mb-1">Score over Time</h3>
        <LineChart
          steps={steps}
          agents={agents}
          lookup={lookup}
          valueFn={(m) => m.own_score}
          yLabel="Score"
        />
      </div>

      {/* Rank over time (rank 1 = top) */}
      <div>
        <h3 className="text-sm font-semibold mb-1">Rank over Time</h3>
        <LineChart
          steps={steps}
          agents={agents}
          lookup={lookup}
          valueFn={(m) => m.own_rank}
          invertY
          yLabel="Rank"
        />
      </div>

      {/* Action distribution */}
      <div>
        <h3 className="text-sm font-semibold mb-1">Action Distribution (avg across agents)</h3>
        <ActionDistributionChart steps={steps} history={history} />
      </div>
    </div>
  );
}
