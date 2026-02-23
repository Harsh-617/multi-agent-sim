"use client";

/**
 * Pure-SVG heatmap: policies (rows) x sweeps (cols), cell color by mean_total_reward.
 */

interface Props {
  perSweepResults: Record<string, Record<string, { mean_total_reward?: number; available?: boolean; n_episodes?: number }>>;
}

function rewardColor(value: number, min: number, max: number): string {
  if (max === min) return "hsl(210,50%,70%)";
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  // red (0) -> yellow (0.5) -> green (1)
  const hue = Math.round(t * 120);
  return `hsl(${hue},70%,45%)`;
}

export default function RobustHeatmap({ perSweepResults }: Props) {
  const sweeps = Object.keys(perSweepResults);
  if (sweeps.length === 0) return <p className="text-gray-500">No sweep data.</p>;

  // Collect all policy names across sweeps
  const policySet = new Set<string>();
  for (const sweepData of Object.values(perSweepResults)) {
    for (const name of Object.keys(sweepData)) policySet.add(name);
  }
  const policies = Array.from(policySet).sort();

  // Build value matrix and find range
  const values: (number | null)[][] = [];
  let globalMin = Infinity;
  let globalMax = -Infinity;

  for (const policy of policies) {
    const row: (number | null)[] = [];
    for (const sweep of sweeps) {
      const entry = perSweepResults[sweep]?.[policy];
      if (entry && entry.available !== false && entry.n_episodes && entry.n_episodes > 0 && entry.mean_total_reward != null) {
        const v = entry.mean_total_reward;
        row.push(v);
        if (v < globalMin) globalMin = v;
        if (v > globalMax) globalMax = v;
      } else {
        row.push(null);
      }
    }
    values.push(row);
  }

  const cellW = 80;
  const cellH = 32;
  const labelW = 160;
  const headerH = 80;
  const svgW = labelW + sweeps.length * cellW + 10;
  const svgH = headerH + policies.length * cellH + 10;

  return (
    <div className="overflow-x-auto">
      <svg width={svgW} height={svgH} className="font-mono text-xs">
        {/* Column headers (sweep names) */}
        {sweeps.map((s, ci) => (
          <text
            key={s}
            x={labelW + ci * cellW + cellW / 2}
            y={headerH - 6}
            textAnchor="end"
            fontSize={10}
            transform={`rotate(-35,${labelW + ci * cellW + cellW / 2},${headerH - 6})`}
          >
            {s}
          </text>
        ))}

        {/* Rows */}
        {policies.map((policy, ri) => (
          <g key={policy}>
            {/* Row label */}
            <text
              x={labelW - 8}
              y={headerH + ri * cellH + cellH / 2 + 4}
              textAnchor="end"
              fontSize={11}
            >
              {policy}
            </text>

            {/* Cells */}
            {sweeps.map((_, ci) => {
              const v = values[ri][ci];
              return (
                <g key={ci}>
                  <rect
                    x={labelW + ci * cellW}
                    y={headerH + ri * cellH}
                    width={cellW - 2}
                    height={cellH - 2}
                    rx={3}
                    fill={v != null ? rewardColor(v, globalMin, globalMax) : "#e5e7eb"}
                  />
                  <text
                    x={labelW + ci * cellW + (cellW - 2) / 2}
                    y={headerH + ri * cellH + cellH / 2 + 4}
                    textAnchor="middle"
                    fontSize={10}
                    fill={v != null ? "#fff" : "#999"}
                  >
                    {v != null ? v.toFixed(1) : "—"}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}
