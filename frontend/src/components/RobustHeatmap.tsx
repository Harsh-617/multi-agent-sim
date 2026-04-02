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

  const cellW = sweeps.length > 15 ? 52 : sweeps.length > 10 ? 60 : 72;
  const cellH = 36;
  const labelW = 140;
  const headerH = 0;
  const svgW = labelW + sweeps.length * cellW + 10;
  const svgH = headerH + policies.length * cellH + 10;

  return (
    <div>
      {/* Single scrollable container for headers + heatmap */}
      <div style={{ overflowX: "auto" }}>
      {/* Column headers as HTML above SVG */}
      <div style={{
        display: "flex",
        paddingLeft: labelW,
        marginBottom: 2,
        minWidth: labelW + sweeps.length * cellW,
      }}>
        {sweeps.map((s) => (
          <div
            key={s}
            style={{
              width: cellW,
              flexShrink: 0,
              height: 100,
              display: "flex",
              alignItems: "flex-end",
              justifyContent: "center",
              paddingBottom: 4,
              overflow: "hidden",
            }}
          >
            <div style={{
              writingMode: "vertical-rl" as const,
              transform: "rotate(180deg)",
              fontSize: 9,
              color: "#666666",
              fontFamily: "monospace",
              whiteSpace: "nowrap",
              lineHeight: 1,
            }}>
              {s}
            </div>
          </div>
        ))}
      </div>
      <svg width={svgW} height={svgH} className="font-mono text-xs">
        <rect width={svgW} height={svgH} fill="#0d0d0d" />

        {/* Rows */}
        {policies.map((policy, ri) => (
          <g key={policy}>
            {/* Row label */}
            <text
              x={labelW - 8}
              y={headerH + ri * cellH + cellH / 2 + 4}
              textAnchor="end"
              fill="#888888"
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
                    fill={v != null ? rewardColor(v, globalMin, globalMax) : "#1a1a1a"}
                  />
                  <text
                    x={labelW + ci * cellW + (cellW - 2) / 2}
                    y={headerH + ri * cellH + cellH / 2 + 4}
                    textAnchor="middle"
                    fontSize={10}
                    fill={v != null ? "#fff" : "#444444"}
                  >
                    {v != null ? v.toFixed(1) : "—"}
                  </text>
                </g>
              );
            })}
          </g>
        ))}

        {/* Grid lines between rows */}
        {Array.from({ length: policies.length - 1 }, (_, i) => i + 1).map((i) => (
          <line
            key={`grid-${i}`}
            x1={labelW}
            y1={headerH + i * cellH}
            x2={labelW + sweeps.length * cellW}
            y2={headerH + i * cellH}
            stroke="#0d0d0d"
            strokeWidth={2}
          />
        ))}
      </svg>
      </div>{/* end single scroll container */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        marginTop: 8,
        paddingLeft: labelW,
      }}>
        <span style={{ fontSize: 10, color: "#555555", fontFamily: "monospace" }}>
          {globalMin.toFixed(1)}
        </span>
        <div style={{
          flex: 1,
          maxWidth: 160,
          height: 6,
          borderRadius: 3,
          background: "linear-gradient(to right, hsl(0,70%,45%), hsl(60,70%,45%), hsl(120,70%,45%))",
        }} />
        <span style={{ fontSize: 10, color: "#555555", fontFamily: "monospace" }}>
          {globalMax.toFixed(1)}
        </span>
        <span style={{ fontSize: 10, color: "#444444", marginLeft: 4 }}>
          mean reward
        </span>
      </div>
    </div>
  );
}
