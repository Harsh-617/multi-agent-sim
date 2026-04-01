"use client";

/**
 * Horizontal grouped bar chart: overall mean reward vs worst-case mean reward per policy.
 */

interface PolicyRobustness {
  policy_name: string;
  overall_mean_reward: number;
  worst_case_mean_reward: number;
  robustness_score: number;
  n_sweeps_evaluated: number;
}

interface Props {
  perPolicyRobustness: Record<string, PolicyRobustness>;
}

const COLORS = [
  "#14b8a6",
  "#f97316",
  "#8b5cf6",
  "#22c55e",
  "#ef4444",
  "#f59e0b",
  "#3b82f6",
  "#ec4899",
];

export default function RobustScatter({ perPolicyRobustness }: Props) {
  const entries = Object.values(perPolicyRobustness).filter(
    (p) => p.n_sweeps_evaluated > 0
  );
  if (entries.length === 0)
    return <p style={{ color: "#666666", fontSize: 13 }}>No data.</p>;

  const rowHeight = 44;
  const pad = { top: 40, right: 120, bottom: 40, left: 140 };
  const plotWidth = 560 - pad.left - pad.right; // 300
  const svgHeight = pad.top + entries.length * rowHeight + pad.bottom;

  const allValues = entries.flatMap((e) => [
    e.overall_mean_reward,
    e.worst_case_mean_reward,
  ]);
  const xMin = Math.min(0, ...allValues);
  const rawMax = Math.max(...allValues);
  const xMax = rawMax * 1.1 || 1;

  const scaleX = (v: number) =>
    pad.left + ((v - xMin) / (xMax - xMin)) * plotWidth;

  const fmt = (v: number) => v.toFixed(1);

  const ticks = Array.from({ length: 4 }, (_, i) => xMin + (i * (xMax - xMin)) / 3);
  const gridXs = ticks.map((v) => scaleX(v));

  const axisY = pad.top + entries.length * rowHeight;

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={560} height={svgHeight}>
        <rect width={560} height={svgHeight} fill="#0d0d0d" />

        {/* Legend */}
        <rect x={pad.left} y={14} width={8} height={8} fill="#14b8a6" rx={1} />
        <text x={pad.left + 12} y={21} fontSize={10} fill="#666666" fontFamily="monospace">
          Mean reward
        </text>
        <rect x={pad.left + 120} y={14} width={8} height={8} fill="#14b8a6" opacity={0.4} rx={1} />
        <text x={pad.left + 132} y={21} fontSize={10} fill="#666666" fontFamily="monospace">
          Worst-case
        </text>

        {/* Vertical grid lines */}
        {gridXs.map((gx, i) => (
          <line
            key={`g${i}`}
            x1={gx}
            y1={pad.top}
            x2={gx}
            y2={axisY}
            stroke="#1a1a1a"
            strokeWidth={1}
          />
        ))}

        {/* X axis line */}
        <line x1={pad.left} y1={axisY} x2={pad.left + plotWidth} y2={axisY} stroke="#2a2a2a" />

        {/* X axis ticks */}
        {ticks.map((v, i) => (
          <text
            key={`t${i}`}
            x={scaleX(v)}
            y={axisY + 14}
            textAnchor="middle"
            fontSize={9}
            fill="#444444"
            fontFamily="monospace"
          >
            {fmt(v)}
          </text>
        ))}

        {/* X axis label */}
        <text
          x={pad.left + plotWidth / 2}
          y={axisY + 30}
          textAnchor="middle"
          fontSize={10}
          fill="#555555"
          fontFamily="monospace"
        >
          Reward
        </text>

        {/* Rows */}
        {entries.map((e, i) => {
          const rowY = pad.top + i * rowHeight;
          const color = COLORS[i % COLORS.length];
          const zeroX = scaleX(0);
          const meanX = scaleX(e.overall_mean_reward);
          const worstX = scaleX(e.worst_case_mean_reward);

          const meanBarX = Math.min(zeroX, meanX);
          const meanBarW = Math.abs(meanX - zeroX);
          const worstBarX = Math.min(zeroX, worstX);
          const worstBarW = Math.abs(worstX - zeroX);

          return (
            <g key={e.policy_name}>
              {/* Policy label */}
              <text
                x={pad.left - 8}
                y={rowY + rowHeight / 2}
                textAnchor="end"
                fontSize={11}
                fill="#888888"
                fontFamily="monospace"
              >
                {e.policy_name}
              </text>

              {/* Mean reward bar */}
              <rect
                x={meanBarX}
                y={rowY + 8}
                width={meanBarW}
                height={10}
                fill={color}
                opacity={1}
                rx={2}
              />
              <text
                x={meanX + (e.overall_mean_reward >= 0 ? 4 : -4)}
                y={rowY + 16}
                textAnchor={e.overall_mean_reward >= 0 ? "start" : "end"}
                fontSize={9}
                fill="#666666"
                fontFamily="monospace"
              >
                {fmt(e.overall_mean_reward)}
              </text>

              {/* Worst-case bar */}
              <rect
                x={worstBarX}
                y={rowY + 22}
                width={worstBarW}
                height={10}
                fill={color}
                opacity={0.4}
                rx={2}
              />
              <text
                x={worstX + (e.worst_case_mean_reward >= 0 ? 4 : -4)}
                y={rowY + 30}
                textAnchor={e.worst_case_mean_reward >= 0 ? "start" : "end"}
                fontSize={9}
                fill="#666666"
                fontFamily="monospace"
              >
                {fmt(e.worst_case_mean_reward)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
