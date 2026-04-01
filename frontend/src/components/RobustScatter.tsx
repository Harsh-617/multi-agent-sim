"use client";

/**
 * Pure-SVG scatter plot: x = overall_mean_reward, y = worst_case_mean_reward.
 * Each point is a policy, labeled.
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
  "#14b8a6", // teal
  "#f97316", // orange
  "#8b5cf6", // purple
  "#22c55e", // green
  "#ef4444", // red
  "#f59e0b", // amber
  "#3b82f6", // blue
  "#ec4899", // pink
];

export default function RobustScatter({ perPolicyRobustness }: Props) {
  const entries = Object.values(perPolicyRobustness).filter(
    (p) => p.n_sweeps_evaluated > 0
  );
  if (entries.length === 0)
    return <p style={{ color: "#666666" }}>No data.</p>;

  const pad = { top: 30, right: 90, bottom: 50, left: 70 };
  const w = 560;
  const h = 360;
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const xs = entries.map((e) => e.overall_mean_reward);
  const ys = entries.map((e) => e.worst_case_mean_reward);
  const xMin = Math.min(...xs) * 0.9;
  const xMax = Math.max(...xs) * 1.1 || 1;
  const yMin = Math.min(...ys) * 0.9;
  const yMax = Math.max(...ys) * 1.1 || 1;

  const scaleX = (v: number) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const scaleY = (v: number) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));

  // Tick values
  const xTicks = Array.from({ length: 4 }, (_, i) => xMin + (i * (xMax - xMin)) / 3);
  const yTicks = Array.from({ length: 4 }, (_, i) => yMin + (i * (yMax - yMin)) / 3);

  // Grid positions (4 lines each)
  const hGridLines = Array.from({ length: 4 }, (_, i) => pad.top + (i * plotH) / 3);
  const vGridLines = Array.from({ length: 4 }, (_, i) => pad.left + (i * plotW) / 3);

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={w} height={h}>
        {/* Background */}
        <rect width={w} height={h} fill="#0d0d0d" />

        {/* Grid lines */}
        {hGridLines.map((y, i) => (
          <line key={`hg${i}`} x1={pad.left} y1={y} x2={pad.left + plotW} y2={y} stroke="#1e1e1e" strokeWidth={1} />
        ))}
        {vGridLines.map((x, i) => (
          <line key={`vg${i}`} x1={x} y1={pad.top} x2={x} y2={pad.top + plotH} stroke="#1e1e1e" strokeWidth={1} />
        ))}

        {/* Axes */}
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + plotH} stroke="#2a2a2a" strokeWidth={1} />
        <line x1={pad.left} y1={pad.top + plotH} x2={pad.left + plotW} y2={pad.top + plotH} stroke="#2a2a2a" strokeWidth={1} />

        {/* X axis ticks */}
        {xTicks.map((v, i) => (
          <text key={`xt${i}`} x={scaleX(v)} y={pad.top + plotH + 16} textAnchor="middle" fontSize={9} fill="#444444" fontFamily="monospace">
            {fmt(v)}
          </text>
        ))}

        {/* Y axis ticks */}
        {yTicks.map((v, i) => (
          <text key={`yt${i}`} x={pad.left - 8} y={scaleY(v) + 3} textAnchor="end" fontSize={9} fill="#444444" fontFamily="monospace">
            {fmt(v)}
          </text>
        ))}

        {/* Axis labels */}
        <text x={pad.left + plotW / 2} y={h - 8} textAnchor="middle" fontSize={11} fill="#666666" fontFamily="monospace">
          Overall Mean Reward
        </text>
        <text
          x={14}
          y={pad.top + plotH / 2}
          textAnchor="middle"
          fontSize={11}
          fill="#666666"
          fontFamily="monospace"
          transform={`rotate(-90,14,${pad.top + plotH / 2})`}
        >
          Worst-Case Mean Reward
        </text>

        {/* Diagonal reference (y = x) */}
        <line
          x1={scaleX(Math.max(xMin, yMin))}
          y1={scaleY(Math.max(xMin, yMin))}
          x2={scaleX(Math.min(xMax, yMax))}
          y2={scaleY(Math.min(xMax, yMax))}
          stroke="#2a2a2a"
          strokeDasharray="3,3"
        />

        {/* Points */}
        {entries.map((e, i) => {
          const cx = scaleX(e.overall_mean_reward);
          const cy = scaleY(e.worst_case_mean_reward);
          const color = COLORS[i % COLORS.length];

          // Smart label placement
          const inRightZone = cx > pad.left + plotW * 0.7;
          const inTopZone = cy < pad.top + plotH * 0.2;

          const labelX = inRightZone ? cx - 10 : cx + 10;
          const labelY = inTopZone ? cy + 16 : cy;
          const anchor = inRightZone ? "end" : "start";
          const labelW = e.policy_name.length * 6;
          const bgX = inRightZone ? labelX - labelW - 2 : labelX - 2;

          return (
            <g key={e.policy_name}>
              {/* Hit area */}
              <circle cx={cx} cy={cy} r={12} fill="transparent" />
              {/* Glow ring */}
              <circle cx={cx} cy={cy} r={7} fill="none" stroke={color} strokeWidth={1} opacity={0.3} />
              {/* Point */}
              <circle cx={cx} cy={cy} r={5} fill={color} opacity={0.9} />
              {/* Label background */}
              <rect x={bgX} y={labelY - 10} width={labelW + 4} height={14} fill="#0d0d0d" opacity={0.85} />
              {/* Label text */}
              <text x={labelX} y={labelY} fontSize={10} fill={color} fontFamily="monospace" textAnchor={anchor}>
                {e.policy_name}
              </text>
            </g>
          );
        })}

      </svg>
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "8px 16px",
        marginTop: 12,
        paddingLeft: pad.left,
      }}>
        {entries.map((e, i) => {
          const color = COLORS[i % COLORS.length];
          return (
            <div key={e.policy_name} style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}>
              <div style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: color,
                flexShrink: 0,
              }} />
              <span style={{
                fontSize: 11,
                color: "#888888",
                fontFamily: "monospace",
              }}>
                {e.policy_name}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
