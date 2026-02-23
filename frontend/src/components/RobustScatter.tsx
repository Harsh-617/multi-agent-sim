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
  "#2563eb", "#dc2626", "#16a34a", "#9333ea",
  "#ea580c", "#0891b2", "#ca8a04", "#be185d",
];

export default function RobustScatter({ perPolicyRobustness }: Props) {
  const entries = Object.values(perPolicyRobustness).filter(
    (p) => p.n_sweeps_evaluated > 0
  );
  if (entries.length === 0) return <p className="text-gray-500">No data.</p>;

  const pad = { top: 30, right: 30, bottom: 50, left: 70 };
  const w = 500;
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

  return (
    <div className="overflow-x-auto">
      <svg width={w} height={h} className="font-mono text-xs">
        {/* Axes */}
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + plotH} stroke="#ccc" />
        <line x1={pad.left} y1={pad.top + plotH} x2={pad.left + plotW} y2={pad.top + plotH} stroke="#ccc" />

        {/* Axis labels */}
        <text x={pad.left + plotW / 2} y={h - 8} textAnchor="middle" fontSize={11}>
          Overall Mean Reward
        </text>
        <text
          x={14}
          y={pad.top + plotH / 2}
          textAnchor="middle"
          fontSize={11}
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
          stroke="#e5e7eb"
          strokeDasharray="4,4"
        />

        {/* Points */}
        {entries.map((e, i) => {
          const cx = scaleX(e.overall_mean_reward);
          const cy = scaleY(e.worst_case_mean_reward);
          const color = COLORS[i % COLORS.length];
          return (
            <g key={e.policy_name}>
              <circle cx={cx} cy={cy} r={6} fill={color} opacity={0.85} />
              <text x={cx + 9} y={cy + 4} fontSize={10} fill={color}>
                {e.policy_name}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
