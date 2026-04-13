"use client";

interface PolicyEntry {
  policy_name: string;
  mean_completion_ratio: number;
  worst_case_completion_ratio: number;
  robustness_score: number;
}

interface Props {
  entries: PolicyEntry[];
}

const BAR_COLORS: Record<string, string> = {
  cooperative_champion: "#14b8a6",
  cooperative_ppo: "#3b82f6",
};

function entryColor(name: string): string {
  return BAR_COLORS[name] ?? "#8b5cf6";
}

export default function CooperativeRobustScatter({ entries }: Props) {
  if (entries.length === 0) {
    return (
      <div style={{ color: "#555555", fontSize: 13, padding: "24px 0" }}>
        No robustness data available.
      </div>
    );
  }

  const BAR_H = 30;
  const BAR_GAP = 8;
  const LABEL_W = 160;
  const CHART_W = 280;

  // Sort by robustness score descending
  const sorted = [...entries].sort(
    (a, b) => b.robustness_score - a.robustness_score
  );

  const maxVal = Math.max(
    ...sorted.map((e) =>
      Math.max(e.mean_completion_ratio, e.worst_case_completion_ratio)
    ),
    0.01,
  );

  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ fontSize: 11, color: "#666666", marginBottom: 8 }}>
        Horizontal bars: mean (filled) vs worst-case (outline) completion ratio
      </div>
      <svg
        width={LABEL_W + CHART_W + 80}
        height={sorted.length * (BAR_H + BAR_GAP) + 20}
        style={{ display: "block" }}
      >
        {sorted.map((e, i) => {
          const y = i * (BAR_H + BAR_GAP);
          const color = entryColor(e.policy_name);
          const meanW = (e.mean_completion_ratio / maxVal) * CHART_W;
          const worstW = (e.worst_case_completion_ratio / maxVal) * CHART_W;
          return (
            <g key={e.policy_name}>
              {/* Policy name */}
              <text
                x={LABEL_W - 6}
                y={y + BAR_H / 2 + 4}
                textAnchor="end"
                fontSize={11}
                fill="#aaaaaa"
              >
                {e.policy_name.replace(/_/g, " ")}
              </text>

              {/* Mean bar (filled) */}
              <rect
                x={LABEL_W}
                y={y + 6}
                width={meanW}
                height={BAR_H - 12}
                fill={color}
                rx={2}
                opacity={0.85}
              />

              {/* Worst-case bar (outline) */}
              <rect
                x={LABEL_W}
                y={y + 2}
                width={worstW}
                height={BAR_H - 4}
                fill="none"
                stroke={color}
                strokeWidth={1}
                rx={2}
                opacity={0.5}
              />

              {/* Values */}
              <text
                x={LABEL_W + Math.max(meanW, worstW) + 6}
                y={y + BAR_H / 2 + 4}
                fontSize={10}
                fill="#666666"
              >
                {e.mean_completion_ratio.toFixed(3)} / {e.worst_case_completion_ratio.toFixed(3)}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Summary table */}
      <table style={{
        marginTop: 16,
        width: "100%",
        borderCollapse: "collapse",
        fontSize: 12,
        color: "#aaaaaa",
      }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #222222" }}>
            <th style={{ textAlign: "left", padding: "4px 8px" }}>Policy</th>
            <th style={{ textAlign: "right", padding: "4px 8px" }}>Mean CR</th>
            <th style={{ textAlign: "right", padding: "4px 8px" }}>Worst CR</th>
            <th style={{ textAlign: "right", padding: "4px 8px" }}>Robustness</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((e) => (
            <tr key={e.policy_name} style={{ borderBottom: "1px solid #1a1a1a" }}>
              <td style={{ padding: "5px 8px", color: entryColor(e.policy_name) }}>
                {e.policy_name.replace(/_/g, " ")}
              </td>
              <td style={{ textAlign: "right", padding: "5px 8px", fontFamily: "monospace" }}>
                {e.mean_completion_ratio.toFixed(4)}
              </td>
              <td style={{ textAlign: "right", padding: "5px 8px", fontFamily: "monospace" }}>
                {e.worst_case_completion_ratio.toFixed(4)}
              </td>
              <td style={{ textAlign: "right", padding: "5px 8px", fontFamily: "monospace", color: "#14b8a6" }}>
                {e.robustness_score.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
