"use client";

interface BenchmarkRow {
  policy: string;
  mean_completion_ratio?: number;
  mean_return?: number;
  [key: string]: unknown;
}

interface Props {
  champion: { member_id: string; rating: number } | null;
  results: BenchmarkRow[];
}

const BAR_COLORS: Record<string, string> = {
  cooperative_champion: "#14b8a6",
  cooperative_ppo: "#3b82f6",
  random: "#6b7280",
  always_cooperate: "#22c55e",
  always_extract: "#ef4444",
  tit_for_tat: "#f59e0b",
};

function barColor(policy: string): string {
  return BAR_COLORS[policy] ?? "#14b8a6";
}

function displayName(policy: string): string {
  return policy.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function CooperativeChampionBenchmark({ champion, results }: Props) {
  if (!champion || champion.member_id == null) {
    return (
      <div style={{ color: "#555555", fontSize: 13, padding: "24px 0" }}>
        No champion available — run the pipeline first.
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div style={{ color: "#555555", fontSize: 13, padding: "24px 0" }}>
        No benchmark results yet.
      </div>
    );
  }

  const BAR_H = 28;
  const BAR_GAP = 8;
  const LABEL_W = 160;
  const VAL_W = 60;
  const CHART_W = 260;
  const maxVal = Math.max(
    ...results.map((r) =>
      Math.abs((r.mean_completion_ratio ?? r.mean_return ?? 0) as number)
    ),
    0.01,
  );

  return (
    <div>
      {/* Champion info */}
      <div style={{
        background: "#0d1f1f",
        border: "1px solid #14b8a6",
        borderRadius: 6,
        padding: "10px 14px",
        marginBottom: 16,
        display: "flex",
        alignItems: "center",
        gap: 12,
      }}>
        <div>
          <div style={{ fontSize: 11, color: "#888888" }}>Champion</div>
          <div style={{ fontSize: 13, fontFamily: "monospace", color: "#14b8a6" }}>
            {champion.member_id}
          </div>
        </div>
        <div style={{ marginLeft: "auto" }}>
          <div style={{ fontSize: 11, color: "#888888" }}>Elo Rating</div>
          <div style={{ fontSize: 16, fontWeight: 600, color: "#ededed" }}>
            {champion.rating.toFixed(1)}
          </div>
        </div>
      </div>

      {/* Horizontal bar chart */}
      <svg
        width={LABEL_W + CHART_W + VAL_W + 8}
        height={results.length * (BAR_H + BAR_GAP)}
        style={{ display: "block", overflow: "visible" }}
      >
        {results.map((r, i) => {
          const val = Math.abs(
            (r.mean_completion_ratio ?? r.mean_return ?? 0) as number,
          );
          const barW = (val / maxVal) * CHART_W;
          const y = i * (BAR_H + BAR_GAP);
          const color = barColor(r.policy);
          return (
            <g key={r.policy}>
              <text
                x={LABEL_W - 6}
                y={y + BAR_H / 2 + 4}
                textAnchor="end"
                fontSize={11}
                fill="#aaaaaa"
              >
                {displayName(r.policy)}
              </text>
              <rect
                x={LABEL_W}
                y={y + 4}
                width={barW}
                height={BAR_H - 8}
                fill={color}
                rx={3}
                opacity={0.85}
              />
              <text
                x={LABEL_W + barW + 6}
                y={y + BAR_H / 2 + 4}
                fontSize={11}
                fill={color}
              >
                {val.toFixed(3)}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Table */}
      <div style={{ marginTop: 20, overflowX: "auto" }}>
        <table style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 12,
          color: "#aaaaaa",
        }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #222222" }}>
              <th style={{ textAlign: "left", padding: "4px 8px" }}>Policy</th>
              <th style={{ textAlign: "right", padding: "4px 8px" }}>Completion Ratio</th>
              <th style={{ textAlign: "right", padding: "4px 8px" }}>Mean Return</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.policy} style={{ borderBottom: "1px solid #1a1a1a" }}>
                <td style={{ padding: "5px 8px", color: barColor(r.policy) }}>
                  {displayName(r.policy)}
                </td>
                <td style={{ textAlign: "right", padding: "5px 8px", fontFamily: "monospace" }}>
                  {r.mean_completion_ratio != null
                    ? r.mean_completion_ratio.toFixed(4)
                    : "—"}
                </td>
                <td style={{ textAlign: "right", padding: "5px 8px", fontFamily: "monospace" }}>
                  {r.mean_return != null
                    ? (r.mean_return as number).toFixed(4)
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
