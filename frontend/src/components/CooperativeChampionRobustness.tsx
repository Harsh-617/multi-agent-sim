"use client";

import { CooperativeRobustnessHeatmapResponse } from "@/lib/api";

interface Props {
  data: CooperativeRobustnessHeatmapResponse | null;
}

function cellColor(val: number | null | undefined): string {
  if (val == null) return "#1a1a1a";
  // Scale: 0 = dark red, 0.5 = teal-dark, 1 = bright teal
  const t = Math.max(0, Math.min(1, val));
  if (t < 0.3) return `hsl(0, 60%, ${10 + t * 30}%)`;
  if (t < 0.6) return `hsl(160, 40%, ${15 + (t - 0.3) * 30}%)`;
  return `hsl(175, 60%, ${20 + (t - 0.6) * 40}%)`;
}

function labelShort(name: string): string {
  return name.length > 16 ? name.slice(0, 15) + "…" : name;
}

export default function CooperativeChampionRobustness({ data }: Props) {
  if (!data) {
    return (
      <div style={{ color: "#555555", fontSize: 13, padding: "24px 0" }}>
        No robustness data — run robustness sweep first.
      </div>
    );
  }

  const { sweep_names, policies, heatmap, per_policy_robustness } = data;

  if (sweep_names.length === 0) {
    return (
      <div style={{ color: "#555555", fontSize: 13, padding: "24px 0" }}>
        No sweep results found.
      </div>
    );
  }

  const CELL_W = 40;
  const CELL_H = 22;
  const LABEL_W = 100;
  const HEADER_H = 0;

  return (
    <div>
      {/* Per-policy summary */}
      {Object.entries(per_policy_robustness || {}).map(([name, pr]: [string, unknown]) => {
        const prObj = pr as Record<string, unknown>;
        return (
          <div key={name} style={{
            background: "#0d1f1f",
            border: "1px solid #14b8a6",
            borderRadius: 6,
            padding: "10px 14px",
            marginBottom: 16,
            display: "flex",
            gap: 24,
            flexWrap: "wrap",
          }}>
            <div>
              <div style={{ fontSize: 11, color: "#888888" }}>Policy</div>
              <div style={{ fontSize: 13, color: "#14b8a6", fontFamily: "monospace" }}>{name}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#888888" }}>Mean CR</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#ededed" }}>
                {typeof prObj.mean_completion_ratio === "number"
                  ? prObj.mean_completion_ratio.toFixed(4)
                  : "—"}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#888888" }}>Worst CR</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#ededed" }}>
                {typeof prObj.worst_case_completion_ratio === "number"
                  ? prObj.worst_case_completion_ratio.toFixed(4)
                  : "—"}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#888888" }}>Robustness Score</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#14b8a6" }}>
                {typeof prObj.robustness_score === "number"
                  ? prObj.robustness_score.toFixed(4)
                  : "—"}
              </div>
            </div>
          </div>
        );
      })}

      {/* Heatmap */}
      <div style={{ overflowX: "auto" }}>
        {/* Column headers as HTML above SVG — same approach as RobustHeatmap */}
        <div style={{
          display: "flex",
          paddingLeft: LABEL_W,
          marginBottom: 2,
          minWidth: LABEL_W + sweep_names.length * CELL_W + 20,
        }}>
          {sweep_names.map((sn) => (
            <div
              key={sn}
              style={{
                width: CELL_W,
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
                {labelShort(sn)}
              </div>
            </div>
          ))}
        </div>

        <svg
          width={LABEL_W + sweep_names.length * CELL_W + 20}
          height={HEADER_H + policies.length * CELL_H + 10}
          style={{ display: "block" }}
        >
          {/* Column headers moved to HTML above */}

          {/* Row labels + cells */}
          {policies.map((policy, i) => (
            <g key={policy}>
              <text
                x={LABEL_W - 4}
                y={HEADER_H + i * CELL_H + CELL_H / 2 + 4}
                fontSize={9}
                fill="#aaaaaa"
                textAnchor="end"
              >
                {labelShort(policy)}
              </text>
              {sweep_names.map((sn, j) => {
                const val = heatmap[policy]?.[sn];
                return (
                  <g key={sn}>
                    <rect
                      x={LABEL_W + j * CELL_W}
                      y={HEADER_H + i * CELL_H}
                      width={CELL_W - 1}
                      height={CELL_H - 1}
                      fill={cellColor(val)}
                      rx={2}
                    />
                    {val != null && (
                      <text
                        x={LABEL_W + j * CELL_W + CELL_W / 2}
                        y={HEADER_H + i * CELL_H + CELL_H / 2 + 3}
                        fontSize={7}
                        fill="#ffffff"
                        textAnchor="middle"
                        opacity={0.8}
                      >
                        {val.toFixed(2)}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>

      {/* Color scale legend */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        marginTop: 8,
        fontSize: 10,
        color: "#666666",
      }}>
        <span>0.0</span>
        <div style={{
          width: 120,
          height: 8,
          borderRadius: 4,
          background: "linear-gradient(to right, hsl(0,60%,10%), hsl(175,60%,40%))",
        }} />
        <span>1.0 completion ratio</span>
      </div>
    </div>
  );
}
