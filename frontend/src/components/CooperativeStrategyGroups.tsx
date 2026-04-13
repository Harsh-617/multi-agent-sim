"use client";

import { StrategyResponse } from "@/lib/api";

interface Props {
  data: StrategyResponse | null;
}

const LABEL_COLORS: Record<string, string> = {
  "Dedicated Specialist": "#14b8a6",
  "Adaptive Generalist": "#3b82f6",
  "Free Rider": "#ef4444",
  "Overcontributor": "#f59e0b",
  "Opportunist": "#8b5cf6",
  Developing: "#6b7280",
};

function labelColor(label: string): string {
  return LABEL_COLORS[label] ?? "#6b7280";
}

const LABEL_DESCRIPTIONS: Record<string, string> = {
  "Dedicated Specialist":
    "Consistently applies effort to one task type with high role stability and specialization.",
  "Adaptive Generalist":
    "Spreads effort across task types; efficient but lacks deep specialization.",
  "Free Rider":
    "High idle rate — contributes little to group output; benefits from others' work.",
  "Overcontributor":
    "Maximizes effort utilization; works hard but may not specialize optimally.",
  "Opportunist":
    "Selectively contributes based on context; moderate on most dimensions.",
};

export default function CooperativeStrategyGroups({ data }: Props) {
  if (!data || Object.keys(data.clusters).length === 0) {
    return (
      <div style={{ color: "#555555", fontSize: 13, padding: "24px 0" }}>
        No strategy cluster data available.
      </div>
    );
  }

  const { clusters, labels, features } = data;

  // Group agents by cluster
  const byCluster: Record<string, string[]> = {};
  Object.entries(clusters).forEach(([agentId, clusterId]) => {
    const cid = String(clusterId);
    if (!byCluster[cid]) byCluster[cid] = [];
    byCluster[cid].push(agentId);
  });

  return (
    <div>
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
        gap: 12,
        marginBottom: 24,
      }}>
        {Object.entries(byCluster).map(([cid, agents]) => {
          const label = labels[cid] ?? "Unknown";
          const color = labelColor(label);
          const desc = LABEL_DESCRIPTIONS[label] ?? "";
          return (
            <div
              key={cid}
              style={{
                background: "#111111",
                border: "1px solid #1e1e1e",
                borderLeft: `3px solid ${color}`,
                borderRadius: 6,
                padding: "12px 14px",
              }}
            >
              <div style={{
                fontSize: 13,
                fontWeight: 600,
                color,
                marginBottom: 4,
              }}>
                {label}
              </div>
              <div style={{
                fontSize: 11,
                color: "#666666",
                marginBottom: 8,
                lineHeight: 1.4,
              }}>
                {desc}
              </div>
              <div style={{
                fontSize: 11,
                color: "#555555",
                marginBottom: 6,
              }}>
                {agents.length} agent{agents.length !== 1 ? "s" : ""}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                {agents.map((aid) => {
                  const feat = features[aid] as Record<string, unknown> | undefined;
                  return (
                    <div key={aid} style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}>
                      <span style={{
                        fontSize: 10,
                        fontFamily: "monospace",
                        color: "#888888",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        maxWidth: 120,
                      }}>
                        {aid}
                      </span>
                      {feat && (
                        <span style={{ fontSize: 10, color: "#555555" }}>
                          eu:{typeof feat.effort_utilization === "number"
                            ? (feat.effort_utilization as number).toFixed(2)
                            : "—"}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Feature table */}
      <div style={{ overflowX: "auto" }}>
        <table style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 11,
          color: "#aaaaaa",
        }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #222222" }}>
              <th style={{ textAlign: "left", padding: "4px 8px" }}>Agent / Policy</th>
              <th style={{ textAlign: "right", padding: "4px 8px" }}>Effort Util</th>
              <th style={{ textAlign: "right", padding: "4px 8px" }}>Idle Rate</th>
              <th style={{ textAlign: "right", padding: "4px 8px" }}>Dom Frac</th>
              <th style={{ textAlign: "right", padding: "4px 8px" }}>Spec Score</th>
              <th style={{ textAlign: "right", padding: "4px 8px" }}>Role Stab</th>
              <th style={{ textAlign: "left", padding: "4px 8px" }}>Label</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(clusters).sort().map(([agentId, cid]) => {
              const label = labels[String(cid)] ?? "Unknown";
              const color = labelColor(label);
              const feat = features[agentId] as Record<string, unknown> | undefined;
              const fmt = (v: unknown) =>
                typeof v === "number" ? (v as number).toFixed(3) : "—";
              return (
                <tr key={agentId} style={{ borderBottom: "1px solid #1a1a1a" }}>
                  <td style={{ padding: "5px 8px", fontFamily: "monospace", fontSize: 10 }}>
                    {agentId}
                  </td>
                  <td style={{ textAlign: "right", padding: "5px 8px" }}>
                    {fmt(feat?.effort_utilization)}
                  </td>
                  <td style={{ textAlign: "right", padding: "5px 8px" }}>
                    {fmt(feat?.idle_rate)}
                  </td>
                  <td style={{ textAlign: "right", padding: "5px 8px" }}>
                    {fmt(feat?.dominant_type_fraction)}
                  </td>
                  <td style={{ textAlign: "right", padding: "5px 8px" }}>
                    {fmt(feat?.final_specialization_score)}
                  </td>
                  <td style={{ textAlign: "right", padding: "5px 8px" }}>
                    {fmt(feat?.role_stability)}
                  </td>
                  <td style={{ padding: "5px 8px", color }}>{label}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
