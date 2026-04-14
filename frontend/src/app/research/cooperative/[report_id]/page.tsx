"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  getCooperativeReport,
  getCooperativeReportRobustness,
  getCooperativeReportStrategies,
  CooperativeRobustnessHeatmapResponse,
  StrategyResponse,
} from "@/lib/api";
import CooperativeChampionRobustness from "@/components/CooperativeChampionRobustness";
import CooperativeStrategyGroups from "@/components/CooperativeStrategyGroups";

/* eslint-disable @typescript-eslint/no-explicit-any */

export default function CooperativeReportDetailPage() {
  const params = useParams<{ report_id: string }>();
  const reportId = params.report_id;

  const [data, setData] = useState<Record<string, any> | null>(null);
  const [robustness, setRobustness] = useState<CooperativeRobustnessHeatmapResponse | null>(null);
  const [strategies, setStrategies] = useState<StrategyResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!reportId) return;

    getCooperativeReport(reportId)
      .then((report) => {
        setData(report as Record<string, any>);
        const isRobust = (report as any).kind === "cooperative_robust";
        return Promise.all([
          isRobust
            ? getCooperativeReportRobustness(reportId).catch(() => null)
            : Promise.resolve(null),
          getCooperativeReportStrategies(reportId).catch(() => null),
        ]);
      })
      .then(([rob, strats]) => {
        setRobustness(rob);
        setStrategies(strats);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [reportId]);

  if (loading) {
    return (
      <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, background: "var(--bg-base)", color: "var(--text-primary)" }}>
        <p>Loading…</p>
      </main>
    );
  }
  if (error) {
    return (
      <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, background: "var(--bg-base)" }}>
        <p style={{ color: "#f87171" }}>Error: {error}</p>
      </main>
    );
  }
  if (!data) {
    return (
      <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, background: "var(--bg-base)", color: "var(--text-primary)" }}>
        <p>No data.</p>
      </main>
    );
  }

  const kind = data.kind as string;
  const isRobust = kind === "cooperative_robust";

  return (
    <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, background: "var(--bg-base)" }}>
      {/* Breadcrumb */}
      <div style={{ marginBottom: 16, display: "flex", gap: 8, fontSize: 13, color: "var(--text-secondary)" }}>
        <Link href="/research" style={{ color: "var(--text-secondary)", textDecoration: "none" }}>
          ← Research
        </Link>
        <span>/</span>
        <Link href="/research/cooperative" style={{ color: "var(--text-secondary)", textDecoration: "none" }}>
          Cooperative
        </Link>
      </div>

      {/* Title */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <h1 style={{ fontSize: 20, fontWeight: 600, fontFamily: "var(--font-mono)", color: "var(--text-primary)", margin: 0 }}>
          {data.report_id as string}
        </h1>
      </div>

      {/* Meta card */}
      <div style={{ background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: 8, padding: 20, marginBottom: 24 }}>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "0 0 4px" }}>
          <span style={{
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 11,
            fontWeight: 500,
            marginRight: 8,
            background: isRobust ? "rgba(20,184,166,0.15)" : "rgba(59,130,246,0.15)",
            color: isRobust ? "#14b8a6" : "#60a5fa",
          }}>
            {kind}
          </span>
          Config hash: <code style={{ color: "var(--text-primary)" }}>{data.config_hash as string}</code>
        </p>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "4px 0 0" }}>
          Generated: {data.timestamp as string}
        </p>
      </div>

      {/* Body */}
      <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
        {isRobust ? (
          <RobustBody data={data} robustness={robustness} />
        ) : (
          <EvalBody data={data} />
        )}

        {strategies && Object.keys(strategies.clusters).length > 0 && (
          <section>
            <h2 style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)", marginBottom: 12 }}>
              Strategy Groups
            </h2>
            <CooperativeStrategyGroups data={strategies} />
          </section>
        )}
      </div>
    </main>
  );
}

/* ------------------------------------------------------------------ */
/* Robustness report body                                               */
/* ------------------------------------------------------------------ */

function RobustBody({
  data,
  robustness,
}: {
  data: Record<string, any>;
  robustness: CooperativeRobustnessHeatmapResponse | null;
}) {
  // Prefer robustness endpoint data; fall back to report data if endpoint failed
  const heatmapData: CooperativeRobustnessHeatmapResponse | null =
    robustness ?? buildHeatmapFromReport(data);

  return (
    <>
      <section>
        <h2 style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)", marginBottom: 12 }}>
          Robustness Heatmap
        </h2>
        <CooperativeChampionRobustness data={heatmapData} />
      </section>

      {data.per_sweep_results && (
        <section>
          <h2 style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)", marginBottom: 12 }}>
            Per-Sweep Results
          </h2>
          <SweepTable perSweep={data.per_sweep_results as Record<string, any>} />
        </section>
      )}
    </>
  );
}

function buildHeatmapFromReport(data: Record<string, any>): CooperativeRobustnessHeatmapResponse | null {
  const perSweep = data.per_sweep_results as Record<string, any> | undefined;
  const perPolicy = data.per_policy_robustness as Record<string, any> | undefined;
  if (!perSweep || !perPolicy) return null;

  const sweepNames = Object.keys(perSweep).sort();
  const policyNames = [...new Set(Object.values(perSweep).map((s: any) => s.policy as string))];

  const heatmap: Record<string, Record<string, number | null>> = {};
  for (const policy of policyNames) {
    heatmap[policy] = {};
    for (const sn of sweepNames) {
      const entry = perSweep[sn];
      heatmap[policy][sn] = entry?.policy === policy ? entry.mean_completion_ratio ?? null : null;
    }
  }

  return { sweep_names: sweepNames, policies: policyNames, heatmap, per_policy_robustness: perPolicy };
}

function SweepTable({ perSweep }: { perSweep: Record<string, any> }) {
  const rows = Object.values(perSweep) as any[];
  if (rows.length === 0) return <p style={{ color: "var(--text-tertiary)", fontSize: 13 }}>No sweep data.</p>;

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: 8 }}>
        <thead>
          <tr>
            {["Sweep", "Description", "Mean CR", "Worst CR", "Episodes"].map((h) => (
              <th key={h} style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", fontSize: 11, textTransform: "uppercase", padding: "8px 12px", textAlign: "left", fontWeight: 500 }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r: any) => (
            <tr key={r.sweep_name}>
              <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-primary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
                {r.sweep_name}
              </td>
              <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-secondary)", fontSize: 12 }}>
                {r.description ?? "—"}
              </td>
              <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-primary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
                {r.mean_completion_ratio != null ? (r.mean_completion_ratio as number).toFixed(4) : "—"}
              </td>
              <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-primary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
                {r.worst_case_completion_ratio != null ? (r.worst_case_completion_ratio as number).toFixed(4) : "—"}
              </td>
              <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-secondary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
                {r.n_episodes ?? "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Eval report body                                                     */
/* ------------------------------------------------------------------ */

function EvalBody({ data }: { data: Record<string, any> }) {
  const summary = (data.summary ?? {}) as Record<string, any>;
  const perSeed = (data.per_seed ?? []) as any[];

  const metrics: Array<{ label: string; key: string; pct?: boolean }> = [
    { label: "Mean Completion Ratio", key: "mean_completion_ratio", pct: true },
    { label: "Worst-Case CR", key: "worst_case_completion_ratio", pct: true },
    { label: "Mean Group Efficiency", key: "mean_group_efficiency_ratio", pct: true },
    { label: "Mean Effort Utilization", key: "mean_effort_utilization", pct: true },
    { label: "Mean System Stress", key: "mean_system_stress" },
    { label: "Mean Return", key: "mean_return" },
  ];

  return (
    <>
      {/* Summary metrics */}
      <section>
        <h2 style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)", marginBottom: 12 }}>
          Summary
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12 }}>
          {metrics.map(({ label, key, pct }) => {
            const v = summary[key] as number | undefined;
            return (
              <div key={key} style={{ background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: 8, padding: "14px 16px" }}>
                <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                  {v != null ? (pct ? `${(v * 100).toFixed(2)}%` : v.toFixed(4)) : "—"}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Per-seed breakdown */}
      {perSeed.length > 0 && (
        <section>
          <h2 style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)", marginBottom: 12 }}>
            Per-Seed Results
          </h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: 8 }}>
              <thead>
                <tr>
                  {["Seed", "Mean CR", "Group Efficiency", "Effort Util", "System Stress", "Mean Return"].map((h) => (
                    <th key={h} style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", fontSize: 11, textTransform: "uppercase", padding: "8px 12px", textAlign: "left", fontWeight: 500 }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {perSeed.map((row: any) => (
                  <tr key={row.seed}>
                    <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-primary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>{row.seed}</td>
                    <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-primary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
                      {row.mean_completion_ratio != null ? `${(row.mean_completion_ratio * 100).toFixed(2)}%` : "—"}
                    </td>
                    <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-primary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
                      {row.mean_group_efficiency_ratio != null ? `${(row.mean_group_efficiency_ratio * 100).toFixed(2)}%` : "—"}
                    </td>
                    <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-primary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
                      {row.mean_effort_utilization != null ? `${(row.mean_effort_utilization * 100).toFixed(2)}%` : "—"}
                    </td>
                    <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-primary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
                      {row.mean_system_stress != null ? (row.mean_system_stress as number).toFixed(4) : "—"}
                    </td>
                    <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--bg-border)", color: "var(--text-primary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
                      {row.mean_return != null ? (row.mean_return as number).toFixed(4) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}
