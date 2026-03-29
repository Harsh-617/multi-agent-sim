"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  getReport,
  getReportStrategies,
  getCompetitiveReport,
  getCompetitiveReportStrategies,
} from "@/lib/api";
import type { StrategyResponse } from "@/lib/api";
import RobustHeatmap from "@/components/RobustHeatmap";
import RobustScatter from "@/components/RobustScatter";
import RobustSummaryTable from "@/components/RobustSummaryTable";

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Unified research report detail page.
 * - If report_id starts with "competitive_": fetch from competitive endpoints
 *   and render the Competitive report detail view.
 * - Otherwise: fetch from mixed endpoints and render the Mixed report detail view.
 */
export default function ResearchReportDetailPage() {
  const params = useParams<{ report_id: string }>();
  const reportId = params.report_id;
  const isCompetitive = reportId?.startsWith("competitive_");

  const [data, setData] = useState<Record<string, any> | null>(null);
  const [strategies, setStrategies] = useState<StrategyResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!reportId) return;

    if (isCompetitive) {
      Promise.all([
        getCompetitiveReport(reportId),
        getCompetitiveReportStrategies(reportId).catch(() => null),
      ])
        .then(([reportData, strats]) => {
          setData(reportData as Record<string, any>);
          setStrategies(strats);
        })
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    } else {
      Promise.all([
        getReport(reportId),
        getReportStrategies(reportId).catch(() => null),
      ])
        .then(([reportData, strats]) => {
          setData(reportData);
          setStrategies(strats);
        })
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    }
  }, [reportId, isCompetitive]);

  if (loading) return <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, background: "var(--bg-base)", color: "var(--text-primary)" }}><p>Loading...</p></main>;
  if (error) return <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, background: "var(--bg-base)" }}><p style={{ color: "#f87171" }}>Error: {error}</p></main>;
  if (!data) return <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, background: "var(--bg-base)", color: "var(--text-primary)" }}><p>No data.</p></main>;

  if (isCompetitive) {
    return <CompetitiveReportView data={data} strategies={strategies} />;
  }
  return <MixedReportView data={data} strategies={strategies} />;
}

/* ------------------------------------------------------------------ */
/* Mixed report detail (copied from reports/[report_id]/page.tsx)      */
/* ------------------------------------------------------------------ */

function BackToReportsLink() {
  const [hover, setHover] = useState(false);
  return (
    <Link
      href="/research"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        color: hover ? "var(--text-primary)" : "var(--text-secondary)",
        fontSize: 13,
        textDecoration: "none",
      }}
    >
      &larr; Back to Reports
    </Link>
  );
}

const sectionHeadingStyle: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 500,
  color: "var(--text-primary)",
  marginBottom: 12,
};

const reportTableStyle: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  background: "var(--bg-surface)",
  border: "1px solid var(--bg-border)",
  borderRadius: 8,
};

const reportThStyle: React.CSSProperties = {
  background: "var(--bg-elevated)",
  color: "var(--text-secondary)",
  fontSize: 11,
  textTransform: "uppercase",
  padding: "8px 12px",
  textAlign: "left",
  fontWeight: 500,
};

const reportTdStyle: React.CSSProperties = {
  padding: "8px 12px",
  borderBottom: "1px solid var(--bg-border)",
  color: "var(--text-primary)",
  fontSize: 13,
};

function MixedReportView({ data, strategies }: { data: Record<string, any>; strategies: StrategyResponse | null }) {
  const kind = data.kind as string;

  return (
    <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, background: "var(--bg-base)" }}>
      <div style={{ marginBottom: 16 }}>
        <BackToReportsLink />
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <h1 style={{ fontSize: 20, fontWeight: 600, fontFamily: "var(--font-mono)", color: "var(--text-primary)", margin: 0 }}>{data.report_id as string}</h1>
      </div>

      <div style={{ background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: 8, padding: 20, marginBottom: 24 }}>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "0 0 4px" }}>
          <span style={{
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 11,
            fontWeight: 500,
            marginRight: 8,
            background: kind === "robust" ? "rgba(168,85,247,0.15)" : "rgba(59,130,246,0.15)",
            color: kind === "robust" ? "#c084fc" : "#60a5fa",
          }}>
            {kind}
          </span>
          Config hash: <code style={{ color: "var(--text-primary)" }}>{data.config_hash as string}</code>
        </p>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "4px 0 0" }}>Generated: {data.timestamp as string}</p>
      </div>

      {kind === "robust" ? (
        <RobustReportView data={data} />
      ) : (
        <EvalReportView data={data} />
      )}

      {strategies && <MixedStrategyGroupsSection strategies={strategies} />}
    </main>
  );
}

function RobustReportView({ data }: { data: Record<string, any> }) {
  const perSweep = (data.per_sweep_results ?? {}) as Record<string, Record<string, any>>;
  const perPolicy = (data.per_policy_robustness ?? {}) as Record<string, any>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
      <section>
        <h2 style={sectionHeadingStyle}>Reward Heatmap (Policy x Sweep)</h2>
        <RobustHeatmap perSweepResults={perSweep} />
      </section>

      <section>
        <h2 style={sectionHeadingStyle}>Mean vs Worst-Case Reward</h2>
        <RobustScatter perPolicyRobustness={perPolicy} />
      </section>

      <section>
        <h2 style={sectionHeadingStyle}>Robustness Summary</h2>
        <RobustSummaryTable
          perPolicyRobustness={perPolicy}
          perSweepResults={perSweep}
        />
      </section>
    </div>
  );
}

function fmt(v: number | null | undefined, pct = false): string {
  if (v == null) return "—";
  return pct ? `${(v * 100).toFixed(2)}%` : v.toFixed(2);
}

function MixedStrategyGroupsSection({ strategies }: { strategies: StrategyResponse }) {
  const { features, clusters, labels, summaries } = strategies;
  const policies = Object.keys(clusters);

  // Group policies by cluster id
  const byCluster: Record<number, string[]> = {};
  for (const policy of policies) {
    const cid = clusters[policy];
    (byCluster[cid] ??= []).push(policy);
  }
  const clusterIds = Object.keys(byCluster).map(Number).sort((a, b) => a - b);

  // Check if optional columns have any data
  const hasWorstCase = policies.some((p) => features[p]?.worst_case_return != null);
  const hasRobustness = policies.some((p) => features[p]?.robustness_score != null);

  return (
    <div style={{ marginTop: 40, display: "flex", flexDirection: "column", gap: 24 }}>
      <h2 style={sectionHeadingStyle}>Strategy Groups</h2>

      {/* Cluster cards */}
      <div style={{ display: "grid", gap: 16, gridTemplateColumns: "repeat(3, 1fr)" }}>
        {clusterIds.map((cid) => (
          <div key={cid} style={{ background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: 8, padding: 16 }}>
            <h3 style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, color: "var(--text-primary)" }}>
              Cluster {cid}: {labels[String(cid)] ?? `Group ${cid}`}
            </h3>
            <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 8 }}>{summaries[String(cid)] ?? ""}</p>
            <ul style={{ fontSize: 12, color: "var(--text-primary)", listStyle: "disc", paddingLeft: 16 }}>
              {byCluster[cid].map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Full policy table */}
      <div style={{ overflowX: "auto" }}>
        <table style={reportTableStyle}>
          <thead>
            <tr>
              <th style={reportThStyle}>Policy</th>
              <th style={reportThStyle}>Cluster</th>
              <th style={reportThStyle}>Label</th>
              <th style={reportThStyle}>Mean Return</th>
              <th style={reportThStyle}>Collapse Rate</th>
              <th style={reportThStyle}>Mean Final Pool</th>
              {hasWorstCase && <th style={reportThStyle}>Worst-Case Return</th>}
              {hasRobustness && <th style={reportThStyle}>Robustness Score</th>}
            </tr>
          </thead>
          <tbody>
            {policies.map((p) => {
              const f = features[p] ?? {};
              const cid = clusters[p];
              return (
                <tr key={p}>
                  <td style={{ ...reportTdStyle, fontWeight: 500 }}>{p}</td>
                  <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{cid}</td>
                  <td style={reportTdStyle}>{labels[String(cid)] ?? "—"}</td>
                  <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{fmt(f.mean_return)}</td>
                  <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{fmt(f.collapse_rate, true)}</td>
                  <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{fmt(f.mean_final_pool)}</td>
                  {hasWorstCase && <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{fmt(f.worst_case_return)}</td>}
                  {hasRobustness && <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{fmt(f.robustness_score)}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EvalReportView({ data }: { data: Record<string, any> }) {
  const results = (data.results ?? []) as any[];
  const available = results.filter((r) => r.available);
  const skipped = results.filter((r) => !r.available);

  // Sort by mean_total_reward descending
  const ranked = [...available].sort(
    (a, b) => (b.mean_total_reward ?? 0) - (a.mean_total_reward ?? 0)
  );

  return (
    <div>
      <table style={{ ...reportTableStyle, marginBottom: 24 }}>
        <thead>
          <tr>
            <th style={reportThStyle}>#</th>
            <th style={reportThStyle}>Policy</th>
            <th style={reportThStyle}>Source</th>
            <th style={reportThStyle}>Mean Reward</th>
            <th style={reportThStyle}>Std</th>
            <th style={reportThStyle}>Final Pool</th>
            <th style={reportThStyle}>Collapse %</th>
            <th style={reportThStyle}>Episodes</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((r, i) => (
            <tr key={r.policy_name}>
              <td style={{ ...reportTdStyle, color: "var(--text-secondary)" }}>{i + 1}</td>
              <td style={{ ...reportTdStyle, fontWeight: 500 }}>{r.policy_name}</td>
              <td style={{ ...reportTdStyle, color: "var(--text-secondary)" }}>{r.source ?? "—"}</td>
              <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>
                {(r.mean_total_reward ?? 0).toFixed(4)}
              </td>
              <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>
                {(r.std_total_reward ?? 0).toFixed(4)}
              </td>
              <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>
                {(r.mean_final_shared_pool ?? 0).toFixed(2)}
              </td>
              <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>
                {((r.collapse_rate ?? 0) * 100).toFixed(1)}%
              </td>
              <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{r.n_episodes ?? 0}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {skipped.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ ...sectionHeadingStyle, fontSize: 13 }}>Skipped Policies</h3>
          <ul style={{ fontSize: 13, color: "var(--text-secondary)", listStyle: "disc", paddingLeft: 16 }}>
            {skipped.map((s: any) => (
              <li key={s.policy_name}>
                <strong style={{ color: "var(--text-primary)" }}>{s.policy_name}</strong> ({s.source}): {s.skip_reason ?? "unavailable"}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Competitive report detail (copied from competitive/reports/[report_id]/page.tsx) */
/* ------------------------------------------------------------------ */

function CompetitiveReportView({ data, strategies }: { data: Record<string, any>; strategies: StrategyResponse | null }) {
  // Transform per_sweep_results: backend returns `mean_reward` but
  // RobustHeatmap expects `mean_total_reward`.
  const rawPerSweep = (data.per_sweep_results ?? {}) as Record<string, Record<string, any>>;
  const perSweep: Record<string, Record<string, any>> = {};
  for (const [sweepName, policies] of Object.entries(rawPerSweep)) {
    const mapped: Record<string, any> = {};
    for (const [policyName, entry] of Object.entries(policies)) {
      mapped[policyName] = {
        ...entry,
        mean_total_reward: entry.mean_total_reward ?? entry.mean_reward,
      };
    }
    perSweep[sweepName] = mapped;
  }
  const perPolicy = (data.per_policy_robustness ?? {}) as Record<string, any>;

  return (
    <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, background: "var(--bg-base)" }}>
      <div style={{ marginBottom: 16 }}>
        <BackToReportsLink />
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <h1 style={{ fontSize: 20, fontWeight: 600, fontFamily: "var(--font-mono)", color: "var(--text-primary)", margin: 0 }}>{data.report_id as string}</h1>
      </div>

      <div style={{ background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: 8, padding: 20, marginBottom: 24 }}>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "0 0 4px" }}>
          <span style={{
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 11,
            fontWeight: 500,
            marginRight: 8,
            background: "rgba(249,115,22,0.15)",
            color: "#fb923c",
          }}>
            competitive
          </span>
          Config hash: <code style={{ color: "var(--text-primary)" }}>{data.config_hash as string}</code>
        </p>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "4px 0 0" }}>Generated: {data.timestamp as string}</p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
        {/* Section 1 — Summary table */}
        <section>
          <h2 style={sectionHeadingStyle}>Robustness Summary</h2>
          <CompetitiveSummaryTable perPolicy={perPolicy} />
        </section>

        {/* Section 2 — Heatmap */}
        <section>
          <h2 style={sectionHeadingStyle}>Reward Heatmap (Policy x Sweep)</h2>
          <RobustHeatmap perSweepResults={perSweep} />
        </section>

        {/* Section 3 — Scatter plot */}
        <section>
          <h2 style={sectionHeadingStyle}>Mean vs Worst-Case Reward</h2>
          <RobustScatter perPolicyRobustness={perPolicy} />
        </section>
      </div>

      {/* Section 4 — Strategy groups */}
      {strategies && <CompetitiveStrategyGroupsSection strategies={strategies} />}
    </main>
  );
}

/* ------------------------------------------------------------------ */
/* Competitive summary table with winner_rate column                   */
/* ------------------------------------------------------------------ */

function CompetitiveSummaryTable({ perPolicy }: { perPolicy: Record<string, any> }) {
  const entries = Object.values(perPolicy).filter(
    (p: any) => p.n_sweeps_evaluated > 0
  ) as any[];

  if (entries.length === 0) return <p style={{ color: "var(--text-tertiary)" }}>No policy data.</p>;

  // Find best per column
  const bestMean = Math.max(...entries.map((p) => p.overall_mean_reward ?? -Infinity));
  const bestRobust = Math.max(...entries.map((p) => p.robustness_score ?? -Infinity));
  const bestWinner = Math.max(...entries.map((p) => p.mean_winner_rate ?? -Infinity));
  const bestWorst = Math.max(...entries.map((p) => p.worst_case_mean_reward ?? -Infinity));

  const ranked = [...entries].sort(
    (a, b) => (b.robustness_score ?? 0) - (a.robustness_score ?? 0)
  );

  const bestStyle = (isBest: boolean): React.CSSProperties => ({
    ...reportTdStyle,
    fontFamily: "var(--font-mono)",
    ...(isBest ? { color: "#4ade80", fontWeight: 700 } : {}),
  });

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={reportTableStyle}>
        <thead>
          <tr>
            <th style={reportThStyle}>#</th>
            <th style={reportThStyle}>Policy</th>
            <th style={reportThStyle}>Mean Reward</th>
            <th style={reportThStyle}>Robustness</th>
            <th style={reportThStyle}>Winner Rate</th>
            <th style={reportThStyle}>Worst-Case</th>
            <th style={reportThStyle}>Sweeps</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((p: any, i: number) => (
            <tr key={p.policy_name}>
              <td style={{ ...reportTdStyle, color: "var(--text-secondary)" }}>{i + 1}</td>
              <td style={{ ...reportTdStyle, fontWeight: 500 }}>{p.policy_name}</td>
              <td style={bestStyle(p.overall_mean_reward === bestMean)}>
                {(p.overall_mean_reward ?? 0).toFixed(4)}
              </td>
              <td style={bestStyle(p.robustness_score === bestRobust)}>
                {(p.robustness_score ?? 0).toFixed(4)}
              </td>
              <td style={bestStyle(p.mean_winner_rate === bestWinner)}>
                {p.mean_winner_rate != null ? `${(p.mean_winner_rate * 100).toFixed(1)}%` : "—"}
              </td>
              <td style={bestStyle(p.worst_case_mean_reward === bestWorst)}>
                {(p.worst_case_mean_reward ?? 0).toFixed(4)}
              </td>
              <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{p.n_sweeps_evaluated}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Strategy groups for competitive reports                              */
/* ------------------------------------------------------------------ */

function competitiveFmt(v: number | null | undefined, pct = false): string {
  if (v == null) return "—";
  return pct ? `${(v * 100).toFixed(2)}%` : v.toFixed(2);
}

function CompetitiveStrategyGroupsSection({ strategies }: { strategies: StrategyResponse }) {
  const { features, clusters, labels, summaries } = strategies;
  const policies = Object.keys(clusters);

  // Group policies by cluster id
  const byCluster: Record<number, string[]> = {};
  for (const policy of policies) {
    const cid = clusters[policy];
    (byCluster[cid] ??= []).push(policy);
  }
  const clusterIds = Object.keys(byCluster).map(Number).sort((a, b) => a - b);

  return (
    <div style={{ marginTop: 40, display: "flex", flexDirection: "column", gap: 24 }}>
      <h2 style={sectionHeadingStyle}>Strategy Groups</h2>

      {/* Cluster cards */}
      <div style={{ display: "grid", gap: 16, gridTemplateColumns: "repeat(3, 1fr)" }}>
        {clusterIds.map((cid) => (
          <div key={cid} style={{ background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: 8, padding: 16 }}>
            <h3 style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, color: "var(--text-primary)" }}>
              Cluster {cid}: {labels[String(cid)] ?? `Group ${cid}`}
            </h3>
            <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 8 }}>{summaries[String(cid)] ?? ""}</p>
            <ul style={{ fontSize: 12, color: "var(--text-primary)", listStyle: "disc", paddingLeft: 16 }}>
              {byCluster[cid].map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Full policy table */}
      <div style={{ overflowX: "auto" }}>
        <table style={reportTableStyle}>
          <thead>
            <tr>
              <th style={reportThStyle}>Policy</th>
              <th style={reportThStyle}>Cluster</th>
              <th style={reportThStyle}>Label</th>
              <th style={reportThStyle}>Mean Reward</th>
              <th style={reportThStyle}>Winner Rate</th>
              <th style={reportThStyle}>Robustness</th>
              <th style={reportThStyle}>Worst-Case</th>
            </tr>
          </thead>
          <tbody>
            {policies.map((p) => {
              const f = (features[p] ?? {}) as Record<string, unknown>;
              const cid = clusters[p];
              return (
                <tr key={p}>
                  <td style={{ ...reportTdStyle, fontWeight: 500 }}>{p}</td>
                  <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{cid}</td>
                  <td style={reportTdStyle}>{labels[String(cid)] ?? "—"}</td>
                  <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{competitiveFmt(f.mean_reward as number | null)}</td>
                  <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{competitiveFmt(f.winner_rate as number | null, true)}</td>
                  <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{competitiveFmt(f.robustness_score as number | null)}</td>
                  <td style={{ ...reportTdStyle, fontFamily: "var(--font-mono)" }}>{competitiveFmt(f.worst_case_reward as number | null)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
