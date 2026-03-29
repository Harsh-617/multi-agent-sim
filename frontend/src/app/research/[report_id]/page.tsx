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

  if (loading) return <main className="max-w-5xl mx-auto p-8"><p>Loading...</p></main>;
  if (error) return <main className="max-w-5xl mx-auto p-8"><p className="text-red-600">Error: {error}</p></main>;
  if (!data) return <main className="max-w-5xl mx-auto p-8"><p>No data.</p></main>;

  if (isCompetitive) {
    return <CompetitiveReportView data={data} strategies={strategies} />;
  }
  return <MixedReportView data={data} strategies={strategies} />;
}

/* ------------------------------------------------------------------ */
/* Mixed report detail (copied from reports/[report_id]/page.tsx)      */
/* ------------------------------------------------------------------ */

function MixedReportView({ data, strategies }: { data: Record<string, any>; strategies: StrategyResponse | null }) {
  const kind = data.kind as string;

  return (
    <main className="max-w-5xl mx-auto p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-bold font-mono">{data.report_id as string}</h1>
        <Link
          href="/research"
          className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
        >
          Back to Reports
        </Link>
      </div>

      <div className="text-sm text-gray-600 mb-6 space-y-1">
        <p>
          <span className={`px-2 py-0.5 rounded text-xs font-medium mr-2 ${
            kind === "robust" ? "bg-purple-100 text-purple-800" : "bg-blue-100 text-blue-800"
          }`}>
            {kind}
          </span>
          Config hash: <code>{data.config_hash as string}</code>
        </p>
        <p>Generated: {data.timestamp as string}</p>
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
    <div className="space-y-8">
      <section>
        <h2 className="text-lg font-semibold mb-3">Reward Heatmap (Policy x Sweep)</h2>
        <RobustHeatmap perSweepResults={perSweep} />
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Mean vs Worst-Case Reward</h2>
        <RobustScatter perPolicyRobustness={perPolicy} />
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Robustness Summary</h2>
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
    <div className="mt-10 space-y-6">
      <h2 className="text-lg font-semibold">Strategy Groups</h2>

      {/* Cluster cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {clusterIds.map((cid) => (
          <div key={cid} className="border rounded-lg p-4 bg-white shadow-sm">
            <h3 className="font-semibold text-sm mb-1">
              Cluster {cid}: {labels[String(cid)] ?? `Group ${cid}`}
            </h3>
            <p className="text-xs text-gray-600 mb-2">{summaries[String(cid)] ?? ""}</p>
            <ul className="text-xs text-gray-800 list-disc list-inside">
              {byCluster[cid].map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Full policy table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b text-left">
              <th className="py-2 pr-3">Policy</th>
              <th className="py-2 pr-3">Cluster</th>
              <th className="py-2 pr-3">Label</th>
              <th className="py-2 pr-3">Mean Return</th>
              <th className="py-2 pr-3">Collapse Rate</th>
              <th className="py-2 pr-3">Mean Final Pool</th>
              {hasWorstCase && <th className="py-2 pr-3">Worst-Case Return</th>}
              {hasRobustness && <th className="py-2 pr-3">Robustness Score</th>}
            </tr>
          </thead>
          <tbody>
            {policies.map((p) => {
              const f = features[p] ?? {};
              const cid = clusters[p];
              return (
                <tr key={p} className="border-b hover:bg-gray-50">
                  <td className="py-2 pr-3 font-medium">{p}</td>
                  <td className="py-2 pr-3 font-mono">{cid}</td>
                  <td className="py-2 pr-3">{labels[String(cid)] ?? "—"}</td>
                  <td className="py-2 pr-3 font-mono">{fmt(f.mean_return)}</td>
                  <td className="py-2 pr-3 font-mono">{fmt(f.collapse_rate, true)}</td>
                  <td className="py-2 pr-3 font-mono">{fmt(f.mean_final_pool)}</td>
                  {hasWorstCase && <td className="py-2 pr-3 font-mono">{fmt(f.worst_case_return)}</td>}
                  {hasRobustness && <td className="py-2 pr-3 font-mono">{fmt(f.robustness_score)}</td>}
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
      <table className="w-full text-sm border-collapse mb-6">
        <thead>
          <tr className="border-b text-left">
            <th className="py-2 pr-3">#</th>
            <th className="py-2 pr-3">Policy</th>
            <th className="py-2 pr-3">Source</th>
            <th className="py-2 pr-3">Mean Reward</th>
            <th className="py-2 pr-3">Std</th>
            <th className="py-2 pr-3">Final Pool</th>
            <th className="py-2 pr-3">Collapse %</th>
            <th className="py-2">Episodes</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((r, i) => (
            <tr key={r.policy_name} className="border-b hover:bg-gray-50">
              <td className="py-2 pr-3 text-gray-500">{i + 1}</td>
              <td className="py-2 pr-3 font-medium">{r.policy_name}</td>
              <td className="py-2 pr-3 text-gray-600">{r.source ?? "—"}</td>
              <td className="py-2 pr-3 font-mono">
                {(r.mean_total_reward ?? 0).toFixed(4)}
              </td>
              <td className="py-2 pr-3 font-mono">
                {(r.std_total_reward ?? 0).toFixed(4)}
              </td>
              <td className="py-2 pr-3 font-mono">
                {(r.mean_final_shared_pool ?? 0).toFixed(2)}
              </td>
              <td className="py-2 pr-3 font-mono">
                {((r.collapse_rate ?? 0) * 100).toFixed(1)}%
              </td>
              <td className="py-2 font-mono">{r.n_episodes ?? 0}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {skipped.length > 0 && (
        <div className="mt-4">
          <h3 className="font-semibold mb-2">Skipped Policies</h3>
          <ul className="text-sm text-gray-600 list-disc list-inside">
            {skipped.map((s: any) => (
              <li key={s.policy_name}>
                <strong>{s.policy_name}</strong> ({s.source}): {s.skip_reason ?? "unavailable"}
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
    <main className="max-w-5xl mx-auto p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-bold font-mono">{data.report_id as string}</h1>
        <Link
          href="/research"
          className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
        >
          Back to Reports
        </Link>
      </div>

      <div className="text-sm text-gray-600 mb-6 space-y-1">
        <p>
          <span className="px-2 py-0.5 rounded text-xs font-medium mr-2 bg-orange-100 text-orange-800">
            competitive
          </span>
          Config hash: <code>{data.config_hash as string}</code>
        </p>
        <p>Generated: {data.timestamp as string}</p>
      </div>

      <div className="space-y-8">
        {/* Section 1 — Summary table */}
        <section>
          <h2 className="text-lg font-semibold mb-3">Robustness Summary</h2>
          <CompetitiveSummaryTable perPolicy={perPolicy} />
        </section>

        {/* Section 2 — Heatmap */}
        <section>
          <h2 className="text-lg font-semibold mb-3">Reward Heatmap (Policy x Sweep)</h2>
          <RobustHeatmap perSweepResults={perSweep} />
        </section>

        {/* Section 3 — Scatter plot */}
        <section>
          <h2 className="text-lg font-semibold mb-3">Mean vs Worst-Case Reward</h2>
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

  if (entries.length === 0) return <p className="text-gray-500">No policy data.</p>;

  // Find best per column
  const bestMean = Math.max(...entries.map((p) => p.overall_mean_reward ?? -Infinity));
  const bestRobust = Math.max(...entries.map((p) => p.robustness_score ?? -Infinity));
  const bestWinner = Math.max(...entries.map((p) => p.mean_winner_rate ?? -Infinity));
  const bestWorst = Math.max(...entries.map((p) => p.worst_case_mean_reward ?? -Infinity));

  const ranked = [...entries].sort(
    (a, b) => (b.robustness_score ?? 0) - (a.robustness_score ?? 0)
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b text-left">
            <th className="py-2 pr-3">#</th>
            <th className="py-2 pr-3">Policy</th>
            <th className="py-2 pr-3">Mean Reward</th>
            <th className="py-2 pr-3">Robustness</th>
            <th className="py-2 pr-3">Winner Rate</th>
            <th className="py-2 pr-3">Worst-Case</th>
            <th className="py-2">Sweeps</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((p: any, i: number) => (
            <tr key={p.policy_name} className="border-b hover:bg-gray-50">
              <td className="py-2 pr-3 text-gray-500">{i + 1}</td>
              <td className="py-2 pr-3 font-medium">{p.policy_name}</td>
              <td className={`py-2 pr-3 font-mono ${p.overall_mean_reward === bestMean ? "text-green-700 font-bold" : ""}`}>
                {(p.overall_mean_reward ?? 0).toFixed(4)}
              </td>
              <td className={`py-2 pr-3 font-mono ${p.robustness_score === bestRobust ? "text-green-700 font-bold" : ""}`}>
                {(p.robustness_score ?? 0).toFixed(4)}
              </td>
              <td className={`py-2 pr-3 font-mono ${p.mean_winner_rate === bestWinner ? "text-green-700 font-bold" : ""}`}>
                {p.mean_winner_rate != null ? `${(p.mean_winner_rate * 100).toFixed(1)}%` : "—"}
              </td>
              <td className={`py-2 pr-3 font-mono ${p.worst_case_mean_reward === bestWorst ? "text-green-700 font-bold" : ""}`}>
                {(p.worst_case_mean_reward ?? 0).toFixed(4)}
              </td>
              <td className="py-2 font-mono">{p.n_sweeps_evaluated}</td>
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
    <div className="mt-10 space-y-6">
      <h2 className="text-lg font-semibold">Strategy Groups</h2>

      {/* Cluster cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {clusterIds.map((cid) => (
          <div key={cid} className="border rounded-lg p-4 bg-white shadow-sm">
            <h3 className="font-semibold text-sm mb-1">
              Cluster {cid}: {labels[String(cid)] ?? `Group ${cid}`}
            </h3>
            <p className="text-xs text-gray-600 mb-2">{summaries[String(cid)] ?? ""}</p>
            <ul className="text-xs text-gray-800 list-disc list-inside">
              {byCluster[cid].map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Full policy table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b text-left">
              <th className="py-2 pr-3">Policy</th>
              <th className="py-2 pr-3">Cluster</th>
              <th className="py-2 pr-3">Label</th>
              <th className="py-2 pr-3">Mean Reward</th>
              <th className="py-2 pr-3">Winner Rate</th>
              <th className="py-2 pr-3">Robustness</th>
              <th className="py-2 pr-3">Worst-Case</th>
            </tr>
          </thead>
          <tbody>
            {policies.map((p) => {
              const f = (features[p] ?? {}) as Record<string, unknown>;
              const cid = clusters[p];
              return (
                <tr key={p} className="border-b hover:bg-gray-50">
                  <td className="py-2 pr-3 font-medium">{p}</td>
                  <td className="py-2 pr-3 font-mono">{cid}</td>
                  <td className="py-2 pr-3">{labels[String(cid)] ?? "—"}</td>
                  <td className="py-2 pr-3 font-mono">{competitiveFmt(f.mean_reward as number | null)}</td>
                  <td className="py-2 pr-3 font-mono">{competitiveFmt(f.winner_rate as number | null, true)}</td>
                  <td className="py-2 pr-3 font-mono">{competitiveFmt(f.robustness_score as number | null)}</td>
                  <td className="py-2 pr-3 font-mono">{competitiveFmt(f.worst_case_reward as number | null)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
