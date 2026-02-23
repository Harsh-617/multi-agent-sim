"use client";

/**
 * Summary table for robustness results: collapse rate, top-3 callout, hardest sweep.
 */

interface PolicyRobustness {
  policy_name: string;
  overall_mean_reward: number;
  worst_case_mean_reward: number;
  robustness_score: number;
  collapse_rate_overall: number;
  n_sweeps_evaluated: number;
}

interface Props {
  perPolicyRobustness: Record<string, PolicyRobustness>;
  perSweepResults: Record<string, Record<string, { mean_total_reward?: number; available?: boolean; n_episodes?: number }>>;
}

export default function RobustSummaryTable({
  perPolicyRobustness,
  perSweepResults,
}: Props) {
  const ranked = Object.values(perPolicyRobustness)
    .filter((p) => p.n_sweeps_evaluated > 0)
    .sort((a, b) => b.robustness_score - a.robustness_score);

  // Compute hardest sweep
  let hardestSweep = "";
  let hardestAvg = Infinity;
  for (const [sweepName, policyData] of Object.entries(perSweepResults)) {
    const rewards: number[] = [];
    for (const v of Object.values(policyData)) {
      if (v.available !== false && (v.n_episodes ?? 0) > 0 && v.mean_total_reward != null) {
        rewards.push(v.mean_total_reward);
      }
    }
    if (rewards.length > 0) {
      const avg = rewards.reduce((a, b) => a + b, 0) / rewards.length;
      if (avg < hardestAvg) {
        hardestAvg = avg;
        hardestSweep = sweepName;
      }
    }
  }

  return (
    <div>
      {/* Main table */}
      <table className="w-full text-sm border-collapse mb-6">
        <thead>
          <tr className="border-b text-left">
            <th className="py-2 pr-3">#</th>
            <th className="py-2 pr-3">Policy</th>
            <th className="py-2 pr-3">Robustness</th>
            <th className="py-2 pr-3">Mean Reward</th>
            <th className="py-2 pr-3">Worst-Case</th>
            <th className="py-2 pr-3">Collapse %</th>
            <th className="py-2">Sweeps</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((p, i) => (
            <tr key={p.policy_name} className="border-b hover:bg-gray-50">
              <td className="py-2 pr-3 text-gray-500">{i + 1}</td>
              <td className="py-2 pr-3 font-medium">{p.policy_name}</td>
              <td className="py-2 pr-3 font-mono">{p.robustness_score.toFixed(4)}</td>
              <td className="py-2 pr-3 font-mono">{p.overall_mean_reward.toFixed(4)}</td>
              <td className="py-2 pr-3 font-mono">{p.worst_case_mean_reward.toFixed(4)}</td>
              <td className="py-2 pr-3 font-mono">
                {(p.collapse_rate_overall * 100).toFixed(1)}%
              </td>
              <td className="py-2 font-mono">{p.n_sweeps_evaluated}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Top-3 callout */}
      {ranked.length > 0 && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded">
          <h4 className="font-semibold text-green-800 mb-1">Top-3 Most Robust</h4>
          <ol className="list-decimal list-inside text-sm text-green-900">
            {ranked.slice(0, 3).map((p) => (
              <li key={p.policy_name}>
                <strong>{p.policy_name}</strong> &mdash; score{" "}
                {p.robustness_score.toFixed(4)}
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Hardest sweep */}
      {hardestSweep && (
        <div className="p-3 bg-red-50 border border-red-200 rounded">
          <h4 className="font-semibold text-red-800 mb-1">Hardest Sweep</h4>
          <p className="text-sm text-red-900">
            <strong>{hardestSweep}</strong> &mdash; avg reward across policies:{" "}
            {hardestAvg.toFixed(4)}
          </p>
        </div>
      )}
    </div>
  );
}
