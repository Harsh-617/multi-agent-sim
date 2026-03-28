"use client";

import { CompetitiveEpisodeSummary } from "@/lib/api";

interface Props {
  summary: CompetitiveEpisodeSummary;
}

export default function CompetitiveRunSummary({ summary }: Props) {
  const rankings = summary.final_rankings ?? [];
  const scores = summary.final_scores ?? {};

  return (
    <div className="mt-6 space-y-4">
      {/* Winner banner */}
      {summary.winner_id && (
        <div className="p-4 bg-yellow-50 border border-yellow-300 rounded text-center">
          <span className="text-lg font-bold text-yellow-800">
            Winner: {summary.winner_id}
          </span>
        </div>
      )}

      {/* Episode info */}
      <div className="p-4 border border-gray-300 rounded">
        <h2 className="text-lg font-semibold mb-2">Episode Summary</h2>
        <dl className="grid grid-cols-2 gap-2 text-sm">
          <dt className="font-medium">Episode Length</dt>
          <dd>{summary.episode_length} steps</dd>
          <dt className="font-medium">Termination</dt>
          <dd>
            <span className="inline-block px-2 py-0.5 rounded bg-gray-100 text-gray-700 text-xs font-mono">
              {summary.termination_reason}
            </span>
          </dd>
          <dt className="font-medium">Eliminations</dt>
          <dd>{summary.num_eliminations ?? 0}</dd>
          <dt className="font-medium">Score Spread</dt>
          <dd>{(summary.score_spread ?? 0).toFixed(2)}</dd>
        </dl>
      </div>

      {/* Final rankings table */}
      {rankings.length > 0 && (
        <div className="p-4 border border-gray-300 rounded">
          <h2 className="text-lg font-semibold mb-2">Final Rankings</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left">
                <th className="py-1 pr-4">Rank</th>
                <th className="py-1 pr-4">Agent</th>
                <th className="py-1 text-right">Score</th>
              </tr>
            </thead>
            <tbody>
              {rankings.map((agentId, idx) => (
                <tr key={agentId} className="border-b border-gray-100">
                  <td className="py-1 pr-4 font-mono">{idx + 1}</td>
                  <td className="py-1 pr-4 font-mono">{agentId}</td>
                  <td className="py-1 text-right font-mono">
                    {(scores[agentId] ?? 0).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Total reward per agent */}
      {summary.total_reward_per_agent &&
        Object.keys(summary.total_reward_per_agent).length > 0 && (
          <div className="p-4 border border-gray-300 rounded">
            <h2 className="text-lg font-semibold mb-2">Total Reward per Agent</h2>
            <ul className="text-sm space-y-1">
              {Object.entries(summary.total_reward_per_agent).map(
                ([agentId, reward]) => (
                  <li key={agentId} className="font-mono">
                    {agentId}: {reward.toFixed(3)}
                  </li>
                ),
              )}
            </ul>
          </div>
        )}
    </div>
  );
}
