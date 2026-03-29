"use client";

import { useParams } from "next/navigation";
import { useEffect, useRef, useState, useCallback } from "react";
import {
  connectReplay,
  WsMessage,
  StepMetric,
  EpisodeSummary,
  CompetitiveEpisodeSummary,
  isCompetitiveSummary,
} from "@/lib/api";
import MetricsChart from "@/components/MetricsChart";
import CompetitiveReplayView, {
  CompetitiveStepMetric,
} from "@/components/CompetitiveReplayView";
import Link from "next/link";

/**
 * Detect whether the replay history belongs to a competitive run by checking
 * for competitive-specific fields on the first metric record.
 */
function isCompetitiveRun(history: StepMetric[]): boolean {
  if (history.length === 0) return false;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const first = history[0] as any;
  return "own_score" in first || "own_rank" in first;
}

export default function ReplayPage() {
  const { run_id } = useParams<{ run_id: string }>();

  const [currentStep, setCurrentStep] = useState<number>(0);
  const [history, setHistory] = useState<StepMetric[]>([]);
  const [done, setDone] = useState(false);
  const [terminationReason, setTerminationReason] = useState<string | null>(null);
  const [summary, setSummary] = useState<EpisodeSummary | CompetitiveEpisodeSummary | null>(null);
  const [status, setStatus] = useState<"connecting" | "streaming" | "done">("connecting");

  const esRef = useRef<EventSource | null>(null);

  const handleMessage = useCallback((msg: WsMessage) => {
    if (msg.type === "step") {
      setStatus("streaming");
      setCurrentStep(msg.t);
      setHistory((prev) => [...prev, ...msg.metrics]);
    } else if (msg.type === "done") {
      setDone(true);
      setTerminationReason(msg.termination_reason);
      setSummary(msg.episode_summary);
      setStatus("done");
    }
  }, []);

  useEffect(() => {
    if (!run_id) return;

    setStatus("connecting");
    const es = connectReplay(run_id, handleMessage, () => setStatus("done"));
    esRef.current = es;

    return () => {
      es.close();
    };
  }, [run_id, handleMessage]);

  return (
    <main className="max-w-4xl mx-auto p-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link href="/simulate/resource-sharing" className="text-blue-500 hover:underline text-sm">
            &larr; Runs
          </Link>
          <h1 className="text-2xl font-bold">
            Replay <span className="font-mono">{run_id}</span>
          </h1>
        </div>
      </div>

      {/* Status bar */}
      <div className="flex gap-4 items-center mb-6 text-sm">
        <span>
          Status:{" "}
          <span
            className={
              status === "streaming"
                ? "text-green-500"
                : status === "connecting"
                ? "text-yellow-500"
                : "text-blue-500"
            }
          >
            {status}
          </span>
        </span>
        <span>Step: {currentStep}</span>
        {done && (
          <span className="font-semibold text-orange-500">
            Done &mdash; {terminationReason}
          </span>
        )}
      </div>

      {/* Chart — choose component based on detected archetype */}
      {isCompetitiveRun(history) ? (
        <CompetitiveReplayView
          history={history as unknown as CompetitiveStepMetric[]}
        />
      ) : (
        <MetricsChart history={history} />
      )}

      {/* Episode summary */}
      {summary && (
        <div className="mt-6 p-4 border border-gray-300 rounded">
          <h2 className="text-lg font-semibold mb-2">Episode Summary</h2>
          <dl className="grid grid-cols-2 gap-2 text-sm">
            <dt className="font-medium">Length</dt>
            <dd>{summary.episode_length} steps</dd>
            <dt className="font-medium">Termination</dt>
            <dd>{summary.termination_reason}</dd>
            {summary && isCompetitiveSummary(summary) ? (
              <>
                <dt className="font-medium">Winner</dt>
                <dd>{summary.winner_id ?? "none"}</dd>
                <dt className="font-medium">Score Spread</dt>
                <dd>{summary.score_spread.toFixed(2)}</dd>
                <dt className="font-medium">Eliminations</dt>
                <dd>{summary.num_eliminations}</dd>
              </>
            ) : (
              <>
                <dt className="font-medium">Final Shared Pool</dt>
                <dd>{(summary as EpisodeSummary).final_shared_pool.toFixed(2)}</dd>
              </>
            )}
          </dl>
          <h3 className="text-sm font-semibold mt-3 mb-1">Total Reward per Agent</h3>
          <ul className="text-sm space-y-1">
            {Object.entries(summary.total_reward_per_agent).map(([agentId, reward]) => (
              <li key={agentId} className="font-mono">
                {agentId.slice(0, 8)}: {reward.toFixed(3)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </main>
  );
}
