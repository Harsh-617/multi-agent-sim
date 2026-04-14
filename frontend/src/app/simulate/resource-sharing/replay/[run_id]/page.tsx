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
    <main style={{ maxWidth: 896, margin: "0 auto", padding: "48px 24px", paddingTop: 96 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <div>
          <Link
            href="/simulate/resource-sharing"
            style={{ fontSize: 13, color: "var(--text-tertiary)", textDecoration: "none" }}
          >
            &larr; Resource Sharing
          </Link>
          <h1 style={{ fontSize: 22, fontWeight: 500, color: "var(--text-primary)", margin: "4px 0 0" }}>
            Replay{" "}
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 18 }}>{run_id}</span>
          </h1>
        </div>
      </div>

      {/* Status bar */}
      <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 24, fontSize: 13, color: "var(--text-secondary)" }}>
        <span>
          Status:{" "}
          <span
            style={{
              color:
                status === "streaming"
                  ? "var(--accent)"
                  : status === "connecting"
                  ? "#eab308"
                  : "var(--text-tertiary)",
            }}
          >
            {status}
          </span>
        </span>
        <span>Step: {currentStep}</span>
        {done && (
          <span style={{ fontWeight: 500, color: "var(--text-secondary)" }}>
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
        <div style={{ marginTop: 24, padding: 16, border: "1px solid var(--bg-border)", borderRadius: 6, background: "var(--bg-surface)" }}>
          <h2 style={{ fontSize: 15, fontWeight: 500, color: "var(--text-primary)", marginBottom: 8 }}>Episode Summary</h2>
          <dl style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 8px", fontSize: 13 }}>
            <dt style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Length</dt>
            <dd style={{ color: "var(--text-primary)" }}>{summary.episode_length} steps</dd>
            <dt style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Termination</dt>
            <dd style={{ color: "var(--text-primary)" }}>{summary.termination_reason}</dd>
            {summary && isCompetitiveSummary(summary) ? (
              <>
                <dt style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Winner</dt>
                <dd style={{ color: "var(--text-primary)" }}>{summary.winner_id ?? "none"}</dd>
                <dt style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Score Spread</dt>
                <dd style={{ color: "var(--text-primary)" }}>{summary.score_spread.toFixed(2)}</dd>
                <dt style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Eliminations</dt>
                <dd style={{ color: "var(--text-primary)" }}>{summary.num_eliminations}</dd>
              </>
            ) : (
              <>
                <dt style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Final Shared Pool</dt>
                <dd style={{ color: "var(--text-primary)" }}>{(summary as EpisodeSummary).final_shared_pool.toFixed(2)}</dd>
              </>
            )}
          </dl>
          <h3 style={{ fontSize: 13, fontWeight: 500, color: "var(--text-secondary)", marginTop: 12, marginBottom: 6 }}>Total Reward per Agent</h3>
          <ul style={{ fontSize: 12, listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 3 }}>
            {Object.entries(summary.total_reward_per_agent).map(([agentId, reward]) => (
              <li key={agentId} style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                {agentId.slice(0, 8)}: {reward.toFixed(3)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </main>
  );
}
