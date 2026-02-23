"use client";

import { useParams } from "next/navigation";
import { useEffect, useRef, useState, useCallback } from "react";
import {
  connectMetrics,
  WsMessage,
  StepMetric,
  EpisodeSummary,
  StepEvent,
} from "@/lib/api";
import MetricsChart from "@/components/MetricsChart";
import StopRunButton from "@/components/StopRunButton";
import Link from "next/link";

export default function RunPage() {
  const { run_id } = useParams<{ run_id: string }>();

  const [currentStep, setCurrentStep] = useState<number>(0);
  const [history, setHistory] = useState<StepMetric[]>([]);
  const [events, setEvents] = useState<StepEvent[]>([]);
  const [done, setDone] = useState(false);
  const [terminationReason, setTerminationReason] = useState<string | null>(null);
  const [summary, setSummary] = useState<EpisodeSummary | null>(null);
  const [wsStatus, setWsStatus] = useState<"connecting" | "open" | "closed">("connecting");

  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const seenRef = useRef(new Set<string>());

  const handleMessage = useCallback((msg: WsMessage) => {
    if (msg.type === "step") {
      setCurrentStep(msg.t);
      const fresh = msg.metrics.filter((m) => {
        const key = `${m.step}:${m.agent_id}`;
        if (seenRef.current.has(key)) return false;
        seenRef.current.add(key);
        return true;
      });
      if (fresh.length > 0) {
        setHistory((prev) => [...prev, ...fresh]);
      }
      if (msg.events) {
        setEvents((prev) => [...prev, ...msg.events!]);
      }
    } else if (msg.type === "done") {
      setDone(true);
      setTerminationReason(msg.termination_reason);
      setSummary(msg.episode_summary);
    }
  }, []);

  useEffect(() => {
    if (!run_id) return;

    function connect() {
      setWsStatus("connecting");
      const ws = connectMetrics(
        run_id,
        (msg) => {
          retryRef.current = 0;
          setWsStatus("open");
          handleMessage(msg);
        },
        () => setWsStatus("closed"),
        () => {
          setWsStatus("closed");
          // Simple retry with backoff (max 5 attempts)
          if (retryRef.current < 5) {
            retryRef.current++;
            setTimeout(connect, 1000 * retryRef.current);
          }
        }
      );
      wsRef.current = ws;
    }

    connect();

    return () => {
      wsRef.current?.close();
    };
  }, [run_id, handleMessage]);

  return (
    <main className="max-w-4xl mx-auto p-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link href="/" className="text-blue-500 hover:underline text-sm">
            &larr; Back
          </Link>
          <h1 className="text-2xl font-bold">
            Run <span className="font-mono">{run_id}</span>
          </h1>
        </div>
        {!done && <StopRunButton onStopped={() => setDone(true)} />}
      </div>

      {/* Status bar */}
      <div className="flex gap-4 items-center mb-6 text-sm">
        <span>
          WebSocket:{" "}
          <span
            className={
              wsStatus === "open"
                ? "text-green-500"
                : wsStatus === "connecting"
                ? "text-yellow-500"
                : "text-red-500"
            }
          >
            {wsStatus}
          </span>
        </span>
        <span>Step: {currentStep}</span>
        {done && (
          <span className="font-semibold text-orange-500">
            Done &mdash; {terminationReason}
          </span>
        )}
      </div>

      {/* Live chart */}
      <MetricsChart history={history} />

      {/* Events log */}
      {events.length > 0 && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold mb-2">Events</h2>
          <ul className="text-sm space-y-1 max-h-40 overflow-y-auto">
            {events.map((ev, i) => (
              <li key={i} className="font-mono">
                [step {ev.step}] {ev.event}
                {ev.agent_id ? ` agent=${ev.agent_id}` : ""}
                {ev.shared_pool !== undefined ? ` pool=${ev.shared_pool.toFixed(2)}` : ""}
              </li>
            ))}
          </ul>
        </div>
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
            <dt className="font-medium">Final Shared Pool</dt>
            <dd>{summary.final_shared_pool.toFixed(2)}</dd>
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
