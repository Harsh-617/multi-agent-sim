"use client";

import { useParams } from "next/navigation";
import { useEffect, useRef, useState, useCallback } from "react";
import {
  connectMetrics,
  getRunDetail,
  WsMessage,
  StepMetric,
  EpisodeSummary,
  CompetitiveEpisodeSummary,
  isCompetitiveSummary,
  StepEvent,
  RunDetail,
} from "@/lib/api";
import CompetitiveMetricsChart from "@/components/CompetitiveMetricsChart";
import CompetitiveRunSummary from "@/components/CompetitiveRunSummary";
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
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const seenRef = useRef(new Set<string>());
  const doneRef = useRef(false);

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
      doneRef.current = true;
      setDone(true);
      setTerminationReason(msg.termination_reason);
      setSummary(msg.episode_summary);
    }
  }, []);

  /** Fetch run detail from REST API (fallback when WS misses the run). */
  const fetchRunDetail = useCallback(async () => {
    if (!run_id || doneRef.current) return;
    try {
      const detail = await getRunDetail(run_id);
      setRunDetail(detail);
      if (detail.episode_summary) {
        doneRef.current = true;
        setDone(true);
        setTerminationReason(detail.termination_reason ?? detail.episode_summary.termination_reason);
      }
    } catch {
      // Run may not exist yet — ignore
    }
  }, [run_id]);

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
        () => {
          setWsStatus("closed");
          // Exponential backoff, max 5 attempts; stop if run completed normally.
          if (!doneRef.current && retryRef.current < 5) {
            retryRef.current++;
            setTimeout(connect, 1000 * Math.pow(2, retryRef.current - 1));
          } else {
            // Retries exhausted — try REST fallback
            fetchRunDetail();
          }
        },
        () => {
          setWsStatus("closed");
          // WS closed cleanly — fetch detail in case run already finished
          fetchRunDetail();
        }
      );
      wsRef.current = ws;
    }

    connect();

    return () => {
      wsRef.current?.close();
    };
  }, [run_id, handleMessage, fetchRunDetail]);

  // Resolve the episode summary — prefer WS-delivered, fall back to REST detail
  const rawSummary = summary ?? (runDetail?.episode_summary as EpisodeSummary | CompetitiveEpisodeSummary | null);
  const competitiveSummary =
    rawSummary && isCompetitiveSummary(rawSummary) ? rawSummary : null;
  const mixedSummary =
    rawSummary && !isCompetitiveSummary(rawSummary) ? (rawSummary as EpisodeSummary) : null;

  return (
    <main style={{ maxWidth: 896, margin: "0 auto", padding: "48px 24px", paddingTop: 96 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <div>
          <Link
            href="/simulate/head-to-head"
            style={{ fontSize: 13, color: "var(--text-tertiary)", textDecoration: "none" }}
          >
            &larr; Head-to-Head
          </Link>
          <h1 style={{ fontSize: 22, fontWeight: 500, color: "var(--text-primary)", margin: "4px 0 0" }}>
            Run{" "}
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 18 }}>{run_id}</span>
          </h1>
        </div>
        {!done && <StopRunButton onStopped={() => setDone(true)} />}
      </div>

      {/* Status bar */}
      <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 24, fontSize: 13, color: "var(--text-secondary)" }}>
        <span>
          WebSocket:{" "}
          <span
            style={{
              color:
                wsStatus === "open"
                  ? "var(--accent)"
                  : wsStatus === "connecting"
                  ? "#eab308"
                  : "#ef4444",
            }}
          >
            {wsStatus}
          </span>
        </span>
        <span>Step: {currentStep}</span>
        {done && (
          <span style={{ fontWeight: 500, color: "var(--text-secondary)" }}>
            Done &mdash; {terminationReason}
          </span>
        )}
      </div>

      {/* Live chart */}
      <CompetitiveMetricsChart history={history} />

      {/* Events log */}
      {events.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h2 style={{ fontSize: 15, fontWeight: 500, color: "var(--text-primary)", marginBottom: 8 }}>Events</h2>
          <ul style={{ fontSize: 12, listStyle: "none", padding: 0, margin: 0, maxHeight: 160, overflowY: "auto", display: "flex", flexDirection: "column", gap: 3 }}>
            {events.map((ev, i) => (
              <li key={i} style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                [step {ev.step}] {ev.event}
                {ev.agent_id ? ` agent=${ev.agent_id}` : ""}
                {ev.shared_pool !== undefined ? ` pool=${ev.shared_pool.toFixed(2)}` : ""}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Episode summary — competitive or mixed */}
      {competitiveSummary ? (
        <CompetitiveRunSummary summary={competitiveSummary} />
      ) : mixedSummary ? (
        <div style={{ marginTop: 24, padding: 16, border: "1px solid var(--bg-border)", borderRadius: 6, background: "var(--bg-surface)" }}>
          <h2 style={{ fontSize: 15, fontWeight: 500, color: "var(--text-primary)", marginBottom: 8 }}>Episode Summary</h2>
          <dl style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 8px", fontSize: 13 }}>
            <dt style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Length</dt>
            <dd style={{ color: "var(--text-primary)" }}>{mixedSummary.episode_length} steps</dd>
            <dt style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Termination</dt>
            <dd style={{ color: "var(--text-primary)" }}>{mixedSummary.termination_reason}</dd>
            <dt style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Final Shared Pool</dt>
            <dd style={{ color: "var(--text-primary)" }}>{mixedSummary.final_shared_pool.toFixed(2)}</dd>
          </dl>
          <h3 style={{ fontSize: 13, fontWeight: 500, color: "var(--text-secondary)", marginTop: 12, marginBottom: 6 }}>Total Reward per Agent</h3>
          <ul style={{ fontSize: 12, listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 3 }}>
            {Object.entries(mixedSummary.total_reward_per_agent).map(([agentId, reward]) => (
              <li key={agentId} style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                {agentId.slice(0, 8)}: {reward.toFixed(3)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </main>
  );
}
