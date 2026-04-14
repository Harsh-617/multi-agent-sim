"use client";

import { useParams } from "next/navigation";
import { useEffect, useRef, useState, useCallback } from "react";
import {
  connectMetrics,
  WsMessage,
  CooperativeStepMetric,
  CooperativeEpisodeSummary,
  CooperativeWsMessage,
  CooperativeRunDetail,
  getCooperativeRunDetail,
} from "@/lib/api";
import CooperativeMetricsChart from "@/components/CooperativeMetricsChart";
import CooperativeRunSummary from "@/components/CooperativeRunSummary";
import StopRunButton from "@/components/StopRunButton";
import Link from "next/link";

export default function CooperativeRunPage() {
  const { run_id } = useParams<{ run_id: string }>();

  const [currentStep, setCurrentStep] = useState<number>(0);
  const [history, setHistory] = useState<CooperativeStepMetric[]>([]);
  const [done, setDone] = useState(false);
  const [terminationReason, setTerminationReason] = useState<string | null>(null);
  const [summary, setSummary] = useState<CooperativeEpisodeSummary | null>(null);
  const [wsStatus, setWsStatus] = useState<"connecting" | "open" | "closed">("connecting");
  const [runDetail, setRunDetail] = useState<CooperativeRunDetail | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const seenRef = useRef(new Set<string>());
  const doneRef = useRef(false);

  const handleMessage = useCallback((msg: WsMessage) => {
    const m = msg as unknown as CooperativeWsMessage;
    if (m.type === "step") {
      setCurrentStep(m.t);
      const fresh = m.metrics.filter((metric) => {
        const key = `${metric.step}:${metric.agent_id}`;
        if (seenRef.current.has(key)) return false;
        seenRef.current.add(key);
        return true;
      });
      if (fresh.length > 0) {
        setHistory((prev) => [...prev, ...fresh]);
      }
    } else if (m.type === "done") {
      doneRef.current = true;
      setDone(true);
      setTerminationReason(m.termination_reason);
      if (m.episode_summary) {
        setSummary(m.episode_summary);
      }
    }
  }, []);

  const fetchRunDetail = useCallback(async () => {
    if (!run_id || doneRef.current) return;
    try {
      const detail = await getCooperativeRunDetail(run_id);
      setRunDetail(detail);
      if (detail.episode_summary) {
        doneRef.current = true;
        setDone(true);
        setTerminationReason(detail.episode_summary.termination_reason ?? null);
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
          if (!doneRef.current && retryRef.current < 5) {
            retryRef.current++;
            setTimeout(connect, 1000 * Math.pow(2, retryRef.current - 1));
          } else {
            fetchRunDetail();
          }
        },
        () => {
          setWsStatus("closed");
          fetchRunDetail();
        },
      );
      wsRef.current = ws;
    }

    connect();

    return () => {
      wsRef.current?.close();
    };
  }, [run_id, handleMessage, fetchRunDetail]);

  const episodeSummary =
    summary ?? (runDetail?.episode_summary ?? null);

  return (
    <main style={{ maxWidth: 896, margin: "0 auto", padding: "48px 24px", paddingTop: 96 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <div>
          <Link
            href="/simulate/cooperative"
            style={{ fontSize: 13, color: "var(--text-tertiary)", textDecoration: "none" }}
          >
            &larr; Cooperative Task Arena
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
      <CooperativeMetricsChart history={history} />

      {/* Episode summary */}
      {episodeSummary && (
        <div style={{ marginTop: 24 }}>
          <CooperativeRunSummary summary={episodeSummary} />
        </div>
      )}
    </main>
  );
}
