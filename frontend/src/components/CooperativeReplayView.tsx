"use client";

/**
 * CooperativeReplayView
 *
 * Replay viewer for completed cooperative runs.
 * Streams stored metrics.jsonl via SSE, feeds data into the four-panel
 * CooperativeMetricsChart.  Play/pause controls included.
 *
 * The parent page calls streamCooperativeReplay() and manages the EventSource.
 * This component receives the accumulated step history and plays/pauses it.
 */

import { useState, useEffect, useRef } from "react";
import CooperativeMetricsChart from "./CooperativeMetricsChart";
import SpecializationChart from "./SpecializationChart";
import ContributionVarianceChart from "./ContributionVarianceChart";
import {
  streamCooperativeReplay,
  CooperativeStepMetric,
  CooperativeEpisodeSummary,
} from "@/lib/api";
import CooperativeRunSummary from "./CooperativeRunSummary";

interface Props {
  runId: string;
}

export default function CooperativeReplayView({ runId }: Props) {
  const [history, setHistory] = useState<CooperativeStepMetric[]>([]);
  const [displayedHistory, setDisplayedHistory] = useState<CooperativeStepMetric[]>([]);
  const [summary, setSummary] = useState<CooperativeEpisodeSummary | null>(null);
  const [playing, setPlaying] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);

  const esRef = useRef<EventSource | null>(null);

  // Load all frames on mount
  useEffect(() => {
    if (!runId) return;
    setHistory([]);
    setDisplayedHistory([]);
    setSummary(null);
    setLoaded(false);
    setError(null);
    setCurrentFrame(0);

    const frames: CooperativeStepMetric[][] = [];

    const es = streamCooperativeReplay(
      runId,
      (msg) => {
        if (msg.type === "step") {
          frames.push(msg.metrics);
        } else if (msg.type === "done") {
          if (msg.episode_summary) setSummary(msg.episode_summary);
          // Flatten all frames into history
          setHistory(frames.flat());
          setLoaded(true);
        }
      },
      () => {
        setHistory(frames.flat());
        setLoaded(true);
      },
    );
    esRef.current = es;

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [runId]);

  // Play animation: reveal frames one at a time
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!loaded) return;

    // Group history by step
    const steps = Array.from(new Set(history.map((m) => m.step))).sort(
      (a, b) => a - b,
    );

    if (playing) {
      intervalRef.current = setInterval(() => {
        setCurrentFrame((prev) => {
          const next = prev + 1;
          if (next >= steps.length) {
            setPlaying(false);
            return prev;
          }
          const visibleSteps = new Set(steps.slice(0, next + 1));
          setDisplayedHistory(history.filter((m) => visibleSteps.has(m.step)));
          return next;
        });
      }, 50);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, loaded, history]);

  // When not playing, show up to currentFrame
  useEffect(() => {
    if (!loaded) return;
    const steps = Array.from(new Set(history.map((m) => m.step))).sort(
      (a, b) => a - b,
    );
    const visibleSteps = new Set(steps.slice(0, currentFrame + 1));
    setDisplayedHistory(history.filter((m) => visibleSteps.has(m.step)));
  }, [currentFrame, loaded, history]);

  const totalFrames = Array.from(new Set(history.map((m) => m.step))).length;

  const btnStyle = (active: boolean): React.CSSProperties => ({
    padding: "6px 16px",
    fontSize: 12,
    fontWeight: 500,
    border: `1px solid ${active ? "var(--accent)" : "var(--bg-border)"}`,
    borderRadius: 6,
    background: active ? "var(--accent)" : "transparent",
    color: active ? "white" : "var(--text-secondary)",
    cursor: "pointer",
    transition: "all 120ms",
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Status / controls */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        {!loaded && (
          <span style={{ fontSize: 13, color: "var(--text-tertiary)" }}>
            Loading replay…
          </span>
        )}

        {loaded && (
          <>
            <button
              onClick={() => {
                if (playing) {
                  setPlaying(false);
                } else {
                  if (currentFrame >= totalFrames - 1) {
                    setCurrentFrame(0);
                    setDisplayedHistory([]);
                  }
                  setPlaying(true);
                }
              }}
              style={btnStyle(playing)}
            >
              {playing ? "⏸ Pause" : currentFrame >= totalFrames - 1 ? "↺ Replay" : "▶ Play"}
            </button>

            <button
              onClick={() => {
                setPlaying(false);
                setCurrentFrame(totalFrames - 1);
                setDisplayedHistory(history);
              }}
              style={btnStyle(false)}
            >
              ⏭ Show all
            </button>

            <span style={{ fontSize: 11, color: "var(--text-tertiary)" }}>
              Step {currentFrame + 1} / {totalFrames}
            </span>

            {/* Scrubber */}
            <input
              type="range"
              min={0}
              max={Math.max(0, totalFrames - 1)}
              value={currentFrame}
              onChange={(e) => {
                setPlaying(false);
                setCurrentFrame(Number(e.target.value));
              }}
              style={{ flex: 1, minWidth: 100, accentColor: "var(--accent)" }}
            />
          </>
        )}
        {error && (
          <span style={{ fontSize: 12, color: "#ef4444" }}>{error}</span>
        )}
      </div>

      {/* Charts */}
      <CooperativeMetricsChart history={displayedHistory} />

      {displayedHistory.length > 0 && (
        <>
          <div
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--bg-border)",
              borderRadius: 6,
              padding: "16px 20px",
            }}
          >
            <SpecializationChart history={displayedHistory} />
          </div>
          <div
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--bg-border)",
              borderRadius: 6,
              padding: "16px 20px",
            }}
          >
            <ContributionVarianceChart history={displayedHistory} />
          </div>
        </>
      )}

      {/* Episode summary (shown after all frames loaded) */}
      {summary && (
        <div>
          <div
            style={{
              fontSize: 11,
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              color: "var(--text-tertiary)",
              marginBottom: 12,
            }}
          >
            Episode Summary
          </div>
          <CooperativeRunSummary summary={summary} />
        </div>
      )}
    </div>
  );
}
