"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  listConfigs,
  getConfigDetail,
  startTransferExperiment,
  getTransferStatus,
  getTransferReport,
  ConfigListItem,
  TransferStatus,
  TransferReport,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

type Archetype = "mixed" | "competitive" | "cooperative";

const ARCHETYPE_LABELS: Record<Archetype, string> = {
  mixed: "Resource Sharing Arena",
  competitive: "Head-to-Head Strategy",
  cooperative: "Cooperative Task Arena",
};

const ARCHETYPE_BADGE_COLORS: Record<Archetype, { bg: string; fg: string; border: string }> = {
  mixed: { bg: "rgba(20,184,166,0.1)", fg: "#14b8a6", border: "rgba(20,184,166,0.2)" },
  competitive: { bg: "rgba(249,115,22,0.1)", fg: "#f97316", border: "rgba(249,115,22,0.2)" },
  cooperative: { bg: "rgba(139,92,246,0.1)", fg: "#8b5cf6", border: "rgba(139,92,246,0.2)" },
};

const PRIMARY_METRIC_NAME: Record<Archetype, string> = {
  mixed: "cooperation_rate",
  competitive: "normalized_rank",
  cooperative: "completion_ratio",
};

const STAGE_LABELS: Record<string, string> = {
  pending: "Queued...",
  running_transfer: "Running transferred agent...",
  running_baseline: "Running random baseline...",
  saving: "Saving report...",
  done: "Complete",
  error: "Error",
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface Props {
  sourceArchetype: Archetype;
  sourceMemberId: string;
  sourceStrategyLabel: string | null;
  sourceElo: number | null;
}

// ---------------------------------------------------------------------------
// Small sub-components
// ---------------------------------------------------------------------------

function ArchetypeBadge({ archetype }: { archetype: Archetype }) {
  const c = ARCHETYPE_BADGE_COLORS[archetype];
  return (
    <span style={{
      fontSize: 11,
      borderRadius: 4,
      padding: "2px 8px",
      background: c.bg,
      color: c.fg,
      border: `1px solid ${c.border}`,
    }}>
      {ARCHETYPE_LABELS[archetype]}
    </span>
  );
}

function InfoCard({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      background: "#0d1f1f",
      border: "1px solid #14b8a6",
      borderRadius: 6,
      padding: "10px 14px",
    }}>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function TransferExperiment({
  sourceArchetype,
  sourceMemberId,
  sourceStrategyLabel,
  sourceElo,
}: Props) {
  // Derive the two non-source archetypes in a stable order
  const targetOptions = (["mixed", "competitive", "cooperative"] as Archetype[]).filter(
    (a) => a !== sourceArchetype,
  );

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const [targetArchetype, setTargetArchetype] = useState<Archetype>(targetOptions[0]);
  const [allConfigs, setAllConfigs] = useState<ConfigListItem[]>([]);
  const [filteredConfigs, setFilteredConfigs] = useState<ConfigListItem[]>([]);
  const [filteringConfigs, setFilteringConfigs] = useState(false);
  const [targetConfigId, setTargetConfigId] = useState("");
  const [episodes, setEpisodes] = useState(5);
  const [seed, setSeed] = useState(42);

  const [running, setRunning] = useState(false);
  const [transferId, setTransferId] = useState<string | null>(null);
  const [status, setStatus] = useState<TransferStatus | null>(null);
  const [report, setReport] = useState<TransferReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---------------------------------------------------------------------------
  // Fetch all configs once
  // ---------------------------------------------------------------------------
  useEffect(() => {
    listConfigs()
      .then(setAllConfigs)
      .catch(() => {/* silently ignore */});
  }, []);

  // ---------------------------------------------------------------------------
  // Filter configs when targetArchetype or allConfigs changes
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (allConfigs.length === 0) return;
    setFilteringConfigs(true);
    setFilteredConfigs([]);
    setTargetConfigId("");

    const envType = targetArchetype; // "mixed" | "competitive" | "cooperative"

    Promise.all(
      allConfigs.map(async (c) => {
        try {
          const detail = await getConfigDetail(c.config_id);
          const identity = detail.identity as Record<string, unknown> | undefined;
          if (identity?.environment_type === envType) return c;
        } catch {
          /* skip */
        }
        return null;
      }),
    )
      .then((results) => {
        const valid = results.filter(Boolean) as ConfigListItem[];
        setFilteredConfigs(valid);
        setTargetConfigId(valid[0]?.config_id ?? "");
      })
      .finally(() => setFilteringConfigs(false));
  }, [targetArchetype, allConfigs]);

  // ---------------------------------------------------------------------------
  // Polling
  // ---------------------------------------------------------------------------
  function stopPolling() {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  useEffect(() => {
    return () => stopPolling();
  }, []);

  async function pollStatus(tid: string) {
    try {
      const s = await getTransferStatus(tid);
      setStatus(s);
      if (s.status === "done") {
        stopPolling();
        setRunning(false);
        if (s.report_id) {
          try {
            const r = await getTransferReport(s.report_id);
            setReport(r);
          } catch {
            setError("Experiment complete but could not load report.");
          }
        }
      } else if (s.status === "error") {
        stopPolling();
        setRunning(false);
        setError(s.error ?? "Transfer experiment failed.");
      }
    } catch {
      // keep polling — transient network error
    }
  }

  // ---------------------------------------------------------------------------
  // Run handler
  // ---------------------------------------------------------------------------
  async function handleRun() {
    if (!sourceMemberId || !targetConfigId) return;
    setError(null);
    setReport(null);
    setStatus(null);
    setRunning(true);

    try {
      const { transfer_id } = await startTransferExperiment({
        source_member_id: sourceMemberId,
        source_archetype: sourceArchetype,
        target_archetype: targetArchetype,
        target_config_id: targetConfigId,
        episodes,
        seed,
      });
      setTransferId(transfer_id);
      stopPolling();
      pollRef.current = setInterval(() => pollStatus(transfer_id), 2000);
    } catch (e) {
      setRunning(false);
      setError(String(e));
    }
  }

  // ---------------------------------------------------------------------------
  // Early-exit: no champion
  // ---------------------------------------------------------------------------
  if (!sourceMemberId) {
    return (
      <div style={{ color: "#555555", fontSize: 13, padding: "24px 0" }}>
        No champion available — run the pipeline first.
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Derived display values
  // ---------------------------------------------------------------------------
  const metricName = PRIMARY_METRIC_NAME[targetArchetype];
  const canRun = !running && filteredConfigs.length > 0 && !filteringConfigs;

  const vsSign = report
    ? report.vs_baseline_pct >= 0 ? "+" : ""
    : "";

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Form */}
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", gap: 12 }}>

        {/* Target environment */}
        <div>
          <label style={{ display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>
            Target Environment
          </label>
          <select
            value={targetArchetype}
            onChange={(e) => setTargetArchetype(e.target.value as Archetype)}
            disabled={running}
            style={{
              border: "1px solid var(--bg-border)",
              borderRadius: 4,
              padding: "4px 8px",
              fontSize: 13,
              background: "var(--bg-base)",
              color: "var(--text-primary)",
              cursor: running ? "not-allowed" : "pointer",
            }}
          >
            {targetOptions.map((a) => (
              <option key={a} value={a}>
                {ARCHETYPE_LABELS[a]}
              </option>
            ))}
          </select>
        </div>

        {/* Target config */}
        <div>
          <label style={{ display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>
            Config
          </label>
          {filteringConfigs ? (
            <span style={{ fontSize: 12, color: "var(--text-tertiary)" }}>Loading configs...</span>
          ) : filteredConfigs.length === 0 ? (
            <span style={{ fontSize: 12, color: "#f87171" }}>No {targetArchetype} configs</span>
          ) : (
            <select
              value={targetConfigId}
              onChange={(e) => setTargetConfigId(e.target.value)}
              disabled={running}
              style={{
                border: "1px solid var(--bg-border)",
                borderRadius: 4,
                padding: "4px 8px",
                fontSize: 13,
                background: "var(--bg-base)",
                color: "var(--text-primary)",
                maxWidth: 240,
                cursor: running ? "not-allowed" : "pointer",
              }}
            >
              {filteredConfigs.map((c) => (
                <option key={c.config_id} value={c.config_id}>
                  {c.config_id} (agents={c.num_agents}, steps={c.max_steps})
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Episodes */}
        <div>
          <label style={{ display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>
            Episodes
          </label>
          <input
            type="number"
            min={1}
            max={20}
            value={episodes}
            onChange={(e) => setEpisodes(Math.min(20, Math.max(1, Number(e.target.value))))}
            disabled={running}
            style={{
              border: "1px solid var(--bg-border)",
              borderRadius: 4,
              padding: "4px 8px",
              fontSize: 13,
              width: 64,
              background: "var(--bg-base)",
              color: "var(--text-primary)",
            }}
          />
        </div>

        {/* Seed */}
        <div>
          <label style={{ display: "block", fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>
            Seed
          </label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            disabled={running}
            style={{
              border: "1px solid var(--bg-border)",
              borderRadius: 4,
              padding: "4px 8px",
              fontSize: 13,
              width: 80,
              background: "var(--bg-base)",
              color: "var(--text-primary)",
            }}
          />
        </div>

        {/* Run button */}
        <button
          onClick={handleRun}
          disabled={!canRun}
          style={{
            padding: "4px 14px",
            background: "#14b8a6",
            color: "#fff",
            borderRadius: 6,
            fontSize: 13,
            border: "none",
            cursor: canRun ? "pointer" : "not-allowed",
            opacity: canRun ? 1 : 0.5,
            fontWeight: 500,
          }}
        >
          {running ? "Running..." : "Run Transfer"}
        </button>
      </div>

      {/* Obs mismatch note */}
      <p style={{ fontSize: 12, color: "#666666", margin: 0 }}>
        Observation spaces may differ between archetypes.
        Transfer uses truncation/padding to bridge the gap — results reflect raw policy generalization.
      </p>

      {/* Status while running */}
      {running && status && (
        <div style={{
          background: "var(--bg-elevated)",
          border: "1px solid var(--bg-border)",
          borderRadius: 6,
          padding: "10px 14px",
          fontSize: 13,
          color: "var(--text-secondary)",
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}>
          <span style={{
            display: "inline-block",
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: "#14b8a6",
            animation: "pulse 1.2s ease-in-out infinite",
          }} />
          {STAGE_LABELS[status.status] ?? status.status}
          {transferId && (
            <span style={{ fontSize: 11, color: "#555555", fontFamily: "monospace", marginLeft: 4 }}>
              {transferId.slice(0, 16)}…
            </span>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <p style={{ fontSize: 13, color: "#f87171", margin: 0 }}>{error}</p>
      )}

      {/* Results panel */}
      {report && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 4 }}>

          {/* Source agent + target env cards */}
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <InfoCard>
              <div style={{ fontSize: 11, color: "#888888", marginBottom: 6 }}>Source Agent</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div><ArchetypeBadge archetype={report.source_archetype} /></div>
                <div style={{ fontSize: 12, fontFamily: "monospace", color: "#ededed", marginTop: 2 }}>
                  {report.source_member_id}
                </div>
                {report.source_strategy_label && (
                  <div style={{ fontSize: 11, color: "#888888" }}>{report.source_strategy_label}</div>
                )}
                <div style={{ fontSize: 12, color: "#ededed" }}>
                  Elo: <span style={{ color: "#14b8a6", fontWeight: 600 }}>{report.source_elo.toFixed(1)}</span>
                </div>
              </div>
            </InfoCard>

            <InfoCard>
              <div style={{ fontSize: 11, color: "#888888", marginBottom: 6 }}>Target Environment</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div><ArchetypeBadge archetype={report.target_archetype} /></div>
                <div style={{ fontSize: 11, fontFamily: "monospace", color: "#888888", marginTop: 2 }}>
                  {report.target_config_hash}
                </div>
              </div>
            </InfoCard>
          </div>

          {/* Obs mismatch note */}
          {report.source_obs_dim !== report.target_obs_dim && (
            <div style={{
              background: "rgba(249,115,22,0.07)",
              border: "1px solid rgba(249,115,22,0.2)",
              borderRadius: 6,
              padding: "8px 12px",
              fontSize: 12,
              color: "#f97316",
            }}>
              Source expects {report.source_obs_dim}d, target produces {report.target_obs_dim}d
              {" "}— {report.obs_mismatch_strategy === "pad" ? "zero-padded" : report.obs_mismatch_strategy === "truncate" ? "truncated" : "matched"} to bridge gap.
            </div>
          )}

          {/* Results table */}
          <div style={{ overflowX: "auto" }}>
            <table style={{
              width: "100%",
              borderCollapse: "collapse",
              background: "var(--bg-surface)",
              border: "1px solid var(--bg-border)",
              borderRadius: 8,
              fontSize: 13,
            }}>
              <thead>
                <tr>
                  {["Metric", "Transferred Agent", "Random Baseline", "vs Baseline"].map((h) => (
                    <th key={h} style={{
                      background: "var(--bg-elevated)",
                      color: "var(--text-secondary)",
                      fontSize: 11,
                      textTransform: "uppercase" as const,
                      padding: "8px 12px",
                      textAlign: "left" as const,
                      fontWeight: 500,
                    }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: "10px 12px", color: "var(--text-secondary)", fontFamily: "monospace" }}>
                    {metricName}
                  </td>
                  <td style={{ padding: "10px 12px", color: "var(--text-primary)", fontWeight: 600, fontFamily: "monospace" }}>
                    {report.transferred_mean.toFixed(4)}
                  </td>
                  <td style={{ padding: "10px 12px", color: "var(--text-secondary)", fontFamily: "monospace" }}>
                    {report.baseline_mean.toFixed(4)}
                  </td>
                  <td style={{
                    padding: "10px 12px",
                    fontWeight: 600,
                    fontFamily: "monospace",
                    color: report.vs_baseline_pct >= 0 ? "#14b8a6" : "#f87171",
                  }}>
                    {vsSign}{report.vs_baseline_pct.toFixed(1)}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Interpretation */}
          <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: 0 }}>
            Transferred agent performed{" "}
            <span style={{ color: report.vs_baseline_pct >= 0 ? "#14b8a6" : "#f87171", fontWeight: 600 }}>
              {Math.abs(report.vs_baseline_pct).toFixed(1)}%{" "}
              {report.vs_baseline_pct >= 0 ? "above" : "below"}
            </span>
            {" "}random baseline.
          </p>

          {/* Report link */}
          {status?.report_id && (
            <div>
              <Link
                href={`/research/transfer/${encodeURIComponent(status.report_id)}`}
                style={{
                  fontSize: 13,
                  color: "#14b8a6",
                  textDecoration: "underline",
                  fontWeight: 500,
                }}
              >
                View full report →
              </Link>
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}
