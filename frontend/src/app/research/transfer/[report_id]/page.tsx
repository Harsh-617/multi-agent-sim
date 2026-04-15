"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getTransferReport, TransferReport } from "@/lib/api";

/* eslint-disable @typescript-eslint/no-explicit-any */

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

function MetaRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <>
      <dt style={{ color: "var(--text-secondary)", fontSize: 13 }}>{label}</dt>
      <dd style={{ fontFamily: "monospace", fontSize: 12, color: "var(--text-primary)" }}>{value}</dd>
    </>
  );
}

export default function TransferReportDetailPage() {
  const params = useParams<{ report_id: string }>();
  const reportId = params.report_id;

  const [data, setData] = useState<TransferReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!reportId) return;
    getTransferReport(reportId)
      .then(setData)
      .catch((e: any) => setError(e.message ?? String(e)))
      .finally(() => setLoading(false));
  }, [reportId]);

  if (loading) {
    return (
      <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, paddingTop: 96, color: "var(--text-primary)" }}>
        <p>Loading…</p>
      </main>
    );
  }

  if (error) {
    return (
      <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, paddingTop: 96 }}>
        <p style={{ color: "#f87171" }}>Error: {error}</p>
        <Link href="/research" style={{ color: "var(--text-secondary)", fontSize: 13 }}>← Research</Link>
      </main>
    );
  }

  if (!data) {
    return (
      <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, paddingTop: 96, color: "var(--text-primary)" }}>
        <p>No data.</p>
      </main>
    );
  }

  const metricName = PRIMARY_METRIC_NAME[data.target_archetype];
  const vsSign = data.vs_baseline_pct != null ? (data.vs_baseline_pct >= 0 ? "+" : "") : "";
  const obsDiff = data.source_obs_dim - data.target_obs_dim;
  const hasMismatch = data.source_obs_dim !== data.target_obs_dim;

  return (
    <main style={{ maxWidth: 1024, margin: "0 auto", padding: 32, paddingTop: 80, background: "var(--bg-base)" }}>

      {/* Breadcrumb */}
      <div style={{ marginBottom: 20, display: "flex", gap: 8, fontSize: 13, color: "var(--text-secondary)" }}>
        <Link href="/research" style={{ color: "var(--text-secondary)", textDecoration: "none" }}>
          ← Research
        </Link>
        <span>/</span>
        <span style={{ color: "var(--text-tertiary)" }}>Transfer</span>
        <span>/</span>
        <span style={{ color: "var(--text-tertiary)", fontFamily: "monospace", fontSize: 11 }}>
          {reportId}
        </span>
      </div>

      {/* Title */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <span style={{
            fontSize: 11,
            borderRadius: 4,
            padding: "2px 8px",
            background: "rgba(20,184,166,0.1)",
            color: "#14b8a6",
            border: "1px solid rgba(20,184,166,0.2)",
            fontWeight: 500,
          }}>
            Transfer
          </span>
          <ArchetypeBadge archetype={data.source_archetype} />
          <span style={{ fontSize: 12, color: "var(--text-tertiary)" }}>→</span>
          <ArchetypeBadge archetype={data.target_archetype} />
        </div>
        <h1 style={{ fontSize: 18, fontWeight: 600, fontFamily: "monospace", color: "var(--text-primary)", margin: 0 }}>
          {reportId}
        </h1>
        {data.timestamp && (
          <p style={{ fontSize: 12, color: "var(--text-tertiary)", margin: "4px 0 0" }}>
            {data.timestamp}
          </p>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

        {/* Source agent card */}
        <section>
          <h2 style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)", marginBottom: 12 }}>
            Source Agent
          </h2>
          <div style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--bg-border)",
            borderRadius: 8,
            padding: 20,
          }}>
            <div style={{ marginBottom: 10 }}>
              <ArchetypeBadge archetype={data.source_archetype} />
            </div>
            <dl style={{ display: "grid", gridTemplateColumns: "1fr 2fr", columnGap: 24, rowGap: 6 }}>
              <MetaRow label="Member ID" value={data.source_member_id} />
              {data.source_strategy_label && (
                <MetaRow label="Strategy Label" value={data.source_strategy_label} />
              )}
              <MetaRow label="Elo Rating" value={
                <span style={{ color: "#14b8a6", fontWeight: 600 }}>{data.source_elo != null ? data.source_elo.toFixed(1) : "—"}</span>
              } />
              <MetaRow label="Obs Dim" value={`${data.source_obs_dim}d`} />
            </dl>
          </div>
        </section>

        {/* Target environment card */}
        <section>
          <h2 style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)", marginBottom: 12 }}>
            Target Environment
          </h2>
          <div style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--bg-border)",
            borderRadius: 8,
            padding: 20,
          }}>
            <div style={{ marginBottom: 10 }}>
              <ArchetypeBadge archetype={data.target_archetype} />
            </div>
            <dl style={{ display: "grid", gridTemplateColumns: "1fr 2fr", columnGap: 24, rowGap: 6 }}>
              <MetaRow label="Config Hash" value={data.target_config_hash} />
              <MetaRow label="Obs Dim" value={`${data.target_obs_dim}d`} />
              <MetaRow label="Episodes" value={String(data.episodes)} />
              <MetaRow label="Seed" value={String(data.seed)} />
            </dl>
          </div>
        </section>

        {/* Obs mismatch explanation */}
        {hasMismatch && (
          <section>
            <h2 style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)", marginBottom: 12 }}>
              Observation Space Mismatch
            </h2>
            <div style={{
              background: "rgba(249,115,22,0.07)",
              border: "1px solid rgba(249,115,22,0.2)",
              borderRadius: 8,
              padding: 16,
              fontSize: 13,
              color: "#f97316",
            }}>
              <p style={{ margin: "0 0 8px" }}>
                Source policy expects <strong>{data.source_obs_dim}</strong>d, target environment produces{" "}
                <strong>{data.target_obs_dim}</strong>d (difference: {Math.abs(obsDiff)}d).
              </p>
              <p style={{ margin: 0 }}>
                Strategy used:{" "}
                <strong>
                  {data.obs_mismatch_strategy === "pad"
                    ? "Zero-padding"
                    : data.obs_mismatch_strategy === "truncate"
                    ? "Truncation"
                    : "Matched (no mismatch)"}
                </strong>
                {data.obs_mismatch_strategy === "pad" && (
                  <> — target obs padded with zeros to reach {data.source_obs_dim}d</>
                )}
                {data.obs_mismatch_strategy === "truncate" && (
                  <> — target obs truncated from {data.target_obs_dim}d to {data.source_obs_dim}d</>
                )}
                . Results reflect raw policy generalization, not engineered compatibility.
              </p>
            </div>
          </section>
        )}

        {/* Results comparison table */}
        <section>
          <h2 style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)", marginBottom: 12 }}>
            Results
          </h2>
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
                  {["Metric", "Transferred Agent", "Random Baseline", "Delta", "vs Baseline"].map((h) => (
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
                    {data.transferred_mean != null ? data.transferred_mean.toFixed(4) : "—"}
                  </td>
                  <td style={{ padding: "10px 12px", color: "var(--text-secondary)", fontFamily: "monospace" }}>
                    {data.baseline_mean != null ? data.baseline_mean.toFixed(4) : "—"}
                  </td>
                  <td style={{
                    padding: "10px 12px",
                    fontFamily: "monospace",
                    color: data.vs_baseline_delta != null ? (data.vs_baseline_delta >= 0 ? "#14b8a6" : "#f87171") : "var(--text-secondary)",
                  }}>
                    {data.vs_baseline_delta != null
                      ? `${data.vs_baseline_delta >= 0 ? "+" : ""}${data.vs_baseline_delta.toFixed(4)}`
                      : "—"}
                  </td>
                  <td style={{
                    padding: "10px 12px",
                    fontWeight: 600,
                    fontFamily: "monospace",
                    color: data.vs_baseline_pct != null ? (data.vs_baseline_pct >= 0 ? "#14b8a6" : "#f87171") : "var(--text-secondary)",
                  }}>
                    {data.vs_baseline_pct != null ? `${vsSign}${data.vs_baseline_pct.toFixed(1)}%` : "—"}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Interpretation summary */}
          <p style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 12 }}>
            Transferred agent performed{" "}
            {data.vs_baseline_pct != null ? (
              <span style={{ color: data.vs_baseline_pct >= 0 ? "#14b8a6" : "#f87171", fontWeight: 600 }}>
                {Math.abs(data.vs_baseline_pct).toFixed(1)}%{" "}
                {data.vs_baseline_pct >= 0 ? "above" : "below"}
              </span>
            ) : (
              <span style={{ color: "var(--text-secondary)", fontWeight: 600 }}>—</span>
            )}
            {" "}random baseline on {metricName} in the {ARCHETYPE_LABELS[data.target_archetype]} environment.
          </p>
        </section>

        {/* Per-episode breakdown */}
        {data.transferred_results && data.transferred_results.length > 0 && (
          <section>
            <h2 style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)", marginBottom: 12 }}>
              Per-Episode Breakdown
            </h2>
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
                    {["Episode", `Transferred (${metricName})`, `Baseline (${metricName})`].map((h) => (
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
                  {data.transferred_results.map((t, idx) => {
                    const b = data.baseline_results[idx];
                    return (
                      <tr key={t.episode} style={{ borderBottom: "1px solid var(--bg-border)" }}>
                        <td style={{ padding: "8px 12px", color: "var(--text-tertiary)", fontFamily: "monospace" }}>
                          {t.episode}
                        </td>
                        <td style={{ padding: "8px 12px", color: "var(--text-primary)", fontFamily: "monospace" }}>
                          {(t as any)[metricName] != null ? ((t as any)[metricName] as number).toFixed(4) : "—"}
                        </td>
                        <td style={{ padding: "8px 12px", color: "var(--text-secondary)", fontFamily: "monospace" }}>
                          {b && (b as any)[metricName] != null ? ((b as any)[metricName] as number).toFixed(4) : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
