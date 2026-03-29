"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ReportEntry {
  report_id: string;
  timestamp: string;
  archetype: "Resource Sharing" | "Head-to-Head";
}

type ArchetypeFilter = "All" | "Resource Sharing" | "Head-to-Head";
type TypeFilter = "All" | "Robustness" | "Strategy" | "Benchmark";
type SortMode = "latest" | "robustness";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function inferType(id: string): "Robustness" | "Strategy" | "Benchmark" | "Other" {
  if (id.startsWith("robust_")) return "Robustness";
  if (id.startsWith("eval_")) return "Strategy";
  if (id.startsWith("benchmark_")) return "Benchmark";
  return "Other";
}

function formatTimestamp(ts: string): string {
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  const day = d.getDate();
  const month = d.toLocaleString("en-US", { month: "short" });
  const year = d.getFullYear();
  const hours = d.getHours();
  const minutes = d.getMinutes().toString().padStart(2, "0");
  const ampm = hours >= 12 ? "pm" : "am";
  const h = hours % 12 || 12;
  return `${day} ${month} ${year}, ${h}:${minutes} ${ampm}`;
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const pillBase: React.CSSProperties = {
  borderRadius: 9999,
  padding: "4px 12px",
  fontSize: 12,
  cursor: "pointer",
  border: "1px solid var(--bg-border)",
  transition: "all 150ms",
};

const pillActive: React.CSSProperties = {
  ...pillBase,
  background: "var(--accent)",
  color: "white",
  borderColor: "var(--accent)",
};

const pillInactive: React.CSSProperties = {
  ...pillBase,
  background: "var(--bg-elevated)",
  color: "var(--text-secondary)",
};

const filterLabel: React.CSSProperties = {
  fontSize: 11,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: "var(--text-tertiary)",
  marginBottom: 8,
};

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function Pill({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button onClick={onClick} style={active ? pillActive : pillInactive}>
      {label}
    </button>
  );
}

function ArchetypeBadge({ archetype }: { archetype: "Resource Sharing" | "Head-to-Head" }) {
  const isTeal = archetype === "Resource Sharing";
  return (
    <span
      style={{
        fontSize: 11,
        borderRadius: 4,
        padding: "2px 8px",
        background: isTeal ? "var(--accent-subtle)" : "rgba(249,115,22,0.1)",
        color: isTeal ? "var(--accent)" : "#f97316",
        border: `1px solid ${isTeal ? "var(--accent-border)" : "rgba(249,115,22,0.2)"}`,
      }}
    >
      {archetype}
    </span>
  );
}

function TypeBadge({ type }: { type: string }) {
  return (
    <span
      style={{
        fontSize: 11,
        borderRadius: 4,
        padding: "2px 8px",
        background: "var(--bg-elevated)",
        color: "var(--text-tertiary)",
        border: "1px solid var(--bg-border)",
      }}
    >
      {type}
    </span>
  );
}

function ReportCard({ report }: { report: ReportEntry }) {
  const [hover, setHover] = useState(false);
  const type = inferType(report.report_id);

  return (
    <div
      style={{
        background: "var(--bg-surface)",
        border: "1px solid var(--bg-border)",
        borderRadius: 8,
        padding: 20,
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Badges */}
      <div style={{ display: "flex", gap: 6 }}>
        <ArchetypeBadge archetype={report.archetype} />
        <TypeBadge type={type} />
      </div>

      {/* Report ID */}
      <div
        style={{
          fontSize: 13,
          fontFamily: "var(--font-mono)",
          color: "var(--text-secondary)",
          marginTop: 8,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {report.report_id}
      </div>

      {/* Timestamp */}
      <div style={{ fontSize: 12, color: "var(--text-tertiary)", marginTop: 4 }}>
        {report.timestamp ? formatTimestamp(report.timestamp) : "—"}
      </div>

      {/* Open button */}
      <Link
        href={`/research/${encodeURIComponent(report.report_id)}`}
        style={{ textDecoration: "none", marginTop: 16 }}
      >
        <button
          onMouseEnter={() => setHover(true)}
          onMouseLeave={() => setHover(false)}
          style={{
            background: "transparent",
            border: `1px solid ${hover ? "var(--accent)" : "var(--bg-border)"}`,
            color: hover ? "var(--text-primary)" : "var(--text-secondary)",
            padding: "6px 14px",
            borderRadius: 6,
            fontSize: 12,
            cursor: "pointer",
            transition: "all 150ms",
          }}
        >
          Open →
        </button>
      </Link>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function ResearchPage() {
  const [mixedReports, setMixedReports] = useState<ReportEntry[]>([]);
  const [compReports, setCompReports] = useState<ReportEntry[]>([]);
  const [loadingMixed, setLoadingMixed] = useState(true);
  const [loadingComp, setLoadingComp] = useState(true);
  const [errorMixed, setErrorMixed] = useState(false);
  const [errorComp, setErrorComp] = useState(false);

  const [archetypeFilter, setArchetypeFilter] = useState<ArchetypeFilter>("All");
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("All");
  const [sort, setSort] = useState<SortMode>("latest");

  useEffect(() => {
    fetch("/api/reports")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: { report_id: string; timestamp: string }[]) =>
        setMixedReports(
          data.map((d) => ({
            report_id: d.report_id,
            timestamp: d.timestamp,
            archetype: "Resource Sharing" as const,
          })),
        ),
      )
      .catch(() => setErrorMixed(true))
      .finally(() => setLoadingMixed(false));

    fetch("/api/competitive/reports")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: { report_id: string; timestamp: string }[]) =>
        setCompReports(
          data.map((d) => ({
            report_id: d.report_id,
            timestamp: d.timestamp,
            archetype: "Head-to-Head" as const,
          })),
        ),
      )
      .catch(() => setErrorComp(true))
      .finally(() => setLoadingComp(false));
  }, []);

  const loading = loadingMixed || loadingComp;
  const bothErrored = errorMixed && errorComp;

  // Merge and filter
  let merged = [...mixedReports, ...compReports];

  if (archetypeFilter !== "All") {
    merged = merged.filter((r) => r.archetype === archetypeFilter);
  }
  if (typeFilter !== "All") {
    merged = merged.filter((r) => inferType(r.report_id) === typeFilter);
  }

  // Sort
  // TODO: "Highest robustness score" sort requires fetching individual report
  // data for scores. For now, both modes sort by timestamp descending.
  merged.sort((a, b) => {
    const ta = new Date(a.timestamp).getTime() || 0;
    const tb = new Date(b.timestamp).getTime() || 0;
    return tb - ta;
  });

  return (
    <main
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        padding: "48px 24px",
        paddingTop: 96,
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1
          style={{
            fontSize: 24,
            fontWeight: 500,
            color: "var(--text-primary)",
            margin: 0,
          }}
        >
          Research
        </h1>
        <p
          style={{
            fontSize: 14,
            color: "var(--text-secondary)",
            margin: "8px 0 0",
          }}
        >
          Robustness reports, strategy analysis, and evaluation results
        </p>
      </div>

      {/* Filter bar */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 24,
          marginBottom: 32,
          alignItems: "flex-end",
        }}
      >
        {/* Archetype filter */}
        <div>
          <div style={filterLabel}>Environment</div>
          <div style={{ display: "flex", gap: 6 }}>
            {(["All", "Resource Sharing", "Head-to-Head"] as ArchetypeFilter[]).map((v) => (
              <Pill
                key={v}
                label={v}
                active={archetypeFilter === v}
                onClick={() => setArchetypeFilter(v)}
              />
            ))}
          </div>
        </div>

        {/* Type filter */}
        <div>
          <div style={filterLabel}>Type</div>
          <div style={{ display: "flex", gap: 6 }}>
            {(["All", "Robustness", "Strategy", "Benchmark"] as TypeFilter[]).map((v) => (
              <Pill
                key={v}
                label={v}
                active={typeFilter === v}
                onClick={() => setTypeFilter(v)}
              />
            ))}
          </div>
        </div>

        {/* Sort */}
        <div>
          <div style={filterLabel}>Sort</div>
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortMode)}
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--bg-border)",
              color: "var(--text-primary)",
              borderRadius: 6,
              padding: "4px 8px",
              fontSize: 12,
              height: 28,
              cursor: "pointer",
            }}
          >
            <option value="latest">Latest first</option>
            <option value="robustness">Highest robustness score</option>
          </select>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <p style={{ fontSize: 13, color: "var(--text-tertiary)" }}>Loading reports...</p>
      )}

      {/* Error state */}
      {!loading && bothErrored && (
        <p style={{ fontSize: 13, color: "var(--text-tertiary)" }}>Could not load reports</p>
      )}

      {/* Empty state */}
      {!loading && !bothErrored && merged.length === 0 && (
        <p
          style={{
            fontSize: 14,
            color: "var(--text-tertiary)",
            textAlign: "center",
            padding: "60px 0",
          }}
        >
          No reports found
        </p>
      )}

      {/* Card grid */}
      {!loading && merged.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, 1fr)",
            gap: 16,
          }}
        >
          {merged.map((r) => (
            <ReportCard key={r.report_id} report={r} />
          ))}
        </div>
      )}

      {/* Responsive: single column on narrow viewports */}
      <style>{`
        @media (max-width: 640px) {
          main > div:last-of-type {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </main>
  );
}
