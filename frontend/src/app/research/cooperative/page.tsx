"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  CooperativeReportListItem,
  getCooperativeReports,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Filter state
// ---------------------------------------------------------------------------

type FilterState = {
  date: string;
  configHash: string;
  kind: string;
};

// ---------------------------------------------------------------------------
// Report card
// ---------------------------------------------------------------------------

function ReportCard({ report }: { report: CooperativeReportListItem }) {
  const kindLabel =
    report.kind === "cooperative_robust"
      ? "Robustness"
      : report.kind === "cooperative_eval"
      ? "Eval"
      : report.kind;

  const kindColor =
    report.kind === "cooperative_robust" ? "#14b8a6"
    : report.kind === "cooperative_eval" ? "#3b82f6"
    : "#8b5cf6";

  const ts = report.timestamp
    ? new Date(report.timestamp).toLocaleString()
    : "—";

  return (
    <div style={{
      background: "#111111",
      border: "1px solid #1e1e1e",
      borderLeft: `3px solid ${kindColor}`,
      borderRadius: 6,
      padding: "14px 16px",
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
        <div>
          <span style={{
            fontSize: 11,
            background: kindColor + "22",
            color: kindColor,
            padding: "2px 6px",
            borderRadius: 3,
            marginRight: 8,
          }}>
            {kindLabel}
          </span>
          <span style={{ fontSize: 11, color: "#555555" }}>{ts}</span>
        </div>
        <Link
          href={`/research/cooperative/${encodeURIComponent(report.report_id)}`}
          style={{
            fontSize: 11,
            color: "#14b8a6",
            textDecoration: "none",
            padding: "3px 8px",
            border: "1px solid #14b8a6",
            borderRadius: 3,
          }}
        >
          View →
        </Link>
      </div>

      {/* Report ID */}
      <div style={{
        fontFamily: "monospace",
        fontSize: 11,
        color: "#666666",
        marginBottom: 8,
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
      }}>
        {report.report_id}
      </div>

      {/* Metrics row */}
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        {report.mean_completion_ratio != null && (
          <div>
            <div style={{ fontSize: 10, color: "#555555" }}>Mean CR</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#ededed" }}>
              {report.mean_completion_ratio.toFixed(4)}
            </div>
          </div>
        )}
        {report.robustness_score != null && (
          <div>
            <div style={{ fontSize: 10, color: "#555555" }}>Robustness Score</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#14b8a6" }}>
              {report.robustness_score.toFixed(4)}
            </div>
          </div>
        )}
        <div>
          <div style={{ fontSize: 10, color: "#555555" }}>Config Hash</div>
          <div style={{ fontSize: 11, fontFamily: "monospace", color: "#666666" }}>
            {report.config_hash || "—"}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function CooperativeResearchPage() {
  const [reports, setReports] = useState<CooperativeReportListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    date: "",
    configHash: "",
    kind: "",
  });
  const [sortKey, setSortKey] = useState<"timestamp" | "robustness_score" | "mean_completion_ratio">(
    "timestamp",
  );

  useEffect(() => {
    getCooperativeReports()
      .then((r) => { setReports(r); setLoading(false); })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, []);

  // Filter
  const filtered = reports.filter((r) => {
    if (filters.date && r.timestamp) {
      const ts = r.timestamp.slice(0, 10);
      if (!ts.includes(filters.date)) return false;
    }
    if (filters.configHash && !r.config_hash.startsWith(filters.configHash)) {
      return false;
    }
    if (filters.kind && r.kind !== filters.kind) return false;
    return true;
  });

  // Sort
  const sorted = [...filtered].sort((a, b) => {
    if (sortKey === "timestamp") {
      return (b.timestamp || "").localeCompare(a.timestamp || "");
    }
    if (sortKey === "robustness_score") {
      return (b.robustness_score ?? -1) - (a.robustness_score ?? -1);
    }
    if (sortKey === "mean_completion_ratio") {
      return (b.mean_completion_ratio ?? -1) - (a.mean_completion_ratio ?? -1);
    }
    return 0;
  });

  return (
    <div style={{
      background: "#0a0a0a",
      minHeight: "100vh",
      padding: "24px 20px",
      color: "#ededed",
    }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        {/* Breadcrumb */}
        <div style={{ marginBottom: 8 }}>
          <Link href="/research" style={{ color: "#555555", fontSize: 12, textDecoration: "none" }}>
            ← Research
          </Link>
        </div>

        {/* Header */}
        <h1 style={{ fontSize: 22, fontWeight: 600, color: "#ededed", marginBottom: 4 }}>
          Cooperative Reports
        </h1>
        <p style={{ fontSize: 13, color: "#666666", marginBottom: 24 }}>
          Evaluation and robustness sweep reports for the Cooperative archetype.
        </p>

        {/* Filter bar */}
        <div style={{
          display: "flex",
          gap: 10,
          flexWrap: "wrap",
          marginBottom: 20,
          background: "#111111",
          border: "1px solid #1e1e1e",
          borderRadius: 6,
          padding: "12px 14px",
          alignItems: "center",
        }}>
          <input
            type="text"
            placeholder="Date (YYYY-MM-DD)"
            value={filters.date}
            onChange={(e) => setFilters((f) => ({ ...f, date: e.target.value }))}
            style={{
              background: "#0a0a0a",
              border: "1px solid #222222",
              borderRadius: 4,
              color: "#ededed",
              padding: "4px 10px",
              fontSize: 12,
              width: 140,
            }}
          />
          <input
            type="text"
            placeholder="Config hash prefix"
            value={filters.configHash}
            onChange={(e) => setFilters((f) => ({ ...f, configHash: e.target.value }))}
            style={{
              background: "#0a0a0a",
              border: "1px solid #222222",
              borderRadius: 4,
              color: "#ededed",
              padding: "4px 10px",
              fontSize: 12,
              width: 140,
            }}
          />
          <select
            value={filters.kind}
            onChange={(e) => setFilters((f) => ({ ...f, kind: e.target.value }))}
            style={{
              background: "#0a0a0a",
              border: "1px solid #222222",
              borderRadius: 4,
              color: "#ededed",
              padding: "4px 10px",
              fontSize: 12,
            }}
          >
            <option value="">All kinds</option>
            <option value="cooperative_eval">Eval</option>
            <option value="cooperative_robust">Robustness</option>
          </select>

          <select
            value={sortKey}
            onChange={(e) =>
              setSortKey(e.target.value as typeof sortKey)
            }
            style={{
              background: "#0a0a0a",
              border: "1px solid #222222",
              borderRadius: 4,
              color: "#ededed",
              padding: "4px 10px",
              fontSize: 12,
              marginLeft: "auto",
            }}
          >
            <option value="timestamp">Sort: Newest</option>
            <option value="robustness_score">Sort: Robustness</option>
            <option value="mean_completion_ratio">Sort: Completion Ratio</option>
          </select>
        </div>

        {/* Content */}
        {loading && (
          <div style={{ color: "#555555", fontSize: 13 }}>Loading reports…</div>
        )}
        {error && (
          <div style={{ color: "#ef4444", fontSize: 13 }}>Error: {error}</div>
        )}

        {!loading && !error && sorted.length === 0 && (
          <div style={{ color: "#555555", fontSize: 13, textAlign: "center", padding: "48px 0" }}>
            No cooperative reports found.
            <br />
            <span style={{ fontSize: 12, marginTop: 8, display: "block" }}>
              Run the cooperative pipeline to generate reports.
            </span>
          </div>
        )}

        {!loading && !error && sorted.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ fontSize: 12, color: "#555555", marginBottom: 4 }}>
              {sorted.length} report{sorted.length !== 1 ? "s" : ""}
            </div>
            {sorted.map((r) => (
              <ReportCard key={r.report_id} report={r} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
