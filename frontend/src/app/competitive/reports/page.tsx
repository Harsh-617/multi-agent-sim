"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { getCompetitiveReports, type CompetitiveReportListItem } from "@/lib/api";

export default function CompetitiveReportsPage() {
  const [reports, setReports] = useState<CompetitiveReportListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCompetitiveReports()
      .then(setReports)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="max-w-4xl mx-auto p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Competitive Reports</h1>
        <Link
          href="/competitive"
          className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
        >
          Back to Competitive
        </Link>
      </div>

      {loading && <p className="text-gray-500">Loading reports...</p>}
      {error && <p className="text-red-600">Error: {error}</p>}

      {!loading && !error && reports.length === 0 && (
        <p className="text-gray-500">
          No competitive reports yet — run champion robustness from the League page
        </p>
      )}

      {reports.length > 0 && (
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b text-left">
              <th className="py-2 pr-4">Report ID</th>
              <th className="py-2 pr-4">Timestamp</th>
              <th className="py-2" />
            </tr>
          </thead>
          <tbody>
            {reports.map((r) => (
              <tr key={r.report_id} className="border-b hover:bg-gray-50">
                <td className="py-2 pr-4 font-mono text-xs">{r.report_id}</td>
                <td className="py-2 pr-4 text-gray-600">
                  {r.timestamp ? new Date(r.timestamp).toLocaleString() : "—"}
                </td>
                <td className="py-2">
                  <Link
                    href={`/competitive/reports/${encodeURIComponent(r.path_name)}`}
                    className="px-3 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
                  >
                    Open
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}
