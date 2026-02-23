"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { listReports, type ReportListItem } from "@/lib/api";

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listReports()
      .then(setReports)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="max-w-4xl mx-auto p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Evaluation Reports</h1>
        <Link
          href="/"
          className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
        >
          Home
        </Link>
      </div>

      {loading && <p className="text-gray-500">Loading reports...</p>}
      {error && <p className="text-red-600">Error: {error}</p>}

      {!loading && !error && reports.length === 0 && (
        <p className="text-gray-500">No reports found in storage/reports/.</p>
      )}

      {reports.length > 0 && (
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b text-left">
              <th className="py-2 pr-4">Report ID</th>
              <th className="py-2 pr-4">Kind</th>
              <th className="py-2 pr-4">Timestamp</th>
              <th className="py-2" />
            </tr>
          </thead>
          <tbody>
            {reports.map((r) => (
              <tr key={r.report_id} className="border-b hover:bg-gray-50">
                <td className="py-2 pr-4 font-mono text-xs">{r.report_id}</td>
                <td className="py-2 pr-4">
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-medium ${
                      r.kind === "robust"
                        ? "bg-purple-100 text-purple-800"
                        : "bg-blue-100 text-blue-800"
                    }`}
                  >
                    {r.kind}
                  </span>
                </td>
                <td className="py-2 pr-4 text-gray-600">
                  {r.timestamp ? new Date(r.timestamp).toLocaleString() : "—"}
                </td>
                <td className="py-2">
                  <Link
                    href={`/reports/${encodeURIComponent(r.path_name)}`}
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
