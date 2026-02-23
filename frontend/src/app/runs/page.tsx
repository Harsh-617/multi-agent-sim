"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { RunListItem, listRuns } from "@/lib/api";

export default function RunsPage() {
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    listRuns()
      .then(setRuns)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="max-w-4xl mx-auto p-8">
      <div className="flex items-center gap-4 mb-6">
        <Link href="/" className="text-blue-500 hover:underline text-sm">
          &larr; Home
        </Link>
        <h1 className="text-2xl font-bold">Run History</h1>
      </div>

      {error && <p className="text-red-500 mb-2 text-sm">{error}</p>}

      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : runs.length === 0 ? (
        <p className="text-gray-500">No completed runs yet.</p>
      ) : (
        <table className="w-full text-left text-sm border-collapse">
          <thead>
            <tr className="border-b border-gray-300">
              <th className="py-2 pr-4">Run ID</th>
              <th className="py-2 pr-4">Policy</th>
              <th className="py-2 pr-4">Steps</th>
              <th className="py-2 pr-4">Termination</th>
              <th className="py-2 pr-4">Time</th>
              <th className="py-2" />
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id} className="border-b border-gray-200">
                <td className="py-2 pr-4 font-mono">{r.run_id}</td>
                <td className="py-2 pr-4">{r.agent_policy ?? "—"}</td>
                <td className="py-2 pr-4">{r.episode_length ?? "—"}</td>
                <td className="py-2 pr-4">{r.termination_reason ?? "—"}</td>
                <td className="py-2 pr-4 text-xs text-gray-500">
                  {r.timestamp ? new Date(r.timestamp).toLocaleString() : "—"}
                </td>
                <td className="py-2">
                  <Link
                    href={`/replay/${r.run_id}`}
                    className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
                  >
                    Replay
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
