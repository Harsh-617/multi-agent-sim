"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  LeagueMember,
  LeagueRating,
  LineageMember,
  listLeagueMembers,
  getLeagueRatings,
  getLeagueLineage,
  recomputeLeagueRatings,
  listConfigs,
  startRun,
  ConfigListItem,
} from "@/lib/api";
import LeagueLineage from "@/components/LeagueLineage";
import ChampionBenchmark from "@/components/ChampionBenchmark";

type Tab = "members" | "lineage" | "benchmark";

export default function LeaguePage() {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("members");
  const [members, setMembers] = useState<LeagueMember[]>([]);
  const [lineageMembers, setLineageMembers] = useState<LineageMember[]>([]);
  const [ratings, setRatings] = useState<Map<string, number>>(new Map());
  const [configs, setConfigs] = useState<ConfigListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [recomputing, setRecomputing] = useState(false);
  const [startingId, setStartingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [m, r, c, lin] = await Promise.all([
        listLeagueMembers(),
        getLeagueRatings(),
        listConfigs(),
        getLeagueLineage(),
      ]);
      setMembers(m);
      setRatings(new Map(r.map((x) => [x.member_id, x.rating])));
      setConfigs(c);
      setLineageMembers(lin.members);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleRecompute() {
    setRecomputing(true);
    setError(null);
    try {
      const r = await recomputeLeagueRatings();
      setRatings(new Map(r.map((x) => [x.member_id, x.rating])));
      // Refresh lineage too
      const lin = await getLeagueLineage();
      setLineageMembers(lin.members);
    } catch (e) {
      setError(String(e));
    } finally {
      setRecomputing(false);
    }
  }

  async function handleRun(memberId: string) {
    if (configs.length === 0) {
      setError("No configs available. Create one on the home page first.");
      return;
    }
    setStartingId(memberId);
    setError(null);
    try {
      const { run_id } = await startRun(configs[0].config_id, "league_snapshot", memberId);
      router.push(`/run/${run_id}`);
    } catch (e) {
      setError(String(e));
      setStartingId(null);
    }
  }

  // Sort members by rating (descending), then by id
  const sorted = [...members].sort((a, b) => {
    const ra = ratings.get(a.member_id) ?? 0;
    const rb = ratings.get(b.member_id) ?? 0;
    return rb - ra || a.member_id.localeCompare(b.member_id);
  });

  const tabClass = (t: Tab) =>
    `px-4 py-2 text-sm font-medium rounded-t ${
      tab === t
        ? "bg-white border border-b-0 border-gray-300"
        : "text-gray-500 hover:text-gray-700"
    }`;

  return (
    <main className="max-w-5xl mx-auto p-8">
      <div className="flex items-center gap-4 mb-6">
        <Link href="/" className="text-blue-500 hover:underline text-sm">
          &larr; Home
        </Link>
        <h1 className="text-2xl font-bold">League</h1>
        <button
          onClick={handleRecompute}
          disabled={recomputing || members.length === 0}
          className="ml-auto px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm disabled:opacity-50"
        >
          {recomputing ? "Recomputing..." : "Recompute Ratings"}
        </button>
      </div>

      {error && <p className="text-red-500 mb-2 text-sm">{error}</p>}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-300 mb-4">
        <button className={tabClass("members")} onClick={() => setTab("members")}>
          Members
        </button>
        <button className={tabClass("lineage")} onClick={() => setTab("lineage")}>
          Lineage
        </button>
        <button className={tabClass("benchmark")} onClick={() => setTab("benchmark")}>
          Champion Benchmark
        </button>
      </div>

      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : (
        <>
          {/* Members tab */}
          {tab === "members" && (
            sorted.length === 0 ? (
              <p className="text-gray-500">
                No league members yet. Train a policy and save a snapshot to get started.
              </p>
            ) : (
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-gray-300">
                    <th className="py-2 pr-4">#</th>
                    <th className="py-2 pr-4">Member ID</th>
                    <th className="py-2 pr-4">Rating</th>
                    <th className="py-2 pr-4">Parent</th>
                    <th className="py-2 pr-4">Created</th>
                    <th className="py-2 pr-4">Notes</th>
                    <th className="py-2" />
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((m, idx) => (
                    <tr key={m.member_id} className="border-b border-gray-200">
                      <td className="py-2 pr-4 text-gray-400">{idx + 1}</td>
                      <td className="py-2 pr-4 font-mono">{m.member_id}</td>
                      <td className="py-2 pr-4 font-mono">
                        {ratings.has(m.member_id)
                          ? ratings.get(m.member_id)!.toFixed(1)
                          : "—"}
                      </td>
                      <td className="py-2 pr-4 font-mono text-xs">
                        {m.parent_id ?? "—"}
                      </td>
                      <td className="py-2 pr-4 text-xs text-gray-500">
                        {m.created_at
                          ? new Date(m.created_at).toLocaleString()
                          : "—"}
                      </td>
                      <td className="py-2 pr-4 text-xs">{m.notes ?? "—"}</td>
                      <td className="py-2">
                        <button
                          onClick={() => handleRun(m.member_id)}
                          disabled={startingId !== null}
                          className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm disabled:opacity-50"
                        >
                          {startingId === m.member_id ? "Starting..." : "Run"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )
          )}

          {/* Lineage tab */}
          {tab === "lineage" && (
            <LeagueLineage members={lineageMembers} />
          )}

          {/* Champion Benchmark tab */}
          {tab === "benchmark" && (
            configs.length === 0 ? (
              <p className="text-gray-500">
                No configs available. Create one on the home page first.
              </p>
            ) : (
              <ChampionBenchmark configs={configs} />
            )
          )}
        </>
      )}
    </main>
  );
}
