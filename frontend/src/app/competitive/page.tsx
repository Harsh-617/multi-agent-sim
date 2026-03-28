"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import {
  createCompetitiveConfig,
  startCompetitiveRun,
  getCompetitiveLeagueRatings,
  recomputeCompetitiveLeagueRatings,
  CompetitiveAgentPolicy,
  CompetitiveLeagueRating,
} from "@/lib/api";

const POLICY_OPTIONS: CompetitiveAgentPolicy[] = [
  "random",
  "always_attack",
  "always_build",
  "always_defend",
  "competitive_ppo",
];

export default function CompetitivePage() {
  const router = useRouter();

  // -- Create & Run form state --
  const [numAgents, setNumAgents] = useState(4);
  const [maxSteps, setMaxSteps] = useState(200);
  const [seed, setSeed] = useState(42);
  const [agentPolicy, setAgentPolicy] = useState<CompetitiveAgentPolicy>("random");
  const [starting, setStarting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  // -- League ratings state --
  const [ratings, setRatings] = useState<CompetitiveLeagueRating[]>([]);
  const [ratingsLoading, setRatingsLoading] = useState(true);
  const [ratingsError, setRatingsError] = useState<string | null>(null);
  const [recomputing, setRecomputing] = useState(false);

  const fetchRatings = useCallback(async () => {
    setRatingsLoading(true);
    setRatingsError(null);
    try {
      const data = await getCompetitiveLeagueRatings();
      setRatings(data);
    } catch (err: unknown) {
      setRatingsError(err instanceof Error ? err.message : String(err));
    } finally {
      setRatingsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRatings();
  }, [fetchRatings]);

  async function handleStartRun() {
    setStarting(true);
    setFormError(null);
    try {
      const { config_id } = await createCompetitiveConfig({
        num_agents: numAgents,
        max_steps: maxSteps,
        seed,
      });
      const { run_id } = await startCompetitiveRun(config_id, agentPolicy);
      router.push(`/run/${run_id}`);
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  }

  async function handleRecompute() {
    setRecomputing(true);
    setRatingsError(null);
    try {
      const data = await recomputeCompetitiveLeagueRatings();
      setRatings(data);
    } catch (err: unknown) {
      setRatingsError(err instanceof Error ? err.message : String(err));
    } finally {
      setRecomputing(false);
    }
  }

  // Sort ratings by rating descending for rank display
  const sorted = [...ratings].sort((a, b) => b.rating - a.rating);

  return (
    <main className="max-w-3xl mx-auto p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Competitive Arena</h1>
        <div className="flex gap-2">
          <Link
            href="/"
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
          >
            Home
          </Link>
          <Link
            href="/competitive/league"
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
          >
            League
          </Link>
          <Link
            href="/runs"
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
          >
            Run History
          </Link>
        </div>
      </div>

      {/* Section A — Create & Run */}
      <section className="mb-10 border rounded p-6">
        <h2 className="text-lg font-semibold mb-4">Create &amp; Run</h2>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <label className="block">
            <span className="text-sm font-medium">num_agents</span>
            <input
              type="number"
              min={2}
              max={20}
              value={numAgents}
              onChange={(e) => setNumAgents(Number(e.target.value))}
              className="mt-1 block w-full border rounded px-2 py-1 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium">max_steps</span>
            <input
              type="number"
              min={10}
              max={10000}
              value={maxSteps}
              onChange={(e) => setMaxSteps(Number(e.target.value))}
              className="mt-1 block w-full border rounded px-2 py-1 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium">seed</span>
            <input
              type="number"
              min={0}
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="mt-1 block w-full border rounded px-2 py-1 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium">agent_policy</span>
            <select
              value={agentPolicy}
              onChange={(e) => setAgentPolicy(e.target.value as CompetitiveAgentPolicy)}
              className="mt-1 block w-full border rounded px-2 py-1 text-sm"
            >
              {POLICY_OPTIONS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
        </div>
        {formError && (
          <p className="text-red-600 text-sm mb-3">{formError}</p>
        )}
        <button
          onClick={handleStartRun}
          disabled={starting}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm"
        >
          {starting ? "Starting…" : "Start Run"}
        </button>
      </section>

      {/* Section B — League Standings */}
      <section className="border rounded p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">League Standings</h2>
          <button
            onClick={handleRecompute}
            disabled={recomputing}
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50 text-sm"
          >
            {recomputing ? "Recomputing…" : "Recompute Ratings"}
          </button>
        </div>

        {ratingsLoading && <p className="text-sm text-gray-500">Loading…</p>}

        {ratingsError && (
          <p className="text-red-600 text-sm">{ratingsError}</p>
        )}

        {!ratingsLoading && !ratingsError && sorted.length === 0 && (
          <p className="text-sm text-gray-500">
            No league data yet — run the pipeline first
          </p>
        )}

        {!ratingsLoading && sorted.length > 0 && (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left">
                <th className="py-2 pr-4">Rank</th>
                <th className="py-2 pr-4">Member ID</th>
                <th className="py-2">Rating</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, idx) => (
                <tr key={r.member_id} className="border-b">
                  <td className="py-2 pr-4">{idx + 1}</td>
                  <td className="py-2 pr-4 font-mono">{r.member_id}</td>
                  <td className="py-2">{r.rating.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
