"use client";

import { useState } from "react";
import {
  ConfigListItem,
  ChampionBenchmarkResponse,
  ChampionBenchmarkResult,
  runChampionBenchmark,
} from "@/lib/api";

interface Props {
  configs: ConfigListItem[];
}

const BAR_COLORS: Record<string, string> = {
  league_champion: "#8b5cf6",
  random: "#6b7280",
  always_cooperate: "#22c55e",
  always_extract: "#ef4444",
  tit_for_tat: "#3b82f6",
  ppo_shared: "#f59e0b",
};

function BarChart({ results }: { results: ChampionBenchmarkResult[] }) {
  if (results.length === 0) return null;

  const maxVal = Math.max(
    ...results.map((r) => Math.abs(r.mean_total_reward)),
    0.01
  );
  const barW = 50;
  const barGap = 12;
  const chartH = 140;
  const chartW = results.length * (barW + barGap);
  const labelH = 48;

  return (
    <svg
      width={chartW}
      height={chartH + labelH}
      className="block"
    >
      {results.map((r, i) => {
        const h = (Math.abs(r.mean_total_reward) / maxVal) * chartH;
        const x = i * (barW + barGap);
        const color = BAR_COLORS[r.policy] || "#6b7280";
        return (
          <g key={r.policy}>
            <rect
              x={x}
              y={chartH - h}
              width={barW}
              height={h}
              fill={color}
              rx={3}
            />
            <text
              x={x + barW / 2}
              y={chartH - h - 4}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
            >
              {r.mean_total_reward.toFixed(2)}
            </text>
            <text
              x={x + barW / 2}
              y={chartH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
            >
              {r.policy.replace("_", " ")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function ChampionBenchmark({ configs }: Props) {
  const [configId, setConfigId] = useState(configs[0]?.config_id ?? "");
  const [episodes, setEpisodes] = useState(5);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ChampionBenchmarkResponse | null>(null);

  async function handleRun() {
    if (!configId) {
      setError("Select a config first.");
      return;
    }
    setRunning(true);
    setError(null);
    setData(null);
    try {
      const resp = await runChampionBenchmark(configId, episodes);
      setData(resp);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-end gap-3">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Config</label>
          <select
            value={configId}
            onChange={(e) => setConfigId(e.target.value)}
            className="border rounded px-2 py-1 text-sm"
          >
            {configs.map((c) => (
              <option key={c.config_id} value={c.config_id}>
                {c.config_id} (agents={c.num_agents}, steps={c.max_steps})
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Episodes</label>
          <input
            type="number"
            min={1}
            max={50}
            value={episodes}
            onChange={(e) => setEpisodes(Number(e.target.value))}
            className="border rounded px-2 py-1 text-sm w-16"
          />
        </div>
        <button
          onClick={handleRun}
          disabled={running || configs.length === 0}
          className="px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700 text-sm disabled:opacity-50"
        >
          {running ? "Running..." : "Run Benchmark"}
        </button>
      </div>

      {error && <p className="text-red-500 text-sm">{error}</p>}

      {data && (
        <>
          <p className="text-sm text-gray-600">
            Champion: <span className="font-mono">{data.champion.member_id}</span>{" "}
            (rating {data.champion.rating.toFixed(1)})
          </p>

          <h3 className="text-sm font-semibold">Mean Total Reward</h3>
          <BarChart results={data.results} />

          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-gray-300">
                <th className="py-1 pr-3">Policy</th>
                <th className="py-1 pr-3">Mean Reward</th>
                <th className="py-1 pr-3">Mean Pool</th>
                <th className="py-1 pr-3">Collapse Rate</th>
                <th className="py-1 pr-3">Mean Length</th>
              </tr>
            </thead>
            <tbody>
              {data.results.map((r) => (
                <tr key={r.policy} className="border-b border-gray-200">
                  <td className="py-1 pr-3 font-mono">{r.policy}</td>
                  <td className="py-1 pr-3">{r.mean_total_reward.toFixed(4)}</td>
                  <td className="py-1 pr-3">
                    {r.mean_final_shared_pool.toFixed(2)}
                  </td>
                  <td className="py-1 pr-3">
                    {(r.collapse_rate * 100).toFixed(0)}%
                  </td>
                  <td className="py-1 pr-3">{r.mean_episode_length}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
