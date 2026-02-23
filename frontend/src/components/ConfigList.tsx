"use client";

import { useEffect, useState } from "react";
import { AgentPolicy, ConfigListItem, listConfigs, createDefaultConfig, startRun } from "@/lib/api";
import { useRouter } from "next/navigation";

const POLICY_OPTIONS: { value: AgentPolicy; label: string }[] = [
  { value: "random", label: "Random" },
  { value: "always_cooperate", label: "Always Cooperate" },
  { value: "always_extract", label: "Always Extract" },
  { value: "tit_for_tat", label: "Tit for Tat" },
  { value: "ppo_shared", label: "PPO Shared" },
];

export default function ConfigList() {
  const router = useRouter();
  const [configs, setConfigs] = useState<ConfigListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState<string | null>(null);
  const [policy, setPolicy] = useState<AgentPolicy>("random");

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setConfigs(await listConfigs());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function handleCreate() {
    setError(null);
    try {
      await createDefaultConfig();
      await load();
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleStart(configId: string) {
    setError(null);
    setStarting(configId);
    try {
      const { run_id } = await startRun(configId, policy);
      router.push(`/run/${run_id}`);
    } catch (e) {
      setError(String(e));
      setStarting(null);
    }
  }

  return (
    <div>
      <div className="flex items-center gap-4 mb-4">
        <h2 className="text-xl font-semibold">Saved Configs</h2>
        <button
          onClick={handleCreate}
          className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
        >
          Create default config
        </button>
        <label className="text-sm flex items-center gap-1">
          Policy:
          <select
            value={policy}
            onChange={(e) => setPolicy(e.target.value as AgentPolicy)}
            className="border rounded px-2 py-1 text-sm"
          >
            {POLICY_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </label>
      </div>

      {error && <p className="text-red-500 mb-2 text-sm">{error}</p>}

      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : configs.length === 0 ? (
        <p className="text-gray-500">No configs yet. Create one to get started.</p>
      ) : (
        <table className="w-full text-left text-sm border-collapse">
          <thead>
            <tr className="border-b border-gray-300">
              <th className="py-2 pr-4">Config ID</th>
              <th className="py-2 pr-4">Seed</th>
              <th className="py-2 pr-4">Agents</th>
              <th className="py-2 pr-4">Max Steps</th>
              <th className="py-2" />
            </tr>
          </thead>
          <tbody>
            {configs.map((c) => (
              <tr key={c.config_id} className="border-b border-gray-200">
                <td className="py-2 pr-4 font-mono">{c.config_id}</td>
                <td className="py-2 pr-4">{c.seed}</td>
                <td className="py-2 pr-4">{c.num_agents}</td>
                <td className="py-2 pr-4">{c.max_steps}</td>
                <td className="py-2">
                  <button
                    onClick={() => handleStart(c.config_id)}
                    disabled={starting !== null}
                    className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm disabled:opacity-50"
                  >
                    {starting === c.config_id ? "Starting..." : "Start run"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
