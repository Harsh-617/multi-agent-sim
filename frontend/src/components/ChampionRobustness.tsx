"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  ConfigListItem,
  ChampionRobustnessRequest,
  runChampionRobustness,
} from "@/lib/api";

interface Props {
  configs: ConfigListItem[];
}

export default function ChampionRobustness({ configs }: Props) {
  const router = useRouter();
  const [configId, setConfigId] = useState(configs[0]?.config_id ?? "default");
  const [seeds, setSeeds] = useState(3);
  const [episodesPerSeed, setEpisodesPerSeed] = useState(2);
  const [maxSteps, setMaxSteps] = useState<string>("");
  const [limitSweeps, setLimitSweeps] = useState<string>("");
  const [seed, setSeed] = useState(42);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setRunning(true);
    setError(null);
    try {
      const payload: ChampionRobustnessRequest = {
        config_id: configId,
        seeds,
        episodes_per_seed: episodesPerSeed,
        seed,
        ...(maxSteps !== "" ? { max_steps: Number(maxSteps) } : {}),
        ...(limitSweeps !== "" ? { limit_sweeps: Number(limitSweeps) } : {}),
      };
      const resp = await runChampionRobustness(payload);
      router.push(`/reports/${resp.report_id}`);
    } catch (e) {
      setError(String(e));
      setRunning(false);
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Evaluates the league champion against all baseline policies across
        multiple environment variants and saves a robustness report.
      </p>

      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Config</label>
          <select
            value={configId}
            onChange={(e) => setConfigId(e.target.value)}
            className="border rounded px-2 py-1 text-sm"
          >
            <option value="default">default</option>
            {configs.map((c) => (
              <option key={c.config_id} value={c.config_id}>
                {c.config_id} (agents={c.num_agents}, steps={c.max_steps})
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">Seeds</label>
          <input
            type="number"
            min={1}
            max={20}
            value={seeds}
            onChange={(e) => setSeeds(Number(e.target.value))}
            className="border rounded px-2 py-1 text-sm w-16"
          />
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Episodes/seed
          </label>
          <input
            type="number"
            min={1}
            max={10}
            value={episodesPerSeed}
            onChange={(e) => setEpisodesPerSeed(Number(e.target.value))}
            className="border rounded px-2 py-1 text-sm w-16"
          />
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Max steps (opt.)
          </label>
          <input
            type="number"
            min={1}
            placeholder="—"
            value={maxSteps}
            onChange={(e) => setMaxSteps(e.target.value)}
            className="border rounded px-2 py-1 text-sm w-20"
          />
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Limit sweeps (opt.)
          </label>
          <input
            type="number"
            min={1}
            placeholder="—"
            value={limitSweeps}
            onChange={(e) => setLimitSweeps(e.target.value)}
            className="border rounded px-2 py-1 text-sm w-20"
          />
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">Seed</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="border rounded px-2 py-1 text-sm w-20"
          />
        </div>

        <button
          onClick={handleRun}
          disabled={running}
          className="px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 text-sm disabled:opacity-50"
        >
          {running ? "Running..." : "Run Robustness"}
        </button>
      </div>

      {error && <p className="text-red-500 text-sm">{error}</p>}

      {running && (
        <p className="text-sm text-gray-500">
          Running robustness sweep — this may take a moment...
        </p>
      )}
    </div>
  );
}
