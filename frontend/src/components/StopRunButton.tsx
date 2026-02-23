"use client";

import { useState } from "react";
import { stopRun } from "@/lib/api";

interface Props {
  onStopped?: () => void;
}

export default function StopRunButton({ onStopped }: Props) {
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleStop() {
    setStopping(true);
    setError(null);
    try {
      await stopRun();
      onStopped?.();
    } catch (e) {
      setError(String(e));
    } finally {
      setStopping(false);
    }
  }

  return (
    <div>
      <button
        onClick={handleStop}
        disabled={stopping}
        className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
      >
        {stopping ? "Stopping..." : "Stop Run"}
      </button>
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
}
