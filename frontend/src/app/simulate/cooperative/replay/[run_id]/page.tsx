"use client";

/**
 * Cooperative replay page — /simulate/cooperative/replay/[run_id]
 *
 * Renders CooperativeReplayView for the given run.
 */

import { use } from "react";
import Link from "next/link";
import CooperativeReplayView from "@/components/CooperativeReplayView";

interface PageProps {
  params: Promise<{ run_id: string }>;
}

export default function CooperativeReplayPage({ params }: PageProps) {
  const { run_id } = use(params);

  return (
    <main
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        padding: "48px 24px",
        paddingTop: 96,
      }}
    >
      {/* Back link */}
      <div style={{ marginBottom: 24 }}>
        <Link
          href="/simulate/cooperative"
          style={{
            fontSize: 12,
            color: "var(--text-tertiary)",
            textDecoration: "none",
          }}
        >
          ← Back to Cooperative
        </Link>
      </div>

      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1
          style={{
            fontSize: 20,
            fontWeight: 500,
            color: "var(--text-primary)",
            margin: 0,
          }}
        >
          Cooperative Replay
        </h1>
        <p
          style={{
            fontSize: 12,
            fontFamily: "var(--font-mono)",
            color: "var(--text-tertiary)",
            margin: "6px 0 0",
          }}
        >
          {run_id}
        </p>
      </div>

      <CooperativeReplayView runId={run_id} />
    </main>
  );
}
