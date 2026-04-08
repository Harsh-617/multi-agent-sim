"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import NetworkBackground from "@/components/NetworkBackground";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Stats {
  totalRuns: string;
  leagueMembers: string;
  environments: string;
  reports: string;
}

interface RecentRun {
  run_id: string;
  agent_policy: string | null;
  timestamp: string | null;
}

// ---------------------------------------------------------------------------
// Stats hook
// ---------------------------------------------------------------------------

function useHomeStats(): { stats: Stats; hasError: boolean } {
  const [stats, setStats] = useState<Stats>({
    totalRuns: "—",
    leagueMembers: "—",
    environments: "2",
    reports: "—",
  });
  const [failCount, setFailCount] = useState(0);
  const totalFetches = 3;

  useEffect(() => {
    let fails = 0;
    const trackFail = () => {
      fails++;
      if (fails >= totalFetches) setFailCount(fails);
    };

    fetch("/api/runs/history")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: unknown[]) =>
        setStats((s) => ({ ...s, totalRuns: String(data.length) }))
      )
      .catch(trackFail);

    Promise.all([
      fetch("/api/league/members").then((r) =>
        r.ok ? r.json() : Promise.reject()
      ),
      fetch("/api/competitive/league/members").then((r) =>
        r.ok ? r.json() : Promise.reject()
      ),
    ])
      .then(([mixed, comp]: [unknown[], unknown[]]) =>
        setStats((s) => ({
          ...s,
          leagueMembers: String(mixed.length + comp.length),
        }))
      )
      .catch(trackFail);

    Promise.all([
      fetch("/api/reports").then((r) =>
        r.ok ? r.json() : Promise.reject()
      ),
      fetch("/api/competitive/reports").then((r) =>
        r.ok ? r.json() : Promise.reject()
      ),
    ])
      .then(([mixed, comp]: [unknown[], unknown[]]) =>
        setStats((s) => ({
          ...s,
          reports: String(mixed.length + comp.length),
        }))
      )
      .catch(trackFail);
  }, []);

  return { stats, hasError: failCount >= totalFetches };
}

// ---------------------------------------------------------------------------
// Recent runs hook
// ---------------------------------------------------------------------------

function useRecentRuns(): RecentRun[] {
  const [runs, setRuns] = useState<RecentRun[]>([]);

  useEffect(() => {
    fetch("/api/runs/history")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: RecentRun[]) => {
        const sorted = [...data].sort((a, b) => {
          const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0;
          const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0;
          return tb - ta;
        });
        setRuns(sorted.slice(0, 3));
      })
      .catch(() => {});
  }, []);

  return runs;
}

function relativeTime(ts: string | null): string {
  if (!ts) return "—";
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

// ---------------------------------------------------------------------------
// Hover button helpers
// ---------------------------------------------------------------------------

function PrimaryCTA() {
  const [hover, setHover] = useState(false);
  return (
    <Link href="/simulate" style={{ textDecoration: "none" }}>
      <button
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          background: hover ? "#0d9488" : "#14b8a6",
          color: "#0a0a0a",
          padding: "11px 24px",
          borderRadius: 6,
          fontSize: 14,
          fontWeight: 500,
          border: "none",
          cursor: "pointer",
          transition: "background 150ms",
        }}
      >
        Start Simulating →
      </button>
    </Link>
  );
}

function SecondaryCTA() {
  const [hover, setHover] = useState(false);
  return (
    <Link href="/league" style={{ textDecoration: "none" }}>
      <button
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          background: "transparent",
          color: hover ? "#ededed" : "#888888",
          border: `1px solid ${hover ? "#444444" : "#2a2a2a"}`,
          padding: "11px 24px",
          borderRadius: 6,
          fontSize: 14,
          cursor: "pointer",
          transition: "all 150ms",
        }}
      >
        View League →
      </button>
    </Link>
  );
}

function QuickStartButton({ href, label }: { href: string; label: string }) {
  const [hover, setHover] = useState(false);
  return (
    <Link href={href} style={{ textDecoration: "none", display: "block" }}>
      <button
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          background: "transparent",
          color: hover ? "#ededed" : "#888888",
          border: `1px solid ${hover ? "#14b8a6" : "#1a1a1a"}`,
          padding: "14px 20px",
          borderRadius: 6,
          fontSize: 14,
          cursor: "pointer",
          transition: "all 150ms",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          width: "100%",
        }}
      >
        <span>{label}</span>
        <span>→</span>
      </button>
    </Link>
  );
}

// ---------------------------------------------------------------------------
// Feature card data
// ---------------------------------------------------------------------------

const features = [
  {
    tag: "Training",
    title: "PPO Self-Play Training",
    body: "Agents are trained using Proximal Policy Optimization against a league of past policy snapshots. Opponent sampling is weighted toward recent members, producing steady improvement without catastrophic forgetting.",
  },
  {
    tag: "League",
    title: "Elo-Rated League System",
    body: "Every policy snapshot enters a league and is rated using the Elo system. Head-to-head matches determine rankings. The highest-rated member becomes the champion and is used as the baseline for future training.",
  },
  {
    tag: "Robustness",
    title: "20-Variant Robustness Sweeps",
    body: "The champion is stress-tested across 20 environment configurations — varying resource scarcity, information asymmetry, observation noise, and agent count. A weighted score (0.7 × mean + 0.3 × worst-case) summarizes performance under pressure.",
  },
  {
    tag: "Analysis",
    title: "Emergent Strategy Clustering",
    body: "After evaluation, K-means clustering groups policies by behavioral features — cooperation rate, extraction frequency, defense patterns. Clusters are automatically labeled (e.g. Cooperator, Exploiter, Opportunist) with no manual annotation required.",
  },
];

// ---------------------------------------------------------------------------
// Live Snapshot Card
// ---------------------------------------------------------------------------

function LiveSnapshotCard({
  stats,
  recentRuns,
  hasError,
}: {
  stats: Stats;
  recentRuns: RecentRun[];
  hasError: boolean;
}) {
  const mono: React.CSSProperties = {
    fontFamily: "var(--font-mono)",
  };

  const statRows = [
    { label: "TOTAL RUNS", value: stats.totalRuns },
    { label: "LEAGUE MEMBERS", value: stats.leagueMembers },
    { label: "ENVIRONMENTS", value: stats.environments },
    { label: "REPORTS", value: stats.reports },
  ];

  return (
    <div
      style={{
        background: "#0d0d0d",
        border: "1px solid #1a1a1a",
        borderRadius: 10,
        padding: 24,
        ...mono,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 20,
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: hasError ? "#444444" : "#14b8a6",
            display: "inline-block",
            animation: hasError ? "none" : "pulse 2s ease-in-out infinite",
          }}
        />
        <span style={{ fontSize: 10, color: hasError ? "#444444" : "#14b8a6" }}>{hasError ? "offline" : "live"}</span>
        <span style={{ fontSize: 10, color: "#444444" }}>system snapshot</span>
      </div>

      {/* Stat rows */}
      {statRows.map((row, i) => (
        <div
          key={row.label}
          style={{
            display: "flex",
            justifyContent: "space-between",
            padding: "10px 0",
            borderBottom:
              i < statRows.length - 1 ? "1px solid #141414" : "none",
          }}
        >
          <span
            style={{
              fontSize: 11,
              color: "#555555",
              textTransform: "uppercase",
              ...mono,
            }}
          >
            {row.label}
          </span>
          <span
            style={{
              fontSize: 14,
              color: "#ededed",
              fontWeight: 500,
              ...mono,
            }}
          >
            {row.value}
          </span>
        </div>
      ))}

      {/* Recent activity */}
      <div
        style={{
          fontSize: 10,
          color: "#333333",
          textTransform: "uppercase",
          marginTop: 16,
          marginBottom: 8,
        }}
      >
        RECENT ACTIVITY
      </div>
      {recentRuns.length > 0
        ? recentRuns.map((run) => (
            <div
              key={run.run_id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "3px 0",
              }}
            >
              <span style={{ fontSize: 11, color: "#444444", ...mono }}>
                {run.run_id.slice(0, 8)}
              </span>
              <span style={{ fontSize: 10, color: "#555555" }}>
                {run.agent_policy ?? "—"}
              </span>
              <span
                style={{ fontSize: 10, color: "#333333", marginLeft: "auto" }}
              >
                {relativeTime(run.timestamp)}
              </span>
            </div>
          ))
        : [
            { run_id: "--------", agent_policy: "—", timestamp: null },
            { run_id: "--------", agent_policy: "—", timestamp: null },
            { run_id: "--------", agent_policy: "—", timestamp: null },
          ].map((run, i) => (
            <div
              key={`placeholder-${i}`}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "3px 0",
              }}
            >
              <span style={{ fontSize: 11, color: "#444444", ...mono }}>
                {run.run_id.slice(0, 8)}
              </span>
              <span style={{ fontSize: 10, color: "#555555" }}>
                {run.agent_policy ?? "—"}
              </span>
              <span
                style={{ fontSize: 10, color: "#333333", marginLeft: "auto" }}
              >
                {relativeTime(run.timestamp)}
              </span>
            </div>
          ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function HomePage() {
  const { stats, hasError } = useHomeStats();
  const recentRuns = useRecentRuns();

  return (
    <div
      style={{
        position: "relative",
        overflow: "hidden",
        minHeight: "100vh",
      }}
    >
      <NetworkBackground />

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(0.8); }
        }
      `}</style>

      <main style={{ paddingTop: 48 }}>
        {/* ── Hero ── */}
        <section
          className="hero-grid"
          style={{
            maxWidth: 1200,
            margin: "0 auto",
            padding: "100px 48px 80px",
            display: "grid",
            gridTemplateColumns: "1fr 420px",
            gap: 80,
            alignItems: "center",
            position: "relative",
            zIndex: 1,
          }}
        >
          {/* Left column */}
          <div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 13,
                letterSpacing: "0.12em",
                color: "#14b8a6",
                textTransform: "uppercase",
                marginBottom: 24,
              }}
            >
              Multi-Agent Simulation Platform
            </div>

            <h1
              style={{
                fontSize: 48,
                fontWeight: 600,
                lineHeight: 1.1,
                color: "#ededed",
                marginBottom: 20,
                marginTop: 0,
              }}
            >
              Where agents learn to compete, cooperate, and evolve
            </h1>

            <p
              style={{
                fontSize: 16,
                color: "#666666",
                lineHeight: 1.7,
                marginBottom: 40,
                marginTop: 0,
              }}
            >
              A research platform for studying emergent strategy in multi-agent
              environments. Configure environments, train agents using PPO with
              league-based self-play, run Elo-rated tournaments, and analyze how
              strategies evolve across generations — all from a single
              end-to-end pipeline.
            </p>

            <div style={{ display: "flex", gap: 12 }}>
              <PrimaryCTA />
              <SecondaryCTA />
            </div>
          </div>

          {/* Right column — live snapshot */}
          <LiveSnapshotCard stats={stats} recentRuns={recentRuns} hasError={hasError} />
        </section>

        {/* ── How It Works ── */}
        <section
          style={{
            padding: "60px 24px",
            maxWidth: 960,
            margin: "0 auto",
            position: "relative",
            zIndex: 1,
          }}
        >
          <p
            style={{
              fontSize: 11,
              fontWeight: 500,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: "var(--accent)",
              marginBottom: 12,
              textAlign: "center",
            }}
          >
            How it works
          </p>

          <h2
            style={{
              fontSize: 28,
              fontWeight: 500,
              color: "var(--text-primary)",
              textAlign: "center",
              marginBottom: 48,
              lineHeight: 1.2,
            }}
          >
            From configuration to research-grade results
          </h2>

          <div
            className="steps-grid"
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 24,
              position: "relative",
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 20,
                left: "12.5%",
                right: "12.5%",
                height: 1,
                background:
                  "linear-gradient(to right, transparent, var(--bg-border) 20%, var(--bg-border) 80%, transparent)",
                zIndex: 0,
              }}
            />

            {[
              {
                step: "01",
                title: "Configure",
                description:
                  "Pick a simulation template — Resource Sharing or Head-to-Head. Set the number of agents, episode length, seed, and behavioral parameters. Save your config to reuse it across runs.",
              },
              {
                step: "02",
                title: "Simulate",
                description:
                  "Start a run. Agents act in the environment each step, producing rewards and observations. Metrics stream live to the dashboard via WebSocket — shared pool, cooperation rate, reward curves — and are saved to storage for replay.",
              },
              {
                step: "03",
                title: "Train & League",
                description:
                  "Run the full pipeline: PPO trains a shared-policy agent across league-sampled opponents. Periodic snapshots are saved as league members, Elo-rated against each other, and used as future opponents — creating a self-improving population.",
              },
              {
                step: "04",
                title: "Evaluate & Analyze",
                description:
                  "The league champion is tested across 20 environment variants — different resource levels, noise, asymmetry, and pressure. Cross-seed evaluation produces a robustness score. K-means clustering identifies emergent strategy labels across the league population.",
              },
            ].map((item) => (
              <div
                key={item.step}
                style={{
                  position: "relative",
                  zIndex: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  textAlign: "center",
                }}
              >
                <div
                  style={{
                    width: 40,
                    height: 40,
                    borderRadius: "50%",
                    background: "var(--bg-surface)",
                    border: "1px solid var(--bg-border)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    marginBottom: 16,
                    fontSize: 12,
                    fontWeight: 500,
                    color: "var(--accent)",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {item.step}
                </div>
                <h3
                  style={{
                    fontSize: 14,
                    fontWeight: 500,
                    color: "var(--text-primary)",
                    marginBottom: 8,
                  }}
                >
                  {item.title}
                </h3>
                <p
                  style={{
                    fontSize: 13,
                    color: "var(--text-secondary)",
                    lineHeight: 1.6,
                  }}
                >
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Feature Highlights ── */}
        <section
          style={{
            maxWidth: 1200,
            margin: "0 auto",
            padding: "60px 48px",
            position: "relative",
            zIndex: 1,
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              color: "#14b8a6",
              marginBottom: 12,
            }}
          >
            Capabilities
          </div>

          <h2
            style={{
              fontSize: 32,
              fontWeight: 600,
              color: "#ededed",
              marginBottom: 48,
              marginTop: 0,
            }}
          >
            Built for serious research
          </h2>

          <div
            className="feature-grid"
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, 1fr)",
              gap: 1,
              background: "#1a1a1a",
            }}
          >
            {features.map((f) => (
              <div
                key={f.tag}
                style={{
                  background: "#0d0d0d",
                  padding: 32,
                }}
              >
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 10,
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                    color: "#14b8a6",
                    marginBottom: 16,
                    display: "block",
                  }}
                >
                  {f.tag}
                </span>
                <div
                  style={{
                    fontSize: 18,
                    fontWeight: 500,
                    color: "#ededed",
                    marginBottom: 12,
                  }}
                >
                  {f.title}
                </div>
                <div
                  style={{
                    fontSize: 14,
                    color: "#666666",
                    lineHeight: 1.7,
                  }}
                >
                  {f.body}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Research Questions ── */}
        <section
          style={{
            padding: "60px 24px",
            maxWidth: 960,
            margin: "0 auto",
            position: "relative",
            zIndex: 1,
          }}
        >
          <p
            style={{
              fontSize: 11,
              fontWeight: 500,
              letterSpacing: "0.1em",
              textTransform: "uppercase" as const,
              color: "var(--accent)",
              marginBottom: 12,
              textAlign: "center",
            }}
          >
            Research questions
          </p>

          <h2
            style={{
              fontSize: 28,
              fontWeight: 500,
              color: "var(--text-primary)",
              textAlign: "center",
              marginBottom: 48,
              lineHeight: 1.2,
            }}
          >
            What this platform helps you study
          </h2>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, 1fr)",
              gap: 16,
            }}
            className="research-grid"
          >
            {[
              {
                q: "When do agents learn to cooperate instead of defect?",
                desc: "In the Resource Sharing environment, agents choose every step whether to contribute to a shared pool or extract from it. Run experiments to see under what conditions — agent count, memory depth, reputation — sustained cooperation emerges.",
                tag: "Resource Sharing",
                tagColor: "var(--accent)",
              },
              {
                q: "Which strategies hold up under stress?",
                desc: "Robustness sweeps test the league champion across 20 environment variants with shifted parameters. The heatmap and scatter chart show exactly which conditions cause performance to collapse and which strategies are truly robust.",
                tag: "Robustness Analysis",
                tagColor: "#f59e0b",
              },
              {
                q: "How do agent strategies evolve across generations?",
                desc: "The league evolution graph shows the full lineage of trained policies — parent, child, rating, and strategy label at each node. Watch how the population shifts from random play to structured strategies over generations of self-play.",
                tag: "League Evolution",
                tagColor: "#8b5cf6",
              },
            ].map((item) => (
              <div
                key={item.q}
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--bg-border)",
                  borderRadius: 8,
                  padding: 24,
                }}
              >
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 500,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase" as const,
                    color: item.tagColor,
                    marginBottom: 12,
                    display: "block",
                  }}
                >
                  {item.tag}
                </span>
                <h3
                  style={{
                    fontSize: 15,
                    fontWeight: 500,
                    color: "var(--text-primary)",
                    marginBottom: 10,
                    lineHeight: 1.3,
                  }}
                >
                  {item.q}
                </h3>
                <p
                  style={{
                    fontSize: 13,
                    color: "var(--text-secondary)",
                    lineHeight: 1.6,
                  }}
                >
                  {item.desc}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Quick Start ── */}
        <section
          style={{
            maxWidth: 1200,
            margin: "0 auto",
            padding: "60px 48px 100px",
            position: "relative",
            zIndex: 1,
          }}
        >
          <div
            className="quickstart-grid"
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 48,
              alignItems: "center",
            }}
          >
            {/* Left */}
            <div>
              <h2
                style={{
                  fontSize: 28,
                  fontWeight: 500,
                  color: "#ededed",
                  marginTop: 0,
                  marginBottom: 0,
                }}
              >
                Ready to simulate?
              </h2>
              <p
                style={{
                  fontSize: 14,
                  color: "#666666",
                  lineHeight: 1.6,
                  marginTop: 12,
                }}
              >
                Choose a template and run your first simulation in under a
                minute. No configuration required.
              </p>
            </div>

            {/* Right */}
            <div>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <QuickStartButton
                  href="/simulate/resource-sharing"
                  label="Resource Sharing Arena"
                />
                <QuickStartButton
                  href="/simulate/head-to-head"
                  label="Head-to-Head Strategy"
                />
              </div>
              <Link
                href="/simulate"
                style={{
                  fontSize: 13,
                  color: "#444444",
                  marginTop: 12,
                  display: "block",
                  textDecoration: "none",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = "#888888";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = "#444444";
                }}
              >
                Or explore all templates →
              </Link>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
