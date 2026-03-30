"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import LineageAnimation from "@/components/LineageAnimation";

// ---------------------------------------------------------------------------
// Stats hook — fetches counts from multiple endpoints, non-blocking
// ---------------------------------------------------------------------------

interface Stats {
  totalRuns: string;
  leagueMembers: string;
  environments: string;
  reports: string;
}

function useHomeStats(): Stats {
  const [stats, setStats] = useState<Stats>({
    totalRuns: "—",
    leagueMembers: "—",
    environments: "2",
    reports: "—",
  });

  useEffect(() => {
    // Total runs
    fetch("/api/runs/history")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: unknown[]) =>
        setStats((s) => ({ ...s, totalRuns: String(data.length) }))
      )
      .catch(() => {});

    // League members: mixed + competitive
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
      .catch(() => {});

    // Reports: mixed + competitive
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
      .catch(() => {});
  }, []);

  return stats;
}

// ---------------------------------------------------------------------------
// Feature card data
// ---------------------------------------------------------------------------

const features = [
  {
    label: "League system",
    title: "Agents compete, evolve, and develop strategies",
    body: "PPO-trained agents enter a self-play league. Periodic snapshots are Elo-rated, creating a lineage of increasingly capable policies.",
  },
  {
    label: "Evaluation",
    title: "Stress-tested across 20 environment variants",
    body: "Every champion policy is evaluated across systematically varied environments. Robustness score = 0.7 × mean + 0.3 × worst-case.",
  },
  {
    label: "Analysis",
    title: "Emergent playstyles discovered automatically",
    body: "K-means clustering over behavioral features identifies distinct agent strategies — Dominant, Aggressive, Consistent, Weak — without any manual labeling.",
  },
];

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
          background: hover ? "var(--accent-hover)" : "var(--accent)",
          color: "white",
          padding: "10px 20px",
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
          color: hover ? "var(--text-primary)" : "var(--text-secondary)",
          border: `1px solid ${hover ? "var(--accent)" : "var(--bg-border)"}`,
          padding: "10px 20px",
          borderRadius: 6,
          fontSize: 14,
          fontWeight: 500,
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
    <Link href={href} style={{ textDecoration: "none" }}>
      <button
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          background: "transparent",
          color: hover ? "var(--text-primary)" : "var(--text-secondary)",
          border: `1px solid ${hover ? "var(--accent)" : "var(--bg-border)"}`,
          padding: "12px 20px",
          borderRadius: 6,
          fontSize: 13,
          fontWeight: 500,
          cursor: "pointer",
          transition: "all 150ms",
        }}
      >
        {label}
      </button>
    </Link>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function HomePage() {
  const stats = useHomeStats();

  return (
    <div style={{ position: "relative", overflow: "hidden", minHeight: "100vh" }}>
      <LineageAnimation />
      <main style={{ paddingTop: 48 /* account for fixed nav */ }}>
        {/* ── Hero ── */}
        <section
          style={{
            padding: "80px 24px",
            textAlign: "center",
            maxWidth: 680,
            margin: "0 auto",
            position: "relative",
            zIndex: 1,
          }}
        >
          {/* Eyebrow */}
          <div
            style={{
              fontSize: 13,
              fontWeight: 500,
              letterSpacing: "0.06em",
              color: "var(--accent)",
              textTransform: "uppercase" as const,
              marginBottom: 20,
            }}
          >
            Multi-Agent Simulation Platform
          </div>

          {/* Headline */}
          <h1
            style={{
              fontSize: 42,
              fontWeight: 600,
              color: "var(--text-primary)",
              lineHeight: 1.15,
              marginBottom: 16,
              marginTop: 0,
            }}
          >
            A research platform for emergent multi-agent strategy
          </h1>

          {/* Subheadline */}
          <p
            style={{
              fontSize: 16,
              color: "var(--text-secondary)",
              lineHeight: 1.6,
              marginBottom: 32,
              marginTop: 0,
            }}
          >
            Train agents, run leagues, and study how strategies emerge in
            configurable multi-agent environments. Built for researchers.
          </p>

          {/* CTAs */}
          <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
            <PrimaryCTA />
            <SecondaryCTA />
          </div>
        </section>

        {/* ── Stats Bar ── */}
        <section
          style={{
            background: "var(--bg-surface)",
            borderTop: "1px solid var(--bg-border)",
            borderBottom: "1px solid var(--bg-border)",
            padding: "20px 24px",
            marginTop: 60,
            position: "relative",
            zIndex: 1,
          }}
        >
        <div
          style={{
            display: "flex",
            justifyContent: "space-around",
            maxWidth: 960,
            margin: "0 auto",
          }}
        >
          {[
            { value: stats.totalRuns, label: "Total runs" },
            { value: stats.leagueMembers, label: "League members" },
            { value: stats.environments, label: "Environments" },
            { value: stats.reports, label: "Reports" },
          ].map((s) => (
            <div key={s.label} style={{ textAlign: "center" }}>
              <div
                style={{
                  fontSize: 28,
                  fontWeight: 500,
                  color: "var(--text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {s.value}
              </div>
              <div
                style={{
                  fontSize: 12,
                  color: "var(--text-tertiary)",
                  marginTop: 4,
                }}
              >
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── How It Works ── */}
      <section style={{
        padding: "60px 24px",
        maxWidth: 960,
        margin: "0 auto",
        position: "relative",
        zIndex: 1,
      }}>
        <p style={{
          fontSize: 11,
          fontWeight: 500,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "var(--accent)",
          marginBottom: 12,
          textAlign: "center",
        }}>How it works</p>

        <h2 style={{
          fontSize: 28,
          fontWeight: 500,
          color: "var(--text-primary)",
          textAlign: "center",
          marginBottom: 48,
          lineHeight: 1.2,
        }}>
          From configuration to research-grade results
        </h2>

        <div className="steps-grid" style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 24,
          position: "relative",
        }}>
          <div style={{
            position: "absolute",
            top: 20,
            left: "12.5%",
            right: "12.5%",
            height: 1,
            background:
              "linear-gradient(to right, transparent, var(--bg-border) 20%, var(--bg-border) 80%, transparent)",
            zIndex: 0,
          }} />

          {[
            {
              step: "01",
              title: "Choose a template",
              description:
                "Pick an environment — Resource Sharing or Head-to-Head. Configure agents, episode length, and behavioral parameters.",
            },
            {
              step: "02",
              title: "Train with self-play",
              description:
                "PPO agents train against a league of past snapshots. Policies evolve across generations with Elo-based ranking.",
            },
            {
              step: "03",
              title: "League evolves",
              description:
                "Periodic snapshots enter the league. A lineage tree builds — parent policies give rise to children with different strategies.",
            },
            {
              step: "04",
              title: "Analyze results",
              description:
                "Robustness sweeps stress-test the champion. Strategy clustering discovers emergent playstyles automatically.",
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
              <div style={{
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
              }}>
                {item.step}
              </div>
              <h3 style={{
                fontSize: 14,
                fontWeight: 500,
                color: "var(--text-primary)",
                marginBottom: 8,
              }}>
                {item.title}
              </h3>
              <p style={{
                fontSize: 13,
                color: "var(--text-secondary)",
                lineHeight: 1.6,
              }}>
                {item.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Feature Highlights ── */}
      <section
        style={{
          padding: "60px 24px",
          maxWidth: 960,
          margin: "0 auto",
          position: "relative",
          zIndex: 1,
        }}
      >
        <div style={{ display: "flex", gap: 16 }}>
          {features.map((f) => (
            <div
              key={f.label}
              style={{
                flex: "1 1 0",
                minWidth: 0,
                background: "var(--bg-surface)",
                border: "1px solid var(--bg-border)",
                borderRadius: 8,
                padding: 24,
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                  color: "var(--accent)",
                  marginBottom: 12,
                }}
              >
                {f.label}
              </div>
              <div
                style={{
                  fontSize: 15,
                  fontWeight: 500,
                  color: "var(--text-primary)",
                  marginBottom: 8,
                }}
              >
                {f.title}
              </div>
              <div
                style={{
                  fontSize: 13,
                  color: "var(--text-secondary)",
                  lineHeight: 1.6,
                }}
              >
                {f.body}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── What You Can Study ── */}
      <section style={{
        padding: "60px 24px",
        maxWidth: 960,
        margin: "0 auto",
        position: "relative",
        zIndex: 1,
      }}>
        <p style={{
          fontSize: 11, fontWeight: 500,
          letterSpacing: "0.1em",
          textTransform: "uppercase" as const,
          color: "var(--accent)",
          marginBottom: 12,
          textAlign: "center",
        }}>Research questions</p>

        <h2 style={{
          fontSize: 28, fontWeight: 500,
          color: "var(--text-primary)",
          textAlign: "center",
          marginBottom: 48, lineHeight: 1.2,
        }}>What this platform helps you study</h2>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 16,
        }} className="research-grid">
          {[
            {
              q: "When do agents cooperate vs defect?",
              desc: "Configure a shared-resource environment and watch as agents discover whether cooperation or exploitation leads to better long-term outcomes.",
              tag: "Resource Sharing",
              tagColor: "var(--accent)",
            },
            {
              q: "Which strategies survive stress testing?",
              desc: "Run robustness sweeps across 20 environment variants. See which policies hold up under information asymmetry, resource scarcity, and uncertainty.",
              tag: "Robustness",
              tagColor: "#f59e0b",
            },
            {
              q: "How do playstyles evolve across generations?",
              desc: "League-based self-play produces a lineage of agents. Strategy clustering reveals how behavioral patterns shift as policies improve.",
              tag: "Evolution",
              tagColor: "#8b5cf6",
            },
          ].map((item) => (
            <div key={item.q} style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--bg-border)",
              borderRadius: 8,
              padding: 24,
            }}>
              <span style={{
                fontSize: 10, fontWeight: 500,
                letterSpacing: "0.06em",
                textTransform: "uppercase" as const,
                color: item.tagColor,
                marginBottom: 12,
                display: "block",
              }}>{item.tag}</span>
              <h3 style={{
                fontSize: 15, fontWeight: 500,
                color: "var(--text-primary)",
                marginBottom: 10, lineHeight: 1.3,
              }}>{item.q}</h3>
              <p style={{
                fontSize: 13,
                color: "var(--text-secondary)",
                lineHeight: 1.6,
              }}>{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Quick Start ── */}
      <section
        style={{
          padding: "40px 24px",
          maxWidth: 680,
          margin: "0 auto",
          marginBottom: 80,
          textAlign: "center",
          position: "relative",
          zIndex: 1,
        }}
      >
        <div
          style={{
            fontSize: 11,
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            color: "var(--text-tertiary)",
            marginBottom: 16,
          }}
        >
          Get started
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 12,
          }}
        >
          <QuickStartButton
            href="/simulate/resource-sharing"
            label="Resource Sharing Arena →"
          />
          <QuickStartButton
            href="/simulate/head-to-head"
            label="Head-to-Head Strategy →"
          />
        </div>

        <Link
          href="/simulate"
          style={{
            fontSize: 12,
            color: "var(--text-tertiary)",
            marginTop: 12,
            display: "block",
            textAlign: "center",
          }}
        >
          Or explore all environments →
        </Link>
      </section>
      </main>
    </div>
  );
}
