"use client";

import Link from "next/link";
import { useState } from "react";

/* ── shared styles ── */
const mono = { fontFamily: "var(--font-mono)" };
const wrap = {
  maxWidth: 1200,
  margin: "0 auto" as const,
  padding: "0 48px",
};
const sectionLabel: React.CSSProperties = {
  ...mono,
  fontSize: 11,
  textTransform: "uppercase",
  letterSpacing: "0.1em",
  color: "var(--accent)",
  marginBottom: 16,
};
const sectionHeading: React.CSSProperties = {
  fontSize: 24,
  fontWeight: 500,
  color: "var(--text-primary)",
  marginBottom: 16,
  marginTop: 0,
};
const border: React.CSSProperties = {
  borderTop: "1px solid var(--bg-border)",
};

/* ── data ── */

const stats = [
  { value: "2", label: "Environment archetypes" },
  { value: "20", label: "Robustness sweep variants" },
  { value: "5", label: "Baseline agent policies (Mixed)" },
  { value: "5", label: "Baseline agent policies (Competitive)" },
  { value: "3", label: "Strategy clustering labels" },
  { value: "1", label: "Command to run full pipeline" },
];

const steps = [
  {
    num: "01",
    title: "Configure",
    desc: "Define your environment. Choose template, set agent count, episode length, and behavioral parameters.",
  },
  {
    num: "02",
    title: "Simulate",
    desc: "Agents run in the environment. Metrics stream live via WebSocket. Results saved to storage.",
  },
  {
    num: "03",
    title: "Train",
    desc: "PPO shared-policy training with league-based self-play. Periodic snapshots saved as league members.",
  },
  {
    num: "04",
    title: "Evaluate",
    desc: "Champion faces 20-variant robustness sweeps. Cross-seed evaluation produces a score and worst-case profile.",
  },
  {
    num: "05",
    title: "Analyze",
    desc: "K-means clustering identifies emergent strategies. Reports generated in JSON and Markdown.",
  },
];

const capabilities = [
  {
    group: "Simulation",
    items: [
      "Mixed environment (cooperation + competition)",
      "Competitive environment (zero-sum)",
      "PPO shared-policy training",
      "5 baseline agents per archetype",
      "WebSocket live metric streaming",
      "Deterministic seeding and reproducibility",
      "Configurable behavioral layers",
    ],
  },
  {
    group: "League",
    items: [
      "Elo rating system",
      "Self-play with periodic snapshots",
      "Parent-child lineage tracking",
      "Champion identification",
      "League member benchmarking",
      "Run-from-league-member option",
    ],
  },
  {
    group: "Evaluation",
    items: [
      "20-variant robustness sweeps",
      "Cross-seed evaluation",
      "Robustness score (0.7\u00d7mean + 0.3\u00d7worst-case)",
      "Champion vs baseline benchmark",
      "Per-sweep performance breakdown",
      "JSON + Markdown report generation",
    ],
  },
  {
    group: "Analysis",
    items: [
      "K-means strategy clustering",
      "Behavioral feature extraction",
      "Strategy label assignment",
      "Evolution tracking across generations",
      "Lineage visualization",
      "Champion history timeline",
    ],
  },
];

const backendTech = [
  { name: "Python 3.11", desc: "Core simulation engine" },
  { name: "PyTorch", desc: "PPO training" },
  { name: "FastAPI", desc: "REST + WebSocket API" },
  { name: "PettingZoo", desc: "MARL environment interface" },
  { name: "NumPy", desc: "State and reward computation" },
  { name: "Pydantic", desc: "Config validation and schemas" },
];

const frontendTech = [
  { name: "Next.js 14", desc: "React framework + routing" },
  { name: "TypeScript", desc: "Type-safe frontend" },
  { name: "Tailwind CSS", desc: "Utility styling" },
  { name: "Recharts", desc: "Metrics visualization" },
  { name: "Custom SVG", desc: "Lineage graph rendering" },
];

/* ── hover button helpers ── */

function PrimaryCTA() {
  const [hover, setHover] = useState(false);
  return (
    <Link href="/simulate" style={{ textDecoration: "none" }}>
      <button
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          background: hover ? "var(--accent-hover)" : "var(--accent)",
          color: "#0a0a0a",
          padding: "11px 28px",
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
          border: `1px solid ${hover ? "#444444" : "var(--bg-border)"}`,
          padding: "11px 28px",
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

/* ── page ── */

export default function AboutPage() {
  return (
    <main style={{ paddingTop: 48, background: "var(--bg-base)" }}>
      {/* ── Section 1: Hero ── */}
      <section style={{ ...wrap, paddingTop: 80, paddingBottom: 60 }}>
        <p style={{ ...sectionLabel, marginBottom: 16 }}>About this project</p>
        <h1
          style={{
            fontSize: 40,
            fontWeight: 600,
            color: "var(--text-primary)",
            lineHeight: 1.15,
            marginBottom: 20,
            marginTop: 0,
            maxWidth: 700,
          }}
        >
          A research platform for multi-agent strategy
        </h1>
        <p
          style={{
            fontSize: 16,
            color: "var(--text-secondary)",
            lineHeight: 1.7,
            maxWidth: 640,
            marginBottom: 0,
            marginTop: 0,
          }}
        >
          This platform was built to study how intelligent agents develop
          strategies when placed in configurable multi-agent environments. It
          combines reinforcement learning, self-play league systems, and
          research-grade evaluation into a single end-to-end pipeline.
        </p>
      </section>

      {/* ── Section 2: What it does ── */}
      <section style={{ ...wrap, padding: "60px 48px", ...border }}>
        <div
          className="about-two-col"
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 80,
            alignItems: "start",
          }}
        >
          {/* Left */}
          <div>
            <p style={sectionLabel}>Core concept</p>
            <h2 style={sectionHeading}>
              Agents that learn, compete, and evolve
            </h2>
            <p
              style={{
                fontSize: 14,
                color: "var(--text-secondary)",
                lineHeight: 1.7,
                marginBottom: 12,
                marginTop: 0,
              }}
            >
              Agents are trained using Proximal Policy Optimization (PPO) in
              configurable environments. Rather than training once and stopping,
              agents enter a self-play league — competing against snapshots of
              past versions of themselves.
            </p>
            <p
              style={{
                fontSize: 14,
                color: "var(--text-secondary)",
                lineHeight: 1.7,
                marginBottom: 12,
                marginTop: 0,
              }}
            >
              Over time, a lineage of policies forms. Each new policy is
              Elo-rated against the population. Strategy clustering identifies
              emergent behavioral patterns without any manual labeling.
            </p>
            <p
              style={{
                fontSize: 14,
                color: "var(--text-secondary)",
                lineHeight: 1.7,
                marginBottom: 0,
                marginTop: 0,
              }}
            >
              The result is a research artifact: a documented lineage of agent
              strategies, with robustness profiles and behavioral analysis for
              each generation.
            </p>
          </div>

          {/* Right — stat grid */}
          <div>
            <p style={sectionLabel}>Key numbers</p>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 1,
                background: "var(--bg-border)",
              }}
            >
              {stats.map((s) => (
                <div
                  key={s.label}
                  style={{ background: "var(--bg-surface)", padding: 20 }}
                >
                  <div
                    style={{
                      ...mono,
                      fontSize: 28,
                      fontWeight: 600,
                      color: "var(--accent)",
                      marginBottom: 4,
                    }}
                  >
                    {s.value}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                    {s.label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Section 3: Pipeline ── */}
      <section style={{ ...wrap, padding: "60px 48px", ...border }}>
        <p style={sectionLabel}>How it works</p>
        <h2 style={{ ...sectionHeading, marginBottom: 8 }}>
          The full pipeline, end to end
        </h2>
        <p
          style={{
            fontSize: 14,
            color: "var(--text-secondary)",
            marginBottom: 48,
            marginTop: 0,
          }}
        >
          From environment configuration to research-grade evaluation in a
          single automated chain.
        </p>
        <div
          className="about-pipeline"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(5, 1fr)",
            gap: 0,
          }}
        >
          {steps.map((step, i) => (
            <div
              key={step.num}
              style={{
                position: "relative",
                padding: "24px 20px",
                background: "var(--bg-surface)",
                border: "1px solid var(--bg-border)",
                borderRight:
                  i < steps.length - 1 ? "none" : "1px solid var(--bg-border)",
              }}
            >
              <div
                style={{
                  ...mono,
                  fontSize: 11,
                  textTransform: "uppercase",
                  letterSpacing: "0.08em",
                  color: "var(--accent)",
                  marginBottom: 8,
                }}
              >
                {step.num} · {step.title}
              </div>
              <div
                style={{
                  fontSize: 12,
                  color: "var(--text-secondary)",
                  lineHeight: 1.6,
                }}
              >
                {step.desc}
              </div>
              {i < steps.length - 1 && (
                <span
                  style={{
                    position: "absolute",
                    right: -12,
                    top: "50%",
                    transform: "translateY(-50%)",
                    width: 24,
                    height: 24,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "var(--text-tertiary)",
                    fontSize: 18,
                    zIndex: 1,
                  }}
                >
                  →
                </span>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ── Section 4: Capabilities ── */}
      <section style={{ ...wrap, padding: "60px 48px", ...border }}>
        <p style={sectionLabel}>Capabilities</p>
        <h2 style={{ ...sectionHeading, marginBottom: 40 }}>
          Everything that&apos;s built
        </h2>
        <div
          className="about-capabilities"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 32,
          }}
        >
          {capabilities.map((group) => (
            <div key={group.group}>
              <div
                style={{
                  ...mono,
                  fontSize: 11,
                  textTransform: "uppercase",
                  color: "var(--accent)",
                  marginBottom: 16,
                }}
              >
                {group.group}
              </div>
              {group.items.map((item) => (
                <div
                  key={item}
                  style={{
                    fontSize: 13,
                    color: "var(--text-secondary)",
                    lineHeight: 1.8,
                  }}
                >
                  <span style={{ color: "var(--accent)", marginRight: 6 }}>
                    ›
                  </span>
                  {item}
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>

      {/* ── Section 5: Archetypes ── */}
      <section style={{ ...wrap, padding: "60px 48px", ...border }}>
        <p style={sectionLabel}>Environments</p>
        <h2 style={{ ...sectionHeading, marginBottom: 40 }}>
          Two archetypes, two strategic landscapes
        </h2>
        <div
          className="about-archetypes"
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 16,
          }}
        >
          {/* Mixed */}
          <div
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--bg-border)",
              borderTop: "2px solid var(--accent)",
              borderRadius: 8,
              padding: 28,
            }}
          >
            <span
              style={{
                ...mono,
                fontSize: 10,
                textTransform: "uppercase",
                color: "var(--accent)",
                marginBottom: 12,
                display: "block",
              }}
            >
              Mixed interaction
            </span>
            <h3
              style={{
                fontSize: 18,
                fontWeight: 500,
                color: "var(--text-primary)",
                marginBottom: 12,
                marginTop: 0,
              }}
            >
              Resource Sharing Arena
            </h3>
            <p
              style={{
                fontSize: 13,
                color: "var(--text-secondary)",
                lineHeight: 1.7,
                marginBottom: 20,
                marginTop: 0,
              }}
            >
              Agents compete for a shared resource pool while facing pressure to
              cooperate for long-term survival. The environment produces emergent
              behaviors: conditional cooperation, free-riding, retaliation, and
              alliance formation.
            </p>
            <div
              style={{
                fontSize: 11,
                color: "var(--text-tertiary)",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                marginBottom: 8,
              }}
            >
              Core mechanics
            </div>
            {[
              "4 action types: cooperate, extract, defend, conditional",
              "3-component reward: individual + group + relational",
              "5 behavioral layers: information asymmetry, memory, reputation, incentives, uncertainty",
              "Termination: max steps, system collapse, no active agents",
              "Strategy labels: Cooperative, Exploitative, Robust, Unstable",
            ].map((m) => (
              <div
                key={m}
                style={{
                  fontSize: 12,
                  color: "var(--text-secondary)",
                  lineHeight: 1.8,
                }}
              >
                · {m}
              </div>
            ))}
          </div>

          {/* Competitive */}
          <div
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--bg-border)",
              borderTop: "2px solid #f97316",
              borderRadius: 8,
              padding: 28,
            }}
          >
            <span
              style={{
                ...mono,
                fontSize: 10,
                textTransform: "uppercase",
                color: "#f97316",
                marginBottom: 12,
                display: "block",
              }}
            >
              Competitive
            </span>
            <h3
              style={{
                fontSize: 18,
                fontWeight: 500,
                color: "var(--text-primary)",
                marginBottom: 12,
                marginTop: 0,
              }}
            >
              Head-to-Head Strategy
            </h3>
            <p
              style={{
                fontSize: 13,
                color: "var(--text-secondary)",
                lineHeight: 1.7,
                marginBottom: 20,
                marginTop: 0,
              }}
            >
              Pure zero-sum competition. Agents accumulate scores through build,
              attack, defend, and gamble actions. Rankings update each step. One
              winner per episode, determined by final score.
            </p>
            <div
              style={{
                fontSize: 11,
                color: "var(--text-tertiary)",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                marginBottom: 8,
              }}
            >
              Core mechanics
            </div>
            {[
              "4 action types: build, attack, defend, gamble",
              "Score-based reward with relative gain component",
              "Opponent history tracking and information asymmetry",
              "Elimination mechanic (configurable)",
              "Strategy labels: Dominant, Aggressive, Consistent, Weak",
            ].map((m) => (
              <div
                key={m}
                style={{
                  fontSize: 12,
                  color: "var(--text-secondary)",
                  lineHeight: 1.8,
                }}
              >
                · {m}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Section 6: Tech Stack ── */}
      <section style={{ ...wrap, padding: "60px 48px", ...border }}>
        <p style={sectionLabel}>Built with</p>
        <h2 style={{ ...sectionHeading, marginBottom: 32 }}>Tech stack</h2>
        <div
          className="about-tech"
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 48,
          }}
        >
          {/* Backend */}
          <div>
            <div
              style={{
                ...mono,
                fontSize: 11,
                textTransform: "uppercase",
                color: "var(--text-tertiary)",
                marginBottom: 12,
              }}
            >
              Backend &amp; ML
            </div>
            {backendTech.map((t) => (
              <div
                key={t.name}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "8px 0",
                  borderBottom: "1px solid var(--bg-border)",
                  fontSize: 13,
                }}
              >
                <span style={{ color: "var(--text-primary)" }}>{t.name}</span>
                <span style={{ color: "var(--text-secondary)" }}>{t.desc}</span>
              </div>
            ))}
          </div>

          {/* Frontend */}
          <div>
            <div
              style={{
                ...mono,
                fontSize: 11,
                textTransform: "uppercase",
                color: "var(--text-tertiary)",
                marginBottom: 12,
              }}
            >
              Frontend
            </div>
            {frontendTech.map((t) => (
              <div
                key={t.name}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "8px 0",
                  borderBottom: "1px solid var(--bg-border)",
                  fontSize: 13,
                }}
              >
                <span style={{ color: "var(--text-primary)" }}>{t.name}</span>
                <span style={{ color: "var(--text-secondary)" }}>{t.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Section 7: CTA ── */}
      <section
        style={{
          ...wrap,
          padding: "60px 48px 100px",
          ...border,
          textAlign: "center",
        }}
      >
        <h2
          style={{
            fontSize: 28,
            fontWeight: 500,
            color: "var(--text-primary)",
            marginBottom: 12,
            marginTop: 0,
          }}
        >
          See it in action
        </h2>
        <p
          style={{
            fontSize: 14,
            color: "var(--text-secondary)",
            marginBottom: 32,
            marginTop: 0,
          }}
        >
          Start a simulation and watch agents compete and evolve in real time.
        </p>
        <div
          style={{
            display: "flex",
            gap: 12,
            justifyContent: "center",
          }}
        >
          <PrimaryCTA />
          <SecondaryCTA />
        </div>
      </section>
    </main>
  );
}
