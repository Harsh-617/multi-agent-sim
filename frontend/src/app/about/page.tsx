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
  { value: "11", label: "Strategy cluster labels" },
  { value: "20", label: "Robustness sweep variants" },
  { value: "5+5", label: "Baseline agent policies" },
  { value: "266", label: "Automated tests passing" },
  { value: "1", label: "Command to run full pipeline" },
];

const steps = [
  {
    num: "01",
    title: "Configure",
    desc: "Choose a simulation template — Resource Sharing or Head-to-Head. Set the number of agents, episode length, random seed, and behavioral parameters such as memory depth, information asymmetry, and observation noise. Save your config to reuse it across multiple runs.",
  },
  {
    num: "02",
    title: "Simulate",
    desc: "Start a run. Agents step through the environment, selecting actions and receiving rewards each timestep. Metrics stream live to the dashboard via WebSocket — shared pool level, cooperation rate, reward curves, and elimination events. All data is saved to storage for replay.",
  },
  {
    num: "03",
    title: "Train",
    desc: "PPO trains a shared-policy neural network across league-sampled opponents. Every N steps, a snapshot of the current policy is saved as a league member. Opponent sampling is weighted toward recent members to maintain competitive pressure without losing diversity.",
  },
  {
    num: "04",
    title: "Evaluate",
    desc: "The league champion is identified by Elo rating and tested across 20 environment variants — different resource levels, noise magnitudes, asymmetry conditions, and agent counts. Cross-seed evaluation runs multiple seeds per variant. The robustness score is computed as 0.7 × mean performance + 0.3 × worst-case performance.",
  },
  {
    num: "05",
    title: "Analyze",
    desc: "K-means clustering groups all league members by behavioral features extracted from their evaluation episodes. Each cluster is automatically assigned a strategy label. A lineage graph shows the full parent-child chain of policy snapshots with ratings, labels, and robustness scores at each node. Reports are generated in both JSON and Markdown.",
  },
];

const pages = [
  {
    label: "/ simulate",
    url: "/simulate",
    heading: "Simulate — configure and run environments",
    description:
      "The Simulate page is your entry point for running experiments. It has two modes toggled at the top: Templates and Advanced. Templates mode shows two live environment cards — Resource Sharing Arena and Head-to-Head Strategy — each with a launch button, a run count, and a brief description. Advanced mode exposes the full configuration form for both environments directly, giving you control over every parameter.",
    sections: [
      {
        name: "Templates tab",
        detail:
          "pick an environment and launch directly. Each card shows the environment name, a description, how many runs have been completed, and a Launch button that takes you to the full configuration page.",
      },
      {
        name: "Advanced tab",
        detail:
          "shows the complete config form for both archetypes side by side. Use this when you want to set specific values for parameters like memory depth, information asymmetry, observation noise, or reward weights before starting a run.",
      },
    ],
  },
  {
    label: "/ simulate / resource-sharing",
    url: "/simulate/resource-sharing",
    heading: "Resource Sharing Arena — configure and start a run",
    description:
      "This is the full configuration page for the Resource Sharing environment. The left panel contains the configuration form. The right panel shows your run history for this template. Fill in the config, hit Start Run, and you will be taken to the live run page.",
    sections: [
      {
        name: "Config panel (left)",
        detail:
          "set number of agents (2–10), episode length (max steps), random seed, and agent policy. The policy selector lets you choose from baseline agents (Always Cooperate, Always Extract, Tit-for-Tat, Random, Conditional) or the trained PPO agent if artifacts exist. Advanced parameters include memory depth, reputation tracking, information asymmetry, observation noise, cooperation pressure, and reward weights.",
      },
      {
        name: "Run history panel (right)",
        detail:
          "shows all past Resource Sharing runs sorted newest first. Each row shows the policy used, number of steps, termination reason, and timestamp. Click any row to open the replay for that run.",
      },
    ],
  },
  {
    label: "/ simulate / head-to-head",
    url: "/simulate/head-to-head",
    heading: "Head-to-Head Strategy — configure and start a run",
    description:
      "The full configuration page for the Head-to-Head competitive environment. Same layout as Resource Sharing — config on the left, run history on the right.",
    sections: [
      {
        name: "Config panel (left)",
        detail:
          "set number of agents, episode length, seed, and agent policy. Competitive-specific parameters include elimination threshold (resource level below which an agent is eliminated), opponent observation window (how many steps of opponent history agents can see), and history sensitivity (how strongly agents react to opponent attack patterns).",
      },
      {
        name: "Run history panel (right)",
        detail:
          "shows all past Head-to-Head runs. Each row shows policy, steps completed, termination reason (max steps or elimination), and timestamp. Click any row to open the replay.",
      },
    ],
  },
  {
    label: "/ simulate / [template] / run / [id]",
    url: "",
    heading: "Live run page — watch a simulation in real time",
    description:
      "After starting a run, you are taken to the live run page. This page streams metrics from the backend via WebSocket and updates in real time as the simulation progresses. When the run finishes, the final summary is displayed.",
    sections: [
      {
        name: "Live metrics chart",
        detail:
          "shows reward curves, shared pool level (Resource Sharing), and cooperation rate updating every step in real time. The chart stops updating when the run ends.",
      },
      {
        name: "Run summary",
        detail:
          "displayed when the run completes. Shows total steps, termination reason, episode outcome, and per-agent statistics. For Head-to-Head runs, shows final rankings, scores, and elimination order.",
      },
      {
        name: "Replay button",
        detail:
          "once the run is complete, a Replay button appears that takes you to the replay page for this run.",
      },
    ],
  },
  {
    label: "/ simulate / [template] / replay / [id]",
    url: "",
    heading: "Replay page — step through a completed run",
    description:
      "The replay page lets you play back any completed run step by step. Metrics are streamed from storage via Server-Sent Events (SSE) and replayed at a controlled pace, giving you the same view you would have seen during the live run.",
    sections: [
      {
        name: "Replay chart",
        detail:
          "shows the same metrics as the live run page but replayed from saved data. You can see exactly how the shared pool evolved, how cooperation rates shifted, and when key events like resource collapse or agent elimination occurred.",
      },
    ],
  },
  {
    label: "/ league",
    url: "/league",
    heading: "League — manage the self-play population and run the pipeline",
    description:
      "The League page is the control center for the self-play league system. At the top is the Pipeline panel, which lets you run the full training pipeline for either environment. Below that is the archetype switcher — Resource Sharing and Head-to-Head — with separate league data for each. Each archetype tab has four sub-tabs: Ratings, Champion, Evolution, and Robustness.",
    sections: [
      {
        name: "Pipeline panel",
        detail:
          "at the top of the page. Configure training steps, snapshot interval, and seed, then click Run Pipeline. The panel polls the backend and shows live stage updates (training → snapshotting → rating → evaluating → done). When complete, a link to the generated report appears. Both archetypes have their own pipeline controls.",
      },
      {
        name: "Ratings tab",
        detail:
          "shows all league members as a ranked table sorted by Elo rating. Each row shows member ID, rating, creation timestamp, and a Run button to run a league episode from that snapshot. A Recompute Ratings button re-runs all head-to-head matches and recalculates the full Elo table.",
      },
      {
        name: "Champion tab",
        detail:
          "shows the current highest-rated league member. Displays their Elo rating, member ID, and a benchmark comparison chart showing their performance against all baseline policies. Also contains a Run Robustness button that triggers a full 20-variant robustness sweep for the champion.",
      },
      {
        name: "Evolution tab",
        detail:
          "shows the full lineage graph of the league as an SVG tree. Each node represents a policy snapshot with its Elo rating and strategy label. Edges connect parent policies to their children. Clicking a node opens a detail panel showing rating, label, robustness score, and creation time. Below the graph is the Champion History — a chronological list of every snapshot with its label and rating.",
      },
      {
        name: "Robustness results",
        detail:
          "after running robustness from the Champion tab, the results appear as a heatmap and a bar chart. The heatmap shows performance across all 20 environment variants for each policy. The bar chart shows mean and worst-case performance per policy side by side.",
      },
    ],
  },
  {
    label: "/ research",
    url: "/research",
    heading: "Research — browse and filter all generated reports",
    description:
      "The Research page is a unified browser for all reports generated by the platform — robustness reports, strategy analysis reports, and benchmark reports from both environments. Use the filter bar to narrow by environment or report type, and the sort control to order by date or robustness score.",
    sections: [
      {
        name: "Filter bar",
        detail:
          "filter reports by environment (Resource Sharing, Head-to-Head, or All) and by report type (Robustness, Strategy, Benchmark, or All). Filters can be combined.",
      },
      {
        name: "Sort control",
        detail:
          "sort reports by Latest first or by Highest robustness score. Useful when comparing champion performance across multiple pipeline runs.",
      },
      {
        name: "Report cards",
        detail:
          "each card shows the report type badge, environment badge, date, robustness score if available, and an Open button. Click Open to go to the full report detail page.",
      },
    ],
  },
  {
    label: "/ research / [report_id]",
    url: "",
    heading: "Report detail — full research output for a single run",
    description:
      "The report detail page shows the complete output of a robustness evaluation or strategy analysis. This is the primary research artifact produced by the platform — a structured breakdown of how the champion policy performed across all test conditions.",
    sections: [
      {
        name: "Report header",
        detail:
          "shows report type, environment, date generated, and overall robustness score.",
      },
      {
        name: "Robustness heatmap",
        detail:
          "a grid of policy × environment variant cells, color-coded by performance. Darker cells indicate lower performance. Use this to identify which environment conditions are most challenging for each policy.",
      },
      {
        name: "Performance bar chart",
        detail:
          "shows mean and worst-case performance for each policy side by side. Policies with a large gap between mean and worst-case are fragile under distributional shift.",
      },
      {
        name: "Strategy groups",
        detail:
          "shows the strategy cluster assignments for all evaluated policies, with the cluster label, behavioral description, and which policies belong to each cluster.",
      },
    ],
  },
];

const capabilityDescriptions: Record<string, string> = {
  "Simulation Engine":
    "The simulation engine is the core of the platform. It defines the rules, state, observations, and reward structure for each environment. Both environments are fully deterministic given a seed — the same config and seed will always produce the same trajectory, making experiments reproducible. The engine is built on a common BaseEnvironment interface and exposes a PettingZoo-compatible adapter for standard multi-agent interoperability.",
  "Training & League":
    "The training system handles everything from raw PPO optimization to the self-play league that drives long-term improvement. Training is not a one-shot process — agents are trained continuously against a population of past versions of themselves, with the league growing as new snapshots are added. The full pipeline from training to rating to evaluation can be triggered with a single button from the dashboard.",
  "Evaluation & Analysis":
    "Evaluation goes beyond reward curves. The robustness system tests the champion policy across 20 systematically varied environment configurations, producing a score that accounts for both average performance and worst-case resilience. Strategy clustering then extracts behavioral features from evaluation episodes and groups policies into labeled archetypes — making emergent behavior legible without manual inspection.",
  "Platform & API":
    "The backend is a FastAPI application with async task execution for long-running operations like pipeline runs and robustness sweeps. All results are stored on the filesystem — no database required. The frontend is a Next.js 14 dashboard with live WebSocket streaming for run metrics and SSE streaming for replay. The full system is covered by 266 automated tests across unit and integration modules.",
};

const capabilities = [
  {
    group: "Simulation Engine",
    items: [
      "Configurable shared-resource environment (Resource Sharing)",
      "Configurable zero-sum competitive environment (Head-to-Head)",
      "4 action types per archetype with amount validation",
      "7 behavioral layers: memory, reputation, asymmetry, noise, incentives",
      "3-component reward model: individual + group + relational",
      "Deterministic seeding \u2014 same config + seed = identical trajectory",
      "PettingZoo-compatible adapter for both archetypes",
    ],
  },
  {
    group: "Training & League",
    items: [
      "PPO shared-policy training with configurable hyperparameters",
      "League-based self-play with periodic policy snapshots",
      "Elo rating system for all league members",
      "Weighted opponent sampling (recent-biased)",
      "5 baseline agent policies per archetype",
      "Full pipeline automation: train \u2192 snapshot \u2192 rate \u2192 evaluate \u2192 report",
      "One-command pipeline execution from the dashboard",
    ],
  },
  {
    group: "Evaluation & Analysis",
    items: [
      "Cross-seed robustness evaluation across 20 environment variants",
      "Robustness score: 0.7 \u00d7 mean + 0.3 \u00d7 worst-case",
      "K-means strategy clustering with 11 labeled behavioral archetypes",
      "Lineage graph showing full parent-child policy chain",
      "Champion benchmark against all baseline policies",
      "Reports exported as JSON + Markdown",
      "Robustness heatmap and scatter chart per policy",
    ],
  },
  {
    group: "Platform & API",
    items: [
      "FastAPI backend with 7 route modules",
      "Async task execution for pipeline and robustness runs",
      "WebSocket streaming for live run metrics",
      "SSE streaming for replay playback",
      "Filesystem-backed storage \u2014 no database required",
      "Next.js 14 dashboard with dark theme",
      "266 automated tests: 22 unit modules + 4 integration modules",
    ],
  },
];

const backendTech = [
  { name: "Python 3.11", desc: "Core runtime" },
  { name: "FastAPI", desc: "Async REST API with 7 route modules" },
  { name: "PyTorch", desc: "PPO neural network training and inference" },
  { name: "NumPy", desc: "Environment simulation and RNG isolation" },
  { name: "Pydantic", desc: "Config validation and serialization" },
  { name: "PettingZoo", desc: "Multi-agent environment interface standard" },
];

const frontendTech = [
  { name: "Next.js 14", desc: "App Router" },
  { name: "TypeScript", desc: "Type-safe frontend" },
  { name: "Recharts", desc: "Reward curves and metric charts" },
  { name: "Custom SVG", desc: "Lineage graph and robustness heatmap" },
  { name: "WebSocket + SSE", desc: "Live metrics and replay streaming" },
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
          A research platform for multi-agent strategy and emergent behavior
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
          strategies when placed in competitive and cooperative environments. It
          combines reinforcement learning, self-play league systems, robustness
          evaluation, and strategy analysis into a single end-to-end research
          pipeline — from environment configuration to a documented report of
          emergent behavior.
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
              configurable multi-agent environments. At each training step,
              agents observe the environment state, select actions, and receive
              rewards shaped by individual performance, group outcomes, and
              relational dynamics.
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
              Rather than training once and stopping, agents enter a self-play
              league. Periodic policy snapshots are saved, Elo-rated against the
              full population, and used as opponents in future training runs.
              Over time, a lineage of policies forms — each generation improving
              on the last.
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
              Strategy clustering runs K-means on behavioral features extracted
              from evaluation episodes — cooperation rate, extraction frequency,
              defense ratio — and automatically assigns human-readable labels
              like Cooperator, Exploiter, or Opportunist. No manual annotation
              required.
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
              The result is a fully documented research artifact: a lineage
              graph of agent generations, Elo ratings across the population,
              robustness profiles across 20 environment variants, and strategy
              labels showing how behavior evolved.
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

      {/* ── Section 3.5: User Guide ── */}
      <section style={{ ...wrap, padding: "60px 48px", ...border }}>
        <p style={sectionLabel}>User guide</p>
        <h2 style={{ ...sectionHeading, marginBottom: 8 }}>
          What each page does and how to use it
        </h2>
        <p
          style={{
            fontSize: 14,
            color: "var(--text-secondary)",
            marginBottom: 48,
            marginTop: 0,
            lineHeight: 1.7,
          }}
        >
          The platform has three main sections accessible from the navigation
          bar — Simulate, League, and Research. Here is a complete breakdown of
          every page, every tab, and every panel.
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 40 }}>
          {pages.map((page) => (
            <div key={page.label}>
              <div
                style={{
                  ...mono,
                  fontSize: 13,
                  color: "var(--accent)",
                  marginBottom: 4,
                }}
              >
                {page.label}
              </div>
              {page.url && (
                <div
                  style={{
                    fontSize: 12,
                    color: "var(--text-tertiary)",
                    marginBottom: 12,
                  }}
                >
                  {page.url}
                </div>
              )}
              {!page.url && (
                <div
                  style={{
                    fontSize: 12,
                    color: "var(--text-tertiary)",
                    marginBottom: 12,
                  }}
                >
                  {page.label}
                </div>
              )}
              <h3
                style={{
                  fontSize: 16,
                  fontWeight: 500,
                  color: "var(--text-primary)",
                  marginBottom: 8,
                  marginTop: 0,
                }}
              >
                {page.heading}
              </h3>
              <p
                style={{
                  fontSize: 13,
                  color: "var(--text-secondary)",
                  lineHeight: 1.7,
                  marginBottom: 16,
                  marginTop: 0,
                }}
              >
                {page.description}
              </p>
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 8,
                }}
              >
                {page.sections.map((s) => (
                  <div
                    key={s.name}
                    style={{
                      fontSize: 13,
                      color: "var(--text-secondary)",
                      lineHeight: 1.7,
                    }}
                  >
                    <span
                      style={{
                        color: "var(--accent)",
                        marginRight: 6,
                      }}
                    >
                      ›
                    </span>
                    <strong style={{ color: "var(--text-primary)" }}>
                      {s.name}
                    </strong>{" "}
                    — {s.detail}
                  </div>
                ))}
              </div>
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
              {capabilityDescriptions[group.group] && (
                <p
                  style={{
                    fontSize: 13,
                    color: "var(--text-secondary)",
                    lineHeight: 1.7,
                    marginTop: 0,
                    marginBottom: 12,
                  }}
                >
                  {capabilityDescriptions[group.group]}
                </p>
              )}
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
          Two archetypes, two research questions
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
              Cooperative / Competitive
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
              Agents share a common resource pool that regenerates each step.
              Every agent chooses an action — Cooperate (contribute to the
              pool), Extract (take from it), Defend (protect their share), or
              Conditional (mirror the group&apos;s recent behavior). Cooperation
              increases the pool for everyone; extraction benefits the
              individual at the group&apos;s expense. The environment supports
              configurable memory depth, reputation tracking, information
              asymmetry, observation noise, and incentive structures. The core
              research question: under what conditions does sustained
              cooperation emerge as the rational strategy?
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
              "4 action types: Cooperate, Extract, Defend, Conditional",
              "3-component reward: individual + group + relational",
              "7 behavioral layers: memory, reputation, asymmetry, noise, incentives, and more",
              "Termination: max steps, system collapse, no active agents",
              "Core question: when does sustained cooperation become rational?",
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
              Zero-Sum Competitive
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
              Agents compete directly for resources in a zero-sum environment.
              Every step, each agent chooses to Build (accumulate resources),
              Attack (take from opponents), Defend (reduce incoming damage), or
              Gamble (high-variance build with a random multiplier). Agents with
              resources below the elimination threshold are removed from the
              episode. A terminal bonus rewards relative ranking at episode end.
              The environment supports configurable opponent observation windows,
              history sensitivity for retaliation behavior, and elimination
              thresholds. The core research question: which competitive
              strategies survive elimination pressure and dominate across
              generations?
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
              "4 action types: Build, Attack, Defend, Gamble",
              "Terminal bonus rewards relative ranking at episode end",
              "Configurable opponent observation and history sensitivity",
              "Elimination mechanic with configurable threshold",
              "Core question: which strategies survive elimination pressure?",
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
        <p style={sectionLabel}>Tech stack</p>
        <h2 style={{ ...sectionHeading, marginBottom: 32 }}>What it&apos;s built with</h2>
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
              Backend
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
