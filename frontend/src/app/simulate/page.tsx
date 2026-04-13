"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

// ---------------------------------------------------------------------------
// Archetype-aware run counting
// ---------------------------------------------------------------------------

const MIXED_ONLY = new Set([
  "always_cooperate",
  "always_extract",
  "tit_for_tat",
  "ppo_shared",
  "league_snapshot",
]);

const COMPETITIVE_ONLY = new Set([
  "always_attack",
  "always_build",
  "always_defend",
  "competitive_ppo",
]);

function useRunCounts() {
  const [resourceRuns, setResourceRuns] = useState<string>("—");
  const [headToHeadRuns, setHeadToHeadRuns] = useState<string>("—");
  const [cooperativeRuns, setCooperativeRuns] = useState<string>("—");

  useEffect(() => {
    fetch("/api/runs/history")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: Array<{ agent_policy?: string | null }>) => {
        let mixed = 0;
        let competitive = 0;
        let ambiguous = 0;

        for (const run of data) {
          const policy = run.agent_policy ?? "";
          if (MIXED_ONLY.has(policy)) mixed++;
          else if (COMPETITIVE_ONLY.has(policy)) competitive++;
          else ambiguous++; // 'random', null, empty, or unknown
        }

        setResourceRuns(String(mixed + Math.ceil(ambiguous / 2)));
        setHeadToHeadRuns(String(competitive + Math.floor(ambiguous / 2)));
      })
      .catch(() => {
        setResourceRuns("—");
        setHeadToHeadRuns("—");
      });

    fetch("/api/cooperative/runs")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: unknown[]) => setCooperativeRuns(String(data.length)))
      .catch(() => setCooperativeRuns("—"));
  }, []);

  return { resourceRuns, headToHeadRuns, cooperativeRuns };
}

// ---------------------------------------------------------------------------
// League member count hook
// ---------------------------------------------------------------------------

function useMembers(membersUrl: string) {
  const [members, setMembers] = useState<string>("—");

  useEffect(() => {
    fetch(membersUrl)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: unknown[]) => setMembers(String(data.length)))
      .catch(() => setMembers("—"));
  }, [membersUrl]);

  return members;
}

// ---------------------------------------------------------------------------
// Coming-soon templates
// ---------------------------------------------------------------------------

const comingSoon = [
  {
    name: "Algorithmic Auction Arena",
    desc: "Agents bid strategically in repeated auctions",
  },
  {
    name: "Team-Based Market Simulation",
    desc: "Firms compete while employees cooperate internally",
  },
  {
    name: "Multi-Team Resource Control",
    desc: "Squad-based territory and resource competition",
  },
  {
    name: "Negotiation Arena",
    desc: "Agents form alliances, make deals, and defect",
  },
];

// ---------------------------------------------------------------------------
// Shared styles
// ---------------------------------------------------------------------------

const sectionLabel: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: "var(--text-tertiary)",
  marginBottom: 16,
};

const cardBase: React.CSSProperties = {
  background: "var(--bg-surface)",
  border: "1px solid var(--bg-border)",
  borderRadius: 8,
  padding: 24,
};

// ---------------------------------------------------------------------------
// Hero template card
// ---------------------------------------------------------------------------

function HeroCard({
  title,
  subtitle,
  tags,
  accentColor,
  accentSubtle,
  accentBorder,
  runs,
  membersUrl,
  href,
}: {
  title: string;
  subtitle: string;
  tags: string[];
  accentColor: string;
  accentSubtle: string;
  accentBorder: string;
  runs: string;
  membersUrl: string;
  href: string;
}) {
  const members = useMembers(membersUrl);
  const [hoverBtn, setHoverBtn] = useState(false);

  const hoverColor =
    accentColor === "var(--accent)"
      ? "var(--accent-hover)"
      : accentColor === "#f97316"
      ? "#ea580c"
      : "#7c3aed";

  return (
    <div
      style={{
        ...cardBase,
        flex: "1 1 0",
        minWidth: 0,
        borderTop: `2px solid ${accentColor}`,
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div style={{ fontSize: 16, fontWeight: 500, color: "var(--text-primary)" }}>
        {title}
      </div>
      <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>
        {subtitle}
      </div>

      {/* Tag pills */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {tags.map((t) => (
          <span
            key={t}
            style={{
              fontSize: 11,
              background: accentSubtle,
              color: accentColor,
              border: `1px solid ${accentBorder}`,
              borderRadius: 9999,
              padding: "2px 8px",
            }}
          >
            {t}
          </span>
        ))}
      </div>

      {/* Stats */}
      <div style={{ fontSize: 13, color: "var(--text-tertiary)" }}>
        {runs} runs · {members} league members
      </div>

      {/* Launch button */}
      <Link href={href} style={{ textDecoration: "none", marginTop: "auto" }}>
        <button
          onMouseEnter={() => setHoverBtn(true)}
          onMouseLeave={() => setHoverBtn(false)}
          style={{
            width: "100%",
            height: 36,
            background: hoverBtn ? hoverColor : accentColor,
            color: "white",
            fontSize: 13,
            fontWeight: 500,
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            transition: "background 150ms",
          }}
        >
          Launch →
        </button>
      </Link>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Advanced archetype card
// ---------------------------------------------------------------------------

function AdvancedCard({
  title,
  description,
  accentColor,
  accentSubtle,
  href,
}: {
  title: string;
  description: string;
  accentColor: string;
  accentSubtle: string;
  href: string;
}) {
  const [hover, setHover] = useState(false);

  return (
    <div style={{ ...cardBase, flex: "1 1 0", minWidth: 0 }}>
      <div
        style={{
          fontSize: 15,
          fontWeight: 500,
          color: "var(--text-primary)",
          marginBottom: 8,
        }}
      >
        {title}
      </div>
      <div
        style={{
          fontSize: 13,
          color: "var(--text-secondary)",
          lineHeight: 1.5,
          marginBottom: 20,
        }}
      >
        {description}
      </div>
      <Link href={href} style={{ textDecoration: "none" }}>
        <button
          onMouseEnter={() => setHover(true)}
          onMouseLeave={() => setHover(false)}
          style={{
            height: 36,
            padding: "0 16px",
            background: hover ? accentSubtle : "transparent",
            color: accentColor,
            fontSize: 13,
            fontWeight: 500,
            border: `1px solid ${accentColor}`,
            borderRadius: 6,
            cursor: "pointer",
            transition: "background 150ms",
          }}
        >
          Open full config →
        </button>
      </Link>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function SimulatePage() {
  const [tab, setTab] = useState<"templates" | "advanced">("templates");
  const { resourceRuns, headToHeadRuns, cooperativeRuns } = useRunCounts();

  return (
    <main
      style={{
        maxWidth: 960,
        margin: "0 auto",
        padding: "48px 24px",
        paddingTop: 96, // account for fixed nav
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1
          style={{
            fontSize: 24,
            fontWeight: 500,
            color: "var(--text-primary)",
            margin: 0,
          }}
        >
          Simulate
        </h1>
        <p
          style={{
            fontSize: 14,
            color: "var(--text-secondary)",
            margin: "8px 0 0",
          }}
        >
          Choose an environment to simulate, or configure directly
        </p>
      </div>

      {/* Tab toggle */}
      <div
        style={{
          display: "inline-flex",
          background: "var(--bg-elevated)",
          borderRadius: 8,
          padding: 3,
          marginBottom: 32,
        }}
      >
        {(["templates", "advanced"] as const).map((t) => {
          const active = tab === t;
          return (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: "6px 18px",
                fontSize: 13,
                fontWeight: 500,
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                transition: "all 150ms",
                background: active ? "var(--accent)" : "transparent",
                color: active ? "white" : "var(--text-secondary)",
              }}
            >
              {t === "templates" ? "Templates" : "Advanced"}
            </button>
          );
        })}
      </div>

      {/* ---- Templates tab ---- */}
      {tab === "templates" && (
        <>
          {/* Live environments */}
          <div style={sectionLabel}>Live environments</div>
          <div style={{ display: "flex", gap: 16, marginBottom: 40 }}>
            <HeroCard
              title="Resource Sharing Arena"
              subtitle="Agents share a common resource pool and decide when to cooperate or compete. Watch emergent strategies develop over time."
              tags={["Cooperation", "Defection", "Emergent Strategy"]}
              accentColor="var(--accent)"
              accentSubtle="var(--accent-subtle)"
              accentBorder="var(--accent-border)"
              runs={resourceRuns}
              membersUrl="/api/league/members"
              href="/simulate/resource-sharing"
            />
            <HeroCard
              title="Head-to-Head Strategy"
              subtitle="Pure zero-sum competition. Agents fight for score dominance. One winner per episode."
              tags={["Zero-sum", "Rankings", "Score Spread"]}
              accentColor="#f97316"
              accentSubtle="rgba(249,115,22,0.1)"
              accentBorder="rgba(249,115,22,0.2)"
              runs={headToHeadRuns}
              membersUrl="/api/competitive/league/members"
              href="/simulate/head-to-head"
            />
            <HeroCard
              title="Cooperative Task Simulation"
              subtitle="Agents share a task queue and coordinate effort to prevent system collapse. Emergent specialization and free-rider dynamics unfold over time."
              tags={["Coordination", "Specialization", "Collective Goal"]}
              accentColor="#8b5cf6"
              accentSubtle="rgba(139,92,246,0.1)"
              accentBorder="rgba(139,92,246,0.2)"
              runs={cooperativeRuns}
              membersUrl="/api/cooperative/league/members"
              href="/simulate/cooperative"
            />
          </div>

          {/* Coming soon */}
          <div style={sectionLabel}>Coming soon</div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
              gap: 16,
            }}
          >
            {comingSoon.map((c) => (
              <div
                key={c.name}
                style={{
                  ...cardBase,
                  opacity: 0.5,
                  cursor: "default",
                  position: "relative",
                }}
              >
                {/* Badge */}
                <span
                  style={{
                    position: "absolute",
                    top: 12,
                    right: 12,
                    fontSize: 10,
                    background: "var(--bg-elevated)",
                    color: "var(--text-tertiary)",
                    padding: "2px 6px",
                    borderRadius: 4,
                  }}
                >
                  In development
                </span>
                <div
                  style={{
                    fontSize: 14,
                    fontWeight: 500,
                    color: "var(--text-secondary)",
                    marginBottom: 6,
                    paddingRight: 80,
                  }}
                >
                  {c.name}
                </div>
                <div style={{ fontSize: 12, color: "var(--text-tertiary)", lineHeight: 1.4 }}>
                  {c.desc}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* ---- Advanced tab ---- */}
      {tab === "advanced" && (
        <>
          <div style={sectionLabel}>Direct archetype access</div>
          <p
            style={{
              fontSize: 13,
              color: "var(--text-secondary)",
              margin: "0 0 24px",
            }}
          >
            Full parameter control. For researchers who want to configure
            environments directly.
          </p>
          <div style={{ display: "flex", gap: 16 }}>
            <AdvancedCard
              title="Resource Sharing"
              description="Mixed interaction archetype — cooperation, defection, shared resources."
              accentColor="var(--accent)"
              accentSubtle="var(--accent-subtle)"
              href="/simulate/resource-sharing?mode=advanced"
            />
            <AdvancedCard
              title="Head-to-Head"
              description="Competitive archetype — zero-sum, scoring, rankings, elimination."
              accentColor="#f97316"
              accentSubtle="rgba(249,115,22,0.1)"
              href="/simulate/head-to-head?mode=advanced"
            />
            <AdvancedCard
              title="Cooperative"
              description="Cooperative archetype — shared task queue, collective goal, specialization."
              accentColor="#8b5cf6"
              accentSubtle="rgba(139,92,246,0.1)"
              href="/simulate/cooperative?mode=advanced"
            />
          </div>
        </>
      )}
    </main>
  );
}
