"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  CooperativeLeagueMember,
  CooperativeChampionInfo,
  CooperativeEvolutionResponse,
  CooperativeRobustnessHeatmapResponse,
  CooperativeRobustnessStatusResponse,
  getCooperativeLeagueMembers,
  getCooperativeChampion,
  getCooperativeLeagueLineage,
  getCooperativeLeagueEvolution,
  runCooperativeChampionRobustness,
  getCooperativeRobustnessStatus,
  CooperativeLineageMember,
} from "@/lib/api";
import CooperativeLeagueLineage from "@/components/CooperativeLeagueLineage";
import CooperativeChampionBenchmark from "@/components/CooperativeChampionBenchmark";
import CooperativeChampionRobustness from "@/components/CooperativeChampionRobustness";
import CooperativeRobustScatter from "@/components/CooperativeRobustScatter";
import CooperativeStrategyGroups from "@/components/CooperativeStrategyGroups";

type Tab = "lineage" | "champion" | "robustness" | "evolution";

const LABEL_COLORS: Record<string, string> = {
  "Dedicated Specialist": "#14b8a6",
  "Adaptive Generalist": "#3b82f6",
  "Free Rider": "#ef4444",
  "Overcontributor": "#f59e0b",
  "Opportunist": "#8b5cf6",
  Developing: "#6b7280",
};

function labelColor(label: string): string {
  return LABEL_COLORS[label] ?? "#6b7280";
}

// ---------------------------------------------------------------------------
// Archetype switcher header
// ---------------------------------------------------------------------------

function ArchetypeSwitcher() {
  return (
    <div style={{
      display: "flex",
      gap: 8,
      marginBottom: 20,
    }}>
      <Link
        href="/league"
        style={{
          padding: "6px 14px",
          borderRadius: 4,
          fontSize: 13,
          background: "#111111",
          border: "1px solid #222222",
          color: "#888888",
          textDecoration: "none",
          cursor: "pointer",
        }}
      >
        Resource Sharing
      </Link>
      <Link
        href="/league/competitive"
        style={{
          padding: "6px 14px",
          borderRadius: 4,
          fontSize: 13,
          background: "#111111",
          border: "1px solid #222222",
          color: "#888888",
          textDecoration: "none",
          cursor: "pointer",
        }}
      >
        Head-to-Head
      </Link>
      <span style={{
        padding: "6px 14px",
        borderRadius: 4,
        fontSize: 13,
        background: "#0d1f1f",
        border: "1px solid #14b8a6",
        color: "#14b8a6",
        cursor: "default",
        fontWeight: 600,
      }}>
        Cooperative
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab bar
// ---------------------------------------------------------------------------

const TABS: { key: Tab; label: string }[] = [
  { key: "lineage", label: "Lineage" },
  { key: "champion", label: "Champion Benchmark" },
  { key: "robustness", label: "Robustness" },
  { key: "evolution", label: "Evolution" },
];

// ---------------------------------------------------------------------------
// Ratings table
// ---------------------------------------------------------------------------

function RatingsTable({ members }: { members: CooperativeLeagueMember[] }) {
  if (members.length === 0) {
    return (
      <div style={{ color: "#555555", fontSize: 13, padding: "16px 0" }}>
        No league members yet — run the cooperative pipeline to create snapshots.
      </div>
    );
  }

  const sorted = [...members].sort(
    (a, b) => (b.rating ?? 1000) - (a.rating ?? 1000)
  );

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{
        width: "100%",
        borderCollapse: "collapse",
        fontSize: 13,
        color: "#aaaaaa",
      }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #222222" }}>
            <th style={{ textAlign: "left", padding: "6px 8px" }}>#</th>
            <th style={{ textAlign: "left", padding: "6px 8px" }}>Member ID</th>
            <th style={{ textAlign: "right", padding: "6px 8px" }}>Elo Rating</th>
            <th style={{ textAlign: "left", padding: "6px 8px" }}>Parent</th>
            <th style={{ textAlign: "left", padding: "6px 8px" }}>Created</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m, i) => (
            <tr key={m.member_id} style={{ borderBottom: "1px solid #1a1a1a" }}>
              <td style={{ padding: "5px 8px", color: "#555555" }}>{i + 1}</td>
              <td style={{ padding: "5px 8px", fontFamily: "monospace", fontSize: 11, color: "#14b8a6" }}>
                {m.member_id}
              </td>
              <td style={{ textAlign: "right", padding: "5px 8px", fontWeight: 600, color: "#ededed" }}>
                {(m.rating ?? 1000).toFixed(1)}
              </td>
              <td style={{ padding: "5px 8px", fontFamily: "monospace", fontSize: 10, color: "#555555" }}>
                {m.parent_id ?? "—"}
              </td>
              <td style={{ padding: "5px 8px", fontSize: 11, color: "#555555" }}>
                {m.created_at ? new Date(m.created_at).toLocaleDateString() : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function CooperativeLeaguePage() {
  const [tab, setTab] = useState<Tab>("lineage");
  const [members, setMembers] = useState<CooperativeLeagueMember[]>([]);
  const [champion, setChampion] = useState<CooperativeChampionInfo | null>(null);
  const [lineage, setLineage] = useState<CooperativeLineageMember[]>([]);
  const [evolution, setEvolution] = useState<CooperativeEvolutionResponse | null>(null);
  const [heatmapData, setHeatmapData] = useState<CooperativeRobustnessHeatmapResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Robustness sweep state
  const [robustnessId, setRobustnessId] = useState<string | null>(null);
  const [robustnessStatus, setRobustnessStatus] = useState<CooperativeRobustnessStatusResponse | null>(null);
  const [robustnessRunning, setRobustnessRunning] = useState(false);
  const [robustnessError, setRobustnessError] = useState<string | null>(null);

  const [selectedMemberId, setSelectedMemberId] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      getCooperativeLeagueMembers().catch(() => [] as CooperativeLeagueMember[]),
      getCooperativeChampion().catch(() => null),
      getCooperativeLeagueLineage().catch(() => ({ members: [] })),
      getCooperativeLeagueEvolution().catch(() => null),
    ]).then(([m, c, lin, evo]) => {
      setMembers(m);
      setChampion(c);
      setLineage(lin.members);
      setEvolution(evo);
      setLoading(false);
    }).catch((e) => {
      setError(String(e));
      setLoading(false);
    });
  }, []);

  // Poll robustness status
  useEffect(() => {
    if (!robustnessId || !robustnessRunning) return;
    const interval = setInterval(() => {
      getCooperativeRobustnessStatus(robustnessId).then((s) => {
        setRobustnessStatus(s);
        if (!s.running) {
          setRobustnessRunning(false);
          clearInterval(interval);
          if (s.error) setRobustnessError(s.error);
          // Could load heatmap here if report_id is available
        }
      }).catch(() => {});
    }, 3000);
    return () => clearInterval(interval);
  }, [robustnessId, robustnessRunning]);

  async function handleRunRobustness() {
    setRobustnessError(null);
    setRobustnessRunning(true);
    try {
      const res = await runCooperativeChampionRobustness({
        seeds: 3,
        episodes_per_seed: 2,
        limit_sweeps: 5,
        seed: 42,
      });
      setRobustnessId(res.robustness_id);
    } catch (e) {
      setRobustnessError(String(e));
      setRobustnessRunning(false);
    }
  }

  // Derive robustness scatter data from evolution
  const robustEntries = evolution?.members
    ? evolution.members
        .filter((m) => m.robustness_score != null)
        .map((m) => ({
          policy_name: m.member_id,
          mean_completion_ratio: m.rating / 1200,
          worst_case_completion_ratio: m.robustness_score ?? 0,
          robustness_score: m.robustness_score ?? 0,
        }))
    : [];

  return (
    <div style={{
      background: "#0a0a0a",
      minHeight: "100vh",
      padding: "24px 20px",
      color: "#ededed",
    }}>
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 8 }}>
          <Link href="/league" style={{ color: "#555555", fontSize: 12, textDecoration: "none" }}>
            ← League
          </Link>
        </div>
        <h1 style={{ fontSize: 22, fontWeight: 600, color: "#ededed", marginBottom: 4 }}>
          Cooperative League
        </h1>
        <p style={{ fontSize: 13, color: "#666666", marginBottom: 20 }}>
          Self-play league for the Cooperative archetype — shared goal, collective outcome.
        </p>

        {/* Archetype switcher */}
        <ArchetypeSwitcher />

        {/* Tab bar */}
        <div style={{ display: "flex", gap: 0, marginBottom: 24, borderBottom: "1px solid #1e1e1e" }}>
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              style={{
                padding: "8px 16px",
                fontSize: 13,
                background: "none",
                border: "none",
                borderBottom: tab === t.key ? "2px solid #14b8a6" : "2px solid transparent",
                color: tab === t.key ? "#14b8a6" : "#666666",
                cursor: "pointer",
                marginBottom: -1,
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {loading && (
          <div style={{ color: "#555555", fontSize: 13 }}>Loading league data…</div>
        )}
        {error && (
          <div style={{ color: "#ef4444", fontSize: 13 }}>Error: {error}</div>
        )}

        {!loading && !error && (
          <>
            {/* Lineage tab */}
            {tab === "lineage" && (
              <div>
                <div style={{ marginBottom: 20 }}>
                  <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>
                    Lineage Graph
                  </h2>
                  <CooperativeLeagueLineage
                    members={lineage}
                    selectedId={selectedMemberId}
                    onSelect={setSelectedMemberId}
                  />
                </div>
                <div>
                  <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>
                    Ratings Table
                  </h2>
                  <RatingsTable members={members} />
                </div>
              </div>
            )}

            {/* Champion Benchmark tab */}
            {tab === "champion" && (
              <div>
                <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>
                  Champion Benchmark
                </h2>
                <CooperativeChampionBenchmark
                  champion={champion
                    ? { member_id: champion.member_id ?? "", rating: champion.rating ?? 1000 }
                    : null}
                  results={[]}
                />
              </div>
            )}

            {/* Robustness tab */}
            {tab === "robustness" && (
              <div>
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 16,
                }}>
                  <h2 style={{ fontSize: 16, fontWeight: 600 }}>
                    Champion Robustness
                  </h2>
                  <button
                    onClick={handleRunRobustness}
                    disabled={robustnessRunning}
                    style={{
                      padding: "6px 14px",
                      fontSize: 12,
                      background: robustnessRunning ? "#1a2a2a" : "#0d1f1f",
                      border: "1px solid #14b8a6",
                      color: "#14b8a6",
                      borderRadius: 4,
                      cursor: robustnessRunning ? "not-allowed" : "pointer",
                    }}
                  >
                    {robustnessRunning ? "Running…" : "Run Robustness Sweep"}
                  </button>
                </div>

                {robustnessError && (
                  <div style={{ color: "#ef4444", fontSize: 12, marginBottom: 12 }}>
                    Error: {robustnessError}
                  </div>
                )}
                {robustnessStatus && robustnessRunning && (
                  <div style={{ color: "#888888", fontSize: 12, marginBottom: 12 }}>
                    Stage: {robustnessStatus.stage}
                  </div>
                )}

                {heatmapData ? (
                  <>
                    <CooperativeChampionRobustness data={heatmapData} />
                    <div style={{ marginTop: 32 }}>
                      <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
                        Mean vs Worst-Case
                      </h3>
                      <CooperativeRobustScatter entries={robustEntries} />
                    </div>
                  </>
                ) : (
                  <div style={{ color: "#555555", fontSize: 13 }}>
                    No robustness data yet. Click "Run Robustness Sweep" to start.
                  </div>
                )}
              </div>
            )}

            {/* Evolution tab */}
            {tab === "evolution" && (
              <div>
                <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>
                  Strategy Evolution
                </h2>
                {evolution ? (
                  <>
                    {/* Evolution member cards */}
                    <div style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
                      gap: 10,
                      marginBottom: 24,
                    }}>
                      {evolution.members.map((m) => {
                        const color = labelColor(m.strategy?.label ?? "Developing");
                        return (
                          <div
                            key={m.member_id}
                            style={{
                              background: "#111111",
                              border: "1px solid #1e1e1e",
                              borderLeft: `3px solid ${color}`,
                              borderRadius: 6,
                              padding: "10px 12px",
                            }}
                          >
                            <div style={{ fontSize: 11, color: "#555555", fontFamily: "monospace" }}>
                              {m.member_id}
                            </div>
                            <div style={{ fontSize: 13, fontWeight: 600, color, margin: "2px 0" }}>
                              {m.strategy?.label ?? "Developing"}
                            </div>
                            <div style={{ fontSize: 12, color: "#888888" }}>
                              {m.rating.toFixed(1)} Elo
                            </div>
                            {m.robustness_score != null && (
                              <div style={{ fontSize: 11, color: "#555555", marginTop: 2 }}>
                                Robust: {m.robustness_score.toFixed(3)}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    {/* Champion history */}
                    <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
                      Champion History
                    </h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 320, overflowY: "auto" }}>
                      {evolution.champion_history.map((entry, idx) => {
                        const color = labelColor(entry.label);
                        return (
                          <div key={entry.member_id} style={{
                            background: "#111111",
                            border: "1px solid #1e1e1e",
                            borderLeft: `3px solid ${color}`,
                            borderRadius: 6,
                            padding: "8px 12px",
                            display: "flex",
                            alignItems: "center",
                            gap: 12,
                          }}>
                            <span style={{ fontSize: 11, color: "#555555" }}>#{idx + 1}</span>
                            <span style={{ fontFamily: "monospace", fontSize: 11, color: "#888888" }}>
                              {entry.member_id}
                            </span>
                            <span style={{ fontSize: 12, color, fontWeight: 600 }}>
                              {entry.label}
                            </span>
                            <span style={{ marginLeft: "auto", fontSize: 13, fontWeight: 600, color: "#ededed" }}>
                              {entry.rating.toFixed(1)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <div style={{ color: "#555555", fontSize: 13 }}>
                    No evolution data available.
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
