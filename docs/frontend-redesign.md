# Frontend Redesign

> This document captures all decisions for the full frontend redesign.
> Nothing gets built until this document is complete and approved.
> Same pattern as docs/architecture-decisions.md — decisions made here,
> then implemented. This document is authoritative.

---

## 1. Current State — Feature Inventory

### Global nav (current)
- Mixed side: Competitive | League | Run History | Reports
- Competitive side: Home | League | Reports | Run History
- No unified nav — two separate nav structures

### Home page (/)
- "Create default config" button
- Policy dropdown: random, always_cooperate, always_extract, tit_for_tat, ppo_shared
- List of saved configs — each row: Config ID, Seed, Agents, Max Steps, Start Run button

### Run live page (/run/[run_id])
- Back button
- Run ID, WebSocket status, step counter, termination reason
- For Mixed runs: MetricsChart (reward bars, shared pool sparkline)
- For Competitive runs: winner banner, episode info grid, final rankings table,
  total reward per agent

### Run History page (/runs)
- Table: Run ID, Policy, Steps, Termination, Time
- Replay button per row
- Mixed and Competitive runs mixed together, no archetype indicator

### Replay page (/replay/[run_id])
- Back to Runs link
- Status, step, termination reason
- For Competitive: Score over Time chart, Rank over Time chart,
  Action Distribution chart (BUILD/ATTACK/DEFEND/GAMBLE)
- Episode Summary: length, termination, winner, score spread, eliminations,
  total reward per agent

### Mixed League page (/league)
- Back to Home, Recompute Ratings button
- 5 tabs: Members, Lineage, Champion Benchmark, Robustness, Evolution
- Members tab: ratings table (rank, member ID, rating, parent, created, notes),
  Run button per row
- Lineage tab: SVG graph, nodes with Elo number inside and rating below,
  click to see detail panel (member ID, rating, parent, created, notes)
- Champion Benchmark tab: config dropdown, episodes input, Run Benchmark button,
  bar chart + table results after running
- Robustness tab: config dropdown, seeds, episodes/seed, max steps, limit sweeps,
  seed inputs, Run Robustness button, redirects to report on completion
- Evolution tab: lineage graph with strategy label colors (Champion/Competitive/
  Developing), champion history timeline with cluster labels and robustness scores

### Mixed Reports page (/reports)
- Table: Report ID, Kind, Timestamp, Open button

### Mixed Report detail (/reports/[report_id])
- Back to Reports, report ID, config hash, generated timestamp
- Reward Heatmap (policy × sweep grid)
- Mean vs Worst-Case scatter plot
- Robustness Summary table (policy, robustness, mean reward, worst-case,
  collapse %, sweeps)
- Top-3 Most Robust list
- Hardest Sweep label
- Strategy Groups (cluster cards with member list + full policy table)

### Competitive main page (/competitive)
- Nav: Home | League | Reports | Run History
- Config form: num_agents, max_steps, seed, agent_policy dropdown
  (random, always_attack, always_build, always_defend, competitive_ppo)
- Start Run button
- League Standings table: Rank, Member ID, Rating, Recompute Ratings button

### Competitive League page (/competitive/league)
- Back to Competitive, Recompute Ratings button
- 4 tabs: Ratings, Lineage, Champion, Evolution
- Ratings tab: table (rank, member ID, rating, parent, created), Run button per row
- Lineage tab: SVG graph with strategy colors (Dominant=gold, Aggressive=red,
  Consistent=green, Weak=gray, Competitive=default), click for detail panel
  (member ID, label, rating, parent, cluster, robustness, created)
- Champion tab: current champion stats (member ID, rating, parent),
  Champion Benchmark section (config dropdown, episodes, Run Champion Benchmark,
  shows bar chart + table after running), Run Robustness on Champion section
  (config, seeds, episodes/seed, limit sweeps, seed, Run Robustness button,
  redirects to report)
- Evolution tab: lineage graph with label colors + champion history timeline
  (rank, label, cluster, member ID, rating, robustness, timestamp)

### Competitive Reports page (/competitive/reports)
- Back to Competitive, table: Report ID, Timestamp, Open button

### Competitive Report detail (/competitive/reports/[report_id])
- Back to Reports, report ID, config hash, generated timestamp
- Robustness Summary table (policy, mean reward, robustness, winner rate,
  worst-case, sweeps)
- Reward Heatmap (policy × sweep grid)
- Mean vs Worst-Case scatter plot
- Strategy Groups (cluster cards + policy table with cluster, label, mean reward,
  winner rate, robustness, worst-case)

---

## 2. Known Bugs

- **Home page creates wrong config type**: "Create default config" on the Mixed
  home page creates a Competitive config. Runs show Competitive results (winner,
  score spread, rankings) instead of Mixed results (shared pool, cooperation rate).
  Will be fixed naturally by the redesign.

- **Run History has no archetype indicator**: Mixed and Competitive runs appear
  in the same table with no way to distinguish which archetype produced each run.

- **Config list shows cryptic IDs**: No archetype label, no human-readable name.
  Users cannot tell which configs belong to which archetype.

---

## 3. Problems with Current Structure

- Mixed and Competitive are two completely disconnected experiences with different
  nav structures, different URL schemes, and no shared entry point
- No template system — users must understand archetypes directly to use the platform
- Navigation is inconsistent — Mixed uses top-level nav, Competitive has its own
  separate nav with different items
- No unified entry point — unclear where a new user should start
- Archetypes are exposed directly to users ("Mixed", "Competitive") instead of
  being hidden behind meaningful template names
- Run History is shared but unmarked — no way to tell Mixed from Competitive runs
- Config dropdown in league pages shows raw config IDs (cryptic hashes), not
  human-readable names
- Reports are siloed — Mixed reports at /reports, Competitive at /competitive/reports
  with no unified view
- League is siloed — same problem, no unified view across archetypes

---

## 4. Design Goals

- Users never see the words "Mixed" or "Competitive" — these are internal terms
- Templates are the user-facing layer — "Resource Sharing Arena", "Head-to-Head
  Strategy", etc.
- Archetypes are engine-level abstractions, hidden from all UI
- Power users can access full archetype configs via an Advanced mode toggle —
  still no archetype language, just full parameter control
- One coherent product — unified nav, unified league, unified research/reports
- Dark theme, data-forward, technically credible — Weights & Biases density,
  Vercel dark mode typography discipline, Linear interaction polish
- Accent color: teal
- Charts and data tables are the hero — large, readable, not squished
- Flat surfaces, subtle borders, no gradients, no shadows, no glow effects
- Typography: one font, two weights (400 regular, 500 medium), nothing decorative
- Must not look AI-generated — intentional visual decisions throughout

---

## 5. New Information Architecture

### Navigation (global, every page)
```
[logo/wordmark]    Simulate    League    Research
```
Three top-level destinations. Clean. No sub-menus in the nav itself.

### URL structure
```
/                           — Home
/simulate                   — Template picker + Advanced mode toggle
/simulate/resource-sharing  — Resource Sharing Arena (Mixed archetype)
/simulate/head-to-head      — Head-to-Head Strategy (Competitive archetype)
/simulate/[template]/run/[id] — Live run page
/simulate/[template]/replay/[id] — Replay page
/league                     — Unified league (archetype switcher inside)
/research                   — Unified reports browser (filter bar inside)
/research/[report_id]       — Report detail
```

### Old URLs → redirects
```
/runs               → /league (or a dedicated run history section)
/replay/[id]        → /simulate/[template]/replay/[id]
/reports            → /research
/reports/[id]       → /research/[id]
/competitive        → /simulate/head-to-head
/competitive/league → /league (with Head-to-Head tab active)
/competitive/reports → /research (filtered to Head-to-Head)
```

---

## 6. Template System Design

### What users see on /simulate

**Default view — Templates tab:**

Two live template cards (full width, prominent):
1. "Resource Sharing Arena" — backed by Mixed archetype, teal accent
   - Tagline: "Agents share a common resource pool and decide when to cooperate
     or compete. Watch emergent strategies develop over time."
   - Tags: Cooperation · Defection · Emergent Strategy
   - Live stats: X runs · Current champion · Top strategy label
   - Launch button → /simulate/resource-sharing

2. "Head-to-Head Strategy" — backed by Competitive archetype, coral accent
   - Tagline: "Pure zero-sum competition. Agents fight for score dominance
     across 200 steps. One winner per episode."
   - Tags: Zero-sum · Rankings · Score Spread
   - Live stats: X runs · Current champion · Top strategy label
   - Launch button → /simulate/head-to-head

Four coming-soon cards (smaller, grayed out, no interaction):
- "Algorithmic Auction Arena" — Competitive variant
- "Team-Based Market Simulation" — Mixed variant
- "Multi-Team Resource Control" — Mixed variant
- "Negotiation Arena" — future Cooperative archetype
Each shows: name, one-line description, "In development" badge

**Advanced mode — toggle from Templates tab:**

Shows the two archetypes directly with full config exposed:
- "Resource Sharing — full config" (Mixed archetype, all knobs)
- "Head-to-Head — full config" (Competitive archetype, all knobs)
Still uses template language, not archetype language.
This is the existing config form, better presented.

### Template page layout (/simulate/resource-sharing and /simulate/head-to-head)

Left panel (config):
- Template name + tagline at top
- Key parameters: num_agents, max_steps, seed
- Agent policy selector
- Template-specific knobs (e.g. for Resource Sharing: cooperation pressure slider;
  for Head-to-Head: elimination on/off)
- Start Run button

Right panel (run history for this template):
- Table of past runs: run ID, policy, result, timestamp
- Click row → replay page
- Empty state: "No runs yet — start your first simulation"

---

## 7. Per-Page Spec

### Home page (/)
- Hero: headline, 2-line description, two CTAs (Start Simulating → /simulate,
  View League → /league)
- Live stats bar (from API): total runs, active league members, templates
  available, reports generated
- Three feature highlight cards: "League-based self-play", "Robustness analysis",
  "Emergent strategy clustering"
- Quick-start: pick a template and go directly to it

### Simulate index (/simulate)
- Tab toggle: Templates | Advanced
- Templates tab: 2 live cards + 4 coming-soon cards (as above)
- Advanced tab: full config forms for both archetypes

### Template pages (/simulate/resource-sharing, /simulate/head-to-head)
- Left/right split layout (config + run history) as above

### Live run page (/simulate/[template]/run/[id])
- Unchanged from current — reuse existing components
- URL moves under /simulate/[template]/run/[id]
- "Back" goes to the template page, not home

### Replay page (/simulate/[template]/replay/[id])
- Unchanged from current — reuse existing components
- URL moves under /simulate/[template]/replay/[id]

### League page (/league)
- Archetype switcher at top: "Resource Sharing" | "Head-to-Head"
  (tab-style, not archetype language)
- Each tab shows the full league experience for that archetype:
  - Ratings sub-tab: Elo table, recompute button, run button per row
  - Lineage sub-tab: SVG lineage graph (reuse existing component)
  - Champion sub-tab: champion stats, benchmark chart, robustness shortcut
  - Evolution sub-tab: strategy change timeline (reuse LeagueEvolution.tsx)
- Recompute Ratings button scoped to active archetype tab

### Research page (/research)
- Filter bar: All | Resource Sharing | Head-to-Head (archetype filter)
- Type filter: All | Robustness | Strategy | Benchmark
- Sort: Latest | Highest robustness score
- Report cards grid: type badge, archetype badge, date, robustness score,
  Open button
- Empty state: "No reports yet — run robustness analysis from the League page"

### Report detail (/research/[report_id])
- Unchanged from current — reuse existing components
- URL moves under /research/[report_id]
- "Back" goes to /research

---

## 8. What Gets Kept vs Rebuilt

### New from scratch
- Home page (/)
- Simulate index (/simulate) with Templates + Advanced toggle
- Global nav component (replaces both current nav structures)
- League page (/league) — unified wrapper with archetype switcher
- Research page (/research) — filter bar + unified report cards

### Modify / move (keep logic, change URL + nav)
- Template pages — reskin existing Mixed and Competitive config forms
- Run live page — move URL, update back-link target
- Replay page — move URL, update back-link target
- Report detail — move URL, update back-link target

### Reuse as-is (no changes to component logic)
- All league tab content components (LeagueEvolution, ChampionBenchmark,
  ChampionRobustness, lineage SVG)
- MetricsChart (Mixed live run)
- CompetitiveRunSummary, CompetitiveReplayView
- RobustHeatmap, RobustScatter, strategy group cards
- All api.ts functions (additive only — no existing functions removed)

### Explicitly removed
- /competitive/* routes (replaced by /simulate/head-to-head and /league)
- /reports/* routes (replaced by /research/*)
- /runs route (run history moves inside template pages)
- Separate Mixed and Competitive nav structures

---

## 9. Build Order

1. Global nav component — everything depends on this
2. Routing restructure — set up new URL structure, add redirects for old routes
3. Simulate index — template picker (most visible page, first impression)
4. Template pages — resource-sharing and head-to-head config + run history panels
5. Home page — hero, stats bar, feature highlights, quick-start
6. Unified League page — archetype switcher wrapping existing tab components
7. Research index — filter bar over existing report cards
8. Move run/replay/report detail pages to new URLs
9. Delete old /competitive/* and /reports/* routes
10. Final pass — verify all redirects work, all back-links point correctly,
    no broken routes

---

## 10. Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| Navigation structure | B+ (Simulate / League / Research) | Feature-first, hides technical internals |
| Theme | Dark, technical, data-forward | Matches research tool use case |
| Accent color | Teal | Distinctive, works well on dark, data tool feel |
| Template names | Resource Sharing Arena, Head-to-Head Strategy | User-facing, no archetype language |
| Coming-soon templates | 4 grayed-out cards | Shows roadmap, makes platform feel bigger |
| Advanced mode | Full config toggle on /simulate | Power users get raw access without breaking default UX |
| Archetypes in UI | Never exposed directly | Users see templates, archetypes are engine-internal |
| Branch | feat/frontend-redesign | Isolated from main until complete |
| Bug fix timing | Fix during redesign, not before | Bug disappears naturally with rebuilt home page |
| Deploy timing | After frontend redesign, before next archetype | Redesign makes it feel like a real product |