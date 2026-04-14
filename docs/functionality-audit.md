# Functionality Audit

**Audit date:** 2026-04-15  
**Branch:** feat/cooperative-archetype  
**Scope:** Full end-to-end platform audit against docs/frontend-redesign.md and the audit checklist

---

## Methodology

This audit was conducted by reading all page source files, component source files, and backend route files systematically. Every page, form, button, chart, and API endpoint listed in the audit scope was checked. No runtime server was started; findings are based on static code analysis.

---

## 1. Home Page (/)

### ✓ Passing
- Hero section renders correctly: headline, 2-line description, live snapshot card.
- "Start Simulating →" button links to `/simulate`. ✓
- "View League →" button links to `/league`. ✓
- Quick Start buttons link to correct pages: `/simulate/resource-sharing`, `/simulate/head-to-head`, `/simulate/cooperative`. ✓
- Feature highlight cards (4 cards in 2-column grid) render correctly. ✓
- "How it works" section renders (4 numbered pipeline steps). ✓
- "Research questions" section renders (4 cards including Cooperative Task Arena). ✓
- Live snapshot card shows pulsing dot and "offline" fallback when API is down. ✓

### ✗ Issues

**[MAJOR] Home-1 — totalRuns stat excludes cooperative runs**  
- Page: Home page `/` — stats snapshot card  
- Expected: TOTAL RUNS shows all runs across all three archetypes  
- Actual: `useHomeStats()` fetches only `/api/runs/history` which returns Mixed (RS) and Competitive (HH) runs. Cooperative runs are stored separately under `/api/cooperative/runs` and are never included. If a user runs only cooperative simulations, TOTAL RUNS shows 0.  
- Severity: **Major**

**[MAJOR] Home-2 — leagueMembers stat excludes cooperative league members**  
- Page: Home page `/` — stats snapshot card  
- Expected: LEAGUE MEMBERS shows members across all three archetypes  
- Actual: `useHomeStats()` sums `/api/league/members` (RS) + `/api/competitive/league/members` (HH) only. Cooperative league members from `/api/cooperative/league/members` are never fetched or counted.  
- Severity: **Major**

**[MAJOR] Home-3 — reports stat excludes cooperative reports**  
- Page: Home page `/` — stats snapshot card  
- Expected: REPORTS shows reports across all three archetypes  
- Actual: `useHomeStats()` sums `/api/reports` + `/api/competitive/reports` only. Cooperative reports from `/api/cooperative/reports` are never counted.  
- Severity: **Major**

**[MAJOR] Home-4 — recent activity excludes cooperative runs**  
- Page: Home page `/` — recent activity panel inside snapshot card  
- Expected: Recent activity shows the 3 most recent runs regardless of archetype  
- Actual: `useRecentRuns()` fetches only from `/api/runs/history` (RS + HH runs). A user who exclusively runs cooperative simulations will see an empty recent activity panel (`--------` placeholder rows) even after completing runs.  
- Severity: **Major**

**[MINOR] Home-5 — "environments" stat is a hardcoded constant**  
- Page: Home page `/` — stats snapshot card  
- Expected: ENVIRONMENTS count is fetched dynamically from an API  
- Actual: `environments: "3"` is a hardcoded string constant set in the initial state of `useHomeStats()`. It is never updated by any API call. The value will not change when new environments are added to the platform.  
- Severity: **Minor**

---

## 2. Simulate Index (/simulate)

### ✓ Passing
- Templates tab shows 3 live template cards: Resource Sharing Arena, Head-to-Head Strategy, Cooperative Task Arena. ✓
- Templates tab shows 4 coming-soon cards (Algorithmic Auction Arena, Team-Based Market Simulation, Multi-Team Resource Control, Negotiation Arena) — grayed out, no interaction, "In development" badge. ✓
- Advanced tab shows full config forms for all 3 archetypes with correct links. ✓
- "Launch →" buttons on all 3 live cards link to correct pages. ✓
- Live stats on cards (runs count, league members count) fetched from correct endpoints per archetype. ✓

### ✗ Issues
None found.

---

## 3. Resource Sharing Arena (/simulate/resource-sharing)

### ✓ Passing
- Back link "← Simulate" present and links to `/simulate`. ✓
- Config form renders: num_agents, max_steps, seed, agent_policy dropdown, league_member_id (conditional). ✓
- All 6 agent policies in dropdown: random, always_cooperate, always_extract, tit_for_tat, ppo_shared, league_snapshot. ✓
- Start Run creates a Mixed config and navigates to `/simulate/resource-sharing/run/${run_id}`. ✓
- Past Runs panel loads correctly, filters to RS-only runs (including archetype resolution for ambiguous runs). ✓
- Replay links in past runs table point to `/simulate/resource-sharing/replay/${run_id}`. ✓
- Error state renders when config creation or run start fails. ✓

### ✗ Issues

**[CRITICAL] RS-1 — Live run page renders with light-theme Tailwind CSS classes**  
- Page: `/simulate/resource-sharing/run/[run_id]`  
- Expected: Dark theme consistent with the rest of the platform (CSS custom properties: `var(--text-primary)`, `var(--accent)`, etc.)  
- Actual: The run page uses Tailwind CSS class names throughout: `max-w-4xl mx-auto p-8`, `text-blue-500`, `text-green-500`, `text-yellow-500`, `text-orange-500`, `text-red-500`, `border-gray-300`. On the dark theme platform these produce incorrect visually broken output — the back link is blue (`text-blue-500`), status badges are light-mode colors, and the episode summary uses a light gray border (`border-gray-300`). The WebSocket status indicators, step counter, and episode summary box do not match the platform's dark design.  
- Severity: **Critical**

**[CRITICAL] RS-2 — Replay page renders with light-theme Tailwind CSS classes**  
- Page: `/simulate/resource-sharing/replay/[run_id]`  
- Expected: Dark theme consistent with the rest of the platform  
- Actual: Same Tailwind CSS light-mode class names as the run page. Back link is `text-blue-500`, status badges are colored with light-mode classes, episode summary box uses `border-gray-300`. Page is visually inconsistent with the dark theme used on all other pages.  
- Severity: **Critical**

---

## 4. Head-to-Head Strategy (/simulate/head-to-head)

### ✓ Passing
- Back link "← Simulate" present and links to `/simulate`. ✓
- Config form renders: num_agents, max_steps, seed, agent_policy dropdown (5 policies). ✓
- Start Run creates a Competitive config and navigates to `/simulate/head-to-head/run/${run_id}`. ✓
- Past Runs panel loads correctly, filters to HH-only runs. ✓
- Replay links in past runs table point to `/simulate/head-to-head/replay/${run_id}`. ✓
- Competitive-specific summary fields (winner, score spread, eliminations) rendered via `CompetitiveRunSummary` after run completes. ✓
- Error state renders on failure. ✓

### ✗ Issues

**[CRITICAL] HH-1 — Live run page renders with light-theme Tailwind CSS classes**  
- Page: `/simulate/head-to-head/run/[run_id]`  
- Expected: Dark theme consistent with the rest of the platform  
- Actual: Identical Tailwind light-theme issue as RS-1. `text-blue-500`, `border-gray-300`, status indicator color classes all render incorrectly against the dark background.  
- Severity: **Critical**

**[MAJOR] HH-2 — Live run page shows wrong metrics chart during active Competitive run**  
- Page: `/simulate/head-to-head/run/[run_id]`  
- Expected: Live chart during an active run should show competitive-relevant metrics (score, rank, actions)  
- Actual: The live run page renders `MetricsChart` for all run types. `MetricsChart` is the Mixed-archetype component — it displays a shared pool sparkline (reads `shared_pool` from step metrics) and per-agent reward bars. Competitive run step metrics have no `shared_pool` field, so the sparkline will be flat at zero or empty. The chart renders nonsensical data during any live Competitive run. `CompetitiveReplayView` (the correct chart component) is only used on the replay pages, not during live runs.  
- Severity: **Major**

**[CRITICAL] HH-3 — Replay page renders with light-theme Tailwind CSS classes**  
- Page: `/simulate/head-to-head/replay/[run_id]`  
- Expected: Dark theme consistent with the rest of the platform  
- Actual: Same Tailwind light-mode class issue as RS-2. Visually inconsistent with all other pages.  
- Severity: **Critical**

---

## 5. Cooperative Task Arena (/simulate/cooperative)

### ✓ Passing
- Back link "← Simulate" present and links to `/simulate`. ✓
- Config form renders: num_agents, max_steps, num_task_types, seed, agent_policy dropdown. ✓
- All 6 agent policies in dropdown: random, always_work, always_idle, specialist, balancer, cooperative_ppo. ✓
- Live run shows 4 chart panels (Backlog Level, System Stress, Group Completion Rate, Effort Utilization) via `CooperativeMetricsChart`. ✓
- Episode summary shows cooperative fields via `CooperativeRunSummary` after run completes. ✓
- Past Runs panel loads from `/api/cooperative/runs`. ✓
- Replay links point to `/simulate/cooperative/replay/${run_id}`. ✓
- Replay page at `/simulate/cooperative/replay/[run_id]` renders correctly with back link, dark theme, and `CooperativeReplayView`. ✓

### ✗ Issues

**[MAJOR] CP-1 — Start Run does not navigate to a dedicated live run page**  
- Page: `/simulate/cooperative`  
- Expected: Per spec, "Start Run button starts a run and navigates to live run page." RS and HH both navigate to `/simulate/[template]/run/[run_id]` on Start Run.  
- Actual: The cooperative page starts a run and shows the live chart inline on the same config page — beneath the config form in the right panel. There is no `/simulate/cooperative/run/[run_id]` route. The user stays on the config page during the entire run. This is inconsistent with RS and HH behavior and prevents users from bookmarking or sharing a live run URL.  
- Severity: **Major**

---

## 6. Advanced Mode

### ✓ Passing
- `/simulate/resource-sharing?mode=advanced` shows Advanced parameters section with 13 advanced fields in a 2-column grid. ✓
- `/simulate/head-to-head?mode=advanced` shows Advanced parameters section with 18 advanced fields. ✓
- `/simulate/cooperative?mode=advanced` shows Advanced parameters section with 18 fields (population, layers, task, rewards) in a 2-column grid. ✓
- Start Run works from advanced mode for all three archetypes. ✓

### ✗ Issues
None found.

---

## 7. League Page (/league)

### ✓ Passing
- Pipeline panel visible above tabs, shows all 3 archetypes (Resource Sharing, Head-to-Head, Cooperative) each with "Run Pipeline →" button and configurable settings. ✓
- Archetype switcher shows Resource Sharing, Head-to-Head, Cooperative pills. ✓
- Recompute Ratings button present for all archetypes (shown in header row, scoped to active archetype). ✓
- RS tab: Ratings, Champion, Evolution sub-tabs all render. ✓
- RS Ratings tab: member table with rank, member ID, rating, parent, created, notes, Run button per row. ✓
- RS Champion tab: champion info panel, ChampionBenchmark component, ChampionRobustness component. ✓
- RS Evolution tab: `LeagueEvolution` component renders. ✓
- HH tab: Ratings, Champion, Evolution sub-tabs all render. ✓
- HH Ratings tab: member table with rank, member ID, rating, parent, created, Run button per row. ✓
- HH Champion tab: champion info panel, inline benchmark form with config dropdown and Run Champion Benchmark button, inline robustness form with config/seeds/episodes/seed/limit-sweeps fields and Run Robustness button. ✓
- HH Champion tab: config dropdown filtered to competitive configs only. ✓
- HH Evolution tab: LineageGraph + scrollable Champion History list. ✓
- Cooperative tab: Ratings, Champion, Evolution sub-tabs all render. ✓
- Cooperative Champion tab: champion info, benchmark form with config dropdown, robustness form. ✓
- Cooperative Champion config dropdown filtered to cooperative configs only. ✓
- Cooperative Evolution tab: LineageGraph + Champion History. ✓
- Cooperative robustness "View report →" link points to `/research/cooperative/${reportId}`. ✓
- HH robustness "View report →" link points to `/research/${reportId}`. ✓
- RS pipeline "View report →" link points to `/research/${reportId}` (default `reportBasePath`). ✓
- Cooperative pipeline "View report →" link points to `/research/cooperative/${reportId}` (`reportBasePath="/research/cooperative/"`). ✓
- Pipeline polling for all 3 archetypes implemented with 2-second interval polling. ✓
- Error states and stage display working for pipelines and robustness. ✓

### ✗ Issues

**[MAJOR] LG-1 — Cooperative ratings tab has no Run button per member row**  
- Page: `/league` → Cooperative tab → Ratings sub-tab  
- Expected: Per spec and parity with RS and HH, each league member row should have a Run button to run that member  
- Actual: The Cooperative ratings table renders columns: #, Member ID, Rating, Parent, Created, Notes — with no Run button column. RS and HH both have a Run button per row. The cooperative ratings tab omits this action entirely. Users cannot launch a run using a specific cooperative league member from the League page.  
- Severity: **Major**

---

## 8. Research Page (/research)

### ✓ Passing
- All RS, HH, and Cooperative reports listed (fetched from 3 endpoints: `/api/reports`, `/api/competitive/reports`, `/api/cooperative/reports`). ✓
- Filter by archetype works: Resource Sharing, Head-to-Head, Cooperative, All pills render and filter correctly. ✓
- Filter by type works: All, Robustness, Strategy, Benchmark pills. ✓
- Sort dropdown renders. ✓
- "Open →" button for RS/HH reports links to `/research/${report_id}`. ✓
- "Open →" button for Cooperative reports links to `/research/cooperative/${report_id}`. ✓
- Archetype badge and type badge render on each report card. ✓
- Loading state, error state, and empty state all have implementations. ✓

### ✗ Issues

**[MINOR] RS-Page-1 — "Sort by highest robustness score" uses alphabetical report_id as a proxy**  
- Page: `/research`  
- Expected: Sort "Highest robustness score" should sort by actual robustness score values  
- Actual: The list endpoints return no robustness score field. When sort = "robustness" is selected, the code explicitly falls back to `b.report_id.localeCompare(a.report_id)` — an alphabetical sort of report IDs. The UI label says "Highest robustness score" which is misleading. A note does appear below the dropdown ("Robustness scores available in report detail") but the sorting still doesn't function as labeled. Users selecting this sort expect meaningful ordering but get alphabetical report ID ordering.  
- Severity: **Minor**

---

## 9. Report Detail Pages

### ✓ Passing
- RS report detail (`/research/[report_id]`): back link to `/research`, report ID, timestamp, `RobustHeatmap`, `RobustScatter` (horizontal bar chart), `RobustSummaryTable`, strategy groups. ✓
- Competitive report detection works via `reportId.startsWith("competitive_")` — routes to competitive endpoints automatically. ✓
- HH report detail renders with `CompetitiveRunSummary`-compatible fields. ✓
- Cooperative report detail (`/research/cooperative/[report_id]`): loads via `getCooperativeReport`, conditionally fetches robustness heatmap for `cooperative_robust` kind reports, renders `CooperativeChampionRobustness` and `CooperativeStrategyGroups`. ✓

### ✗ Issues
None found.

---

## 10. Navigation

### ✓ Passing
- Nav bar fixed at top, shows wordmark "masp" and nav links. ✓
- Active state highlights correct nav item: Simulate active when path starts with `/simulate`, League active only at `/league` (exact match), Research active when path starts with `/research`. ✓
- All nav links work. ✓
- Hover states work on all nav links. ✓

### ✗ Issues

**[MINOR] NAV-1 — Nav shows 4 links instead of 3 per spec**  
- Feature: Global navigation  
- Expected: Spec (docs/frontend-redesign.md, Section 5) specifies exactly 3 nav items: Simulate, League, Research  
- Actual: Nav shows 4 items: Simulate, League, Research, About. The "About" link was added post-redesign. While the About page is a valid addition, it deviates from the specified nav structure.  
- Severity: **Minor**

---

## 11. API Endpoints

All routes verified by static analysis of backend router prefixes and route definitions:

| Endpoint | Route Defined In | Status |
|---|---|---|
| GET /api/configs | routes_config.py | ✓ Exists (returns list of configs) |
| GET /api/league/members | routes_league.py | ✓ Exists |
| GET /api/league/champion | routes_league.py | ✓ Exists |
| GET /api/competitive/league/members | routes_competitive_league.py | ✓ Exists |
| GET /api/competitive/league/champion | routes_competitive_league.py | ✓ Exists |
| GET /api/cooperative/league/members | routes_cooperative_league.py | ✓ Exists |
| GET /api/cooperative/league/champion | routes_cooperative_league.py | ✓ Exists |
| GET /api/cooperative/runs | routes_cooperative.py | ✓ Exists |
| GET /api/cooperative/reports | routes_cooperative_reports.py | ✓ Exists |
| GET /api/reports | routes_reports.py | ✓ Exists |
| GET /docs | FastAPI auto-generated | ✓ Standard FastAPI behavior |

No missing or incorrectly prefixed routes found. All routers registered in `main.py`.

---

## Summary

### Issue Count by Severity

| Severity | Count | Issues |
|---|---|---|
| **Critical** | 4 | RS-1, RS-2, HH-1, HH-3 |
| **Major** | 6 | Home-1, Home-2, Home-3, Home-4, HH-2, CP-1, LG-1 |
| **Minor** | 3 | Home-5, NAV-1, RS-Page-1 |
| **Total** | 13 | |

### Issue Index

| ID | Page/Feature | Severity | Summary |
|---|---|---|---|
| RS-1 | `/simulate/resource-sharing/run/[run_id]` | Critical | Tailwind light-theme CSS classes — dark theme visual breakage |
| RS-2 | `/simulate/resource-sharing/replay/[run_id]` | Critical | Tailwind light-theme CSS classes — dark theme visual breakage |
| HH-1 | `/simulate/head-to-head/run/[run_id]` | Critical | Tailwind light-theme CSS classes — dark theme visual breakage |
| HH-3 | `/simulate/head-to-head/replay/[run_id]` | Critical | Tailwind light-theme CSS classes — dark theme visual breakage |
| Home-1 | Home page stats | Major | totalRuns excludes cooperative runs |
| Home-2 | Home page stats | Major | leagueMembers excludes cooperative league members |
| Home-3 | Home page stats | Major | reports excludes cooperative reports |
| Home-4 | Home page recent activity | Major | Cooperative runs never appear |
| HH-2 | `/simulate/head-to-head/run/[run_id]` | Major | Live run shows Mixed metrics chart (MetricsChart) for Competitive run — shared_pool sparkline is meaningless |
| CP-1 | `/simulate/cooperative` | Major | Start Run does not navigate to dedicated live run page; run is inline |
| LG-1 | `/league` → Cooperative → Ratings tab | Major | No Run button per member row (unlike RS and HH) |
| Home-5 | Home page stats | Minor | "environments" stat hardcoded as "3", not dynamic |
| NAV-1 | Global nav | Minor | 4 nav items (spec says 3 — "About" not in spec) |
| RS-Page-1 | `/research` | Minor | "Sort by highest robustness score" uses alphabetical report_id proxy |

### Critical Findings Summary

The four Critical issues all share the same root cause: the live run pages and replay pages for Resource Sharing and Head-to-Head were ported from an earlier version without updating their styling from Tailwind CSS class names to the platform's dark-theme CSS custom properties. These pages will render with incorrect colors (light blue back links, light green/yellow/red status indicators, light gray borders) on the otherwise dark platform. Every user who starts a RS or HH run and lands on the live run page will experience this visual regression immediately.

The six Major issues collectively mean that: cooperative runs are invisible in three places on the home page; the Head-to-Head live run experience shows a broken metrics chart; the cooperative page diverges from the RS/HH run flow; and cooperative league members have no "Run" action from the League page.
