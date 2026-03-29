# Build Log

> This document records what was built, in what order, and what was verified at each step.
> Mixed is documented retrospectively from build notes.
> Competitive is documented in real time from the point it was started.
> This log is the authoritative record of build progress.

---

## Mixed Archetype

### Phase 1: Paper Spec
- Defined the 11-step environment framework
- Defined Mixed archetype Parts 1–11 (core tension, layers, agents/episode, state/observation, actions, transitions, rewards, termination, config schema, instrumentation, export/reuse)
- All documented in Notion, condensed into `design/` docs

### Phase 2: Pre-Implementation
- Sanity check: 8 ambiguities found and resolved
  - Conditional actions use aggregate signals, not per-agent targeting
  - Reputation owned by transition logic, not reward logic
  - Stable resolution termination deferred from V1
  - Action type is discrete enum, amount is continuous [0,1]
  - Template boundary deferred
  - (See playbook Section 6 for full list)
- V1 scope locked: Mixed only, no auth, no DB, no other archetypes
- Tech stack locked: Python 3.11, FastAPI, PyTorch, PettingZoo, Next.js/TypeScript, filesystem storage
- 3-stage Claude process: context ingestion → scope lock → structure proposal (no code yet)
- Folder structure proposed, reviewed, and approved

### Phase 3: Implementation

#### Step 2: Environment Core Loop
**Built:**
- `simulation/envs/mixed/actions.py` — ActionType enum + Action dataclass
- `simulation/envs/mixed/state.py` — GlobalState, AgentState, RelationalState
- `simulation/envs/mixed/transition.py` — collective resolution, pool updates, bounded noise
- `simulation/envs/mixed/rewards.py` — 3-component weighted sum
- `simulation/envs/mixed/termination.py` — max_steps, collapse, no_active_agents
- `simulation/envs/mixed/env.py` — MixedEnvironment implementing BaseEnvironment

**Verified:** reset/step contract, determinism, termination triggers. Pytest passed.

---

#### Step 3: Metrics Collector + Run Artifacts
**Built:**
- `simulation/metrics/definitions.py` — step/episode metrics, event types
- `simulation/metrics/collector.py` — ingests state+actions each step, accumulates episode metrics, emits semantic events
- `simulation/runner/run_logger.py` — writes config.json, metrics.jsonl, events.jsonl, episode_summary.json to `storage/runs/{run_id}/`

**Verified:** Pytest passed.

---

#### Step 4: FastAPI Backend + WebSocket Streaming
**Built:**
- `backend/main.py` — app assembly
- `backend/schemas/api_models.py` — Pydantic request/response models
- `backend/api/routes_config.py` — CRUD for configs
- `backend/api/routes_experiment.py` — start/stop/status
- `backend/api/ws_metrics.py` — WebSocket streaming per step
- `backend/runner/run_manager.py` — singleton, broadcasts to WS clients, non-blocking
- `backend/runner/experiment_runner.py` — async loop driving env

**Verified:** Integration tests passed. Frontend connected, live metrics working.

---

#### Step 5: Minimal Next.js Frontend
**Built:**
- `frontend/src/app/layout.tsx`
- Home page, `/run/[run_id]` live view
- `ConfigList`, `MetricsChart` (Recharts + WebSocket), `StopRunButton`
- `frontend/src/lib/api.ts` — typed REST + WebSocket client

**Verified:** Frontend connected, live metrics working end-to-end.

---

#### Step 6A: Selectable Agent Policies
**Built:**
- `simulation/agents/base.py` — BaseAgent ABC
- `simulation/agents/random_agent.py`
- `simulation/agents/always_cooperate.py`
- `simulation/agents/always_extract.py`
- `simulation/agents/tit_for_tat.py`
- `simulation/agents/__init__.py` — POLICY_REGISTRY + create_agent factory
- Updated `StartRunRequest` with `agent_policy` field, wired into experiment_runner

**Verified:** Pytest passed.

---

#### Step 6B: PettingZoo Adapter
**Built:**
- `simulation/adapters/pettingzoo_mixed.py` — MixedPettingZooParallelEnv wrapping MixedEnvironment as ParallelEnv
- Added pettingzoo + gymnasium dependencies

**Verified:** Pytest passed.

---

#### Step 6C: Shared-Policy PPO Training
**Built:**
- `simulation/training/ppo_shared.py` — CleanRL-style PPO, shared policy, Categorical action_type + Beta amount, rollout buffer + GAE + clipped surrogate, saves policy.pt + metadata.json
- `simulation/training/eval_policy.py` — N episodes, avg return

**Verified:** Pytest passed. Artifacts saved to storage.

---

#### Step 7: PPO Agent in Web App
**Built:**
- `simulation/agents/ppo_shared_agent.py` — loads policy.pt, deterministic inference
- Updated agent registry + StartRunRequest to include "ppo_shared"
- Lazy torch import so server starts without torch if not needed
- Frontend dropdown for policy selection

**Verified:** Pytest passed.

---

#### Step 8: Run History + Replay + Benchmark
**Built:**
- `simulation/runner/run_registry.py` — scans storage/runs/, exposes metadata
- `backend/api/routes_history.py` — GET /api/runs list + detail, replay endpoint (streams stored metrics.jsonl), POST /api/benchmark
- Frontend: Runs page + replay view reusing MetricsChart

**Verified:** Pytest passed.

---

#### Step 9A: League Storage
**Built:**
- `simulation/league/registry.py` — LeagueRegistry: list_members, save_snapshot with monotonic IDs + lineage, load_member, get_member_metadata, filesystem-backed under `storage/agents/league/`
- Backend: GET /api/league/members

**Verified:** Pytest passed.

---

#### Step 9B: Opponent Sampling + League Evaluation
**Built:**
- `simulation/league/sampling.py` — OpponentSampler with weights for recent_vs_old, baseline, league, fixed
- `simulation/league/eval_population.py` — CLI runner, episodes with sampled opponents, summary table
- `simulation/agents/league_snapshot_agent.py` — loads league member policy.pt

**Verified:** Pytest passed.

---

#### Step 9C-1: League Self-Play Training
**Built:**
- Modified `ppo_shared.py` — agent_0 learns, agent_1..N are opponents sampled from OpponentSampler each episode
- Added PPOConfig fields for opponent mix weights
- Tracks opponent source distribution + mean return per bucket
- Updated metadata with training_mode, opponent_mix, opponent_source_counts

**Verified:** Pytest passed.

---

#### Step 9C-2: Periodic Snapshots
**Built:**
- Extended PPOConfig with snapshot_every_timesteps, max_league_members
- During training: saves snapshot to league via LeagueRegistry.save_snapshot at intervals, tracks parent_id chain, trims oldest if over max
- Updated metadata with snapshots_created, last_league_snapshot_id

**Verified:** Pytest passed.

---

#### Step 10A: Elo Ratings + League UI
**Built:**
- `simulation/league/ratings.py` — Elo system: start_rating=1000, k_factor=32, pairwise matches, save/load JSON
- Backend: POST /api/league/ratings/recompute, GET /api/league/ratings
- Extended StartRunRequest for league_snapshot + league_member_id
- Frontend: /league page with ratings table, recompute button, "Run using this member" button

**Verified:** Pytest passed.

---

#### Step 10B: Lineage Graph + Champion Benchmark
**Built:**
- Backend: GET /api/league/lineage, GET /api/league/champion, POST /api/league/champion/benchmark
- `frontend/src/components/LeagueLineage.tsx` — SVG tree, parent→child, Elo-sized nodes, strategy-colored, clickable
- `frontend/src/components/ChampionBenchmark.tsx` — bar chart + table

**Verified:** Pytest passed.

---

#### Step 11A-1: Research-Grade Evaluation
**Built:**
- `simulation/evaluation/policy_set.py` — PolicySpec resolver
- `simulation/evaluation/evaluator.py` — cross-seed evaluation, per-seed breakdown, aggregated stats
- `simulation/evaluation/reporting.py` — JSON + Markdown reports to storage/reports/{report_id}/
- `simulation/evaluation/run_eval.py` — CLI entrypoint

**Verified:** Pytest passed.

---

#### Step 11A-2: Robustness Sweeps
**Built:**
- `simulation/evaluation/sweeps.py` — SweepSpec, build_default_sweeps (~10 variants)
- `simulation/evaluation/robustness.py` — evaluate across all sweeps, robustness_score = 0.7×mean + 0.3×worst_case
- `simulation/evaluation/run_robustness.py` — CLI
- Extended reporting for robustness reports

**Verified:** Pytest passed.

---

#### Step 11B: Reports Browser + Visualizations
**Built:**
- Backend: `routes_reports.py` — GET /api/reports list + detail
- Frontend: /reports page, /reports/[report_id] detail page
- `RobustHeatmap.tsx` — policies × sweeps heatmap
- `RobustScatter.tsx` — overall_mean vs worst_case scatter
- `RobustSummaryTable.tsx` — collapse_rate, top-3 callout, hardest sweep

**Verified:** Pytest passed.

---

#### Step 11C-1: Strategy Clustering
**Built:**
- `simulation/analysis/strategy_features.py` — extract per-policy features from report
- `simulation/analysis/strategy_clustering.py` — deterministic k-means, seed=0
- `simulation/analysis/strategy_labels.py` — rule-based labels: Exploitative/Cooperative/Robust/Unstable
- Backend: GET /api/reports/{report_id}/strategies

**Verified:** Pytest passed.

---

#### Step 11C-2: Strategy Groups UI
**Built:**
- Updated report detail page with cluster cards and policy table
- TypeScript types + getReportStrategies() in api.ts

---

#### Step A: Pipeline Automation
**Built:**
- `simulation/pipeline/pipeline_run.py` — run_pipeline orchestrator: load config → league PPO training → recompute Elo → select champion → robustness evaluation → write pipeline summary
- Backend: `routes_pipeline.py` — POST /api/pipeline/run, GET /api/pipeline/{id}/status
- `backend/runner/pipeline_manager.py` — in-memory status tracking

**Verified:** Pytest passed.

---

#### Step B: One-Click Champion Robustness
**Built:**
- Backend: POST /api/league/champion/robustness
- Frontend: Champion Robustness section on /league page with config selector + "Run Robustness" button

**Verified:** Pytest passed.

---

#### Step C1: Strategy Evolution Backend
**Built:**
- Backend: GET /api/league/evolution — lineage nodes enriched with Elo, strategy label, robustness_score, champion_history

**Verified:** Pytest passed.

---

#### Step C2: Evolution Tab Frontend
**Built:**
- New "Evolution" tab on /league page
- Lineage graph with label colors + rating
- Timeline panel showing label changes across snapshots
- TypeScript types + getLeagueEvolution() in api.ts

---

### Phase 4: Production Readiness
- Deployment readiness audit conducted
- 14 defects found (C1–C5, M1–M7, L1–L4), all fixed and verified
- Verdict: PRODUCTION-READY
- Non-blocking improvements noted: WebSocket reconnection backoff, CORS middleware, rate limiting, structured logging
- Pre-deployment checklist documented in `docs/bug-audit.md`

---

## Competitive Archetype

### Phase 1: Paper Spec
- Designed all 11 parts following the same methodology as Mixed
- Documented in `design/Competitive_archetype.md`

### Phase 2: Pre-Implementation

#### Step A: Sanity Check
- 8 ambiguities found and resolved:
  1. ATTACK targeting — untargeted in V1, distributed across non-defending opponents
  2. GAMBLE distribution — uniform multiplier [0.0, 2.5] on BUILD gain
  3. History sensitivity mechanism — 50% attack frequency → +20% defense bonus
  4. Relative gain denominator — fixed to initial agent count
  5. Terminal bonus + PettingZoo timing — held until global done flag
  6. Dominance threshold — fully deferred to V2
  7. information_asymmetry vs opponent_obs_window — orthogonal, not overlapping
  8. No baseline agents defined — RandomAgent, AlwaysAttack, AlwaysBuild, AlwaysDefend added
- Resolutions written back into `design/Competitive_archetype.md`

#### Step B: Architecture Design
- 3 decisions made and documented in `docs/architecture-decisions.md`
  - Config routing via discriminated union (ADR-010)
  - Separate PettingZoo adapter (ADR-011)
  - Separate metrics collector (ADR-012)
  - Nothing existing modified except pipeline routing (ADR-013)

### Phase 3: Implementation

#### Step C: Minimal Vertical Slice
**Built:**
- `simulation/envs/competitive/actions.py` — ActionType enum (BUILD, ATTACK, DEFEND, GAMBLE) + Action dataclass
- `simulation/envs/competitive/state.py` — AgentState, OpponentHistoryState, GlobalState with rankings and to_observation()
- `simulation/config/competitive_schema.py` — full Pydantic schema with all Part 9 validators
- `simulation/config/competitive_defaults.py` — default_competitive_config(seed=42)
- `simulation/envs/competitive/transition.py` — 8-phase resolve_actions()
- `simulation/envs/competitive/rewards.py` — 3-component reward + terminal bonus
- `simulation/envs/competitive/termination.py` — MAX_STEPS, ELIMINATION, NO_ACTIVE_AGENTS
- `simulation/envs/competitive/env.py` — CompetitiveEnvironment implementing BaseEnvironment
- `simulation/core/types.py` — added ELIMINATION to TerminationReason enum

**Verified:**
- Import check passed
- Full episode runs to MAX_STEPS with 4 random agents
- All 4 agents survive 200 steps with random play
- Termination reason correct
- All 231 existing Mixed tests still green


#### Step D: MARL Integration
**Built:**
- `simulation/adapters/competitive_pettingzoo.py` — PettingZoo ParallelEnv adapter with terminal bonus timing handled correctly
- `simulation/agents/competitive_baselines.py` — 4 baseline agents: RandomAgent, AlwaysAttack, AlwaysBuild, AlwaysDefend + registry
- `simulation/metrics/competitive_definitions.py` — step/episode metric keys, 5 event types
- `simulation/metrics/competitive_collector.py` — CompetitiveMetricsCollector
- `simulation/runner/competitive_experiment_runner.py` — full episode runner with run logger integration and run manager wiring
- `backend/schemas/api_models.py` — EnvironmentConfig union type
- `backend/api/routes_config.py` — config creation routes to handle both archetypes
- `backend/api/routes_experiment.py` — experiment start routes to handle both archetypes

**Verified:**
- PettingZoo parallel_api_test passes
- Full instrumented episode runs correctly (winner, score spread, events logged)
- Backend API accepts competitive configs and starts competitive runs
- Run history shows correct config_id and agent_policy for new runs
- All 231 existing Mixed tests still green


# Add to docs/build-log.md under Competitive Phase 3:

#### Step E: PPO Training
**Built:**
- `simulation/training/competitive_ppo.py` — PPO training loop, CompetitivePPOConfig, CompetitivePolicyNetwork, league snapshots to storage/agents/competitive_league/
- `simulation/agents/competitive_ppo_agent.py` — PPO inference agent, lazy torch import, obs flattening
- Updated `simulation/agents/competitive_baselines.py` — added competitive_ppo to registry

**Verified:**
- Full 50k timestep training run completes in ~72 seconds
- 4 league snapshots created during training
- PPO agent runs episodes correctly via policy registry
- All 231 existing tests still green


#### Step F: League System + Pipeline
**Built:**
- `simulation/league/competitive_sampling.py` — CompetitiveOpponentSampler
- `simulation/league/competitive_eval.py` — population eval across all policies
- `simulation/training/competitive_ppo.py` — updated with league self-play mode
- `backend/api/routes_competitive_league.py` — 6 league endpoints
- `simulation/pipeline/competitive_pipeline_run.py` — end-to-end pipeline
- `backend/api/routes_pipeline.py` — competitive pipeline endpoints
- `backend/main.py` — registered new routers

**Verified:**
- Full pipeline runs end-to-end: training → rating → eval → reporting
- 7 league members rated, champion identified (rating 1027)
- PPO agent outperforms all baselines in population eval
- All 231 existing Mixed tests still green


#### Step E: PPO Training
**Built:**
- `simulation/training/competitive_ppo.py` — PPO training loop, CompetitivePPOConfig, CompetitivePolicyNetwork, league snapshots to storage/agents/competitive_league/
- `simulation/agents/competitive_ppo_agent.py` — PPO inference agent, lazy torch import, obs flattening
- Updated `simulation/agents/competitive_baselines.py` — added competitive_ppo to registry

**Verified:**
- Full 50k timestep training run completes in ~72 seconds
- 4 league snapshots created during training
- PPO agent runs episodes correctly via policy registry
- All tests green

#### Step F: League System + Pipeline
**Built:**
- `simulation/league/competitive_sampling.py` — CompetitiveOpponentSampler
- `simulation/league/competitive_eval.py` — population eval across all policies
- Updated `simulation/training/competitive_ppo.py` — league self-play mode
- `backend/api/routes_competitive_league.py` — 6 league endpoints
- `simulation/pipeline/competitive_pipeline_run.py` — end-to-end pipeline
- Updated `backend/api/routes_pipeline.py` — competitive pipeline endpoints
- Updated `backend/main.py` — registered new routers

**Verified:**
- Full pipeline runs end-to-end: training → rating → eval → reporting
- 7 league members rated, champion identified (rating 1027)
- PPO agent outperforms all baselines in population eval
- All tests green

#### Frontend: Competitive Dashboard
**Built:**
- `frontend/src/app/competitive/page.tsx` — competitive dashboard with config form and league standings
- Updated `frontend/src/lib/api.ts` — competitive API client functions
- Updated `frontend/src/app/page.tsx` — added Competitive nav link

**Verified:**
- Frontend builds clean
- Create config + start run flow works end to end
- Run completes correctly (max_steps, episode_length 200)
- League standings table shows data after recompute

#### Tests
**Built:**
- `tests/unit/test_competitive_env.py` — 35 unit tests across 8 test classes

**Verified:**
- 266 total tests green (35 new + 231 existing)
- Zero regressions


#### Frontend: Complete Competitive UI
**Built:**
- `frontend/src/app/competitive/reports/page.tsx` — reports list
- `frontend/src/app/competitive/reports/[report_id]/page.tsx` — report detail with heatmap, scatter, strategy groups
- `frontend/src/components/CompetitiveRunSummary.tsx` — run summary with winner, rankings, scores
- `frontend/src/components/CompetitiveReplayView.tsx` — replay with score/rank/action charts
- Updated `frontend/src/app/run/[run_id]/page.tsx` — REST fallback + competitive/Mixed detection
- Updated `frontend/src/app/replay/[run_id]/page.tsx` — competitive/Mixed detection
- Updated `frontend/src/app/competitive/league/page.tsx` — full 4-tab league page
- Updated `frontend/src/lib/api.ts` — all competitive API functions

**Verified:**
- Create config + start run works end to end
- Live run page shows competitive summary after run completes
- Replay works for both competitive and Mixed runs
- League page all 4 tabs work correctly
- Champion benchmark and robustness work
- Reports list and detail pages work with heatmap populated
- Mixed functionality unchanged

#### Bug Fixes
- Mixed evolution endpoint was using competitive strategy labels — fixed
- Competitive nav links pointing to wrong paths — fixed
- Champion benchmark endpoint missing — added
- Robustness seeds schema mismatch — fixed
- Heatmap data transformation — fixed
- Replay crash on competitive runs — fixed


## Frontend Redesign

### Phase 1: Design & Planning
- Full feature inventory completed from browser walkthrough
- Decisions documented in docs/frontend-redesign.md:
  - Navigation: B+ structure (Simulate / League / Research)
  - Theme: dark, technical, data-forward
  - Accent color: teal (#14b8a6)
  - Template names: Resource Sharing Arena, Head-to-Head Strategy
  - Coming-soon templates: 4 grayed-out cards
  - Advanced mode: full config toggle on /simulate
  - Branch: feat/frontend-redesign

### Phase 2: Implementation (10 prompts)
- Prompt 1: Dark theme tokens + global Nav component
- Prompt 2: New URL structure + redirects
- Prompt 3: Simulate index page (template picker + advanced mode)
- Prompt 4: Template pages (resource-sharing + head-to-head)
- Prompt 5: Home page (hero, stats bar, feature highlights, quick-start)
- Prompt 6: Unified League page (archetype switcher + all tabs)
- Prompt 7: Research index (filter bar + unified report cards)
- Prompt 8: Move run/replay/report detail pages to new URLs
- Prompt 9: Delete old routes (competitive/*, runs, reports, run, replay)
- Prompt 10: Final verification pass — 2 bugs found and fixed

### Verified
- npm run build passes with zero errors and zero warnings
- All 12 routes compile cleanly
- All redirects fire correctly
- All back links point to correct URLs
- Nav active states correct for all routes