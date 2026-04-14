# Deployment Readiness Audit

This document tracks issues identified during the deployment-readiness audit of the multi-agent simulation platform. Each issue is assigned an ID, a brief description, a current status, and a corresponding GitHub issue number when available.

> **All deployment-readiness issues identified in the audit have now been resolved.**

---

## Recently Resolved

The following deployment-readiness issues were fixed in PR #8:

- C1 — shared_pool initialization crash
- C3 — event loop blocking in benchmark/robustness endpoints
- C4 — empty league champion endpoint returning 404
- C5 — hardcoded backend URL in Next.js config
- M5 — shared_pool initialization in benchmark runner verified safe

The following deployment-readiness issues were fixed in PR #13 / Branch: fix/event-loop-and-report-sorting:

- M1 — ratings recompute blocking the event loop
- M4 — fragile report timestamp sorting

The following deployment-readiness issues were fixed in PR #12 / Branch: fix/minor-runtime-cleanups:

- L1 — recompute_ratings silent empty response with fewer than 2 members
- L2 — numpy dependency alignment between environment.yml and pyproject.toml
- L3 — env.reset observation alignment with active agents

The following deployment-readiness issues were fixed in PR #11 / Branch: fix/frontend-ws-resilience:

- C2 — WebSocket URL hardcodes backend port 8000
- M7 — No WebSocket reconnection logic

The following deployment-readiness issues were fixed in PR #10:

- M2 — path traversal risk in /api/reports/{report_id}
- M3 — storage paths depending on current working directory
- M6 — registry/storage initialization fragile on fresh install
- L4 — storage directories created lazily per-request

---

## Critical

| ID | Description | Status | GitHub Issue |
|----|-------------|--------|--------------|
| C1 | `shared_pool` referenced before assignment in `experiment_runner` — `NameError` crash if the environment terminates before the loop body executes | Fixed — Resolved in PR: #8 / Branch: fix/deploy-blockers | TBD |
| C2 | WebSocket connection hardcodes port `8000` — breaks behind any reverse proxy or non-standard port | Fixed — Resolved in PR: #11 / Branch: fix/frontend-ws-resilience | TBD |
| C3 | Blocking CPU work (`_run_episode_sync`, `evaluate_robustness`, `compute_ratings`) runs directly inside `async def` endpoints, freezing the event loop | Fixed — Resolved in PR: #8 / Branch: fix/deploy-blockers | TBD |
| C4 | `GET /api/league/champion` raises HTTP 404 instead of returning a graceful empty response on a fresh install with no league members | Fixed — Resolved in PR: #8 / Branch: fix/deploy-blockers | TBD |
| C5 | `next.config.ts` hardcodes `http://localhost:8000` as the backend URL — every non-local deployment returns 502 | Fixed — Resolved in PR: #8 / Branch: fix/deploy-blockers | TBD |

---

## Medium

| ID | Description | Status | GitHub Issue |
|----|-------------|--------|--------------|
| M1 | `compute_ratings` called synchronously inside `POST /api/league/ratings/recompute` — blocks the event loop for the duration of the match simulations | Fixed — Resolved in PR: #13 / Branch: fix/event-loop-and-report-sorting | TBD |
| M2 | `report_id` path parameter is used directly in a `Path` join without validation — path traversal risk | Fixed — Resolved in PR: #10 / Branch: fix/storage-paths | TBD |
| M3 | All storage paths are relative to the server's working directory — fails silently when `uvicorn` is started from a directory other than the repo root | Fixed — Resolved in PR: #10 / Branch: fix/storage-paths | TBD |
| M4 | `list_reports` sorts by `timestamp` string; reports with a missing `timestamp` field default to `""` and sort incorrectly | Fixed — Resolved in PR: #13 / Branch: fix/event-loop-and-report-sorting | TBD |
| M5 | Same uninitialized `shared_pool` bug as C1 exists in the benchmark episode runner inside `routes_league.py` | Fixed — Resolved in PR: #8 / Branch: fix/deploy-blockers | TBD |
| M6 | `LeagueRegistry` is instantiated at module level on a path that may not exist — fragile on a fresh install if the registry does not create the directory in `__init__` | Fixed — Resolved in PR: #10 / Branch: fix/storage-paths | TBD |
| M7 | `connectMetrics` has no reconnection logic — a transient WebSocket disconnect silently stops the live metrics chart with no user feedback | Fixed — Resolved in PR: #11 / Branch: fix/frontend-ws-resilience | TBD |

---

## Low

| ID | Description | Status | GitHub Issue |
|----|-------------|--------|--------------|
| L1 | `POST /api/league/ratings/recompute` returns an empty list with no explanation when fewer than 2 league members exist | Fixed — Resolved in PR: #12 / Branch: fix/minor-runtime-cleanups | TBD |
| L2 | PyTorch and NumPy versions are not co-pinned between `pyproject.toml` and `environment.yml` — version drift likely across install methods | Fixed — Resolved in PR: #12 / Branch: fix/minor-runtime-cleanups | TBD |
| L3 | Observation dict in the `routes_league` inline episode runner is populated from `env.reset()` but not re-validated against `env.active_agents()` — mismatch possible | Fixed — Resolved in PR: #12 / Branch: fix/minor-runtime-cleanups | TBD |
| L4 | Storage subdirectories are created lazily per-request rather than at server startup — first request to each endpoint races with `mkdir` | Fixed — Resolved in PR: #10 / Branch: fix/storage-paths | TBD |

---

## Workflow

1. **Issue** — Pick an item from this table and open a GitHub issue (replace TBD with the issue number).
2. **Branch** — Create a branch named `fix/<ID>-short-description` (e.g., `fix/c1-shared-pool`).
3. **Fix** — Make the minimal change required. Do not bundle unrelated changes.
4. **Test** — Ensure all existing tests pass (`pytest tests/unit -v`). Add a regression test if applicable.
5. **PR** — Open a pull request referencing the GitHub issue.
6. **Merge** — After review, merge and update the **Status** column in this file to `Fixed`.

---

### Pre-Deployment Audit (April 2026)

9 issues found and fixed before deployment:

| ID | Severity | File(s) | Description | Status |
|----|----------|---------|-------------|--------|
| P1 | Critical | backend/api/routes_pipeline.py, frontend/src/app/league/page.tsx | Competitive pipeline never returned report_id — backend task and status endpoint fixed, frontend polling now extracts it | Fixed |
| P2 | Major | frontend/src/app/league/page.tsx | League page had 100+ Tailwind light-theme classes — replaced with inline dark theme styles | Fixed |
| P3 | Major | frontend/src/app/league/page.tsx | Head-to-Head robustness seeds sent as [3] instead of 3 — fixed to send int | Fixed |
| P4 | Major | frontend/src/components/ConfigList.tsx | ConfigList navigated to dead route /run/${id} — fixed to correct URL | Fixed |
| P5 | Major | frontend/src/app/league/page.tsx | HH pipeline polling never extracted report_id — fixed | Fixed |
| P6 | Minor | frontend/src/app/page.tsx | Home page "live" indicator now shows "offline" when API is down | Fixed |
| P7 | Minor | frontend/src/app/page.tsx | Recent activity React keys changed from index to run_id | Fixed |
| P8 | Minor | frontend/src/components/ChampionRobustness.tsx | Robustness polling catch block now shows error to user | Fixed |
| P9 | Minor | frontend/src/app/league/page.tsx | Pipeline polling catch blocks now set error stage | Fixed |

---

### Second-Pass Audit (April 2026)

12 issues found and fixed after pre-deployment audit:

| ID | Severity | File(s) | Description | Status |
|----|----------|---------|-------------|--------|
| F1 | Critical | tests/integration/test_league_api.py | 4 tests never updated after champion robustness endpoint was refactored from sync to async — updated all 4 to use polling pattern | Fixed |
| F2 | Critical | frontend/src/lib/api.ts | WebSocket onmessage JSON.parse had no try-catch — malformed message killed all future message processing | Fixed |
| F3 | Critical | frontend/src/lib/api.ts | SSE replay onmessage JSON.parse had no try-catch — malformed event crashed the stream mid-replay | Fixed |
| F4 | Critical | frontend/src/app/simulate/resource-sharing/run/[run_id]/page.tsx, frontend/src/app/simulate/head-to-head/run/[run_id]/page.tsx | WebSocket callback arguments swapped in both run pages — caused race condition on error and unnecessary retry on clean close | Fixed |
| F5 | Major | backend/api/routes_history.py | Replay SSE endpoint crashed on malformed metrics.jsonl lines — added try-except and safe key access | Fixed |
| F6 | Major | backend/api/routes_history.py | Benchmark run_id not unique per request — concurrent requests with same config overwrote each other | Fixed |
| F7 | Major | frontend/src/app/league/page.tsx | HH robustness polling catch block never reset running state — perpetual spinner on network error | Fixed |
| F8 | Major | frontend/src/app/league/page.tsx | Pipeline polling catch blocks set error stage but never set error message — blank error state shown to user | Fixed |
| F9 | Major | backend/api/routes_competitive_league.py | seeds field type mismatch between two ChampionRobustnessRequest schemas — competitive endpoint rejected int with 422 | Fixed |
| F10 | Minor | frontend/src/components/RobustScatter.tsx | Math.min/max called on empty array produced broken SVG scales — added no-data placeholder | Fixed |
| F11 | Minor | backend/runner/run_manager.py | Unbounded asyncio.Queue for WebSocket subscribers — slow clients leaked memory at ~50ms broadcast rate | Fixed |
| F12 | Minor | backend/api/routes_config.py | Malformed config files silently skipped with no logging — added logger.warning so server logs show the cause | Fixed |

---

## Cooperative Archetype Pre-Deployment Audit

Audit conducted 2026-04-15 against all files added or modified for the Cooperative archetype.

**Result: 8 issues found (2 Critical, 3 Major, 3 Minor). All fixed. 500 tests pass. 0 TypeScript errors.**

### Critical

| ID | Severity | File(s) | Description | Status |
|----|----------|---------|-------------|--------|
| CP1 | Critical | simulation/evaluation/cooperative_sweeps.py | `spec_high` sweep variant sets `specialization_scale=0.6`, which violates the schema field constraint `le=0.5`. `apply_coop_sweep` calls `model_copy(update={"specialization_scale": 0.6})`, Pydantic raises `ValidationError`, and every robustness pipeline run crashes when it reaches this sweep (sweep 14 of 20). Fixed: changed `0.6` → `0.5` (the schema maximum). | Fixed |
| CP2 | Critical | backend/api/routes_cooperative_pipeline.py | `run_cooperative_pipeline()` was called with only request-level kwargs (config_id, seed, timesteps…) — none of the path overrides (`ppo_agent_dir`, `pipelines_dir`, `configs_dir`, `reports_dir`) were passed. The function fell back to hardcoded relative `Path("storage/…")` defaults, reproducing the old M3 CWD-dependent path bug for the cooperative pipeline. Fixed: added four STORAGE_ROOT-derived path kwargs to the kwargs dict passed to `run_cooperative_pipeline`. | Fixed |

### Major

| ID | Severity | File(s) | Description | Status |
|----|----------|---------|-------------|--------|
| CP3 | Major | simulation/pipeline/pipeline_run.py | When a saved config has `environment_type == "cooperative"`, `run_pipeline()` loaded it as a `CooperativeEnvironmentConfig` and then passed it directly to `ppo_shared.train` (the Mixed-archetype trainer). That trainer instantiates `MixedPettingZooParallelEnv`, which would crash with a confusing AttributeError deep in the adapter. Any direct call to `run_pipeline()` with a cooperative config_id would fail. Fixed: replaced the `CooperativeEnvironmentConfig.model_validate` branch with an early `ValueError` that directs callers to `run_cooperative_pipeline()` instead. Removed now-unused `CooperativeEnvironmentConfig` import. | Fixed |
| CP4 | Major | frontend/src/components/CooperativeReplayView.tsx, frontend/src/lib/api.ts | `streamCooperativeReplay`'s `onerror` handler called `onDone()` — the same callback as a successful close — so a 404 (run not found) or dropped SSE connection was indistinguishable from a clean finish. The component's `error` state was declared and displayed but never populated. User saw blank charts with no explanation. Fixed: added `onError?: (detail: string) => void` parameter to `streamCooperativeReplay`; `onerror` now calls `onError` with a descriptive message instead of `onDone`. The component passes an error callback that sets `error` state and marks `loaded=true`. | Fixed |
| CP5 | Major | backend/schemas/api_models.py | `EnvironmentConfig` union type only listed `MixedEnvironmentConfig \| CompetitiveEnvironmentConfig`. The comment notes it is for documentation purposes, but omitting the cooperative type means any tooling that inspects this union (API schema generators, type stubs) gives an incomplete picture of accepted configs. Fixed: added `CooperativeEnvironmentConfig` to the union. | Fixed |

### Minor

| ID | Severity | File(s) | Description | Status |
|----|----------|---------|-------------|--------|
| CP-L1 | Minor | simulation/evaluation/cooperative_eval_runner.py, simulation/evaluation/cooperative_robustness.py | Module-level `_REPORTS_ROOT = Path("storage/reports")` is a relative path. Safe in practice because every caller passes an explicit `report_root=` argument, but the default is misleading and would silently fail if these functions were ever called standalone from a non-root directory. | Fixed — changed to `Path(__file__).resolve().parent.parent.parent / "storage" / "reports"` |
| CP-L2 | Minor | backend/api/routes_cooperative_league.py | `GET /api/cooperative/league/champion` returns HTTP 404 when the league is empty. The frontend handles this gracefully with `.catch(() => null)`, but the endpoint semantics differ from the Mixed archetype's league endpoint. Should be normalized to return an empty response (`{}` or `null`) in a future cleanup. | Fixed — endpoint now returns `{"member_id": None}` on empty league, matching Mixed archetype behavior. Test updated accordingly. |
| CP-L3 | Minor | simulation/pipeline/cooperative_pipeline_run.py | Module-level `_AGENTS_DIR`, `_COOPERATIVE_PPO_DIR`, `_PIPELINES_DIR`, `_CONFIGS_DIR`, `_REPORTS_DIR` are all relative paths. Now unreachable through the API (C2 fix ensures callers always supply absolute overrides), but standalone script invocations from non-root directories would still fail. | Fixed — introduced `_STORAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "storage"` and derived all five path constants from it. |

### Scope check — existing archetypes not modified

Verified no Mixed or Competitive environment files, adapters, or metrics collectors were touched:
- `simulation/envs/mixed/` — unchanged
- `simulation/envs/competitive/` — unchanged
- `simulation/adapters/pettingzoo.py` — unchanged
- `simulation/adapters/competitive_pettingzoo.py` — unchanged
- `simulation/metrics/` (mixed and competitive collectors) — unchanged
