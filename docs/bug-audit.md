# Deployment Readiness Audit

This document tracks issues identified during the deployment-readiness audit of the multi-agent simulation platform. Each issue is assigned an ID, a brief description, a current status, and a corresponding GitHub issue number when available.

---

## Critical

| ID | Description | Status | GitHub Issue |
|----|-------------|--------|--------------|
| C1 | `shared_pool` referenced before assignment in `experiment_runner` — `NameError` crash if the environment terminates before the loop body executes | Open | TBD |
| C2 | WebSocket connection hardcodes port `8000` — breaks behind any reverse proxy or non-standard port | Open | TBD |
| C3 | Blocking CPU work (`_run_episode_sync`, `evaluate_robustness`, `compute_ratings`) runs directly inside `async def` endpoints, freezing the event loop | Open | TBD |
| C4 | `GET /api/league/champion` raises HTTP 404 instead of returning a graceful empty response on a fresh install with no league members | Open | TBD |
| C5 | `next.config.ts` hardcodes `http://localhost:8000` as the backend URL — every non-local deployment returns 502 | Open | TBD |

---

## Medium

| ID | Description | Status | GitHub Issue |
|----|-------------|--------|--------------|
| M1 | `compute_ratings` called synchronously inside `POST /api/league/ratings/recompute` — blocks the event loop for the duration of the match simulations | Open | TBD |
| M2 | `report_id` path parameter is used directly in a `Path` join without validation — path traversal risk | Open | TBD |
| M3 | All storage paths are relative to the server's working directory — fails silently when `uvicorn` is started from a directory other than the repo root | Open | TBD |
| M4 | `list_reports` sorts by `timestamp` string; reports with a missing `timestamp` field default to `""` and sort incorrectly | Open | TBD |
| M5 | Same uninitialized `shared_pool` bug as C1 exists in the benchmark episode runner inside `routes_league.py` | Open | TBD |
| M6 | `LeagueRegistry` is instantiated at module level on a path that may not exist — fragile on a fresh install if the registry does not create the directory in `__init__` | Open | TBD |
| M7 | `connectMetrics` has no reconnection logic — a transient WebSocket disconnect silently stops the live metrics chart with no user feedback | Open | TBD |

---

## Low

| ID | Description | Status | GitHub Issue |
|----|-------------|--------|--------------|
| L1 | `POST /api/league/ratings/recompute` returns an empty list with no explanation when fewer than 2 league members exist | Open | TBD |
| L2 | PyTorch and NumPy versions are not co-pinned between `pyproject.toml` and `environment.yml` — version drift likely across install methods | Open | TBD |
| L3 | Observation dict in the `routes_league` inline episode runner is populated from `env.reset()` but not re-validated against `env.active_agents()` — mismatch possible | Open | TBD |
| L4 | Storage subdirectories are created lazily per-request rather than at server startup — first request to each endpoint races with `mkdir` | Open | TBD |

---

## Workflow

1. **Issue** — Pick an item from this table and open a GitHub issue (replace TBD with the issue number).
2. **Branch** — Create a branch named `fix/<ID>-short-description` (e.g., `fix/c1-shared-pool`).
3. **Fix** — Make the minimal change required. Do not bundle unrelated changes.
4. **Test** — Ensure all existing tests pass (`pytest tests/unit -v`). Add a regression test if applicable.
5. **PR** — Open a pull request referencing the GitHub issue.
6. **Merge** — After review, merge and update the **Status** column in this file to `Fixed`.
