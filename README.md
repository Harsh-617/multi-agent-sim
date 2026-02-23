# Multi-Agent Simulation Platform

> **Work in Progress** — This project is under active development. The simulation engine, training pipeline, REST API, and dashboard are functional but evolving. APIs may change without notice. Not suitable for production use.

A research platform for studying emergent behaviour in multi-agent environments. Agents with different decision strategies interact inside configurable environments; their behaviour is recorded, evaluated, and visualised through a web dashboard. Training is done via PPO with optional league-based self-play.

---

## Key Idea: Archetypes vs. Templates

The design distinguishes two concepts:

- **Archetypes** are internal environment blueprints that define the fundamental dynamics of a scenario (e.g. how actions are structured, what the reward signal looks like, what causes an episode to end). They are engineering constructs, not user-facing.
- **Templates** (planned) are user-facing scenario presets built on top of archetypes — named, documented configurations a researcher can pick and run without understanding the internals.

**V1 implements a single archetype: Mixed.** The Mixed archetype models a shared-resource pool where agents can cooperate, extract, or defend. It supports configurable reward weights (individual vs. group vs. relational), temporal memory, reputation dynamics, and information asymmetry.

The design documents in [`design/`](design/) describe a roadmap of 8 archetypes and a template library. Those documents are planning artifacts — the code does not yet implement anything beyond Mixed.

---

## What Is Implemented Now

### Simulation Engine (`simulation/`)

- **`core/`** — Abstract `BaseEnvironment` contract (lifecycle, step, obs/action specs, termination); deterministic seeding utilities; shared type definitions.
- **`envs/mixed/`** — Full `MixedEnvironment` implementation:
  - Four action types: `cooperate`, `extract`, `defend`, plus a continuous `amount` parameter.
  - Three termination conditions: max steps, shared-pool collapse, all-agents-eliminated.
  - Configurable reward with three components (individual, group, relational) plus penalty scaling.
  - Observation includes step, shared pool, own resources, active agent count, per-peer cooperation scores, and a rolling action-history window.
- **`config/`** — Pydantic `MixedEnvironmentConfig` schema: population, layers (information asymmetry, temporal memory, reputation sensitivity, incentive softness, uncertainty), reward weights, agent config, instrumentation. Fully validated; serves as the single source of truth for reproducibility.
- **`adapters/`** — PettingZoo `ParallelEnv` wrapper over `MixedEnvironment`, enabling standard RL library compatibility.

### Agents (`simulation/agents/`)

- `BaseAgent` abstract interface.
- Rule-based baselines: `RandomAgent`, `AlwaysCooperate`, `AlwaysExtract`, `TitForTat`.
- `PPOSharedAgent` — loads a trained `policy.pt` for deterministic or stochastic inference.
- `LeagueSnapshotAgent` — loads a specific league checkpoint for evaluation or self-play.

### Training (`simulation/training/`)

- **Shared-policy PPO** (`ppo_shared.py`) — CleanRL-style implementation:
  - Actor-critic network: shared trunk → discrete action-type head (Categorical) + continuous amount head (Beta distribution) + value head.
  - Rollout buffer with GAE advantage estimation.
  - Clipped surrogate objective, value loss, entropy bonus, gradient clipping.
  - Optional TensorBoard logging.
  - Deterministic seeding. Artifact export: `policy.pt` + `metadata.json` (hyperparameters, config hash, obs layout).
- **League self-play mode** — trains against a mix of opponents sampled from the league registry, random baselines, and fixed policies. Periodic snapshots are saved automatically; old members are trimmed when the registry exceeds a configured size.

### League System (`simulation/league/`)

- `LeagueRegistry` — filesystem-backed snapshot store. Each member is a numbered folder (`league_000001/`) containing `policy.pt` and enriched `metadata.json` (parent lineage, timestamp, notes).
- `OpponentSampler` — weighted sampling across league members, baselines, and fixed policies; configurable recent-vs-old bias.
- Elo-style ratings (`ratings.py`).
- Population evaluation runner (`eval_population.py`).

### Evaluation & Analysis (`simulation/evaluation/`, `simulation/analysis/`)

- Cross-seed policy evaluator: runs N policies × M seeds, aggregates mean/std returns, final pool health, collapse rate, episode length.
- Robustness sweeps: evaluate a policy across a grid of config perturbations.
- Strategy feature extraction, numpy-only k-means clustering (deterministic seed), strategy labeling.
- Report generation.

### Metrics (`simulation/metrics/`)

- `MetricsCollector` — records per-step and per-episode metrics during runs.
- Definitions for standard metric keys.
- `RunLogger` — persists run data to `storage/runs/`.

### Backend API (`backend/`)

FastAPI application with six routers:

| Route group | Endpoints |
|---|---|
| `/api/health` | Health check |
| `/api/runs` | `POST /start`, `POST /stop`, `GET /status` |
| `/api/configs` | List and retrieve saved experiment configs |
| `/api/history` | Past run records |
| `/api/league` | League member listing |
| `/api/reports` | Evaluation reports |
| `ws://…/ws/metrics` | WebSocket stream of live per-step metrics |

Configs are stored as JSON in `storage/configs/`. Runs are one-at-a-time; the run manager tracks the active async task.

### Frontend Dashboard (`frontend/`)

Next.js 14 app with:

- **Home** — config list, navigation to other views.
- **`/run/[run_id]`** — live run view: real-time metrics chart over WebSocket, stop button.
- **`/replay/[run_id]`** — step-through replay of a completed run.
- **`/runs`** — run history list.
- **`/league`** — league browser: agent lineage tree (`LeagueLineage`), champion benchmark (`ChampionBenchmark`).
- **`/reports`** and **`/reports/[report_id]`** — robustness heatmap, scatter plot, summary table.

### Tests (`tests/`)

- **Unit** (22 test files): agents, config validation, evaluation, league ratings/registry/self-play/snapshotting, metrics collector, mixed environment, opponent sampling, PettingZoo adapter, PPO training, robustness, run logger, run registry, strategy analysis.
- **Integration** (4 test files): API contract tests for experiments, history, league, and reports endpoints (using `httpx`).

---

## What Is Planned Next (Roadmap)

These are engineering plans, not promises. They reflect the direction described in [`design/`](design/).

- **7 additional archetypes**: purely cooperative, purely competitive, hierarchical authority, deceptive/hidden-role, negotiation/bargaining, emergency/crisis, and market/exchange.
- **Template library**: user-facing named scenario presets built on top of archetypes (e.g. "commons dilemma", "price war", "evacuation"). Selectable from the UI without editing config JSON.
- **Multi-archetype composition**: allow multiple environment types to coexist in a single session.
- **Improved training tooling**: training runs managed through the UI/API (not just CLI); multi-GPU support; more RL algorithm options.
- **Expanded agent zoo**: recurrent policies (LSTM), population-based training, imitation learning from human play.
- **Richer observability**: agent-level metric breakdowns, action-distribution histograms, interactive replay controls.
- **Productization**: authentication, multi-user support, persistent experiment tracking, packaged deployment.

---

## Architecture Overview

```
multi-agent-sim/
├── simulation/          # Python package — core engine
│   ├── core/            # BaseEnvironment, types, seeding
│   ├── envs/mixed/      # MixedEnvironment (the only archetype in V1)
│   ├── adapters/        # PettingZoo parallel env wrapper
│   ├── agents/          # BaseAgent + rule-based + PPO + league snapshot
│   ├── config/          # Pydantic schema (single source of truth)
│   ├── training/        # PPO training loop (standard + league self-play)
│   ├── league/          # Registry, sampler, ratings, eval population
│   ├── evaluation/      # Cross-seed evaluator, robustness sweeps, reports
│   ├── analysis/        # Strategy feature extraction, clustering, labeling
│   ├── metrics/         # MetricsCollector, RunLogger
│   └── runner/          # Run logger
├── backend/             # FastAPI REST API + WebSocket
│   ├── api/             # Route modules (configs, experiments, history, league, reports, ws)
│   ├── runner/          # ExperimentRunner, RunManager (async)
│   ├── schemas/         # Pydantic API models
│   └── main.py          # FastAPI app assembly
├── frontend/            # Next.js 14 dashboard
│   └── src/
│       ├── app/         # Pages (home, run, replay, runs, league, reports)
│       ├── components/  # React components (charts, tables, lineage)
│       └── lib/api.ts   # Typed API client
├── tests/
│   ├── unit/            # 22 unit test modules
│   └── integration/     # 4 API integration test modules
├── storage/             # Runtime artifacts (gitignored except .gitkeep)
│   ├── configs/         # Saved experiment configs (JSON)
│   ├── runs/            # Run output files
│   └── agents/          # Trained policy artifacts + league snapshots
└── design/              # Architecture docs and roadmap (planning only)
```

**Data flow (happy path):**

1. User saves a config JSON to `storage/configs/` (via the UI or directly).
2. `POST /api/runs/start` loads the config, validates it, and spawns an async `ExperimentRunner` task.
3. The runner steps through `MixedEnvironment`, collects metrics via `MetricsCollector`, and broadcasts step data over the WebSocket.
4. The frontend `/run/[run_id]` page connects to the WebSocket and renders the live chart.
5. On completion, the run log is written to `storage/runs/<run_id>/`.
6. Evaluation and reports are generated separately via the evaluation module or the `/api/reports` endpoint.

---

## Quickstart

### Option A — Conda (recommended, Windows/Mac/Linux)

```bash
# Clone
git clone https://github.com/<your-username>/multi-agent-sim.git
cd multi-agent-sim

# Create and activate the conda environment
conda env create -f environment.yml
conda activate multi-agent-sim

# Install frontend dependencies
cd frontend && npm install && cd ..
```

The conda environment installs Python 3.11, Node 18, and the package in editable mode with dev and training extras (`pip install -e ".[dev,training]"`).

### Option B — pip / venv

```bash
# Clone
git clone https://github.com/<your-username>/multi-agent-sim.git
cd multi-agent-sim

# Python environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Runtime + dev deps
pip install -e ".[dev]"

# Optional: training extras (PyTorch, tqdm, tensorboard)
pip install -e ".[training]"

# Frontend
cd frontend && npm install && cd ..
```

> **Note:** `uvicorn[standard]` is used in `requirements.txt` (vs. bare `uvicorn` in `pyproject.toml`) to ensure the WebSocket transport (`websockets` / `httptools`) is available for the live metrics stream. Both are compatible — the `[standard]` extra is additive.

---

## How to Run

### Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm run dev
```

Dashboard at `http://localhost:3000`.

### Tests

```bash
# All tests
pytest tests/

# Unit tests only (faster, no running server required)
pytest tests/unit/

# With coverage report
pytest tests/ --cov=simulation --cov=backend --cov-report=term-missing
```

### PPO Training (CLI)

```bash
# Standard training (requires [training] extras)
python -m simulation.training.ppo_shared

# Artifacts are saved to storage/agents/ppo_shared/
```

### Evaluation (CLI)

```bash
python -m simulation.evaluation
```

---

## Dependency Notes

| File | Purpose |
|---|---|
| `pyproject.toml` | Single source of truth for all Python deps |
| `requirements.txt` | Convenience file for `pip install -r` (runtime deps) |
| `requirements-dev.txt` | Runtime + test deps |
| `environment.yml` | Conda environment (Python 3.11 + Node 18 + editable install) |

Training dependencies (PyTorch, tqdm, TensorBoard) are in the `[training]` optional group and are **not** installed by default. Install with `pip install -e ".[training]"` or use the conda environment (which installs `.[dev,training]`).

---

## Reproducibility and Disclaimer

All experiments use a root integer seed that propagates deterministically through environment resets, agent initialisation, and training. The `config_hash` stored in `metadata.json` alongside each trained policy identifies the exact configuration used.

**This platform is for research in simulated multi-agent environments only.** The decision models produced here are experimental and untested outside their training distribution. This is not financial software, not a trading system, and not financial advice. Do not use outputs from this platform to make real-world decisions.

---

## Status

This is a solo WIP project. External contributions are not being accepted at this time.

## License

MIT — see [LICENSE](LICENSE).
