# Multi-Agent Simulation Platform

![Status](https://img.shields.io/badge/status-WIP%20%E2%80%93%20stable-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

A research platform for studying emergent strategy and cooperation in multi-agent environments. Agents with heterogeneous policies interact inside a configurable shared-resource environment; their behavior is recorded, evaluated across stress-tested environment variants, and visualized through a full-stack web dashboard. Training is done via PPO with league-based evolutionary self-play, producing a lineage of competitive policy snapshots with Elo ratings and robustness profiles.

The system is designed around a single, well-specified environment archetype — the **Mixed** environment, which models a shared resource pool where agents choose to cooperate, extract, or defend. The configuration surface is deep: five orthogonal behavioral layers control information asymmetry, temporal memory depth, reputation dynamics, incentive regime, and observation uncertainty. Everything downstream — training, evaluation, robustness sweeps, the backend API, and the dashboard — operates on this single source of truth.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        multi-agent-sim · V1 Architecture                      │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌───────────────┐     ┌────────────────────────────────────────────────────┐
  │  Config       │     │                 Simulation Engine                   │
  │  (Pydantic)   │────▶│                                                    │
  │               │     │  ┌──────────────────┐   ┌────────────────────────┐ │
  │ · 5 layers    │     │  │  MixedEnvironment │◀──│        Agents          │ │
  │ · RewardWeights     │  │                  │   │  · Random              │ │
  │ · PopulationCfg│    │  │  4 action types  │   │  · AlwaysCooperate     │ │
  │ · Validation  │     │  │  3-component rwd │   │  · TitForTat           │ │
  └───────────────┘     │  │  3 terminations  │   │  · PPOShared (infer)   │ │
                        │  │  PettingZoo wrap │   │  · LeagueSnapshot      │ │
                        │  └────────┬─────────┘   └────────────────────────┘ │
                        └───────────│────────────────────────────────────────┘
                                    │ rollout data
                        ┌───────────▼────────────────────────────────────────┐
                        │                 PPO Training                        │
                        │  CleanRL-style · Shared trunk (64→64 Tanh)          │
                        │  Categorical action-type head + Beta amount head     │
                        │  GAE advantage · Clipped surrogate · Ent. bonus      │
                        │  League self-play mode + periodic snapshotting       │
                        └───────────┬────────────────────────────────────────┘
                                    │ policy.pt + metadata.json
                        ┌───────────▼────────────────────────────────────────┐
                        │               League Registry                       │
                        │  Filesystem snapshots · Parent–child lineage        │
                        │  Elo ratings (K=32) · Weighted opponent sampler     │
                        │  Recent-vs-old bias · Max-member trimming           │
                        └───────────┬────────────────────────────────────────┘
                                    │ champion + member pool
                        ┌───────────▼────────────────────────────────────────┐
                        │             Robustness Evaluation                   │
                        │  10 environment variants (info, noise, incentive,   │
                        │  scarcity, population, combined stress)             │
                        │  Score = 0.7 × mean_reward + 0.3 × worst_case      │
                        └───────────┬────────────────────────────────────────┘
                                    │ JSON + Markdown reports
                        ┌───────────▼────────────────────────────────────────┐
                        │              FastAPI Backend                        │
                        │  7 routers · WebSocket per-step metrics stream      │
                        │  Async experiment runner · Run state manager        │
                        └───────────┬────────────────────────────────────────┘
                                    │ REST + WebSocket
                        ┌───────────▼────────────────────────────────────────┐
                        │            Next.js 14 Dashboard                     │
                        │  League evolution graph (SVG) · Robustness heatmap  │
                        │  Live metrics chart · Champion benchmark             │
                        │  Run history · Report browser                       │
                        └────────────────────────────────────────────────────┘
```

### Runtime Data Flow

```
  User saves config JSON
          │
          ▼
  POST /api/runs/start  ─────────────────────────────────────────────────┐
          │                                                                │
          ▼                                                                │
  ExperimentRunner (async task)                                            │
  ┌────────────────────────────────────────────────────────────────┐      │
  │  env.reset(seed)  ──▶  initial observations                     │      │
  │  loop each step:                                                │      │
  │    agent.act(obs)   ──▶  action                                 │      │
  │    env.step(actions) ──▶  rewards, next obs, done, info        │      │
  │    metrics.collect_step()                                       │      │
  │    mgr.broadcast({type:"step", metrics, events})  ─────────────┼──▶  ws://…/ws/metrics
  │  on done: write run artifacts                                   │      │
  └────────────────────────────────────────────────────────────────┘      │
          │                                                                │
          ▼                                                                ▼
  storage/runs/{run_id}/                                    /run/[id] live chart
  ├── config.json
  ├── metrics.jsonl
  ├── events.jsonl
  └── episode_summary.json
```

---

## Feature Checklist

### Simulation Engine
- ✅ `BaseEnvironment` abstract contract (lifecycle, obs/action specs, termination)
- ✅ `MixedEnvironment` — shared resource pool with cooperative/competitive/defensive dynamics
- ✅ Four action types: `COOPERATE`, `EXTRACT`, `DEFEND`, `CONDITIONAL`
- ✅ Three-component reward: individual gain + group pool health + relational reputation
- ✅ Three termination conditions: max steps, shared-pool collapse, all-agents-eliminated
- ✅ Five behavioral config layers (see [Configuration](#configuration))
- ✅ PettingZoo `ParallelEnv` wrapper for standard RL library compatibility
- ✅ Fully deterministic seeding via `derive_seed(base, offset)` propagation
- 🚧 Additional environment archetypes (cooperative-only, competitive, hierarchical, negotiation, market, emergency, deceptive)
- 🚧 Multi-archetype composition in a single session

### Agents & Policies
- ✅ `BaseAgent` abstract interface
- ✅ Rule-based baselines: `RandomAgent`, `AlwaysCooperate`, `AlwaysExtract`, `TitForTat`
- ✅ `PPOSharedAgent` — lazy-loaded inference wrapper, deterministic and stochastic modes
- ✅ `LeagueSnapshotAgent` — loads a specific league checkpoint by member ID
- ✅ Observation flattening: step, pool, resources, peer cooperation scores, action-history window
- 🚧 Recurrent policies (LSTM)
- 🚧 Population-based training
- 🚧 Imitation learning from recorded trajectories

### Training
- ✅ Shared-policy PPO (CleanRL-style, homogeneous population)
- ✅ Actor-critic network: shared trunk → Categorical action-type head + Beta amount head + value head
- ✅ Rollout buffer with GAE advantage estimation
- ✅ Clipped surrogate objective, value loss (MSE), entropy bonus, gradient clipping
- ✅ League self-play mode with configurable opponent sampling weights
- ✅ Periodic automatic snapshotting to league registry during training
- ✅ TensorBoard logging (optional)
- ✅ Artifact export: `policy.pt` + `metadata.json` (hyperparams, config hash, obs layout)
- 🚧 Training runs managed through the backend API / UI

### League System
- ✅ `LeagueRegistry` — filesystem-backed snapshot store with monotonic member IDs
- ✅ Parent–child lineage tracking across training generations
- ✅ Elo ratings with pairwise evaluation (K=32)
- ✅ `OpponentSampler` — weighted sampling across league members, baselines, and fixed policies
- ✅ Configurable recent-vs-old bias; registry auto-trims at max member count
- ✅ Population evaluation runner

### Evaluation & Robustness
- ✅ Cross-seed policy evaluator: N policies × M seeds, mean/std returns, collapse rate, episode length
- ✅ 10 default robustness sweep variants (information, noise, incentive, scarcity, population, combined)
- ✅ Robustness score: `0.7 × overall_mean + 0.3 × worst_case_reward`
- ✅ Strategy feature extraction and k-means clustering (NumPy-only, deterministic)
- ✅ Report generation: JSON + Markdown, named by type/config/timestamp
- ✅ Full pipeline orchestrator: training → league creation → evaluation → report

### Backend API
- ✅ FastAPI application with 7 routers
- ✅ Run lifecycle: start, stop (graceful), status
- ✅ WebSocket per-step metrics stream (`ws://…/ws/metrics`)
- ✅ Config CRUD, run history, league endpoints, report endpoints
- ✅ Champion identification, champion benchmark vs. baselines, champion robustness sweep
- ✅ League evolution endpoint: lineage + strategy labels + timeline
- 🚧 Concurrent multi-run support

### Frontend Dashboard
- ✅ Live run view with real-time metrics chart over WebSocket
- ✅ League evolution page: SVG lineage graph with parent–child edges, color by strategy label
- ✅ Robustness heatmap, scatter plot, and summary table
- ✅ Champion benchmark bar chart vs. baselines
- ✅ Run history browser and report browser
- ✅ Typed API client (`lib/api.ts`)
- 🚧 Interactive step-through replay of completed runs
- 🚧 User-defined configuration templates from the UI
- 🚧 Agent-level metric breakdowns and action-distribution histograms

### Infrastructure & Testing
- ✅ 22 unit test modules covering all major simulation and league components
- ✅ 4 integration test modules covering API contracts (using `httpx`)
- ✅ `environment.yml` for reproducible Conda setup (Python 3.11 + Node 18)
- ✅ `pyproject.toml` with optional `[dev]` and `[training]` dependency groups
- 🚧 Authentication and multi-user support
- 🚧 Packaged deployment (Docker)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation core | Python 3.11, NumPy |
| Configuration | Pydantic V2 |
| RL training | PyTorch, CleanRL-style PPO |
| RL compatibility | PettingZoo `ParallelEnv` |
| Backend | FastAPI, Uvicorn, WebSocket (`websockets`) |
| Frontend | Next.js 14, React, Tailwind CSS |
| Storage | Filesystem JSON / JSONL |
| Testing | pytest, httpx |
| Optional logging | TensorBoard |

---

## Repository Layout

```
multi-agent-sim/
├── simulation/                  # Core Python engine
│   ├── core/                    # BaseEnvironment, types, seeding
│   ├── envs/
│   │   └── mixed/               # MixedEnvironment (V1 archetype)
│   │       ├── env.py           # Main environment class
│   │       ├── actions.py       # ActionType enum + Action dataclass
│   │       ├── state.py         # AgentState, RelationalState, GlobalState
│   │       ├── rewards.py       # Three-component reward computation
│   │       ├── transition.py    # Action resolution + state mutation
│   │       └── termination.py   # Episode termination checks
│   ├── adapters/                # PettingZoo parallel env wrapper
│   ├── agents/                  # BaseAgent + baselines + PPO + league agents
│   ├── config/                  # MixedEnvironmentConfig schema + defaults
│   ├── training/                # PPO training loop (standard + league self-play)
│   ├── league/                  # Registry, ratings, sampler, population eval
│   ├── evaluation/              # Cross-seed evaluator, robustness sweeps, reporting
│   ├── analysis/                # Strategy feature extraction + k-means clustering
│   ├── metrics/                 # MetricsCollector, event definitions
│   ├── runner/                  # RunLogger (artifact persistence)
│   └── pipeline/                # End-to-end pipeline orchestrator
│
├── backend/                     # FastAPI server
│   ├── api/                     # Route modules
│   │   ├── routes_config.py
│   │   ├── routes_experiment.py
│   │   ├── routes_history.py
│   │   ├── routes_league.py
│   │   ├── routes_pipeline.py
│   │   ├── routes_reports.py
│   │   └── ws_metrics.py        # WebSocket endpoint
│   ├── runner/                  # ExperimentRunner (async) + RunManager (state)
│   ├── schemas/                 # Pydantic request/response models
│   └── main.py                  # App assembly + router registration
│
├── frontend/                    # Next.js 14 dashboard
│   └── src/
│       ├── app/                 # Pages: home, /run/[id], /runs, /league, /reports/[id]
│       ├── components/          # React components
│       │   ├── LeagueEvolution.tsx    # SVG lineage graph + champion timeline
│       │   ├── ChampionRobustness.tsx # Robustness sweep form
│       │   ├── ChampionBenchmark.tsx  # Baseline comparison bar chart
│       │   ├── MetricsChart.tsx       # Real-time step chart
│       │   ├── RobustHeatmap.tsx      # Policy × sweep heatmap
│       │   ├── RobustScatter.tsx      # Mean vs. worst-case scatter
│       │   └── RobustSummaryTable.tsx # Aggregated stats table
│       └── lib/api.ts           # Typed REST + WebSocket client
│
├── tests/
│   ├── unit/                    # 22 test modules (agents, env, league, training, …)
│   └── integration/             # 4 API contract test modules
│
├── storage/                     # Runtime artifacts (gitignored, .gitkeep present)
│   ├── configs/
│   ├── runs/
│   ├── agents/
│   └── reports/
│
├── design/                      # Architecture docs, roadmap, planning notes
│                                # (not implemented code — treat as design artifacts)
├── environment.yml
└── pyproject.toml
```

---

## Setup

### Option A — Conda (recommended)

```bash
git clone https://github.com/<your-username>/multi-agent-sim.git
cd multi-agent-sim

conda env create -f environment.yml
conda activate multi-agent-sim

cd frontend && npm install && cd ..
```

The Conda environment installs Python 3.11, Node 18, and the Python package in editable mode with both `[dev]` and `[training]` extras.

### Option B — pip + venv

```bash
git clone https://github.com/<your-username>/multi-agent-sim.git
cd multi-agent-sim

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -e ".[dev]"         # runtime + test deps
pip install -e ".[training]"    # adds PyTorch, tqdm, TensorBoard

cd frontend && npm install && cd ..
```

> PyTorch and TensorBoard are **not** installed by default. The backend server and all evaluation code run without them. Only training requires `[training]`.

---

## One-Command Demo

The following sequence runs the full pipeline — training, league creation, robustness evaluation, and report generation — then opens the dashboard to explore the results.

**Step 1 — Run the end-to-end pipeline**

```bash
python -m simulation.pipeline.pipeline_run
```

This executes in order:
1. PPO training on `MixedEnvironment` with league self-play
2. Periodic policy snapshots saved to `storage/agents/league/`
3. Elo ratings computed across all league members
4. Robustness sweep: champion policy evaluated across 10 environment variants
5. Report written to `storage/reports/`

**Step 2 — Start the backend**

```bash
uvicorn backend.main:app --port 8000
```

API docs available at `http://localhost:8000/docs`.

**Step 3 — Start the frontend**

```bash
cd frontend && npm run dev
```

Dashboard available at `http://localhost:3000`.

**Step 4 — Explore the results**

| UI Page | URL | What to look for |
|---|---|---|
| League Evolution | `localhost:3000/league` | SVG lineage graph, Elo ratings, champion timeline |
| Robustness Report | `localhost:3000/reports` | Heatmap of policy × sweep, scatter, worst-case analysis |
| Run History | `localhost:3000/runs` | Episode metrics from training runs |

**Where artifacts are written**

```
storage/
├── agents/
│   ├── ppo_shared/
│   │   ├── policy.pt            # final trained weights
│   │   └── metadata.json        # hyperparams, config_hash, obs_dim
│   └── league/
│       ├── league_000001/       # earliest snapshot
│       │   ├── policy.pt
│       │   └── metadata.json    # parent_id, created_at, notes, elo history
│       └── league_000NNN/       # most recent snapshot
├── reports/
│   └── robust_{config_hash}_{timestamp}/
│       ├── report.json          # full structured results
│       └── report.md            # human-readable summary
└── runs/
    └── {run_id}/
        ├── config.json
        ├── metrics.jsonl        # per-step metrics stream
        ├── events.jsonl         # semantic events (collapse, elimination)
        └── episode_summary.json
```

---

## Configuration

All simulation behavior is controlled by a single `MixedEnvironmentConfig` (Pydantic V2). The config is the canonical unit of reproducibility — its SHA-256 hash is embedded in every trained artifact.

### Behavioral Layers

| Layer | Parameter | Range | Effect |
|---|---|---|---|
| Information asymmetry | `information_asymmetry` | 0.0 – 1.0 | Fraction of observation masked per agent |
| Temporal memory | `temporal_memory_depth` | 1 – 50 | Observation history window depth (steps) |
| Reputation | `reputation_sensitivity` | 0.0 – 1.0 | EMA weight on cooperation history |
| Incentive regime | `incentive_softness` | 0.0 – 1.0 | 0 = hard bans, 1 = soft penalties only |
| Uncertainty | `uncertainty_intensity` | 0.0 – 0.5 | Gaussian noise magnitude on observations |

### Reward Weights

```
reward = individual_weight × individual_component
       + group_weight     × group_component
       + relational_weight × relational_component
       × (1 + penalty_scaling × active_penalties)
```

### Population Defaults

| Parameter | Default |
|---|---|
| `num_agents` | 5 |
| `max_steps` | 200 |
| `initial_shared_pool` | 100.0 |
| `initial_resources` | 20.0 (per agent) |
| `collapse_threshold` | ≤ initial_shared_pool (validated) |

Configs are saved as JSON to `storage/configs/` and referenced by ID in API calls and run logs.

---

## Training

```bash
python -m simulation.training.ppo_shared
```

### Network Architecture

```
Observation (flattened dict)
        │
        ▼
Linear(obs_dim → 64) → Tanh → Linear(64 → 64) → Tanh   (shared trunk)
        │                              │                         │
        ▼                              ▼                         ▼
Linear(64 → 4)              Linear(64 → 1) × 2         Linear(64 → 1)
Categorical dist            softplus + 1 → α, β          state value V(s)
(action type)               Beta dist (amount ∈ [0,1])
```

### PPO Hyperparameters (defaults)

| Parameter | Value |
|---|---|
| `total_timesteps` | 50,000 |
| `rollout_steps` | 256 |
| `ppo_epochs` | 4 |
| `num_minibatches` | 4 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_eps` | 0.2 |
| `vf_coef` | 0.5 |
| `ent_coef` | 0.01 |

### League Self-Play

Training can be run against a dynamically sampled opponent pool:

- **Baselines** — random, always-cooperate, always-extract, tit-for-tat
- **League members** — past policy snapshots (weighted toward recent)
- **Fixed policies** — optional pretrained policy (e.g., `ppo_shared`)

Sampling weights and the recent-vs-old bias are configurable. Every N timesteps, the current policy is snapshotted and added to the league registry. The registry auto-trims at a configured `max_members` limit, preserving the lineage record in metadata.

---

## Evaluation & Robustness

### Cross-Seed Evaluation

```python
evaluate_policies(
    config,
    policy_specs,        # list of {name, source, league_member_id}
    seeds=[42, 43, 44],
    episodes_per_seed=2
) -> list[PolicyResult]
```

Each `PolicyResult` aggregates: `mean_total_reward ± std`, `mean_final_shared_pool ± std`, `collapse_rate`, `mean_episode_length`, and a per-seed breakdown for reproducibility verification.

### Robustness Sweeps

Ten environment variants test generalization beyond the training distribution:

| Sweep | What changes |
|---|---|
| `obs_ia_0.0` | No information asymmetry |
| `obs_ia_0.6` | High information asymmetry |
| `uncertainty_0.0` | No observation noise |
| `uncertainty_0.3` | High observation noise |
| `incentive_soft_0.2` | Hard penalty regime |
| `incentive_soft_0.8` | Soft penalty regime |
| `pool_scarce` | Shared pool × 0.5 |
| `pool_abundant` | Shared pool × 1.5 |
| `pop_10` | 10 agents instead of 5 |
| `combined_hard` | High asymmetry + scarcity + noise |

**Robustness score:**

```
robustness_score = 0.7 × overall_mean_reward + 0.3 × worst_case_reward
```

This explicitly penalizes brittle policies that perform well on average but collapse under specific conditions.

---

## Backend API

| Router | Base Path | Key Endpoints |
|---|---|---|
| Config | `/api/configs` | `GET /` list, `GET /{id}` retrieve |
| Experiment | `/api/runs` | `POST /start`, `POST /stop`, `GET /status` |
| History | `/api/history` | `GET /` past runs list |
| League | `/api/league` | `GET /members`, `GET /champion`, `GET /lineage`, `GET /evolution` |
| League (actions) | `/api/league` | `POST /ratings/recompute`, `POST /champion/benchmark`, `POST /champion/robustness` |
| Reports | `/api/reports` | `GET /` list, `GET /{id}` full report, `GET /{id}/strategies` |
| WebSocket | `ws://…/ws/metrics` | Per-step metrics stream for live runs |

Interactive API documentation: `http://localhost:8000/docs`

---

## Frontend

| Page | Route | Description |
|---|---|---|
| Home | `/` | Config list, navigation |
| Live Run | `/run/[run_id]` | Real-time metrics chart via WebSocket, stop control |
| Run History | `/runs` | Completed run index |
| League | `/league` | Evolution graph + champion benchmark |
| Reports | `/reports` | Report index |
| Report Detail | `/reports/[report_id]` | Robustness heatmap, scatter plot, summary table |

### Key Components

**`LeagueEvolution.tsx`** — Renders an SVG forest layout of the policy lineage tree. Nodes are sized by Elo rating and color-coded by strategy label (Champion = gold, Competitive = blue, Developing = gray). Clicking a node opens a metadata detail panel. A secondary panel renders a chronological champion timeline.

**`ChampionRobustness.tsx`** — Form for triggering a new robustness sweep against the current champion. Accepts config selection, seeds, episodes per seed, and optional sweep limits. On completion, redirects to the generated report.

**`RobustHeatmap.tsx`** — Heatmap with policies on rows and environment sweep variants on columns. Cell color encodes mean reward. Provides immediate visual identification of policy weaknesses.

---

## Artifact Layout

```
storage/
├── configs/
│   └── {config_id}.json              # Full MixedEnvironmentConfig snapshot
│
├── runs/
│   └── {run_id}/
│       ├── config.json               # Config used for this run
│       ├── metrics.jsonl             # Per-step: step, agent_id, reward, action, pool, resources
│       ├── events.jsonl              # Sparse semantic events: COLLAPSE_DETECTED, AGENT_DEACTIVATED
│       └── episode_summary.json     # Aggregated: length, termination_reason, total rewards
│
├── agents/
│   ├── ppo_shared/
│   │   ├── policy.pt                 # Trained weights (state_dict)
│   │   └── metadata.json            # hyperparams, config_hash, obs_dim, timestamp
│   └── league/
│       └── league_{N:06d}/          # Monotonically numbered
│           ├── policy.pt
│           └── metadata.json        # + member_id, parent_id, created_at, notes
│
└── reports/
    └── robust_{config_hash}_{timestamp}/
        ├── report.json              # Structured: metadata, per-sweep results, summary
        └── report.md                # Human-readable markdown
```

---

## Reproducibility

All stochastic components are seeded from a single integer root seed:

- **Environment resets** — `env.reset(seed=seed)` accepts an explicit seed.
- **Agent initialization** — `agent.reset(agent_id, seed=seed)` called at episode start.
- **Seed propagation** — `derive_seed(base_seed, offset)` generates deterministic per-agent seeds from the root, eliminating seed-collision artifacts in multi-agent settings.
- **NumPy RNG isolation** — `make_rng(seed)` creates an independent `numpy.random.Generator` per component.
- **Config hash** — SHA-256 of the serialized `MixedEnvironmentConfig` is written into every `metadata.json`. Given the same config hash and seed, a training run or evaluation is exactly reproducible.
- **Strategy clustering** — k-means runs with a fixed seed, producing deterministic cluster assignments.

To reproduce any result: locate the `config_hash` in the artifact's `metadata.json`, retrieve the matching config from `storage/configs/`, and re-run with the same seed.

---

## Testing

```bash
# Full suite
pytest tests/

# Unit tests only (no running server required)
pytest tests/unit/ -v

# Integration tests (requires backend running on port 8000)
pytest tests/integration/ -v

# Coverage report
pytest tests/ --cov=simulation --cov=backend --cov-report=term-missing
```

**Unit coverage** (22 modules): `mixed_env`, `config_validation`, `agents`, `ppo_training`, `league_registry`, `league_ratings`, `league_sampling`, `league_selfplay`, `evaluation`, `robustness`, `reporting`, `metrics_collector`, `run_logger`, `pipeline_run`, `strategy_analysis`, PettingZoo adapter, seeding utilities, and more.

**Integration coverage** (4 modules): experiment lifecycle, run history, league endpoints, report endpoints — all tested via `httpx` against the live FastAPI application.

---

## Screenshots *(Coming Soon)*

- **League evolution graph** — SVG lineage tree with Elo-sized nodes and strategy color coding
- **Robustness heatmap** — policy × environment variant grid showing performance profile
- **Strategy evolution timeline** — champion lineage over training generations
- **Live metrics chart** — real-time reward, shared pool, and cooperation ratio during a run

---

## Project Status

This is a **solo research project** under active development. The simulation engine, training pipeline, league system, evaluation framework, backend API, and frontend dashboard are all implemented and tested. The codebase is stable and all tests pass.

All functionality described outside the [`design/`](design/) directory is fully implemented and runnable. Files under `design/` contain architectural reasoning, future plans, and exploratory ideas — they should not be interpreted as implemented features.

Planned extensions include additional environment archetypes, expanded strategy analysis, and user-defined configuration templates. These are tracked in the design documents and will be implemented incrementally.

**External contributions are not being accepted at this time.**

See [docs/bug-audit.md](docs/bug-audit.md) for deployment-readiness issues and their current status.

---

## License

MIT — see [LICENSE](LICENSE).
