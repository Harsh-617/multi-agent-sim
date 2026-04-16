# Architecture Decision Records

> This document captures engineering decisions made at the platform level and per-archetype level.
> Each decision records context, the choice made, reasoning, and consequences for future work.
> These are authoritative — when code conflicts with this document, this document wins.

---

## Section 1: Platform-Level Decisions

These apply to all archetypes and were established when building Mixed.

---

### ADR-001: Filesystem storage over a database

**Context:** The platform needs to store configs, run artifacts, agent snapshots, league data, and reports. A database was considered.

**Decision:** All storage uses the filesystem (JSON / JSONL files). No database.

**Reasoning:**
- Zero infrastructure dependency — runs locally without setup
- Run artifacts are naturally file-shaped (one folder per run)
- JSON is human-readable and git-inspectable
- A database adds operational complexity for no query benefit at this scale

**Consequences:**
- All new archetypes must store artifacts under `storage/` following existing layout conventions
- No cross-run querying — aggregation happens in Python at read time
- If the platform ever needs multi-user or cloud deployment, this decision gets revisited

---

### ADR-002: PettingZoo as the MARL interface

**Context:** The training loop needs a standard interface to interact with multi-agent environments. Several MARL libraries were considered (RLlib, PettingZoo, custom).

**Decision:** PettingZoo `ParallelEnv` is the standard adapter interface for all archetypes.

**Reasoning:**
- Widely adopted standard — compatible with most MARL training libraries
- Parallel env (all agents act simultaneously) matches the simultaneous action model used by all archetypes
- Clean separation: internal environment logic is framework-owned, PettingZoo is just a translation layer

**Consequences:**
- Every archetype must implement a PettingZoo `ParallelEnv` adapter in `simulation/adapters/`
- The adapter translates — it never modifies environment logic or reward semantics
- If a future archetype uses turn-based actions, a separate `AECEnv` adapter would be needed

---

### ADR-003: Shared-policy PPO as the first trained agent

**Context:** Multiple RL algorithms could be used. The platform needed a first trained agent type.

**Decision:** Shared-policy PPO (CleanRL-style, custom PyTorch loop) is the standard trained agent for all archetypes V1.

**Reasoning:**
- Shared policy keeps training stable in homogeneous populations
- CleanRL-style means no external RL library dependency — full control over the training loop
- PPO is robust and well-understood — good baseline before exploring alternatives
- Custom loop means training behavior is transparent and debuggable

**Consequences:**
- The training loop structure is fixed — new archetypes adapt config and adapter only, never the loop itself
- The PPO network architecture (shared trunk → action head + value head) is the template for all archetypes
- Recurrent policies (LSTM), population-based training, and other algorithms are explicitly deferred

---

### ADR-004: Per-archetype metrics collectors, not a generic one

**Context:** Metrics collection could be generic (one collector for all archetypes) or archetype-specific.

**Decision:** Each archetype has its own metrics collector and event definitions.

**Reasoning:**
- Mixed tracks cooperation rate, extraction ratio, pool health — meaningless for Competitive
- Competitive tracks attack ratio, rank volatility, score gap — meaningless for Mixed
- A generic collector would either be too sparse (missing archetype-specific signals) or too bloated (tracking everything for everyone)
- Keeping them separate means each archetype's dashboard shows exactly the right metrics

**Consequences:**
- Every new archetype creates its own `simulation/metrics/{archetype}_definitions.py`
- The `MetricsCollector` class can be subclassed or duplicated per archetype — do not modify the Mixed one
- Episode summary schema differs per archetype — the backend must handle this cleanly

---

### ADR-005: BaseEnvironment interface is frozen

**Context:** `BaseEnvironment` defines the contract all environments must follow. It could be extended per-archetype.

**Decision:** `BaseEnvironment` is never modified once established. All archetypes implement it as-is.

**Reasoning:**
- Modifying the base class breaks all existing environments simultaneously
- The interface is deliberately minimal — reset, step, observation_spec, action_spec, is_done
- Archetype-specific behavior lives inside the implementation, not the interface

**Consequences:**
- If a new archetype needs something the interface doesn't support, add it to the archetype's own class — never to BaseEnvironment
- This constraint is enforced by convention, not by the Python type system

---

### ADR-006: Config schema is per-archetype, not shared

**Context:** A single unified config schema was considered to simplify the backend.

**Decision:** Each archetype has its own Pydantic config schema. No shared base config class.

**Reasoning:**
- Mixed and Competitive have fundamentally different parameters — a shared schema would have too many optional fields
- Pydantic validation is most useful when the schema is tight and specific
- `environment_type` field on the identity section handles routing without a shared class

**Consequences:**
- The backend must route config operations based on `environment_type`
- Cross-archetype config comparison is done by humans, not by the system
- Every new archetype adds a new schema file in `simulation/config/`

---

## Section 2: Mixed Archetype Decisions

Decisions made implicitly during Mixed implementation. Documented here retroactively.

---

### ADR-007: routes_config.py hardcodes MixedEnvironmentConfig

**Context:** When Mixed was the only archetype, the config route imported `MixedEnvironmentConfig` directly.

**Decision:** Accepted as technical debt. Will be replaced with archetype-routing when Competitive is added.

**Reasoning:** At the time, there was only one archetype. Over-engineering the routing before a second archetype existed would have been premature.

**Consequences:**
- When Competitive is added, `routes_config.py` must be updated to route by `environment_type`
- See ADR-010 for how this is resolved in Competitive

---

### ADR-008: Conditional action resolves to aggregate signal, not targeted

**Context:** The CONDITIONAL action type in Mixed implies reacting to another agent's behavior. Agents only see aggregate stats, not per-agent identities.

**Decision:** CONDITIONAL reacts to the mean cooperation ratio across all agents — not to any specific agent.

**Reasoning:** Targeted conditional actions would require agents to identify specific opponents, which contradicts the partial observability design (agents see aggregates, not identities).

**Consequences:** Tit-for-tat style behavior in Mixed is population-level, not pairwise. Pairwise conditional behavior requires the Competitive archetype's opponent history mechanism.

---

### ADR-009: Terminated agent reputation persists in Mixed

**Context:** When an agent is eliminated in Mixed, its cooperation scores remain in relational state.

**Decision:** Frozen reputation persists. Remaining agents carry the weight of past interactions with eliminated agents.

**Reasoning:** Removing the reputation on elimination would retroactively change the social history of the episode. Keeping it frozen is simpler and more honest about what happened.

**Consequences:** This is intentional, not a bug. Relational state is never cleaned up mid-episode.

---

## Section 3: Competitive Archetype Decisions

Decisions made explicitly during Step B architecture design before implementation began.

---

### ADR-010: Config routing via environment_type union, not separate routes

**Context:** The backend needs to handle both `MixedEnvironmentConfig` and `CompetitiveEnvironmentConfig`. Options were: (a) separate API routes per archetype, (b) a union type on existing routes, (c) a generic dict-based approach.

**Decision:** Update existing routes to accept a union type `MixedEnvironmentConfig | CompetitiveEnvironmentConfig`, discriminated by the `environment_type` field.

**Reasoning:**
- Separate routes would duplicate all CRUD logic (create, list, get, delete) for each archetype
- A generic dict approach loses Pydantic validation entirely
- A discriminated union keeps one set of routes, full validation, and clean routing logic

**Consequences:**
- `routes_config.py` and `routes_experiment.py` must be updated when Competitive is added
- Every future archetype adds its config type to the union — one line change per archetype
- The `environment_type` field on `EnvironmentIdentity` is the discriminator and must always be set correctly

---

### ADR-011: Competitive gets a separate PettingZoo adapter

**Context:** The existing `MixedParallelEnv` adapter could be modified or subclassed to support Competitive.

**Decision:** Create a separate `CompetitiveParallelEnv` adapter. Do not modify or subclass the Mixed adapter.

**Reasoning:**
- Competitive has a terminal bonus timing requirement (held until global done, not issued at individual elimination) that Mixed does not have
- Modifying the Mixed adapter risks breaking it
- Subclassing would create tight coupling between two unrelated environments
- Separate adapters are independent, testable, and clearly owned by their archetype

**Consequences:**
- `simulation/adapters/competitive_pettingzoo.py` is a new file, not a modification of existing code
- The terminal bonus logic lives entirely inside this adapter
- Mixed adapter (`simulation/adapters/pettingzoo.py`) is never touched

---

### ADR-012: Competitive gets a separate metrics collector

**Context:** The existing `MetricsCollector` is Mixed-specific. Options were to generalize it, subclass it, or create a new one.

**Decision:** Create a separate `CompetitiveMetricsCollector`. Do not modify or subclass the Mixed one.

**Reasoning:**
- Competitive metrics (attack ratio, rank volatility, score gap, elimination events) have no overlap with Mixed metrics (cooperation rate, pool health, betrayal frequency)
- Generalizing the collector would produce a bloated class serving two very different purposes
- A separate collector keeps each archetype's instrumentation clean, independent, and testable

**Consequences:**
- `simulation/metrics/competitive_definitions.py` and a `CompetitiveMetricsCollector` class are new files
- Episode summary schema for Competitive differs from Mixed — the backend pipeline runner must handle both
- Mixed `MetricsCollector` is never touched

---

### ADR-013: Nothing in the existing codebase is modified for Competitive except pipeline routing

**Context:** Adding a second archetype could require modifying many shared files.

**Decision:** The only existing files that change when adding Competitive are `routes_config.py`, `routes_experiment.py`, and `pipeline_run.py`. Everything else is new files only.

**Reasoning:**
- Modifying shared infrastructure risks breaking Mixed, which is already working and tested
- New files are safe — they can't break what already works
- Pipeline routing must change because the pipeline currently hardcodes Mixed

**Consequences:**
- All Competitive-specific code lives under `simulation/envs/competitive/`, `simulation/config/competitive_*`, `simulation/agents/competitive_baselines.py`, `simulation/metrics/competitive_*`, `simulation/adapters/competitive_pettingzoo.py`
- The three modified files get targeted, minimal changes only — no refactoring

---

## Section 4: Cross-Archetype Rules

Rules that govern all current and future archetypes. Established from Mixed patterns and Competitive decisions.

---

### New archetype checklist

Every new archetype must create these files:
```
simulation/envs/{name}/
  actions.py
  state.py
  transition.py
  rewards.py
  termination.py
  env.py
simulation/adapters/{name}_pettingzoo.py
simulation/config/{name}_schema.py
simulation/config/{name}_defaults.py
simulation/agents/{name}_baselines.py
simulation/metrics/{name}_definitions.py
```

And make minimal targeted changes to:
```
routes_config.py          — add to union type
routes_experiment.py      — add env instantiation routing
pipeline_run.py           — add archetype routing
```

### Rules that must never be broken

- Never modify `BaseEnvironment`
- Never modify an existing archetype's environment files
- Never modify an existing archetype's adapter
- Never modify an existing archetype's metrics collector
- Never change the storage layout conventions
- Never change the PPO training loop structure — adapt config and adapter only
- Never let an adapter modify reward semantics or inject hidden state

---

## Section 5: Cooperative Archetype Decisions

Decisions made explicitly during Step B architecture design before implementation began.

---

### ADR-014: Cooperative archetype uses task queue model not shared resource pool

**Context:** The cooperative environment needed a mechanic clearly distinct from Mixed's shared pool.

**Decision:** Task queue with configurable arrival rate, completion via pooled effort, and specialization EMA.

**Reasoning:**
- Clearly distinct from Mixed, maps to real distributed systems problems, enables specialization emergence without hardcoding it.

**Consequences:**
- Cooperative metrics (`completion_ratio`, `backlog_level`, `specialization_score`) are entirely distinct from Mixed and Competitive. A separate metrics collector is required.

---

### ADR-015: No individual agent termination in Cooperative V1

**Context:** Mixed and Competitive both support individual agent elimination mid-episode.

**Decision:** All agents remain active for the full episode in Cooperative V1.

**Reasoning:**
- Eliminating one agent mid-episode punishes the entire group for one agent's failure — a distinct mechanic that would obscure core coordination dynamics before they are validated. Individual penalties exist in the reward model instead.

**Consequences:**
- Episode termination is group-level only: `max_steps`, `system_collapse`, `perfect_clearance`. Individual elimination deferred to V2.

---

### ADR-016: Cooperative Elo recompute runs as background task capped at 10 members

**Context:** `compute_cooperative_ratings` with 30 members blocked the entire FastAPI event loop for minutes because each rating match runs a full cooperative episode.

**Decision:** Recompute runs via FastAPI `BackgroundTasks`, returning immediately. Capped at 10 most recent members and 3 matches per pair.

**Reasoning:**
- Cooperative rating requires running full episodes (not just score comparisons), making it 10-100x slower than Competitive rating. Blocking the event loop breaks all other archetypes while running.

**Consequences:**
- Recompute returns `{"status": "started"}` immediately. Frontend waits 5 seconds then re-fetches. Only the 10 most recent members are rated — older members retain their last known rating.

---

### ADR-017: Cooperative report detail lives at /research/cooperative/[report_id]

**Context:** Mixed and Competitive reports use the shared `/research/[report_id]` detail page.

**Decision:** Cooperative reports get a dedicated detail page at `/research/cooperative/[report_id]`.

**Reasoning:**
- The shared detail page renders Mixed/Competitive-specific metrics (score spread, cooperation rate, elimination events). Adding cooperative support would require significant branching that makes the page unmaintainable.

**Consequences:**
- The Research page Open button branches on `report.archetype === "Cooperative"` to route to the correct path. The shared detail page is unchanged.

---

### ADR-018: Cooperative reports saved as summary.json not report.json

**Context:** Mixed and Competitive evaluation runners save their primary report file as `report.json`.

**Decision:** Cooperative evaluation and robustness runners save as `summary.json`.

**Consequences:**
- `backend/api/routes_cooperative_reports.py` reads `summary.json`. This inconsistency should be normalized in a future cleanup — either rename cooperative to `report.json` or migrate Mixed/Competitive to `summary.json`.

---

### ADR-019: Transfer experiment uses truncate/pad for obs dimension mismatch
Context: Each archetype's trained policy expects a fixed obs_dim stored in metadata.json. Target environments produce different obs shapes.
Decision: If target obs > source expected: truncate. If target obs < source expected: zero-pad. Document clearly in UI.
Reasoning: Learned projection matrices are more principled but require additional training. Truncate/pad is honest about what's happening and produces interpretable results — a policy operating with partial or zero-padded information is a valid research finding, not an engineering failure.
Consequences: Results reflect raw generalization, not engineered compatibility. The UI prominently displays the obs mismatch strategy. Learned projection deferred to V2.

### ADR-020: Transfer experiment is champion-only in V1
Context: Any league member could theoretically be transferred, not just the champion.
Decision: V1 only transfers the current Elo champion from each archetype.
Reasoning: The champion is the most meaningful agent to transfer — it represents the best learned policy. Member-level transfer selection adds UI complexity before the core feature is validated.
Consequences: sourceElo is derived from champion rating. Member-level selection deferred to V2.

### ADR-021: Transfer reports stored under storage/reports/transfer_* 
Context: Transfer reports needed a storage location consistent with existing robustness and eval reports.
Decision: storage/reports/transfer_{src}_{tgt}_{hash}_{timestamp}/summary.json
Reasoning: Follows the same pattern as cooperative reports (summary.json) and sits alongside other report types in the unified storage/reports/ directory.
Consequences: Research page infers report type from directory name prefix. Transfer reports route to /research/transfer/[report_id] detail page.