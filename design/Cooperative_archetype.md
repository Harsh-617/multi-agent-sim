# Cooperative Archetype

# Part 1: Core Tension & Purpose

## 1️⃣ Archetype Name
**Cooperative Archetype**
(Shared Goal, Collective Outcome)

---

## 2️⃣ Core Question
> **How do agents learn to coordinate effort toward a shared goal when free-riding is always individually rational?**

---

## 3️⃣ Fundamental Tension

Three simultaneous pressures — no single strategy dominates:

1. **Contribute vs Free-ride** — working hard costs the individual agent, but the group reward is shared regardless of who contributed. Slacking is individually rational in the short term but collectively disastrous if everyone does it.
2. **Specialize vs Generalize** — focusing effort on one task type produces better group outcomes, but creates dependency on other agents doing their part. Generalists are safer but less efficient.
3. **Trust vs Verification** — agents cannot directly observe others' effort, only the collective outcome. Do you keep contributing when you can't tell if others are pulling their weight, or do you reduce effort in response to poor group results?

---

## 4️⃣ What Makes This Archetype "Cooperative"

**Is:**
- Purely collective outcome — all agents share the same group reward signal
- Interdependent — no agent can succeed alone; group performance depends on the sum of contributions
- Effort-based — the core variable is how much each agent contributes, not who they target
- Coordination problem — the challenge is alignment of effort without communication

**Is NOT:**
- Competitive in any dimension — no agent gains at another's expense
- Mixed incentives — there is no individual reward for defecting or free-riding
- Communication-based — agents cannot send signals or negotiate (that is the Negotiation archetype)
- Resource-sharing in the Mixed sense — no shared pool to extract from; instead a shared task to complete

---

## 5️⃣ Desired Emergent Behaviors

Not hardcoded — must emerge naturally:

- **Specialization** — agents converge on distinct task roles without being assigned them
- **Division of labor** — different agents handling different parts of the workload efficiently
- **Effort matching** — agents calibrating contribution levels to what the group needs
- **Free-rider detection** — agents implicitly reducing effort when group outcomes suggest others are slacking
- **Role stability** — once roles emerge, agents maintain them across timesteps
- **Collective recovery** — group adapting effort upward when the system is failing

If none of these appear, the environment has failed.

---

## 6️⃣ What This Archetype Is NOT Trying to Model

- Competition, ranking, or zero-sum dynamics
- Betrayal, extraction, or individual self-interest mechanics
- Direct communication or negotiation between agents
- Trust and reputation in the Mixed sense (agents don't punish each other — the environment punishes the group)
- Any scenario where one agent can win while others lose

This is a **pure coordination problem**, not a social dilemma.

---

## 7️⃣ Success Criteria

The archetype is working if:

- Agents learn contribution strategies meaningfully better than random
- Specialization emerges without being hardcoded — agents develop distinct roles
- Free-rider dynamics appear — reducing one agent's reward share causes measurable effort drop
- Group performance degrades gracefully under partial failure (one bad agent hurts but doesn't collapse everything)
- Robustness sweeps show clear sensitivity to task difficulty and group size
- League Elo ratings diverge — coordination-aware policies outperform naive ones


# Part 2: Layer Emphasis (Knob Settings)

## Why we do this first
These choices decide what the environment *feels* like, what behaviors can emerge, and how hard training will be.

---

## Primary Layers — HIGH

### ✅ Layer 2: Temporal Structure — HIGH
The defining layer for this archetype. Specialization can only emerge if agents remember their role across timesteps. Effort matching requires tracking group performance history. Role stability is impossible without meaningful memory depth. This is the highest temporal emphasis of all three archetypes — the Cooperative archetype lives in time.

### ✅ Layer 4: Interaction Structure — HIGH
Every agent's effort directly affects the group outcome. Contributions pool additively — one agent slacking hurts everyone, one agent working hard helps everyone. Unlike Competitive (adversarial coupling), this is cooperative coupling: no action is resolved in isolation.

---

## Secondary Layers — MEDIUM

### ⚠️ Layer 1: Information Structure — MEDIUM
Agents observe group-level task progress but NOT individual contributions. This is what makes the trust/verification tension real — free-riding is not instantly detectable. If agents could see each other's effort directly, coordination would be trivial. Partial observability is what makes this a hard problem.

### ⚠️ Layer 3: State Hierarchy — MEDIUM
Global task state (shared progress, task queue, system health) plus agent-local state (effort capacity, contribution history, specialization signal). Minimal relational state — agents track group-level signals, not individual agent behavior. Less relational complexity than Mixed.

### ⚠️ Layer 6: Soft Constraints & Incentives — MEDIUM
Effort costs something — agents have limited effort capacity per step and cannot contribute maximally to everything simultaneously. The group is penalized for poor collective performance, not individual agents directly. Zero effort is allowed but costly to the group outcome. No hard bans — bad behavior is discouraged via group cost, not prohibition.

### ⚠️ Layer 7: Uncertainty & Meta Dynamics — MEDIUM
Higher than both Mixed and Competitive (both LOW). Task difficulty and task composition vary across episodes, forcing agents to adapt rather than memorize a fixed strategy. Enough variance that role rigidity is punished, not so much that effort becomes irrelevant to outcomes. Seeded and reproducible.

---

## Minimal Layers — LOW

### 🟢 Layer 5: Power & Role Asymmetry — LOW
All agents start structurally symmetric. Roles emerge from learning, not from design. Explicit role assignment (e.g. leader/follower, specialist/generalist by design) is deferred to V2 — it would add institutional complexity before the core coordination dynamics are validated.

---

## Summary

| Layer | Emphasis |
|---|---|
| 2. Temporal Structure | HIGH |
| 4. Interaction Structure | HIGH |
| 1. Information Structure | MEDIUM |
| 3. State Hierarchy | MEDIUM |
| 6. Soft Constraints & Incentives | MEDIUM |
| 7. Uncertainty & Meta Dynamics | MEDIUM |
| 5. Power & Role Asymmetry | LOW |

**Key differences from Mixed and Competitive:** Temporal Structure jumps to HIGH — the highest of all three archetypes, and the defining feature of Cooperative. Uncertainty steps up to MEDIUM because task variability is a primary mechanic, not just noise. Information Structure drops to MEDIUM because the observability gap is specifically about effort (not scores or resources), which requires a narrower but more targeted design.


# Part 3: Agents, Roles, and Episode Structure

> This part answers **one narrow question**:
> **Who exists in this world, and what does one episode look like?**
> Everything else (state, actions, rewards) depends on this.

---

## 1️⃣ Agents (Who Exists)

### Number of Agents
- The environment supports **N agents**, where:
  - N ≥ 2
  - N is configurable per experiment

This allows:
- minimal coordination (2–3 agents)
- small team dynamics (4–8 agents)
- larger group coordination challenges later

---

### Agent Symmetry (Important Design Choice)
For the **first Cooperative archetype**:
- All agents are **structurally symmetric**
- Same:
  - action space
  - observation schema
  - reward structure (form, not value)
- Differences in behavior must **emerge from learning**, not roles

This keeps:
- coordination dynamics clean and analyzable
- learning stable
- emergent specialization meaningful — if roles appear, they weren't designed in

(Explicit role asymmetry — e.g. designated leader, specialist by design — is deferred to V2.)

---

### Agent Identity
Each agent has:
- a unique ID
- persistent identity across the episode

Identity matters for:
- tracking contribution history
- measuring specialization over time
- logging which agent handled which task types

Unlike Competitive, identity here is not about tracking who is winning — it is about tracking **who is doing what**.

---

## 2️⃣ Agent Capabilities (High-Level, Not Actions Yet)

At this stage we only define **capability categories**, not concrete actions.

Each agent can:
- **Contribute effort** — apply effort toward one or more task types in the shared task queue
- **Specialize** — repeatedly direct effort toward the same task type, building implicit efficiency
- **Idle** — contribute zero effort (allowed, not prohibited, but costly to the group)

Agents cannot:
- communicate directly with other agents
- observe other agents' individual effort levels
- target or affect other agents directly

---

## 3️⃣ The World (What Agents Operate In)

The environment maintains a **shared task queue** — a pool of tasks that need to be completed collectively.

Key properties:
- Tasks have **types** (e.g. Type A, Type B, Type C — configurable number of types)
- New tasks arrive each timestep at a configurable rate
- Tasks have a **backlog limit** — if uncompleted tasks exceed the limit, the system is under stress
- The group succeeds by keeping the backlog manageable and completing tasks before the episode ends

This maps to real coordination problems: distributed systems, load balancing, team task allocation. The agents are workers. The task queue is the shared environment they operate in.

---

## 4️⃣ Episode Timeline

- **Discrete timesteps** — the episode advances in fixed steps
- **Simultaneous actions** — all agents submit their action at the same time each step
- **No turn-taking** — agents do not observe each other's action before choosing their own
- **Fixed horizon** — episodes run for `max_steps` timesteps (configurable)

One timestep cycle:
1. New tasks arrive in the queue (rate configurable)
2. All agents simultaneously submit their effort allocation
3. Environment resolves contributions — tasks are completed based on collective effort
4. Task queue updates — completed tasks removed, backlog recalculated
5. Group reward computed
6. Observations updated
7. Timestep advances

---

## 5️⃣ Persistence Rules

**Within episode:**
- Task queue state persists and evolves each step
- Agent contribution history accumulates (used for specialization signals)
- Backlog pressure builds if the group underperforms

**Across episodes:**
- No state carries over between episodes
- Each episode starts with a fresh task queue
- Agent policies persist (learning carries over via trained weights, not environment state)

---

## 6️⃣ Agent Termination

**Individual termination: NOT in V1.**

This is a deliberate departure from Mixed and Competitive. Eliminating one agent mid-episode punishes the entire group for one agent's failure — that is a different and more complex mechanic. In V1, all agents remain active for the full episode duration.

Episode ends when:
- `max_steps` is reached (primary termination)
- System collapse: backlog exceeds the critical threshold for a sustained period (configurable)

Individual agent penalties (reduced reward share for low contributors) exist in the reward model — but no agent is removed from the episode.

Agent-level elimination under sustained failure is explicitly deferred to V2.


# Part 4: State & Observation Definition

> This part answers:
> **What exists in the world, and what does each agent get to see?**
> State is the truth. Observations are the projection agents receive.

---

## 1️⃣ Core Design Principle

> **Agents see collective outcomes, never individual effort.**

This single constraint is what makes coordination hard. If agents could see exactly how much each other agent contributed, free-rider detection would be trivial and the core tension would collapse. The observability gap between individual effort (hidden) and group outcome (visible) is the mechanic that drives the archetype.

---

## 2️⃣ State Definition

State is divided into three tiers.

---

### Global Shared State
The true state of the world — visible to the environment, not directly to agents.

| Variable | Description |
|---|---|
| `task_queue` | Current count of pending tasks per type (array of length `num_task_types`) |
| `backlog_level` | Total pending tasks across all types (scalar, derived) |
| `tasks_completed_this_step` | Tasks completed in the current timestep per type |
| `tasks_completed_total` | Cumulative tasks completed across the episode per type |
| `task_arrival_rate` | Current task arrival rate per type (can vary by config) |
| `system_stress` | Normalized backlog pressure (0.0 = no stress, 1.0 = at collapse threshold) |
| `timestep` | Current step index |
| `episode_progress` | Fraction of max_steps elapsed (scalar) |

---

### Agent-Local State
Private to each agent — not visible to other agents or directly observable.

| Variable | Description |
|---|---|
| `effort_spent` | Effort allocated this step per task type (array) |
| `contribution_history` | Rolling window of own effort per task type over last K steps |
| `specialization_score` | Per-task-type specialization index (computed from contribution history) |
| `cumulative_reward` | Own accumulated reward so far this episode |
| `steps_active` | Number of steps this agent has been active |

---

### Relational / Interaction State
Group-level signals only — no per-agent breakdown.

| Variable | Description |
|---|---|
| `group_contribution_last_step` | Total effort contributed by all agents last step (scalar) |
| `group_contribution_by_type` | Aggregate effort per task type last step (array) |
| `group_completion_rate` | Rolling average task completion rate over last K steps |
| `free_rider_pressure` | Normalized signal: how far group contribution is below expected (scalar) |

**Critical:** There is no per-agent contribution tracking in relational state. Agents cannot infer individual effort from these signals. The environment knows, but does not expose it.

---

## 3️⃣ Observation Definition

Observations are projections of state — what each agent actually receives each step.

---

### Public Observation (same for all agents)

| Field | Source |
|---|---|
| `backlog_level` (normalized) | Global state |
| `task_queue_composition` | Global state — counts per type, normalized |
| `system_stress` | Global state |
| `group_completion_rate` | Relational state — rolling average |
| `group_contribution_last_step` (normalized) | Relational state |
| `episode_progress` | Global state |

---

### Private Self Observation (unique per agent)

| Field | Source |
|---|---|
| `effort_capacity_remaining` (normalized) | Agent-local state |
| `own_contribution_last_step` | Agent-local state — own effort last step, per type |
| `own_specialization_score` | Agent-local state — per task type |
| `own_cumulative_reward` (normalized) | Agent-local state |
| `own_action_history` | Agent-local state — last K actions |

---

### Partial Social Observation (group-level only, never per-agent)

| Field | Source |
|---|---|
| `group_contribution_by_type` (normalized) | Relational state — aggregate per type |
| `free_rider_pressure` | Relational state |

---

## 4️⃣ Critical Constraints

- Agents **never** see other agents' individual effort levels
- Agents **never** see other agents' specialization scores
- Agents **never** see other agents' reward accumulation
- Agents **never** see the true task arrival rate (only the observable queue)
- All observations are **normalized** to [0, 1] for training stability
- Observation schema is **fixed and stable** — same shape every step regardless of episode config
- The environment has access to full state at all times for metrics and logging — agents do not

---

## 5️⃣ Observation Vector Structure

Each agent receives a flat, fixed-length observation vector each step, composed of:

1. Public observation fields (same for all agents)
2. Private self observation fields (unique per agent)
3. Partial social observation fields (aggregate group signals)

The exact vector length is determined by `num_task_types` and history window `K` — both configurable. The schema is fixed for a given config. No variable-length observations.


# Part 5: Action Space Definition

> This part answers:
> **What can agents do, and what do their actions mean?**
> Actions express intent — the environment decides what actually happens.

---

## 1️⃣ Core Action Philosophy

In the Cooperative archetype:

> **Actions represent effort allocation decisions, not attacks or negotiations.**

Agents choose:
- **Where** to direct effort (which task type)
- **How much** effort to apply (intensity)

The environment then resolves whether tasks are actually completed — effort is necessary but not always sufficient, depending on backlog pressure and task difficulty.

---

## 2️⃣ Action Timing

- All agents submit **one action per timestep**
- Actions are **simultaneous**
- The environment resolves them jointly

This preserves:
- strategic uncertainty (you don't know what others chose before committing)
- true interdependence (collective outcome depends on the combination of all choices)
- consistent structure with Mixed and Competitive archetypes

---

## 3️⃣ Action Structure

Each action has two components:

| Component | Type | Range | Description |
|---|---|---|---|
| `action_type` | Discrete enum | `WORK_A`, `WORK_B`, ..., `WORK_N`, `IDLE` | Which task type to work on this step |
| `effort_amount` | Continuous float | [0.0, 1.0] | Fraction of available effort capacity to apply |

`action_type` is a discrete choice from the configured task types plus IDLE.
`effort_amount` is continuous and scales against the agent's current effort capacity.

If `action_type` is IDLE, `effort_amount` is ignored and treated as 0.0.

---

## 4️⃣ Action Categories

---

### A) WORK Actions (WORK_A, WORK_B, ..., WORK_N)
Intent:
- Direct effort toward a specific task type in the shared queue

Properties:
- Contributes to completing tasks of the chosen type
- Effective contribution = `effort_amount × agent_effort_capacity`
- Repeated WORK on the same task type over time builds specialization (tracked in agent-local state)
- Does not directly affect other agents
- Cost: effort capacity consumed this step (capacity resets next step)

This enables:
- steady contribution strategies
- specialization development
- division of labor across the group

---

### B) IDLE Action
Intent:
- Contribute zero effort this step

Properties:
- No contribution to any task type
- Effort capacity is not consumed
- Does not directly penalize the agent individually — but reduces group output
- The environment records idle steps in contribution history
- Repeated idling degrades the agent's specialization score

IDLE is **allowed but not banned**. Its cost is collective: the group performs worse.
Repeated idling is implicitly discouraged through the reward model (lower group reward = lower shared signal), not through hard penalties on the individual.

This enables:
- free-rider behavior to emerge naturally
- the group to detect underperformance via outcome signals
- agents to adaptively reduce effort when group outcomes are poor (effort matching)

---

## 5️⃣ Action Constraints

- Each agent submits **exactly one action per step** — one task type only
- Splitting effort across multiple task types simultaneously is **not supported in V1**
  - This would require a vector action per agent, adding significant complexity before core coordination dynamics are validated
  - Deferred to V2
- `effort_amount` is clipped to [0.0, 1.0] — values outside this range are normalized, not rejected
- If a chosen task type has zero tasks in the queue, the effort is applied but yields no completions — agents must learn to read the queue
- IDLE is always valid regardless of queue state

---

## 6️⃣ What Actions Do NOT Do

Actions do **not**:
- directly affect other agents' state or capacity
- send signals or messages to other agents
- modify the task arrival rate
- guarantee task completion (effort is input; completion depends on collective effort and task difficulty)

They only express **effort allocation intent**. The environment resolves outcomes.

---

## 7️⃣ Extensibility

The action model is designed so that:
- new task types can be added via config without changing the action structure
- split-effort allocation can be added in V2 as a vector extension
- role-based action restrictions can be layered on in V2 without breaking existing agents


# Part 6: Transition Dynamics

> This part answers:
> **Given the current state and all agents' actions, what actually happens next?**

---

## 1️⃣ Core Transition Philosophy

In the Cooperative archetype:

> **Outcomes are collective, additive, and history-sensitive.**

No agent's contribution is resolved in isolation. Task completions depend on the sum of effort directed at each type. The environment pools contributions — individual effort matters only insofar as it adds to the group total.

---

## 2️⃣ Transition Cycle (One Timestep)

Each timestep follows this exact sequence:

1. New tasks arrive in the queue
2. Collect all agents' actions
3. Validate and normalize effort amounts
4. Apply specialization multipliers to individual contributions
5. Pool contributions by task type
6. Resolve task completions
7. Update task queue (remove completed, retain backlog)
8. Update system stress
9. Update agent-local state (contribution history, specialization scores)
10. Update relational state (group-level signals)
11. Check termination conditions
12. Advance timestep

This order is **fixed** and deterministic.

---

## 3️⃣ Task Arrival

At the start of each timestep, new tasks arrive before agents act.

- Each task type receives new tasks drawn from a **configurable base rate** per type
- A small seeded noise term is added to simulate variability in demand
- Noise is bounded — arrival rate never drops to zero or explodes
- Total new tasks per step = sum across all task types
- Arriving tasks are added directly to the task queue

Task arrival is the only source of stochasticity in the transition. Everything else is deterministic.

---

## 4️⃣ Specialization Multiplier

Before pooling, each agent's effort is scaled by their specialization score for the chosen task type.

~~~
effective_contribution = effort_amount × capacity × (1 + specialization_bonus)
~~~

Where:
- `specialization_bonus` = agent's specialization score for this task type × `specialization_scale` (config param)
- Specialization score is bounded [0.0, 1.0] — computed from contribution history
- Maximum bonus is capped (e.g. 30% efficiency gain at full specialization) — keeps generalists viable
- An agent working on a type they have never worked on before gets no bonus (score = 0.0)

This makes specialization **rationally emergent** — it is mechanically worth doing but not so dominant that generalists become useless.

---

## 5️⃣ Contribution Pooling

After individual effective contributions are computed, they are summed per task type:

~~~
total_effort[type] = sum of effective_contribution[agent] for all agents choosing WORK_type
~~~

IDLE agents contribute 0.0 to all types.

---

## 6️⃣ Task Completion Resolution

Tasks are completed based on pooled effort vs task difficulty:

~~~
tasks_completed[type] = min(
    floor(total_effort[type] / task_difficulty[type]),
    task_queue[type]
)
~~~

Where:
- `task_difficulty` — effort required to complete one task of this type (configurable per type)
- Completion is floored — partial effort on the last task does not complete it
- Cannot complete more tasks than exist in the queue (capped at `task_queue[type]`)

Completed tasks are removed from the queue immediately.

---

## 7️⃣ Task Queue Update

After completion resolution:

~~~
task_queue[type] = task_queue[type] - tasks_completed[type] + new_arrivals[type]
~~~

Backlog level (total pending tasks across all types) is recomputed:

~~~
backlog_level = sum(task_queue)
~~~

System stress is updated:

~~~
system_stress = min(backlog_level / collapse_threshold, 1.0)
~~~

Where `collapse_threshold` is the configurable backlog level at which the system is considered at full stress.

---

## 8️⃣ Specialization Score Update

After each step, each agent's specialization score per task type is updated using an exponential moving average:

~~~
specialization_score[agent][type] =
    (1 - decay) × specialization_score[agent][type] + decay × contribution_share
~~~

Where:
- `contribution_share` = this agent's effort on this type as a fraction of their total effort this step
- `decay` is a configurable parameter controlling how fast specialization builds and fades
- An agent who switches task types will see their old specialization decay over time

This means specialization is earned through consistent behavior and lost through inconsistency — a natural, history-sensitive dynamic.

---

## 9️⃣ Group-Level Signal Update

Relational state is updated after all resolutions:

- `group_contribution_last_step` = total effective effort across all agents this step
- `group_contribution_by_type` = effective effort per type this step
- `group_completion_rate` = rolling average of (tasks_completed / tasks_arrived) over last K steps — if tasks_arrived == 0 for a step, that step's completion rate is recorded as 1.0
- `free_rider_pressure` = max(0, (expected_contribution - actual_group_contribution) / expected_contribution)

Where `expected_contribution` = N agents × average effort capacity × 0.5 (a configurable baseline).

---

## 🔟 Key Principles

- **No action directly affects another agent** — all effects are mediated through the shared task queue
- **Effort is necessary but not sufficient** — pooling and difficulty mean individual effort alone doesn't guarantee completions
- **History shapes efficiency** — specialization multipliers make past behavior mechanically relevant
- **Stochasticity is narrow** — only task arrival has noise; everything else is deterministic given state and actions
- **No hidden state** — all transition logic is fully determined by the config, seed, and action sequence


# Part 7: Reward Model

> This part answers:
> **How does the environment judge each agent's outcome?**

---

## 1️⃣ Core Reward Philosophy

In the Cooperative archetype:

> **The group's outcome is each agent's outcome. Individual contribution matters only insofar as it shapes the group result.**

All agents receive the same group-level reward signal as the primary component. A small individual component exists to prevent the learning algorithm from being completely blind to free-riding — but it never dominates. If the individual component outweighs the group component, the archetype breaks.

---

## 2️⃣ Reward Decomposition

Final reward = weighted sum of three components:

~~~
reward[agent] = w_group × R_group + w_individual × R_individual + w_efficiency × R_efficiency
~~~

Where weights are configurable but must satisfy:

~~~
w_group >> w_individual >= w_efficiency
~~~

Default weights: `w_group = 0.7`, `w_individual = 0.2`, `w_efficiency = 0.1`

---

## 3️⃣ Group Component — R_group

**What it measures:** How well the group performed this step — task completion rate and backlog health.

~~~
R_group = α × completion_rate_this_step + (1 - α) × (1 - system_stress)
~~~

Where:
- `completion_rate_this_step` = tasks_completed_this_step / tasks_arrived_this_step (capped at 1.0)
- `system_stress` = normalized backlog pressure [0.0, 1.0]
- `α` = configurable balance between throughput and stress (default 0.6)

Properties:
- Identical for all agents — purely collective
- High when the group completes most arriving tasks and backlog is low
- Low when the group falls behind and stress builds
- Bounded [0.0, 1.0]

This is the signal that makes cooperation rational. If you help the group succeed, everyone benefits equally.

---

## 4️⃣ Individual Component — R_individual

**What it measures:** Whether this agent used their capacity meaningfully this step.

~~~
R_individual = effort_amount (if action is WORK) or 0.0 (if IDLE)
~~~

Properties:
- Based solely on own effort level — not compared to other agents
- Does not reward working on the right task type — only that the agent contributed
- IDLE always yields 0.0
- Bounded [0.0, 1.0]
- Kept small (w_individual = 0.2) so it nudges against free-riding without overriding group signal

This component exists for one reason: without it, the learning algorithm cannot distinguish a high-contributing agent from a free-rider when the group signal is identical for both.

---

## 5️⃣ Efficiency Component — R_efficiency

**What it measures:** Whether this agent's specialization paid off for the group this step.

~~~
R_efficiency = specialization_score[agent][chosen_type] × completion_rate_this_step
~~~

Properties:
- Rewards agents who specialized AND whose specialization contributed to group success
- Zero if the agent is generalist (no specialization score built up)
- Zero if the group performed poorly regardless of specialization
- Bounded [0.0, 1.0]
- Kept smallest (w_efficiency = 0.1) — encourages role stability without making generalists unviable

This is what makes specialization self-reinforcing once it emerges — specialists get a small extra reward when the group is succeeding.

---

## 6️⃣ Reward Timing

- **Per-step reward:** All three components are computed and delivered every timestep
- **No terminal bonus in V1** — episode end does not deliver an additional lump reward
- Terminal bonuses (e.g. for completing all tasks or avoiding collapse) are deferred to V2

Per-step rewards keep the learning signal dense and stable. Sparse terminal rewards are harder to learn from and unnecessary given the continuous task completion dynamic.

---

## 7️⃣ Key Rules

- Reward is **bounded** — final reward per step is always in [0.0, 1.0] by construction
- Reward is **stable across episodes** — same config + same behavior = same reward magnitude
- Agents see only the **scalar total reward** — component breakdown is logged for humans but not exposed to agents
- Reward weights are **configurable via schema** — researchers can test different balance points
- The group component must always carry the largest weight — this is a hard constraint, not a soft guideline

---

## 8️⃣ What the Reward Model Does NOT Do

- Does not reward agents for outperforming each other — there is no relative ranking
- Does not penalize agents for choosing the wrong task type — only for not contributing
- Does not reveal other agents' contribution levels — the individual component uses only own effort
- Does not deliver large terminal bonuses — learning is driven by dense per-step signals


# Part 8: Termination Model

> This part answers:
> **When do agents stop, and when does the world end?**

---

## 1️⃣ Core Termination Philosophy

In the Cooperative archetype:

> **The episode ends when the group succeeds decisively, fails sustainably, or runs out of time.**

There is no individual agent termination in V1. All agents remain active for the full episode. The episode is a group outcome — it ends for everyone at the same time.

---

## 2️⃣ Agent-Level Termination

**Not implemented in V1.**

Individual agent elimination — removing one agent mid-episode due to sustained low contribution or capacity exhaustion — is explicitly deferred to V2.

Reasoning:
- Eliminating one agent punishes the entire group for one agent's failure, which is a distinct and more complex mechanic
- V1 must first validate that the core coordination dynamics work with a stable population
- Individual penalties exist in the reward model (low R_individual for free-riders) — elimination is not needed to discourage free-riding in V1

---

## 3️⃣ Episode-Level Termination

Three conditions can end the episode. Any one triggers episode end.

---

### Condition 1: Maximum Horizon Reached (Primary)
~~~
timestep >= max_steps
~~~

- Always triggers if no other condition fires first
- `max_steps` is configurable per experiment
- This is the expected termination path for well-functioning groups
- Logged as termination reason: `"max_steps"`

---

### Condition 2: System Collapse (Failure)
~~~
backlog_level >= collapse_threshold for collapse_sustain_window consecutive steps
~~~

Where:
- `system_stress == 1.0` means backlog has reached or exceeded `collapse_threshold`
- `collapse_sustain_window` — configurable number of consecutive steps at full stress before collapse triggers (default: 10)
- A single bad step does not end the episode — sustained failure does
- Once triggered, the episode ends immediately regardless of remaining steps
- Logged as termination reason: `"system_collapse"`

The sustain window is critical — without it, noise in task arrival could trigger spurious early termination.

---

### Condition 3: Perfect Clearance (Success, Optional)
~~~
backlog_level == 0 for clearance_sustain_window consecutive steps
~~~

Where:
- `clearance_sustain_window` — configurable number of consecutive steps at zero backlog before early success triggers (default: 15)
- **Disabled by default in V1** — set `enable_early_success = false` in config
- When enabled, prevents episodes from dragging after the group has already won decisively
- Logged as termination reason: `"perfect_clearance"`

---

## 4️⃣ Termination Logging

All termination events are logged with:
- Termination reason (`max_steps`, `system_collapse`, `perfect_clearance`)
- Step at which termination fired
- Final backlog level
- Final system stress
- Total tasks completed across episode
- Total tasks arrived across episode
- Completion ratio (tasks_completed / tasks_arrived)

This gives researchers a clean signal for comparing episode outcomes across configs and runs.

---

## 5️⃣ What Termination Does NOT Do

- Does not eliminate individual agents mid-episode
- Does not trigger based on a single bad step — sustained conditions only (for collapse and clearance)
- Does not deliver a terminal reward bonus — reward is per-step only (V1)
- Does not carry any state forward to the next episode


# Part 9: Configuration Schema

> This part answers:
> **How does a user precisely describe this Cooperative environment instance so it can be instantiated, reproduced, compared, and shared?**

---

## 1️⃣ Core Principle

> **A single configuration fully defines one Cooperative environment instance.
> No hidden defaults. No implicit behavior.**

If two configs are identical (including seed), the environment behaves identically.

---

## 2️⃣ Top-Level Config Sections

A Cooperative archetype config has **six sections**.

---

## 3️⃣ Environment Identity

Defines *what* this environment instance is.

| Field | Type | Description |
|---|---|---|
| `environment_type` | string | Always `"cooperative"` |
| `environment_version` | string | Semver, e.g. `"1.0.0"` |
| `archetype` | string | Always `"shared_goal_collective"` |
| `seed` | integer | Global random seed — controls all stochasticity |

Purpose:
- Reproducibility
- Compatibility checks — prevents Competitive or Mixed configs from being loaded as Cooperative
- Experiment tracking

---

## 4️⃣ Population & Episode Parameters

Defines *who exists, for how long, and under what conditions*.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `num_agents` | integer | ≥ 2 | Number of agents in the episode |
| `max_steps` | integer | ≥ 10 | Maximum timesteps per episode |
| `num_task_types` | integer | ≥ 1 | Number of distinct task types in the queue |
| `agent_effort_capacity` | float | > 0.0 | Maximum effort an agent can apply per step |
| `collapse_sustain_window` | integer | ≥ 1 | Consecutive steps at full stress before system collapse triggers (default: 10) |
| `enable_early_success` | boolean | — | Whether perfect clearance termination is active (default: false) |
| `clearance_sustain_window` | integer | ≥ 1 | Consecutive steps at zero backlog before early success triggers (default: 15, ignored if enable_early_success is false) |

---

## 5️⃣ Layer Configuration (Knob Settings)

Defines how strongly each layer is applied. All layers are present — config adjusts their intensity.

| Field | Type | Range | Description |
|---|---|---|---|
| `observation_noise` | float | [0.0, 0.2] | Noise added to public observation signals — simulates imperfect group visibility |
| `history_window` | integer | ≥ 1 | Number of past steps included in agent observation history (K) |
| `specialization_scale` | float | [0.0, 0.5] | Maximum specialization efficiency bonus (0.0 = specialization has no mechanical effect) |
| `specialization_decay` | float | (0.0, 1.0) | Rate at which specialization scores update via EMA — higher = faster build and fade |
| `task_arrival_noise` | float | [0.0, 0.3] | Magnitude of noise on task arrival rates per step |
| `task_difficulty_variance` | float | [0.0, 0.3] | Episode-level variance in task difficulty — draws difficulty per task type at episode start |
| `free_rider_pressure_scale` | float | [0.0, 1.0] | Scales the free_rider_pressure signal in observations — higher = more sensitive group signal |

---

## 6️⃣ Task & World Parameters

Defines the structure of the task environment agents operate in.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `task_arrival_rate` | float or list[float] | > 0.0 | Base task arrival rate per step — scalar applies to all types, list sets per type |
| `task_difficulty` | float or list[float] | > 0.0 | Effort required to complete one task — scalar applies to all types, list sets per type |
| `collapse_threshold` | integer | > 0 | Backlog level at which system_stress reaches 1.0 |
| `initial_backlog` | integer | ≥ 0 | Tasks pre-loaded into the queue at episode start (default: 0) |

---

## 7️⃣ Reward Weights

Defines the balance between reward components. Must satisfy `w_group >> w_individual >= w_efficiency`.

| Field | Type | Range | Description |
|---|---|---|---|
| `w_group` | float | (0.0, 1.0) | Weight on group completion and stress component |
| `w_individual` | float | (0.0, 1.0) | Weight on individual effort component |
| `w_efficiency` | float | (0.0, 1.0) | Weight on specialization efficiency component |

Constraint enforced at instantiation:

~~~
w_group + w_individual + w_efficiency == 1.0
w_group >= 0.5  (hard minimum — group signal must dominate)
~~~

Default: `w_group = 0.7`, `w_individual = 0.2`, `w_efficiency = 0.1`

---

## 8️⃣ Instrumentation Flags

Controls what is logged during the episode. Agents never see these — logging is environment-side only.

| Field | Type | Default | Description |
|---|---|---|---|
| `log_per_agent_contributions` | boolean | true | Log each agent's effort per type per step |
| `log_specialization_scores` | boolean | true | Log per-agent specialization scores each step |
| `log_group_signals` | boolean | true | Log group-level relational state each step |
| `log_task_queue` | boolean | true | Log full task queue state each step |
| `log_reward_components` | boolean | true | Log R_group, R_individual, R_efficiency separately per agent per step |
| `log_termination_detail` | boolean | true | Log full termination state snapshot |


# Part 10: Instrumentation & Metrics

> This part answers:
> **What do we measure so humans can understand, compare, and trust agent behavior?**

---

## 1️⃣ Core Principle

> **Metrics access full state. Agents see none of them.**

All metrics are computed from the environment's internal state — not from agent observations. They exist entirely for human analysis, dashboard visualization, and research output. They are read-only and deterministic.

---

## 2️⃣ Episode-Level Metrics

Computed once per episode. Summarize the overall group outcome.

| Metric | Description |
|---|---|
| `episode_length` | Number of steps before termination |
| `termination_reason` | One of: `max_steps`, `system_collapse`, `perfect_clearance` |
| `total_tasks_arrived` | Total tasks that entered the queue across the episode |
| `total_tasks_completed` | Total tasks completed across the episode |
| `completion_ratio` | `total_tasks_completed / total_tasks_arrived` (primary outcome metric) |
| `final_backlog_level` | Backlog size at episode end |
| `final_system_stress` | System stress at episode end [0.0, 1.0] |
| `mean_system_stress` | Average system stress across all steps |
| `peak_system_stress` | Maximum system stress reached during episode |
| `group_efficiency_ratio` | `total_tasks_completed / (num_agents × agent_effort_capacity × episode_length / mean_task_difficulty)` — how close the group got to the theoretical maximum output |
| `collapse_occurred` | Boolean — whether system_collapse termination fired |

---

## 3️⃣ Agent-Level Metrics

Computed per agent per episode. Reveal individual contribution patterns.

| Metric | Description |
|---|---|
| `cumulative_reward` | Total reward accumulated across the episode |
| `mean_reward_per_step` | Average per-step reward |
| `effort_utilization` | Mean effort_amount across all non-idle steps (how fully the agent used its capacity) |
| `idle_rate` | Fraction of steps where agent chose IDLE |
| `dominant_task_type` | Task type this agent directed the most effort toward |
| `dominant_type_fraction` | Fraction of total effort directed at the dominant task type |
| `final_specialization_score` | Specialization score for the dominant task type at episode end |
| `peak_specialization_score` | Highest specialization score reached across any task type |
| `reward_component_breakdown` | Mean R_group, R_individual, R_efficiency separately across the episode |

---

## 4️⃣ Interaction / Social Metrics

The cooperative-specific signals. These are what make the archetype analytically interesting.

| Metric | Description |
|---|---|
| `contribution_variance` | Variance in cumulative effort across agents — high variance signals free-riding |
| `specialization_divergence` | Mean pairwise difference in dominant task type across agents — high = strong role differentiation |
| `role_stability` | Per agent: fraction of steps spent on their dominant task type — high = stable specialist |
| `mean_role_stability` | Average role stability across all agents |
| `free_rider_count` | Number of agents whose idle_rate exceeded a threshold (configurable, default: 0.4) |
| `free_rider_fraction` | `free_rider_count / num_agents` |
| `effort_gini_coefficient` | Gini coefficient of cumulative effort distribution across agents — 0.0 = perfectly equal, 1.0 = one agent did everything |
| `group_coordination_score` | Composite: high when specialization_divergence is high AND completion_ratio is high — confirms roles helped |

---

## 5️⃣ Resource / System Metrics

Track the health of the shared task environment across the episode.

| Metric | Description |
|---|---|
| `backlog_trajectory` | Backlog level at each step — full time series |
| `stress_trajectory` | System stress at each step — full time series |
| `completion_rate_trajectory` | Per-step completion rate — full time series |
| `task_queue_by_type` | Per-step queue depth per task type — full time series |
| `arrival_vs_completion_balance` | Mean (tasks_completed - tasks_arrived) per step — positive = group keeping up |
| `steps_at_full_stress` | Number of steps where system_stress == 1.0 |
| `steps_near_collapse` | Number of steps where system_stress >= 0.8 |

---

## 6️⃣ Event-Level Logs

Sparse, semantic events logged when they occur. Not every step — only when something meaningful happens.

| Event | Trigger |
|---|---|
| `specialization_threshold_crossed` | Any agent's specialization score exceeds 0.7 for the first time |
| `role_lock` | Any agent maintains the same dominant task type for 20+ consecutive steps |
| `free_rider_detected` | Any agent's idle_rate exceeds threshold over a rolling 10-step window |
| `system_stress_spike` | system_stress jumps by >= 0.3 in a single step |
| `stress_recovery` | system_stress drops from >= 0.8 to <= 0.4 within 5 steps — group recovered |
| `system_collapse` | Collapse termination condition triggered |
| `perfect_clearance` | Clearance termination condition triggered |
| `contribution_imbalance` | effort_gini_coefficient exceeds 0.5 for 5+ consecutive steps |

---

## 7️⃣ Strategy Clustering Inputs

These metrics feed directly into the k-means strategy clustering pipeline (same as Mixed and Competitive).

Per-agent feature vector for clustering:

- `effort_utilization`
- `idle_rate`
- `dominant_type_fraction`
- `final_specialization_score`
- `role_stability`
- `mean_reward_per_step`

Note: `dominant_task_type` is excluded from the clustering feature vector — it is categorical and incompatible with k-means. It is retained as agent metadata and used as a label only.

Expected emergent strategy labels (not hardcoded — must emerge from clustering):
- **Dedicated Specialist** — high specialization, high role stability, low idle rate
- **Adaptive Generalist** — low specialization divergence, switches task types, moderate effort
- **Free Rider** — high idle rate, low effort utilization, low individual reward
- **Opportunist** — switches to whichever task type has the longest queue, moderate specialization
- **Overcontributor** — maximum effort utilization, compensates for free-riders


# Part 11: Agent Export, Reuse & User Value

> This part answers:
> **What can users do with a trained Cooperative agent?**

---

## 1️⃣ Core Principle

> **A trained agent is a portable, analyzable artifact — not just a weights file.**

An exported Cooperative agent carries enough metadata to be reused, compared, and understood without re-running the experiment.

---

## 2️⃣ Exported Agent Artifact

Each exported agent contains:

| Component | Description |
|---|---|
| `policy.pt` | Trained PyTorch weights |
| `metadata.json` | Full artifact metadata (see below) |

`metadata.json` includes:
- `archetype` — `"cooperative"`
- `config_hash` — fingerprint of the config used during training
- `environment_version` — compatibility reference
- `training_steps` — total steps trained
- `league_id` — league snapshot ID if exported from league
- `elo_rating` — Elo rating at time of export
- `performance_summary` — key metrics from evaluation (see below)
- `strategy_label` — cluster label assigned by strategy clustering pipeline
- `compatibility_bounds` — config parameter ranges this agent is valid for

---

## 3️⃣ Performance Summary (Cooperative-Specific)

The performance summary inside metadata captures what kind of cooperative agent this is:

| Field | Description |
|---|---|
| `mean_completion_ratio` | Average group completion ratio across evaluation episodes |
| `mean_effort_utilization` | How fully this agent used its effort capacity |
| `dominant_task_type` | Task type this agent specialized in most consistently |
| `final_specialization_score` | Specialization score at end of evaluation episodes |
| `role_stability` | How consistently this agent maintained its dominant task type |
| `idle_rate` | Fraction of steps this agent chose IDLE |
| `strategy_label` | e.g. Dedicated Specialist, Free Rider, Adaptive Generalist |

This turns agents into **analyzable objects** — not just winners, but characterized strategies.

---

## 4️⃣ Four Reuse Modes

---

### 1. Evaluation / Benchmarking
Use case:
- Compare multiple trained agents in the same Cooperative config
- Run tournaments — which agent contributes most to group success?
- Rank strategies by completion ratio and effort utilization

Value:
- Fair comparison across training runs
- Reproducible leaderboards
- Strategy profiling across agent types

---

### 2. Cross-Environment Transfer
Use case:
- Take an agent trained in one Cooperative config
- Evaluate it in a modified config — more task types, higher arrival rate, harder difficulty

Value:
- Robustness testing — does the specialist hold up when task difficulty increases?
- Generalization analysis — does the agent adapt or collapse under new conditions?
- Directly feeds the robustness sweep pipeline

---

### 3. Population Seeding
Use case:
- Inject a trained Cooperative agent into a population of untrained or random agents
- Study how one strong coordinator influences group performance
- Test whether a free-rider policy degrades group outcomes when seeded into a cooperative population

Value:
- Social dynamics experiments
- Influence analysis — does one good agent lift the group?
- Directly leverages the league self-play system

---

### 4. Agent-as-Module
Use case:
- Treat the trained agent as a black-box effort allocator
- Plug into a different simulation or downstream system
- Study coordination behavior in novel contexts

Value:
- Software reuse across archetypes
- System design credibility
- Aligns with real-world multi-agent deployment thinking

---

## 5️⃣ Compatibility Guarantee

An exported agent guarantees compatibility with:
- Any Cooperative environment instance whose config falls within `compatibility_bounds`
- Rejection with explanation if config is outside bounds or wrong archetype

This prevents:
- Silent failures from loading a Competitive agent into a Cooperative environment
- Invalid comparisons across incompatible configs
- Misleading robustness results

---

## 6️⃣ Explicitly Deferred to V2

The following were considered and deliberately excluded from V1:

| Item | Reason |
|---|---|
| Individual agent termination mid-episode | Adds group punishment mechanic before core coordination is validated |
| Split effort allocation across multiple task types per step | Requires vector action space — adds complexity before single-type specialization is proven |
| Direct agent-to-agent communication | That is the Negotiation archetype |
| Role assignment by design (leader/follower) | Roles must emerge from learning in V1 — explicit asymmetry deferred to V2 |
| Terminal reward bonus | Dense per-step rewards sufficient for V1 — sparse terminal signal deferred |
| Task deadlines (per-task time limits) | Adds urgency mechanic — meaningful extension but out of V1 scope |
| Dynamic team size (agents joining/leaving mid-episode) | Significant added complexity — deferred |
| Multi-team cooperative competition | Hybrid of Cooperative and Competitive — separate archetype consideration |

---

# Ambiguities Found & Resolved

During the sanity check phase, the following contradictions and underspecifications were identified and resolved. These resolutions are authoritative — they take precedence over any ambiguous wording elsewhere in this document.

1. **effort_capacity in agent-local state:** Part 3 stated effort capacity resets each step, making it a constant rather than a tracked state variable. Resolution: `effort_capacity` removed from agent-local state. It is a fixed config param (`agent_effort_capacity`) — the same value is available to the environment each step without being stored in state.

2. **contribution_share always equals 1.0:** Part 6's specialization update formula used `contribution_share` = "this agent's effort on this type as a fraction of their total effort this step." Since agents choose exactly one task type per step (Part 5), their total effort is always 100% on that type — contribution_share is always 1.0 for the chosen type and 0.0 for all others. Resolution: stated explicitly. The EMA simplifies to: score[chosen_type] moves toward 1.0, score[all other types] decay toward 0.0. No complex fraction needed in implementation.

3. **task_difficulty scalar vs per-type:** Part 6's completion formula used `task_difficulty` as a single scalar. Part 9's config defines it as `float or list[float]`. Resolution: formula updated to use `task_difficulty[type]` — per-type difficulty is always used. If config provides a scalar, it is broadcast to all types at instantiation.

4. **Division by zero in completion rate:** `tasks_completed / tasks_arrived` is undefined when no tasks arrive in a step. Resolution: if `tasks_arrived == 0` for a step, that step's completion rate is recorded as 1.0 — the group trivially succeeded since nothing needed to be completed.

5. **system_stress float equality check:** Collapse condition used `system_stress == 1.0` — fragile for floating point. Resolution: condition changed to `backlog_level >= collapse_threshold` — integer comparison, no floating point issue.

6. **group_efficiency_ratio undefined:** Part 10 listed this metric without a formula. Resolution: defined as `total_tasks_completed / (num_agents × agent_effort_capacity × episode_length / mean_task_difficulty)` — the ratio of actual output to the theoretical maximum if all agents contributed fully every step.

7. **dominant_task_type in clustering feature vector:** K-means requires continuous features — a categorical task type label is incompatible. Resolution: `dominant_task_type` removed from the clustering feature vector. Retained as agent metadata and used as a human-readable label only. `dominant_type_fraction` already in the vector captures the same information numerically.