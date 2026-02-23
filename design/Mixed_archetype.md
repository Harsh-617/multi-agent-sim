# Mixed archetype

# Part 1: Core Tension & Purpose
## 1️⃣ Archetype Name
**Mixed Interaction Archetype**
(Competition + Cooperation)
---
## 2️⃣ Core Question This Archetype Is Designed to Answer
> **How do agents decide when to cooperate and when to compete in a shared environment where both behaviors are necessary for long-term success?**
This is the *entire reason* this archetype exists.
---
## 3️⃣ Fundamental Tension (Very Important)
This archetype is built around **three simultaneous pressures**:
1. **Individual Incentive**
    - Agents can gain short-term advantage by acting selfishly
2. **Group Dependence**
    - Long-term survival or growth requires some level of cooperation
3. **Strategic Uncertainty**
    - Agents do not fully know others’ intentions or future actions
No single strategy is dominant.
---
## 4️⃣ What Makes This Archetype “Mixed” (Not Pure)
This archetype is **not**:
- Pure zero-sum (someone must lose)
- Pure cooperative (shared reward only)
Instead:
- Some rewards are **individual**
- Some rewards are **shared**
- Some outcomes depend on **others’ behavior**
An agent can:
- cooperate and be exploited
- defect and be punished later
- form temporary alliances
- switch strategies over time
---
## 5️⃣ Desired Emergent Behaviors (What We Expect to See)
We are not hardcoding these — we want them to **emerge**:
- Conditional cooperation (“I help if you helped before”)
- Free-riding
- Retaliation
- Alliance formation and collapse
- Short-term betrayal vs long-term trust
- Risk-taking under uncertainty
If none of these appear, the environment failed.
---
## 6️⃣ What This Archetype Is NOT Trying to Model
Important exclusions (by design):
- Perfect rationality
- Global coordination
- Centralized control
- Fully observable intentions
- Guaranteed convergence to one equilibrium
This is a **strategic system**, not an optimization problem.
---
## 7️⃣ Success Criteria for the Archetype (High-Level)
This archetype is successful if:
- Different agents learn **different strategies**
- No single policy dominates all others
- Performance depends on **population composition**
- Policies trained here behave differently when moved to:
    - more cooperative variants
    - more competitive variants
This makes it valuable for benchmarking and transfer.

# Part 2: Layer Emphasis (Knob Settings)
## 0) Why we do this first
Because these choices decide:
- what the environment *feels like*
- what behaviors can emerge
- how hard training will be
---
## 1) Primary Layers (High emphasis)
### ✅ Layer 4: Interaction Structure (Social Layer) — **HIGH**
This archetype lives here.
We need:
- externalities (my action affects your reward/state)
- conflicts over shared resources
- group outcomes influenced by individual behavior
Without this, it’s not “mixed”.
---
### ✅ Layer 6: Soft Constraints & Incentives (Economic Layer) — **HIGH**
Mixed environments become trivial if rules are hard bans.
We need:
- cooperation is possible but costly
- betrayal is possible but risky later
- tradeoffs, not prohibitions
This is what creates strategic depth.
---
### ✅ Layer 2: Temporal Structure (History Layer) — **MEDIUM → HIGH**
Mixed behavior requires memory.
We need at least:
- delayed consequences (short-term gain can cause long-term harm)
- reputation/trust signals (even simple)
Otherwise agents just greed-max instantly.
---
## 2) Secondary Layers (Medium emphasis)
### ⚠️ Layer 1: Information Structure (Observability Layer) — **MEDIUM**
To make it realistic/strategic:
- agents should not see everything
- must infer others’ intentions from behavior
But keep it moderate so learning is still feasible.
---
### ⚠️ Layer 3: State Hierarchy (Structural Layer) — **MEDIUM**
We want at least:
- global shared state
- per-agent private state
    Optional later:
- coalition/team state
Not too heavy initially, but present.
---
## 3) Minimal Layers (Low emphasis for v1)
### 🟢 Layer 5: Power & Role Asymmetry (Institutional Layer) — **LOW**
For the first mixed archetype, keep agents mostly symmetric.
Roles can come later (governor/citizen etc.).
This reduces complexity early.
---
### 🟢 Layer 7: Uncertainty & Meta Dynamics (Meta Layer) — **LOW → MEDIUM**
We include only:
- simple seeded randomness (spawn variation)
    Avoid:
- heavy non-stationarity
- evolving rules
We can dial this up later for robustness tests.
---
## Summary (the knob settings)
- **High:** Interaction (4), Incentives (6)
- **Medium/High:** Temporal (2)
- **Medium:** Information (1), State hierarchy (3)
- **Low:** Power/roles (5)
- **Low→Med:** Uncertainty/meta (7)

# Part 3: **Agents, Roles, and Episode Structure**
This part answers **one narrow question**:
> **Who exists in this world, and what does one episode look like?**
Everything else (state, actions, rewards) depends on this.
---
## 1️⃣ Agents (Who Exists)
### Number of Agents
- The environment supports **N agents**, where:
    - N ≥ 2
    - N is configurable per experiment
This allows:
- dyadic interaction (2 agents)
- small-group dynamics (3–10 agents)
- population effects later
---
### Agent Symmetry (Important Design Choice)
For the **first Mixed archetype**:
- All agents are **structurally symmetric**
- Same:
    - action space
    - observation schema
    - reward structure (form, not value)
- Differences in behavior must **emerge from learning**, not roles
This keeps:
- analysis clean
- comparisons fair
- learning stable
(We intentionally deferred role asymmetry to later archetypes.)
---
### Agent Identity
Each agent has:
- a unique ID
- persistent identity across the episode
Identity matters for:
- reputation
- memory
- conditional cooperation (“how *this* agent behaved before”)
---
## 2️⃣ Agent Capabilities (High-Level, Not Actions Yet)
At this stage we only define **capability categories**, not concrete actions.
Each agent is capable of:
- acting every timestep (unless terminated)
- influencing shared resources
- affecting other agents indirectly
- accumulating private history (memory)
Agents are **autonomous**:
- no central controller
- no enforced coordination
---
## 3️⃣ Episode Structure (What One Run Looks Like)
### Episode Timeline
An episode proceeds in **discrete timesteps**:
```
reset
→ step1
→ step2
→ ...
→ step T
→ termination
```
Where:
- T is bounded (max steps)
- early termination is possible
---
### Action Timing
- All agents act **simultaneously** at each step
- No turn-taking in v1
This ensures:
- true strategic interaction
- need for anticipation
- conflict resolution by the environment
---
### Persistence Across Steps
Within an episode:
- agent identity persists
- agent memory persists
- consequences accumulate
Across episodes:
- the environment resets
- agents may or may not retain learned parameters (training vs evaluation)
---

## 4️⃣ Agent Termination (Individual-Level)

An agent may terminate before the episode ends due to:

- resource exhaustion
- elimination condition
- failure state

After termination:

- the agent no longer acts
- its state is frozen or removed (defined later)
- remaining agents continue

This allows:

- dominance
- collapse
- survival dynamics
 

# Part 4: **State & Observation Definition**

This part answers:

> **What exists in the world, and what does each agent get to see?**
> 

We are now instantiating **Step 3 of the framework** for the Mixed archetype.

---

## 1️⃣ Global State (What the World Actually Contains)

The **global state** is the complete internal truth of the environment.

For the Mixed archetype, the state is **intentionally structured**, not flat.

### State is divided into three tiers:

---

### A) Global Shared State

Visible to the environment only (not directly to agents unless exposed).

Includes:

- Current timestep
- Shared resource pool(s)
- Global system indicators (e.g. stress, scarcity level)
- Global constraints (caps, limits)

Purpose:

- Create interdependence
- Enable externalities
- Support delayed consequences

---

### B) Agent-Local State (Private)

Each agent has its own private slice of state.

Includes:

- Individual resources/budget
- Accumulated payoff
- Private counters or cooldowns
- Agent-specific memory variables

Purpose:

- Enable selfish incentives
- Allow divergence in strategy
- Support history-based behavior

---

### C) Relational / Interaction State

State that exists *between* agents.

Includes:

- Pairwise interaction history (who cooperated with whom)
- Trust / reputation scores
- Past conflict markers

Purpose:

- Enable conditional cooperation
- Support retaliation and forgiveness
- Make history matter

---

## 2️⃣ Observation Model (What an Agent Sees)

Each agent receives an **observation**, which is a *projection* of the state.

Observations are **not symmetric** to the state.

---

### Observation is composed of:

### A) Public Observation

Same for all agents.

Examples:

- Current timestep
- Size of shared resource pool
- Aggregate group performance metrics

Purpose:

- Ground agents in the same world
- Enable coordination signals

---

### B) Private Self Observation

Visible only to the agent.

Examples:

- Own resources
- Own accumulated reward
- Own recent actions/effects

Purpose:

- Support self-optimization
- Enable risk assessment

---

### C) Partial Social Observation

Limited information about others.

Examples:

- Aggregate stats (not identities)
- Limited reputation signals
- Recent public actions (but not intentions)

Purpose:

- Force inference
- Prevent perfect prediction
- Enable strategic uncertainty

---

## 3️⃣ What Agents Do NOT See (By Design)

Agents do **not** see:

- Full global state
- Other agents’ private state
- True reward functions of others
- Future events or hidden timers

This ensures:

- no omniscience
- no trivial equilibria
- realistic mixed incentives

---

## 4️⃣ Observation Stability Guarantees

The Mixed archetype guarantees:

- Observation **structure is fixed**
- Fields do not appear/disappear
- Missing info is masked, not removed

This is critical for:

- learning stability
- fair comparison
- reproducibility

---

## 5️⃣ Information Asymmetry (Moderate)

This archetype deliberately includes:

- asymmetric information
- partial observability
- delayed inference

But avoids:

- extreme hidden-state complexity (for v1)

This balances:

- strategic depth
- learnability

---

## 6️⃣ Logging vs Observation (Reiterated)

Important reminder:

- Full state is available to logs
- Observations are the *only* agent inputs
- No leakage from logs to agents

---

# Part 5: **Action Space Definition**

This part instantiates **Step 4 (Action Model)** for the Mixed archetype.

---

## 1️⃣ Core Action Philosophy (Important)

In the Mixed archetype:

> **Actions represent strategic intent, not guaranteed outcomes.**
> 

Agents may:

- attempt to cooperate
- attempt to exploit
- attempt to free-ride
- attempt to protect themselves

The **environment decides** what actually happens.

---

## 2️⃣ Action Timing

- All agents submit **one action per timestep**
- Actions are **simultaneous**
- The environment resolves them jointly

This preserves:

- strategic uncertainty
- true interaction coupling
- non-trivial dynamics

---

## 3️⃣ Action Categories (High-Level)

We define **categories**, not low-level mechanics yet.

Each agent can choose **exactly one** action per step from the following conceptual categories:

---

### A) Contribution / Cooperation Actions

Intent:

- contribute to a shared resource or group outcome

Properties:

- incurs immediate personal cost
- may increase shared state
- benefits may be delayed or indirect

This enables:

- cooperation
- altruism
- long-term group survival

---

### B) Extraction / Competitive Actions

Intent:

- take from shared resources
- prioritize self-gain

Properties:

- increases private state
- may reduce shared state
- can harm others indirectly

This enables:

- selfish behavior
- exploitation
- short-term dominance

---

### C) Neutral / Defensive Actions

Intent:

- preserve current state
- reduce exposure to risk

Properties:

- low reward
- low risk
- may protect against penalties

This enables:

- cautious strategies
- risk management
- waiting behavior

---

### D) Conditional / Strategic Actions

Intent:

- react based on history or signals

Examples (conceptual):

- retaliate
- forgive
- signal cooperation
- withhold contribution

These actions depend on:

- past interactions
- observed behavior
- inferred intent

This enables:

- strategy switching
- adaptive behavior
- conditional cooperation

---

## 4️⃣ Action Constraints

Actions are subject to:

- resource availability
- cooldowns or limits
- soft costs

Important:

- Most actions are **allowed**
- Undesirable actions are discouraged via **costs**, not bans

This aligns with the Mixed archetype’s incentive design.

---

## 5️⃣ Invalid Action Handling

If an agent attempts an invalid action:

- the environment resolves it **deterministically**
- examples:
    - convert to neutral action
    - apply penalty
    - partial execution

No randomness here.

---

## 6️⃣ What Actions Do NOT Do

Actions do **not**:

- directly modify state
- directly change rewards
- directly affect other agents

They only express **intent**.

---

## 7️⃣ Extensibility Guarantee

The action model is designed so that:

- new action types can be added later
- existing agents remain compatible
- observation structure remains stable

This future-proofs the archetype.

---

# Part 6: **Transition Dynamics**

This part instantiates **Step 5 (Transition / Dynamics Model)** for the Mixed archetype.

It answers:

> **Given the current state and all agents’ actions, what actually happens next?**
> 

---

## 1️⃣ Core Transition Philosophy

In the Mixed archetype:

> **Outcomes are collective, interdependent, and history-sensitive.**
> 

No action is resolved in isolation.

Most outcomes depend on:

- how many agents chose similar actions
- who did what in the past
- current scarcity or abundance

---

## 2️⃣ Transition Cycle (One Timestep)

Each timestep follows this exact sequence:

1. Collect all agents’ actions
2. Validate and normalize actions
3. Resolve interactions & conflicts
4. Update shared/global state
5. Update agent-local state
6. Update relational (history/reputation) state
7. Advance time

This order is **fixed** and deterministic.

---

## 3️⃣ Collective Resolution Logic (Key Feature)

Certain actions (especially cooperation & extraction) are resolved **collectively**, not per-agent.

Examples (conceptual):

- If many agents cooperate → shared resource grows efficiently
- If few cooperate → contributors may be exploited
- If too many extract → resource collapses faster

This is where **emergent group dynamics** come from.

---

## 4️⃣ Externalities (Mandatory)

Actions may have **side effects** on others:

- One agent extracting reduces availability for all
- One agent cooperating may benefit others more than itself
- Defensive actions may shield an agent while increasing pressure on others

These effects are **explicitly encoded**, not accidental.

---

## 5️⃣ History-Dependent Dynamics (Temporal Layer Active)

The transition function may depend on:

- recent cooperation frequency
- past betrayals
- trust/reputation scores

Examples:

- Repeated betrayal increases future penalties
- Sustained cooperation unlocks efficiency bonuses
- Retaliation becomes more effective after exploitation

This makes short-term greed risky.

---

## 6️⃣ Resource Constraints & Depletion

Shared resources are:

- finite
- regenerating (possibly)
- sensitive to overuse

The transition model must enforce:

- depletion
- diminishing returns
- possible collapse states

This ensures:

- tragedy-of-the-commons scenarios
- real strategic tradeoffs

---

## 7️⃣ Stochasticity (Controlled, Low)

The Mixed archetype includes **limited randomness**:

- small noise in outcomes
- seeded, reproducible

Used only to:

- prevent brittle policies
- break ties
- simulate uncertainty

Randomness never dominates outcomes.

---

## 8️⃣ State Integrity & Invariants

After every transition:

- no negative resources (unless explicitly allowed)
- no invalid agent states
- relational data remains consistent

If an invariant is violated:

- environment resolves it deterministically
- or triggers termination

---

# Part 7: **Reward Model**

This instantiates **Step 6 (Reward Model)** for the Mixed archetype.

It answers:

> **Given what just happened, how does the environment judge each agent’s outcome?**
> 

---

## 1️⃣ Core Reward Philosophy

In the Mixed archetype:

> **Rewards must create tension between short-term individual gain and long-term collective viability.**
> 

If agents can optimize rewards without caring about others,

the archetype collapses into pure competition.

If agents are forced to share rewards equally,

it collapses into pure cooperation.

So rewards must be **mixed by design**.

---

## 2️⃣ Reward Decomposition (Very Important)

Each agent’s reward at a timestep is composed of **three components**:

### A) Individual Component

Reflects:

- personal resource gain
- personal survival
- efficiency of own actions

This incentivizes:

- selfish optimization
- risk-taking
- exploitation (when possible)

---

### B) Shared / Group Component

Reflects:

- health of shared resources
- group-level outcomes
- long-term sustainability

This incentivizes:

- cooperation
- restraint
- collective planning

---

### C) Relational / History Component

Reflects:

- how the agent’s behavior affects others over time
- trust, betrayal, retaliation dynamics

This incentivizes:

- conditional cooperation
- reputation management
- strategic consistency

---

The final reward is a **weighted combination** of these components.

Weights are:

- fixed per environment instance
- configurable via schema
- known to the environment (not necessarily to agents)

---

## 3️⃣ Reward Timing

The Mixed archetype uses **hybrid timing**:

- Small per-step rewards (feedback)
- Larger delayed effects via:
    - resource collapse
    - trust decay
    - efficiency bonuses

This prevents:

- purely greedy policies
- myopic optimization

---

## 4️⃣ Penalties Instead of Prohibitions

Undesirable behavior is discouraged via:

- penalties
- opportunity costs
- increased future risk

Examples:

- Over-extraction reduces future efficiency
- Betrayal lowers reputation, harming later payoffs
- Hoarding increases collapse probability

Actions are **rarely banned outright**.

---

## 5️⃣ Credit Assignment Clarity (for Humans)

While agents receive only scalar rewards:

- logs must record reward components separately
- humans can see *why* rewards occurred

This supports:

- debugging
- analysis
- explainability

But:

- agents do not see decomposed rewards directly (unless explicitly configured)

---

## 6️⃣ Reward Boundedness & Stability

The environment guarantees:

- rewards are bounded
- scales are stable across episodes
- no runaway positive feedback loops

This is essential for:

- MARL stability
- fair benchmarking
- cross-policy comparison

---

## 7️⃣ What Rewards Do NOT Do

Rewards do **not**:

- change the state
- override transition dynamics
- signal hidden state directly

They only evaluate outcomes after the fact.

---

# Part 8: **Termination Model**

This instantiates **Step 7 (Termination Model)** for the Mixed archetype.

It answers:

> **When does an agent stop, and when does the world end?**
> 

---

## 1️⃣ Core Termination Philosophy

In the Mixed archetype:

> **Termination reflects systemic consequences, not arbitrary cutoffs.**
> 

Episodes end because:

- the system succeeds,
- the system collapses,
- or meaningful interaction is no longer possible.

---

## 2️⃣ Agent-Level Termination

An individual agent may terminate while others continue.

### Agent termination conditions (conceptual):

- Resource exhaustion (can no longer act meaningfully)
- Accumulated penalties exceed a threshold
- Irrecoverable failure state (e.g., exclusion from interaction)

After agent termination:

- the agent no longer submits actions
- its private state is frozen
- its relational impact persists (reputation still matters)

This allows:

- dominance
- elimination
- survivor dynamics

---

## 3️⃣ Episode-Level Termination

The entire episode terminates when **any one** of the following holds:

### A) Maximum Horizon Reached

- A fixed maximum number of timesteps
- Prevents infinite play
- Ensures comparability across runs

---

### B) System Collapse

- Shared resource fully depleted
- Irreversible failure condition reached
- Cooperation breakdown beyond recovery

This captures:

- tragedy-of-the-commons outcomes
- collective failure due to greed

---

### C) Stable Resolution / Saturation

- The system reaches a stable regime
- No significant state change for a sustained period
- All remaining agents are locked into low-impact behavior

This avoids wasting computation after convergence.

---

### D) No Active Agents Remain

- All agents terminated individually
- Episode ends immediately

---

## 4️⃣ Final Rewards at Termination

Upon episode termination:

- final rewards may be issued
- delayed consequences resolve
- cumulative performance is logged

Important:

- termination ≠ success
- termination ≠ failure by default

Success/failure is **evaluative**, not structural.

---

## 5️⃣ Determinism & Transparency

Termination conditions are:

- state-based
- deterministic
- logged explicitly

No hidden or probabilistic endings.

This ensures:

- reproducibility
- explainability
- fair benchmarking

---

## 6️⃣ Post-Termination Guarantees

After termination:

- no further state transitions occur
- no further actions are processed
- logs capture final state snapshot

This makes:

- replay
- auditing
- comparison
    
    possible.
    

---

# Part 9: **Configuration Schema (Archetype-Specific)**

This instantiates **Step 9 (Configuration Schema)** for the Mixed archetype.

It answers:

> **How does a user precisely describe *this* Mixed environment so it can be instantiated, reproduced, compared, and shared?**
> 

---

## 1️⃣ Core Principle

> **A single configuration fully defines one Mixed-environment instance.
No hidden defaults. No implicit behavior.**
> 

If two configs are identical (including seed), the environment behaves identically.

---

## 2️⃣ Top-Level Config Sections (Mixed Archetype)

A Mixed archetype config has **six sections**.

---

## 3️⃣ Environment Identity

Defines *what* this environment instance is.

Includes:

- `environment_type`: `"mixed"`
- `environment_version`
- `archetype`: `"competition_cooperation"`
- `seed`

Purpose:

- reproducibility
- compatibility checks
- experiment tracking

---

## 4️⃣ Population & Episode Parameters

Defines *who exists and for how long*.

Includes:

- number of agents (N)
- max episode length
- termination thresholds
- initial resource levels (global & per-agent)

These parameters shape:

- state initialization
- possible trajectories
- system stability

---

## 5️⃣ Layer Configuration (Knob Settings)

Defines how strongly each layer is applied.

Includes (conceptually):

- information asymmetry level
- temporal memory depth
- relational memory strength
- incentive softness (penalties vs bans)
- uncertainty intensity (low for v1)

Important:

- All layers are present
- Config only adjusts their influence

This allows:

- easy environment variants
- controlled ablations
- systematic benchmarking

---

## 6️⃣ Reward Weights & Incentive Mix

Defines how reward components are combined.

Includes:

- individual reward weight
- group reward weight
- relational/history reward weight
- penalty scaling factors

Purpose:

- tune cooperation pressure
- tune exploitation incentives
- study strategic shifts

These are **environment parameters**, not agent choices.

---

## 7️⃣ Agent Configuration

Defines per-agent settings.

Supports:

- homogeneous agents (same config)
- heterogeneous agents (optional later)

Includes:

- initial private resources
- observation scope modifiers (within allowed bounds)
- optional memory limits

Important:

- Action and observation *structure* stays fixed
- Only parameter values vary

---

## 8️⃣ Instrumentation Flags (Link to Part 10)

Defines:

- what metrics to collect
- logging frequency
- verbosity level
- event logging on/off

This section does **not** define metrics — only enables them.

---

## 9️⃣ Validation Rules (Mandatory)

Before instantiation, the config must be validated:

- reward weights sum to a valid range
- layer settings are compatible
- agent counts ≥ 2
- termination thresholds are reachable
- no contradictory constraints

Invalid configs are rejected early.

---

## 10️⃣ Why This Schema Matters

With this schema:

- users can create many Mixed variants
- experiments are reproducible
- UI forms map directly to config fields
- trained agents can be fairly compared

Without it, your project becomes:

- brittle
- non-scientific
- impossible to scale

---

# Part 10: **Instrumentation & Metrics**

This instantiates **Framework Step 10 (Instrumentation & Logging)** specifically for the Mixed archetype.

It answers:

> **What do we measure so humans can understand, compare, and trust agent behavior in mixed cooperation–competition settings?**
> 

---

## 1️⃣ Core Instrumentation Philosophy

In the Mixed archetype:

> *We don’t just measure “who won” — we measure how they behaved and why they succeeded or failed.**
> 

Instrumentation must expose:

- strategic patterns
- cooperation dynamics
- exploitation vs sustainability tradeoffs

Without this, the archetype is a black box.

---

## 2️⃣ Mandatory Metric Categories (Mixed-Specific)

Instrumentation is grouped into **five layers of metrics**.

---

## 3️⃣ Episode-Level Metrics

Collected once per episode.

### Required metrics:

- Episode length
- Termination reason (horizon / collapse / stability / no agents)
- Total global resource change
- System outcome label (sustainable / collapsed / unstable)

Purpose:

- high-level environment behavior
- detecting tragedy-of-the-commons vs stability

---

## 4️⃣ Agent-Level Metrics (Core)

Tracked per agent across the episode.

### Required metrics:

- Cumulative reward
- Individual reward component
- Group reward contribution
- Relational/history reward contribution
- Survival time
- Action distribution (cooperate / extract / defend / conditional)

Purpose:

- identify dominant strategies
- compare agents fairly
- detect free-riders vs contributors

---

## 5️⃣ Interaction & Social Metrics (Key for “Mixed”)

These metrics are what make this archetype **impressive**.

### Required metrics:

- Cooperation rate over time
- Betrayal frequency
- Reciprocity score (did agent respond in kind?)
- Trust/reputation trajectory
- Pairwise interaction matrices (agent ↔ agent)

Purpose:

- visualize alliances
- show emergent strategy shifts
- support behavioral analysis

---

## 6️⃣ Resource & System Metrics

Tracked at step and episode level.

### Required metrics:

- Shared resource level over time
- Extraction vs contribution balance
- Efficiency metrics (output per unit resource)
- Collapse indicators / stress signals

Purpose:

- link individual behavior to system health
- explain why rewards changed

---

## 7️⃣ Event-Level Logs (Sparse but Semantic)

These are **high-signal events**, not raw data.

Examples:

- Resource collapse triggered
- Reputation threshold crossed
- Retaliation activated
- Stability condition reached
- Agent eliminated

Purpose:

- narrative reconstruction
- debugging
- explainability in dashboards

---

## 8️⃣ Visibility & Access Rules (Strict)

- Agents see **none** of these metrics directly
- Metrics access **full state**
- Metrics are read-only
- Metrics are deterministic

This preserves learning integrity.

---

## 9️⃣ Configurable Granularity

Via the config schema, users can control:

- logging frequency
- which metric groups are enabled
- storage verbosity

This allows:

- fast training runs
- deep analysis runs

without modifying environment logic.

---

## 10️⃣ Why This Matters for Your Resume

With this instrumentation, you can:

- show plots of cooperation over time
- compare strategies quantitatively
- explain agent behavior qualitatively
- demonstrate real MARL analysis skills

This is **far beyond** “reward curves”.

---

# Part 11: **Agent Export, Reuse & User Value**

This instantiates **Framework Step 11 (Compatibility & Adapter Boundary)** for the Mixed archetype.

It answers the most important product question:

> **“A user trained or discovered a strong agent. What can they actually *do* with it?”**
> 

If this part is weak, the whole project feels academic.

If this part is strong, your project feels **industry-grade**.

---

## 1️⃣ Core Philosophy

> **Agents are first-class artifacts, not disposable training byproducts.**
> 

An agent is not “just weights” — it is:

- a learned strategy
- conditioned on an environment archetype
- with measurable behavioral properties

---

## 2️⃣ What “Exporting an Agent” Means (Precisely)

Exporting an agent produces a **portable policy artifact** containing:

- trained policy parameters
- action & observation schema reference
- archetype compatibility metadata
- training configuration fingerprint
- performance summary (metrics snapshot)

Importantly:

- **no environment state**
- **no logs**
- **no training history leakage**

This keeps agents reusable and clean.

---

## 3️⃣ Reuse Modes (This Is Where It Gets Impressive)

A trained agent can be reused in **four distinct ways**.

---

### 1. Evaluation / Benchmarking Mode

Use case:

- Compare multiple trained agents in the *same* Mixed environment
- Run tournaments
- Rank strategies

Value:

- fair comparison
- reproducible leaderboards
- strategy profiling

---

### 2. Cross-Environment Transfer Mode

Use case:

- Take an agent trained in one Mixed config
- Evaluate it in a **modified config**
    - more scarcity
    - more asymmetry
    - stronger penalties

Value:

- robustness testing
- generalization analysis
- shows real ML understanding

This is a *huge* resume signal.

---

### 3. Population Seeding Mode

Use case:

- Inject a trained agent into a population of:
    - untrained agents
    - different strategies
    - random agents

Value:

- study influence
- observe dominance or collapse
- social dynamics experiments

This directly leverages your Mixed archetype.

---

### 4. Agent-as-a-Module Mode (Advanced)

Use case:

- Treat the agent as a black-box decision module
- Plug it into:
    - another simulation
    - a planner
    - a downstream system

Value:

- software reuse
- system design credibility
- aligns with real-world AI deployment thinking

---

## 4️⃣ Compatibility Guarantees

An exported agent guarantees:

- compatibility with:
    - the Mixed archetype
    - any environment instance whose config respects required bounds
- rejection (with explanation) if incompatible

This prevents:

- silent failures
- invalid comparisons
- misleading results

---

## 5️⃣ What Users Can Learn From a “Good” Agent

Users don’t just get a score — they get **insight**.

For each exported agent, the system can summarize:

- cooperation tendency
- risk profile
- exploitative vs sustainable behavior
- sensitivity to scarcity
- dependence on others’ behavior

This turns agents into **analyzable objects**, not just winners.

---

## 6️⃣ Why This Is a Big Deal (Zoom Out)

Most ML projects:

- train a model
- plot a curve
- stop

Your project:

- **creates agents**
- **compares strategies**
- **tests robustness**
- **supports reuse**
- **treats AI as a system component**

That is exactly how:

- research labs
- applied ML teams
- simulation-heavy companies
    
    think.
    