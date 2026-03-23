# Competitive Archetype

# Part 1: Core Tension & Purpose

## 1️⃣ Archetype Name
**Competitive Archetype**
(Zero-Sum Direct Competition)

---

## 2️⃣ Core Question
> **How do agents develop and adapt strategies to outperform rivals when every gain for one is a direct loss for another?**

---

## 3️⃣ Fundamental Tension

Three simultaneous pressures — no single strategy dominates:

1. **Exploitation vs Exploration** — commit to a known-good strategy, or probe for a better one against this opponent?
2. **Adaptation vs Stability** — a winning strategy gets countered; adapting too slowly loses, adapting too fast becomes predictable
3. **Aggression vs Risk Management** — aggressive moves yield large gains but expose the agent; conservative play preserves position but cedes ground

---

## 4️⃣ What Makes This Archetype "Competitive"

**Is:**
- Strictly zero-sum — total reward across all agents is constant per episode
- Direct — agents compete against each other, not just the environment
- Rank-based — outcome is relative standing, not absolute score

**Is NOT:**
- Cooperative in any dimension — no shared reward, no alliances
- Resource-sharing — no common pool to contribute to or deplete
- Reputation-based — trust and long-term social dynamics are not the mechanic here

---

## 5️⃣ Desired Emergent Behaviors

Not hardcoded — must emerge naturally:

- Dominant strategy discovery (finding locally optimal play)
- Counter-strategy development (adapting to beat the current leader)
- Cyclical dominance (rock-paper-scissors style rotations — no permanent champion)
- Bluffing and misdirection (feigning weakness or strength)
- Aggressive early play vs defensive consolidation tradeoffs
- Specialization against specific opponent archetypes

If none of these appear, the environment has failed.

---

## 6️⃣ What This Archetype Is NOT Trying to Model

- Cooperation, alliances, or coalition formation
- Shared resources or tragedy-of-the-commons dynamics
- Trust, reputation, forgiveness, or communication
- Any outcome where multiple agents simultaneously "win"

This is a **pure competitive system**, not a social one.

---

## 7️⃣ Success Criteria

The archetype is working if:

- Agents learn strategies meaningfully better than random
- No single policy dominates all opponents across all configs
- Strategy cycles emerge — the champion that beats A loses to B loses to C
- League Elo ratings diverge meaningfully — agents don't cluster at the same rating
- Agents trained against one opponent pool behave differently against a new pool


# Part 2: Layer Emphasis (Knob Settings)

## Why we do this first
These choices decide what the environment *feels* like, what behaviors can emerge, and how hard training will be.

---

## Primary Layers — HIGH

### ✅ Layer 4: Interaction Structure — HIGH
The archetype lives here. Every action directly affects at least one other agent's position. Gains and losses are coupled — there is no acting in isolation.

### ✅ Layer 1: Information Structure — HIGH
Unlike Mixed, information asymmetry is a *primary* mechanic here, not secondary. Agents who can infer opponent strategy faster gain a decisive edge. Partial observability is what makes counter-play non-trivial.

---

## Secondary Layers — MEDIUM

### ⚠️ Layer 6: Soft Constraints & Incentives — MEDIUM
Constraints exist but are simpler than Mixed — no cooperation cost, no betrayal penalty. Incentives are purely positional: reward what leads to winning, penalize what leads to losing. No prohibitions.

### ⚠️ Layer 2: Temporal Structure — MEDIUM
History matters for counter-strategy development. Agents need enough memory to detect opponent patterns and adapt. But delayed consequences are less important than in Mixed — outcomes are more immediate here.

### ⚠️ Layer 3: State Hierarchy — MEDIUM
Global state (scores, rankings, episode progress) plus agent-local state (resources, position, action history). No relational state in the Mixed sense — there is no trust to track, only competitive standing.

---

## Minimal Layers — LOW

### 🟢 Layer 5: Power & Role Asymmetry — LOW
All agents are structurally symmetric in V1. Role asymmetry (e.g. attacker vs defender) is deferred — it would add complexity before the core zero-sum dynamics are validated.

### 🟢 Layer 7: Uncertainty & Meta Dynamics — LOW
Minimal controlled randomness only — enough to prevent deterministic exploitation, not enough to dominate outcomes. No non-stationarity or evolving rules in V1.

---

## Summary

| Layer | Emphasis |
|---|---|
| 4. Interaction Structure | HIGH |
| 1. Information Structure | HIGH |
| 6. Soft Constraints & Incentives | MEDIUM |
| 2. Temporal Structure | MEDIUM |
| 3. State Hierarchy | MEDIUM |
| 5. Power & Role Asymmetry | LOW |
| 7. Uncertainty & Meta Dynamics | LOW |

**Key difference from Mixed:** Information Structure moves from MEDIUM → HIGH. Interaction Structure stays HIGH but the mechanism shifts from resource coupling to direct score competition. Temporal layer drops from MEDIUM-HIGH → MEDIUM because reputation is not a mechanic.



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
- head-to-head competition (2 agents)
- small-group tournaments (3–10 agents)
- population-level dynamics later

---

### Agent Symmetry (Important Design Choice)
For the **first Competitive archetype**:
- All agents are **structurally symmetric**
- Same:
    - action space
    - observation schema
    - reward structure (form, not value)
- Differences in behavior must **emerge from learning**, not roles

This keeps:
- competition fair and comparable
- analysis clean
- learning stable

(Role asymmetry — e.g. attacker vs defender — is explicitly deferred to later.)

---

### Agent Identity
Each agent has:
- a unique ID
- persistent identity across the episode

Identity matters for:
- tracking competitive standing
- opponent action history
- score attribution

Unlike Mixed, identity here is about **tracking who is winning**, not who to trust.

---

## 2️⃣ Agent Capabilities (High-Level, Not Actions Yet)

At this stage we only define **capability categories**, not concrete actions.

Each agent is capable of:
- acting every timestep (unless eliminated)
- directly affecting their own competitive position
- partially observing opponent behavior and standing
- accumulating private history for opponent modeling

Agents are **autonomous**:
- no central controller
- no enforced coordination
- no communication channel

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
- early termination is possible via elimination or score threshold

---

### Action Timing
- All agents act **simultaneously** at each step
- No turn-taking in V1

This ensures:
- true strategic uncertainty
- need for anticipation, not just reaction
- conflict resolution handled entirely by the environment

---

### Persistence Across Steps
Within an episode:
- agent identity persists
- scores and rankings persist and accumulate
- action history persists (for opponent modeling)

Across episodes:
- the environment resets fully
- scores reset to zero
- agents do not carry state between episodes (learned policy parameters persist only in training context)

---

## 4️⃣ Agent Termination (Individual-Level)

An agent may terminate before the episode ends due to:
- score/resource dropping to zero (elimination)
- reaching an unrecoverable state

After termination:
- the agent no longer acts
- its final score is frozen and counted in rankings
- remaining agents continue competing

This allows:
- elimination dynamics
- last-agent-standing outcomes
- progressive dominance


# Part 4: State & Observation Definition

> This part answers:
> **What exists in the world, and what does each agent get to see?**

---

## 1️⃣ Global State (What the World Actually Contains)

The global state is the complete internal truth of the environment.
For the Competitive archetype, state is structured into **three tiers** — but the content of each tier is fundamentally different from Mixed.

---

### A) Global Shared State

Visible to the environment only (not directly to agents unless exposed).

Includes:
- Current timestep
- Global score/ranking table (all agents' current scores)
- Episode progress indicators (steps remaining, score gap to leader)
- Global constraints (score caps, action limits)

Purpose:
- Define the competitive landscape all agents operate within
- Enable rank-awareness without exposing private strategy
- Support termination conditions (score threshold, elimination)

**Key difference from Mixed:** There is no shared resource pool. Global state tracks *standings*, not *resources*.

---

### B) Agent-Local State (Private)

Each agent has its own private slice of state.

Includes:
- Own current score
- Own accumulated resources or budget (action capacity)
- Own action history (rolling window)
- Own reward history (rolling window)
- Private counters or cooldowns

Purpose:
- Enable independent strategy optimization
- Support self-assessment and risk management
- Allow divergence in tactics across agents

---

### C) Opponent History State (Replaces Relational/Trust State)

In Mixed, relational state tracked **trust and cooperation** between pairs.
In Competitive, there is no trust — but **opponent modeling** is critical.

This tier tracks, per agent-pair:
- Recent actions taken by each opponent (observable window only)
- Win/loss outcome history between this pair
- Frequency of opponent action types (inferred pattern, not hidden intent)

Purpose:
- Enable counter-strategy development
- Support adaptation to specific opponent styles
- Make history matter without introducing cooperation mechanics

**Key difference from Mixed:** This tier is about **predicting opponents to beat them**, not managing relationships.

---

## 2️⃣ Observation Model (What an Agent Sees)

Each agent receives an **observation** — a projection of state, not the state itself.
Observations are partial by design (Information Layer is HIGH here).

---

### A) Public Observation
Same for all agents.

Includes:
- Current timestep
- Current global rankings (scores visible to all)
- Number of active agents remaining

Purpose:
- Ground agents in the competitive landscape
- Allow rank-aware strategy ("I am 2nd, leader has X more points")

---

### B) Private Self Observation
Visible only to the agent.

Includes:
- Own score and rank
- Own recent actions and their outcomes
- Own resource/budget level
- Own reward history (rolling window)

Purpose:
- Support self-optimization
- Enable risk and resource assessment

---

### C) Partial Opponent Observation
Limited information about others — this is the **primary strategic challenge**.

Includes:
- Opponents' scores (public, visible to all)
- Opponents' **most recent actions only** (not full history)
- Aggregate action frequency stats (not per-identity breakdown)

**What agents do NOT see:**
- Opponents' private resources or budgets
- Opponents' full action history
- Opponents' reward values
- The environment's internal resolution logic

Purpose:
- Force inference and opponent modeling
- Prevent trivial best-response computation
- Create the information asymmetry that makes competition non-trivial

---

## 3️⃣ What Agents Do NOT See (By Design)

Agents do **not** see:
- Full global state
- Other agents' private resources or strategy state
- True reward functions of others
- Future events or score thresholds

This ensures:
- No omniscience
- No trivially dominant strategies
- Realistic competitive uncertainty

---

## 4️⃣ Observation Stability Guarantees

The Competitive archetype guarantees:
- Observation **structure is fixed** across all timesteps
- Fields do not appear or disappear mid-episode
- Missing or masked information is represented explicitly (e.g. zeroed), not removed

This is critical for:
- Learning stability
- Fair cross-agent comparison
- Reproducibility

---

## 5️⃣ Information Asymmetry (High — by design)

Unlike Mixed where information asymmetry was MEDIUM, here it is **HIGH** (set in Part 2).

This means:
- Agents see opponents' scores but not their strategies
- Agents see recent actions but not historical patterns directly
- Agents must **infer** opponent intent from limited signals

This is what makes counter-strategy non-trivial and cyclical dominance possible.

---

## 6️⃣ Logging vs Observation (Reiterated)

Important boundary:
- Full state is always available to logs and metrics
- Observations are the **only** agent inputs — no leakage from logs to agents
- This separation is enforced by the environment, not by convention


# Part 5: Action Space Definition

> This part answers:
> **What can an agent do to influence the environment?**

---

## 1️⃣ Core Action Philosophy

In the Competitive archetype:

> **Actions represent strategic intent, not guaranteed outcomes.**

Agents may:
- attempt to grow their own position
- attempt to attack an opponent's position
- attempt to defend against incoming attacks
- attempt high-risk plays for large score swings

The **environment decides** what actually happens — the same action has different outcomes depending on opponent actions and current state.

---

## 2️⃣ Action Timing

- All agents submit **one action per timestep**
- Actions are **simultaneous**
- The environment resolves them jointly

This preserves:
- strategic uncertainty (you don't know what your opponent chose)
- true interaction coupling (outcomes depend on the combination of choices)
- conflict resolution by the environment, not the agents

---

## 3️⃣ Action Categories (High-Level)

We define **categories**, not low-level mechanics yet.
Each agent chooses **exactly one** action per step.

---

### A) Build Actions
Intent:
- Grow own score or resources independently of opponents

Properties:
- Reliable, predictable gain
- Does not directly affect opponents
- Lower ceiling than attack actions

This enables:
- steady accumulation strategies
- conservative play
- baseline competitive pressure

---

### B) Attack Actions
Intent:
- Reduce opponents' scores or resources, transferring advantage to self

Properties:
- **Untargeted in V1** — attack is directed at the environment, not a specific agent
- The environment distributes the effect proportionally across all non-defending opponents
- Outcome depends on how many opponents are defending
- Costs resources whether it succeeds or fails (attacker always pays)
- Targeted attacks (choose a specific opponent) are explicitly deferred to V2

This enables:
- Aggressive play
- Punishing passive opponents
- The attack vs defend strategic tension

---

### C) Defend Actions
Intent:
- Protect own position from incoming attacks

Properties:
- Negates or reduces attack effectiveness
- Low reward when not attacked (opportunity cost)
- Creates the strategic tension: attack vs defend vs build

This enables:
- risk management
- prediction of opponent behavior
- the rock-paper-scissors dynamic (build beats defend, defend beats attack, attack beats build)

---

### D) Gamble Actions
Intent:
- High-variance play — large gain or large loss depending on environment resolution

Properties:
- Outcome = agent's BUILD gain × multiplier drawn uniformly from [0.0, 2.5] (seeded RNG)
- Expected value ≈ BUILD × 1.25 — slightly positive bias to make it tempting
- Capped at 2.5× to prevent runaway gains
- Cannot be defended against — bypasses the clash resolution entirely
- Does not consume resources

This enables:
- Risk-seeking behavior under pressure
- Catching up when trailing
- Unpredictability as a strategy

---

## 4️⃣ Action Structure (Composite)

Each action is a **composite** of two fields:
```
Action:
  type:   discrete enum  — BUILD | ATTACK | DEFEND | GAMBLE
  amount: continuous [0, 1] — intensity (only meaningful for BUILD and ATTACK)
```

- `amount` controls how aggressively to build or attack (resource commitment)
- `amount` is ignored for DEFEND and GAMBLE (forced to 0.0 internally)
- This matches the proven structure from Mixed and keeps the adapter layer compatible

---

## 5️⃣ Action Constraints

Actions are subject to:
- Resource availability (cannot attack or build beyond own budget)
- Soft costs (over-committing resources on a failed attack leaves agent exposed)

Important:
- All action types are **always allowed**
- Undesirable actions are discouraged via **costs and outcomes**, not bans
- This preserves strategic diversity and prevents trivial dominant strategies

---

## 6️⃣ Invalid Action Handling

If an agent submits an invalid action:
- The environment resolves it **deterministically** — no randomness
- Examples:
    - amount out of [0,1] → clamped to nearest valid value
    - unknown type → converted to DEFEND (neutral fallback)
    - action beyond resource budget → scaled down proportionally

---

## 7️⃣ What Actions Do NOT Do

Actions do **not**:
- Directly modify state
- Directly change rewards
- Directly affect other agents

All influence flows through:
> **Action → Environment resolution → State transition → Reward**

---

## 8️⃣ Extensibility Guarantee

The action model is designed so that:
- New action types can be added later without breaking existing agents
- The composite (type + amount) structure is stable
- Observation structure remains compatible across V1 and future versions


# Part 6: Transition Dynamics

> This part answers:
> **Given the current state and all agents' actions, what actually happens next?**

---

## 1️⃣ Core Transition Philosophy

In the Competitive archetype:

> **Outcomes are direct, pairwise, and opponent-sensitive.**

Unlike Mixed where outcomes depend on group-level cooperation,
here outcomes depend on **what your specific opponents chose simultaneously**.

- A BUILD against a DEFEND yields different results than a BUILD against an ATTACK
- The same action can be strong or weak depending on the population composition
- History influences effectiveness but does not dominate it

---

## 2️⃣ Transition Cycle (One Timestep)

Each timestep follows this **fixed, deterministic sequence**:

1. Collect all agents' actions
2. Validate and normalize actions
3. Resolve GAMBLE actions (before pairwise resolution — their outcome is independent)
4. Resolve pairwise ATTACK vs DEFEND clashes
5. Resolve BUILD gains (uncontested, independent)
6. Update scores and resources (global + agent-local state)
7. Update opponent history state
8. Inject controlled noise
9. Deactivate eliminated agents
10. Advance timestep

This order is **fixed**. Changing it changes semantics.

---

## 3️⃣ Pairwise Resolution Logic (Core Mechanic)

Competitive actions are resolved **pairwise**, not collectively.

**ATTACK is untargeted in V1.** The attacker targets the environment — the environment distributes the attack effect proportionally across all non-defending opponents.

### Resolution matrix:

| Attacker \ Defender | DEFEND | BUILD | GAMBLE |
|---|---|---|---|
| **ATTACK** | Attack negated, attacker pays cost, no score gain | Attack succeeds, attacker gains, opponent loses | Partial success — reduced gain |
| **BUILD** | Build proceeds normally, no interaction | Both build independently, no interaction | Build proceeds normally |
| **DEFEND** | No gain, no loss (wasted if no attack incoming) | Defend was wasted — opportunity cost only | No gain |

This creates the **rock-paper-scissors cycle**:
- ATTACK beats BUILD (aggression beats passivity)
- DEFEND beats ATTACK (defense neutralizes aggression)
- BUILD beats DEFEND (productivity beats passive waiting)

No single action dominates. Cyclical dominance emerges from this structure.

---

## 4️⃣ GAMBLE Resolution

GAMBLE is resolved **before** pairwise clashes each step:
- Outcome = agent's BUILD gain × multiplier drawn uniformly from [0.0, 2.5], seeded RNG
- Expected value ≈ BUILD × 1.25 — slight positive bias makes it strategically tempting
- Capped at 2.5× — prevents runaway gains that would break competitive balance
- Cannot be defended against — bypasses clash resolution entirely
- Does not consume resources

This gives trailing agents a legitimate catch-up mechanism without breaking core balance.

---

## 5️⃣ History-Dependent Effectiveness (Temporal Layer)

The transition function is modulated by opponent history state using one rule in V1:

> **If ≥ 50% of an opponent's last `opponent_history_depth` actions were ATTACK, defending against them yields a +20% defense success bonus.**

This is the only history-dependent modifier in V1.

Effects:
- An agent that attacks repeatedly becomes **predictable** — opponents who defend against them get a bonus
- An agent that has only built recently is **underdefended** — attacking them yields higher success rate
- Switching strategy breaks the predictability penalty

More complex pattern detection (e.g. multi-pattern recognition, decay functions) is explicitly deferred to V2.

History modulates outcomes but never overrides the base resolution — it only adjusts margins.

---

## 6️⃣ Resource Constraints

Each agent has a resource budget that limits action intensity:
- BUILD and ATTACK consume resources proportional to `amount`
- Resources regenerate slowly each step (fixed regeneration rate)
- An agent that over-commits on a failed attack is left resource-poor and vulnerable
- DEFEND and GAMBLE do not consume resources

This ensures:
- Agents cannot sustain maximum aggression indefinitely
- Resource management is a genuine strategic dimension
- Elimination occurs naturally when resources are depleted

---

## 7️⃣ Stochasticity (Controlled, Low)

The Competitive archetype includes **limited, seeded randomness**:
- Small noise on BUILD and ATTACK outcomes (prevents brittle deterministic policies)
- GAMBLE outcome drawn from a bounded distribution
- All randomness is seeded and reproducible

Randomness is used only to:
- Prevent perfectly deterministic exploitation
- Break exact ties in simultaneous clashes
- Give GAMBLE its variance

Randomness **never dominates outcomes**. Skill and counter-strategy always matter more.

---

## 8️⃣ State Integrity & Invariants

After every transition, the environment guarantees:
- No agent has negative resources (floored at zero, triggers deactivation check)
- Scores are non-negative and bounded
- Opponent history state is consistent with actual actions taken
- No state change occurs after an agent is deactivated

If an invariant is violated, the environment resolves it deterministically — no randomness, no silent failure.


# Part 7: Reward Model

> This part answers:
> **Given what just happened, how does the environment judge each agent's outcome?**

---

## 1️⃣ Core Reward Philosophy

In the Competitive archetype:

> **Rewards must reflect not just how much you gained, but how well you gained it relative to your opponents.**

If rewards only measure absolute score gain, agents ignore opponents entirely and just spam BUILD.
If rewards only measure rank, the signal is too sparse for stable learning.

So rewards must be **relative and efficient by design** — you are judged against your opponents, not against an absolute standard.

---

## 2️⃣ Reward Decomposition (Three Components)

Each agent's reward at a timestep is composed of **three components**:

---

### A) Absolute Gain Component
Reflects:
- Raw score or resource gain from this timestep
- Whether the action succeeded (attack landed, build completed, gamble paid off)
- Penalized when a failed attack costs resources with no return

This incentivizes:
- Taking meaningful actions each step
- Avoiding reckless resource burning
- Basic competitiveness

---

### B) Relative Gain Component
Reflects:
- Change in this agent's rank or score gap relative to opponents
- Gaining while others also gain = low relative reward
- Gaining while others lose = high relative reward
- Losing ground relative to others = negative relative reward

**Important — fixed denominator rule:**
Relative gain is always computed against the **initial** number of agents at episode start, not the current active count. Eliminated agents count as permanently last-ranked. This keeps reward scale stable and consistent throughout the episode — the same score gap always produces the same relative reward regardless of how many eliminations have occurred.

This incentivizes:
- Targeting opponents when they are vulnerable
- Defending when you are the leader
- Catching up when trailing — falling further behind is punished

This is the **zero-sum signal** — the defining characteristic of this archetype.

---

### C) Efficiency Component
Reflects:
- Score gained per unit of resource spent this step
- An action that costs a lot but gains little is penalized
- An action that costs nothing but gains (BUILD with low amount) is rewarded proportionally

This incentivizes:
- Resource management as a genuine strategy
- Avoiding over-committing on low-probability attacks
- Calibrating GAMBLE and ATTACK amounts relative to expected gain

Without this component, agents would always commit maximum resources regardless of context.

---

The final reward is a **weighted combination**:
```
reward = w_absolute × absolute_gain
       + w_relative × relative_gain
       + w_efficiency × efficiency
       - penalty
```

Weights are:
- Fixed per environment instance
- Configurable via schema
- Known to the environment only — agents receive only the scalar reward

---

## 3️⃣ Reward Timing

The Competitive archetype uses **per-step rewards** with a **terminal bonus**:

- Per-step: small reward each timestep based on the three components above
- Terminal: rank-based bonus issued at the **final step of the episode** to all agents — both surviving and previously eliminated

**Terminal bonus + PettingZoo rule:**
The terminal bonus is held until the global episode done flag is set. It is not issued at the moment of individual agent elimination. The PettingZoo adapter handles this by issuing the terminal bonus to all agents — including eliminated ones — via a final step when the episode ends. This ensures consistent reward attribution and prevents timing conflicts with the adapter layer.

Rank-based terminal bonus scale:
- 1st place: largest bonus (configurable via `terminal_bonus_scale`)
- Last place: zero or small penalty
- Intermediate ranks: linearly interpolated

This prevents:
- Purely myopic play (per-step signal keeps agents active each step)
- Ignoring long-term position (terminal bonus rewards sustained dominance)

---

## 4️⃣ Penalties Instead of Prohibitions

Undesirable behavior is discouraged via penalties, not bans:

- Failed attack (defended) → attacker pays resource cost with no score gain
- Over-committing resources → efficiency component penalizes low return-on-spend
- Passive DEFEND when not under attack → opportunity cost (missed BUILD or ATTACK gain)

Actions are **never banned outright**. Penalties create the strategic tradeoffs.

---

## 5️⃣ Credit Assignment Clarity (for Humans)

Agents receive only the **scalar reward**.

Logs record all three components separately so humans can see:
- Why an agent's reward was high or low
- Whether it won through aggression, defense, or efficiency
- How rank changed contributed to reward at each step

This supports debugging, analysis, and strategy attribution without leaking decomposed signals to agents.

---

## 6️⃣ Reward Boundedness & Stability

The environment guarantees:
- All reward components are normalized (bounded to a fixed range)
- No runaway positive feedback loops — a dominant agent does not get unboundedly rewarded
- Reward scale is consistent across episodes regardless of N agents or config

This is essential for:
- MARL training stability
- Fair cross-policy comparison
- Meaningful Elo ratings in the league

---

## 7️⃣ What Rewards Do NOT Do

Rewards do **not**:
- Change state
- Override transition dynamics
- Signal opponent private state
- Inform agents of the decomposition breakdown

They only evaluate outcomes after the transition has fully resolved.


# Part 8: Termination Model

> This part answers:
> **When does an agent stop, and when does the world end?**

---

## 1️⃣ Core Termination Philosophy

In the Competitive archetype:

> **Termination reflects the completion of competitive resolution — when a winner is decided, or when continued play cannot change the outcome.**

Episodes end because:
- The time limit is reached (most common)
- One agent has eliminated all others
- Only one agent remains active

Unlike Mixed, there is **no systemic collapse** — the world does not fail.
Termination is always about **competitive resolution**, not system health.

---

## 2️⃣ Agent-Level Termination

An individual agent may terminate while others continue.

### Agent termination conditions:
- Resources drop to zero (eliminated — cannot act or influence the episode)
- Score drops below elimination threshold (configurable, can be disabled)

After agent termination:
- The agent no longer submits actions
- Its final score and rank are frozen and logged
- It counts as permanently last-ranked for relative gain computation
- Remaining agents continue competing

**Key difference from Mixed:** A terminated agent's state does NOT persist as a social influence (no reputation). It simply freezes. The competitive landscape shrinks.

---

## 3️⃣ Episode-Level Termination

The entire episode terminates when **any one** of the following holds.

**V1 implements exactly three termination conditions:**

---

### A) Maximum Horizon Reached — ✅ V1
- Fixed maximum timesteps, configurable via schema
- Most common termination in balanced matches
- Ensures all episodes are comparable in length
- Winner is determined by final scores at this point

---

### B) Elimination — One Agent Remains — ✅ V1
- All agents except one have been eliminated
- The surviving agent is the winner by default
- Episode ends immediately — no further steps needed

---

### C) No Active Agents Remain — ✅ V1
- All agents eliminated simultaneously (edge case)
- Episode ends with no winner — logged as a draw

---

### D) Dominance Threshold — ⛔ DEFERRED to V2
- One agent's score lead exceeds a configurable threshold
- Would allow early termination when outcome is decided
- **Not implemented in V1** — `dominance_margin` is disabled (set to 0) in all V1 configs
- Deferred because the threshold formula depends on `max_possible_score_per_step`
  which requires empirical calibration after the environment is running

---

## 4️⃣ Termination Reason Codes

V1 termination reason codes: `MAX_STEPS` | `ELIMINATION` | `NO_ACTIVE_AGENTS`

(`DOMINANCE` is reserved for V2.)

---

## 5️⃣ Final Rewards at Termination

Upon episode termination:
- A rank-based terminal bonus is issued to **all agents** at the final step
- This includes previously eliminated agents — they receive their terminal bonus
  at episode end, not at the step they were eliminated
- Per-step rewards already accumulated are not affected
- The terminal bonus is the primary signal for learning long-term competitive strategy

Termination ≠ success or failure by default — it is a ranked outcome.
A 2nd place finish is not a failure.

---

## 6️⃣ Post-Termination Behavior

After any termination condition fires:
- No further state transitions occur
- No further actions are processed
- Final scores and ranks are computed and logged
- Terminal rewards are issued
- Full state snapshot is captured for replay and auditing

Eliminated agents:
- Receive no further observations after elimination
- Are not given dummy observations — they are simply inactive
- Receive their terminal reward at episode end via the final step

---

## 7️⃣ Determinism & Transparency

All termination conditions are:
- State-based (no hidden triggers)
- Deterministic (same config + seed = same termination point)
- Logged explicitly with reason code

This ensures reproducibility, explainability, and fair benchmarking across runs.


# Part 9: Configuration Schema

> This part answers:
> **How does a user precisely describe this Competitive environment instance so it can be instantiated, reproduced, compared, and shared?**

---

## 1️⃣ Core Principle

> **A single configuration fully defines one Competitive environment instance.
> No hidden defaults. No implicit behavior.**

If two configs are identical (including seed), the environment behaves identically.

---

## 2️⃣ Top-Level Config Sections

A Competitive archetype config has **six sections**.

---

## 3️⃣ Environment Identity

Defines *what* this environment instance is.

Includes:
- `environment_type`: `"competitive"`
- `environment_version`
- `archetype`: `"zero_sum_direct"`
- `seed`

Purpose:
- Reproducibility
- Compatibility checks — prevents Mixed configs from being loaded as Competitive
- Experiment tracking

---

## 4️⃣ Population & Episode Parameters

Defines *who exists, for how long, and under what conditions*.

Includes:
- `num_agents` — number of agents (≥ 2)
- `max_steps` — maximum timesteps per episode
- `initial_score` — starting score for all agents (typically 0)
- `initial_resources` — starting resource budget per agent
- `resource_regeneration_rate` — resources recovered per step (float, ≥ 0)
- `elimination_threshold` — resource level at which an agent is eliminated (default 0.0)
- `dominance_margin` — **V1: always set to 0 (disabled).** Reserved for V2. Do not set to non-zero values in V1 configs.

These parameters shape:
- State initialization
- Elimination dynamics
- Episode length distribution

---

## 5️⃣ Layer Configuration (Knob Settings)

Defines how strongly each layer is applied.
All layers are present — config adjusts their intensity.

Includes:
- `information_asymmetry` — controls masking of opponent **scores and resources** (0.0 = full visibility, 1.0 = heavy masking)
- `opponent_history_depth` — how many past steps of opponent **action history** are tracked (integer, ≥ 1)
- `opponent_obs_window` — how many recent opponent actions are visible in the observation (must be ≤ `opponent_history_depth`)
- `history_sensitivity` — how strongly past opponent patterns modulate transition outcomes (0.0–1.0)
- `incentive_softness` — degree to which bad actions are penalized vs hard-blocked (0.0–1.0, prefer high)
- `uncertainty_intensity` — noise magnitude on action outcomes (capped low for V1, 0.0–0.3)
- `gamble_variance` — spread of GAMBLE action outcomes (0.0 = deterministic multiplier, 1.0 = maximum spread within [0.0, 2.5])

**Clarification — two observation controls are orthogonal, not overlapping:**
- `information_asymmetry` controls what agents can see about opponent **scores and resources** (state-level masking)
- `opponent_obs_window` controls the depth of opponent **action history** visible in observations
- These affect different parts of the observation and do not conflict. A config with `information_asymmetry=0` (full score visibility) and `opponent_obs_window=1` (only 1 step of action history) is valid and unambiguous.

---

## 6️⃣ Reward Weights

Defines how the three reward components are combined.

Includes:
- `absolute_gain_weight` — weight for raw score/resource gain this step
- `relative_gain_weight` — weight for rank/score-gap improvement vs opponents
- `efficiency_weight` — weight for score-gained per resource-spent
- `terminal_bonus_scale` — multiplier on the rank-based terminal bonus at episode end
- `penalty_scaling` — multiplier applied to failed-attack and over-commit penalties

These are **environment parameters**, not agent choices.
Agents receive only the scalar reward — the breakdown is logged for humans.

---

## 7️⃣ Agent Configuration

Defines per-agent settings. V1 uses homogeneous agents.

Includes:
- `observation_memory_steps` — how many past steps appear in the agent's own history window (must be ≤ `opponent_history_depth`)

---

## 8️⃣ Instrumentation Flags

Defines what metrics to collect and at what frequency.
This section enables metrics — it does not define them (that is Part 10).

Includes:
- `enable_step_metrics` — per-step reward, action, score deltas
- `enable_episode_metrics` — episode summary (length, termination reason, final rankings)
- `enable_event_log` — semantic events (elimination, etc.)
- `step_log_frequency` — log every N steps (1 = every step)

---

## 9️⃣ Validation Rules (Mandatory)

Before instantiation, the config must be validated:
- `num_agents` ≥ 2
- `elimination_threshold` ≥ 0 and ≤ `initial_resources`
- `dominance_margin` must be 0 in V1 — non-zero value raises a validation error
- `opponent_obs_window` ≤ `opponent_history_depth`
- `observation_memory_steps` ≤ `opponent_history_depth`
- At least one reward weight must be positive
- `resource_regeneration_rate` ≥ 0
- `gamble_variance` ∈ [0, 1]
- `uncertainty_intensity` ≤ 0.3 (capped for V1)

Invalid configs are rejected before instantiation — no silent failures.

---

## 1️⃣0️⃣ Why This Schema Matters

With this schema:
- Users can create many Competitive variants (aggressive, defensive, high-noise, tournament-style)
- Experiments are reproducible from a single config ID
- UI forms map directly to config fields
- Trained agents from different configs can be fairly compared via Elo
- Cross-archetype experiments are unambiguous — `environment_type` field prevents accidental mixing


# Part 10: Instrumentation & Metrics

> This part answers:
> **What do we measure so humans can understand, compare, and trust agent behavior in zero-sum competitive settings?**

---

## 1️⃣ Core Instrumentation Philosophy

In the Competitive archetype:

> **We don't just measure "who won" — we measure how dominance was established, how strategies evolved, and whether competition was genuine.**

Instrumentation must expose:
- Strategy composition (attack vs defend vs build vs gamble ratios)
- Rank dynamics over time (who was leading when)
- Counter-strategy emergence (did strategy shifts correlate with opponent behavior)
- Resource efficiency (did agents use their budget well)

Without this, the archetype produces only a leaderboard — not insight.

---

## 2️⃣ Mandatory Metric Categories

Instrumentation is grouped into **five layers of metrics**.
All metrics access full state. Agents see none of them. All are read-only and deterministic.

---

## 3️⃣ Episode-Level Metrics

Collected once per episode.

Required metrics:
- `episode_length` — actual steps taken
- `termination_reason` — `MAX_STEPS` | `ELIMINATION` | `DOMINANCE` | `NO_ACTIVE_AGENTS`
- `final_rankings` — ordered list of agent IDs by final score
- `final_scores` — dict of agent_id → final score
- `score_spread` — difference between 1st and last place at episode end
- `winner_id` — agent with highest final score (null on draw)
- `num_eliminations` — how many agents were eliminated before episode end

Purpose:
- High-level competitive outcome tracking
- Tournament result logging
- Detecting runaway dominance vs close matches

---

## 4️⃣ Agent-Level Metrics (Core)

Tracked per agent across the episode.

Required metrics:
- `cumulative_reward` — total scalar reward across all steps
- `absolute_gain_component` — cumulative absolute gain reward component
- `relative_gain_component` — cumulative relative gain reward component
- `efficiency_component` — cumulative efficiency reward component
- `terminal_bonus` — rank-based bonus received at episode end
- `survival_steps` — how many steps the agent was active
- `final_rank` — rank at episode end (1 = winner)
- `action_distribution` — fraction of steps spent on BUILD | ATTACK | DEFEND | GAMBLE
- `resources_spent` — total resources consumed across episode
- `attack_success_rate` — fraction of ATTACK actions that succeeded (not defended)

Purpose:
- Identify dominant strategies
- Compare agents fairly across episodes
- Detect over-aggressive or over-passive policies

---

## 5️⃣ Competitive Dynamics Metrics (Key for "Competitive")

These metrics are what make this archetype's dashboard distinctive.

Required metrics:
- `rank_trajectory` — per-step rank for each agent (list over time)
- `score_gap_trajectory` — per-step gap between 1st and 2nd place
- `rank_volatility` — number of rank-change events per episode (higher = more contested)
- `attack_ratio_over_time` — rolling fraction of ATTACK actions across all agents
- `defend_ratio_over_time` — rolling fraction of DEFEND actions across all agents
- `strategy_switch_count` — per agent, how many times dominant action type changed
- `counter_strategy_correlation` — did an agent's action distribution shift after being attacked?

Purpose:
- Show whether genuine strategic adaptation occurred
- Detect rock-paper-scissors cycles in the league
- Prove that competition is non-trivial (not just "always attack wins")

---

## 6️⃣ Resource & Efficiency Metrics

Tracked at step and episode level.

Required metrics:
- `resource_level_over_time` — per-agent resource trajectory
- `resource_efficiency` — score gained per resource unit spent (per agent)
- `gamble_outcome_distribution` — distribution of GAMBLE payoffs across episode
- `over_commit_events` — steps where an agent committed more resources than available (clamped)

Purpose:
- Link resource management to competitive outcomes
- Show whether GAMBLE is being used strategically or recklessly
- Identify resource-starved agents before elimination

---

## 7️⃣ Event-Level Logs (Sparse but Semantic)

High-signal events only — not raw step data.

Event types:
- `AGENT_ELIMINATED` — agent reaches zero resources; logs step, agent_id, final score, final rank
- `RANK_CHANGE` — any agent's rank changes; logs step, agent_id, old_rank, new_rank
- `ATTACK_SUCCEEDED` — attack landed; logs step, attacker_id, score gained
- `ATTACK_DEFENDED` — attack blocked; logs step, attacker_id, defender_id, cost paid
- `GAMBLE_RESOLVED` — logs step, agent_id, outcome, resources before/after
- `DOMINANCE_TRIGGERED` — dominance threshold crossed (if enabled); logs step, leader_id, score_gap

Purpose:
- Narrative reconstruction of how a match unfolded
- Debugging transition logic
- Explainability in dashboards ("agent_3 eliminated at step 87 after failed attack")

---

## 8️⃣ Visibility & Access Rules (Strict)

- Agents see **none** of these metrics
- Metrics access **full global state**
- Metrics are **read-only** — they never mutate state
- Metrics are **deterministic** — same config + seed = same metric values

This preserves training integrity and ensures reproducible analysis.

---

## 9️⃣ Configurable Granularity

Via the config schema (Part 9), users control:
- Which metric categories are enabled
- Step logging frequency (`step_log_frequency`)
- Event log verbosity

This allows lightweight training runs (metrics off or reduced frequency) and deep analysis runs (all metrics, every step) without touching environment logic.


# Part 11: Agent Export, Reuse & Adapter Boundary

> This part answers two questions:
> **"How does the Competitive environment connect to external tools without being owned by them?"**
> **"What can a user actually do with a trained Competitive agent?"**

---

## 1️⃣ Core Philosophy

> **The framework owns the semantics. Adapters own the translation.**
> **Agents are first-class artifacts, not disposable training byproducts.**

The environment logic never bends to a library.
A trained agent is not "just weights" — it is a learned competitive strategy with measurable behavioral properties.

---

## 2️⃣ Adapter Boundary

The Competitive archetype exposes a clean boundary to the outside world.

**What the framework guarantees at the boundary:**
- Step-based interaction (reset, step, done)
- Per-agent observations (structured, stable format)
- Per-agent actions (composite: type + amount)
- Per-agent rewards (scalar only — no component leakage)
- Termination signals (per-agent and episode-level)
- Deterministic resets given seed
- Full instrumentation hooks (metrics, events)

**What crosses the boundary — nothing else:**
- No internal state leaks to trainers
- No reward decomposition exposed to agents
- No opponent history state accessible externally

---

## 3️⃣ Declared Adapters

Four adapter types are declared for the Competitive archetype:

### 1. MARL Adapter (PettingZoo)
Purpose:
- Wrap `CompetitiveEnvironment` in PettingZoo `ParallelEnv` interface
- Enable standard MARL training libraries to interact with the environment

Constraints:
- Adapter may reshape observation dicts to flat arrays
- Adapter may not modify reward values or add hidden state
- Agent IDs are mapped to PettingZoo agent handles deterministically

### 2. Evaluation Adapter
Purpose:
- Run frozen policies against each other
- Benchmark strategies in controlled tournaments
- Compare Competitive agents against Mixed-trained agents (cross-archetype)

### 3. UI / Backend Adapter
Purpose:
- Stream per-step metrics via WebSocket
- Accept config from frontend
- Control run lifecycle (start, stop, status)

### 4. Export / Import Adapter
Purpose:
- Save trained policy to filesystem artifact
- Reload and run against new opponents or configs
- Transfer agents between Competitive config variants

---

## 4️⃣ What "Exporting an Agent" Means

Exporting produces a **portable policy artifact** containing:
- Trained policy parameters (`policy.pt`)
- Action & observation schema reference
- Archetype compatibility tag (`"competitive"`)
- Training config fingerprint (SHA-256 of config)
- Performance summary (Elo rating, win rate, action distribution, avg rank)

Does **not** contain:
- Environment state
- Opponent histories
- Training logs or episode data

This keeps agents clean, portable, and comparable.

---

## 5️⃣ Reuse Modes

A trained Competitive agent can be reused in four ways:

### 1. Evaluation / Benchmarking
- Run tournaments between multiple trained agents in the same Competitive config
- Produce ranked leaderboards with Elo ratings
- Profile strategy: attack-heavy vs defensive vs adaptive

### 2. Cross-Environment Transfer
- Load a Competitive agent into a modified config (higher noise, more agents, different resource regen)
- Measure performance drop — this is robustness testing
- Reveals whether the agent learned a general strategy or overfit to specific conditions
- Also: load a Competitive agent into a Mixed environment to study how pure competition strategies fare against cooperative-competitive dynamics

### 3. Population Seeding
- Inject a trained Competitive agent into a population of untrained or random agents
- Observe dominance curves — how quickly does it establish a lead?
- Inject multiple trained agents with different strategies and observe which dominates

### 4. Agent-as-Module
- Treat the exported agent as a black-box decision module
- Plug into downstream systems, planners, or other simulations
- The agent only requires a valid Competitive observation dict as input — no environment coupling

---

## 6️⃣ Compatibility Guarantees

An exported Competitive agent guarantees:
- Compatibility with any Competitive environment instance whose config respects the training config bounds
- Explicit rejection (with error message) if loaded into a Mixed environment without cross-archetype adapter
- Observation schema is stable — the agent will not silently receive malformed inputs

---

## 7️⃣ What Users Can Learn From a Trained Agent

Users don't just get a score — they get **behavioral insight**:
- **Attack tendency** — what fraction of steps does it choose ATTACK?
- **Counter-strategy rate** — how quickly does it shift action distribution after being attacked?
- **Risk profile** — how often does it GAMBLE, and under what score conditions?
- **Resource efficiency** — does it win by spending efficiently or by volume?
- **Rank sensitivity** — does its strategy change when it's leading vs trailing?

This turns trained agents into **analyzable competitive strategies**, not just winners.


---

# Baseline Agents (V1)

Four baseline agents are defined for V1. These are required before any PPO training begins — they bootstrap the league and provide meaningful opponents for early training runs.

| Agent | Behavior |
|---|---|
| `RandomAgent` | Random action type and amount each step |
| `AlwaysAttack` | Always chooses ATTACK with amount=0.5 |
| `AlwaysBuild` | Always chooses BUILD with amount=0.5 |
| `AlwaysDefend` | Always chooses DEFEND |

**Deferred:** A reactive/conditional baseline (Competitive equivalent of TitForTat) requires targeted attacks to be meaningful — deferred to V2 alongside targeted ATTACK.

---

# Ambiguities Found & Resolved

During the sanity check phase, the following contradictions and underspecifications were identified and resolved. These resolutions are **authoritative** — they take precedence over any ambiguous wording elsewhere in this document.

1. **ATTACK targeting unspecified:** Parts 5 and 6 implied direct opponent targeting but never defined the mechanism. Resolution: ATTACK is untargeted in V1 — effect is distributed proportionally across all non-defending opponents. Targeted attacks deferred to V2.

2. **GAMBLE outcome distribution unspecified:** Both parts said "high variance, bounded distribution" without defining it. Resolution: GAMBLE outcome = agent's BUILD gain × multiplier drawn uniformly from [0.0, 2.5], seeded. Expected value ≈ BUILD × 1.25. Capped at 2.5× to prevent runaway gains.

3. **History sensitivity mechanism unquantified:** Part 6 described predictability penalties conceptually but gave no concrete rule. Resolution: If ≥ 50% of an opponent's last `opponent_history_depth` actions were ATTACK, defending against them yields a +20% defense success bonus. Only history-dependent modifier in V1. More complex pattern detection deferred to V2.

4. **Relative gain denominator inconsistency:** Relative gain was defined against "all active agents" — denominator shrinks on elimination, causing reward scale drift. Resolution: Relative gain always computed against initial agent count. Eliminated agents count as permanently last-ranked.

5. **Terminal bonus timing conflicts with PettingZoo:** PettingZoo issues rewards at the step an agent is done, not at episode end. Resolution: Terminal bonus is held until global episode done flag. Issued to all agents — including eliminated ones — at the final step via the PettingZoo adapter.

6. **Dominance threshold formula used undefined variable:** Formula referenced `max_possible_score_per_step` which was never defined anywhere. Resolution: Dominance termination fully deferred to V2. `dominance_margin` is disabled (= 0) in all V1 configs. V1 termination conditions: `MAX_STEPS`, `ELIMINATION`, `NO_ACTIVE_AGENTS` only.

7. **`information_asymmetry` vs `opponent_obs_window` overlap:** Two config parameters both affected opponent observability with no clear boundary. Resolution: `information_asymmetry` controls masking of opponent scores and resources. `opponent_obs_window` controls depth of opponent action history visible. Orthogonal — no conflict.

8. **No baseline agents defined:** Unlike Mixed which had 5 baselines defined before training, Competitive had none. Resolution: Four baselines defined for V1 — RandomAgent, AlwaysAttack, AlwaysBuild, AlwaysDefend. Reactive/conditional baseline deferred to V2.