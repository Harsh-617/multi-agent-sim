# Defining the framework of the environment

### 1) Define the Core Entities
Decide what the basic “objects” are in your system:
- Environment
- Agent
- Episode / Step
- Policy
- Experiment (run configuration + results)

---

### 2) Define the Environment Contract
Think of this as a **legal contract** between:
- the **environment**
- the **agents**
- the **rest of the system (training, UI, logging)**
If an environment follows this contract, **everything else can work with it**.
---
## What an “Environment Contract” actually is
It answers exactly this:
> “If I am an agent (or a trainer), what can I *rely on* the environment to do?”
It is **not**:
- how the environment works internally
- what domain it models
- what algorithms are used
It is only:
- **what goes in**
- **what comes out**
- **when**
- **in what structure**
---
## The Environment Contract has 5 REQUIRED parts
Every environment **must** support all 5.
---
## 1️⃣ Environment Lifecycle
The environment must define **how an episode begins and ends**.
### Required guarantees:
- The environment can be **reset** to a clean initial state
- Reset produces **initial observations** for all active agents
- Reset is **deterministic given a seed**
### Conceptually:
- “Start a new world”
- “Clear all history”
- “Spawn agents and initial conditions”
This defines what an **episode** is.
---
## 2️⃣ Step Interface (Core Interaction Loop)
The environment must support a **step operation**.
A step means:
- agents provide actions
- the environment processes them
- the world advances by one timestep
### The environment must return:
For **each agent**:
- observation (what the agent now sees)
- reward (feedback signal)
- done flag (is this agent finished?)
- info (optional metadata, diagnostics)
And also:
- whether the **global episode is finished**
This is the *heart* of the contract.
---
## 3️⃣ Observation Specification
The environment must declare:
- what form observations take
- per-agent, not global
- before training begins
Key rules:
- Observation ≠ state
- Observations can be:
    - partial
    - private
    - noisy
- Observation structure is **stable** across steps
Agents must never guess the format.
---
## 4️⃣ Action Specification
The environment must declare:
- what actions are valid
- per-agent
- before training begins
Key rules:
- Invalid actions must be handled deterministically
- Action space is fixed per agent role/type
- Actions are interpreted **only** by the environment
Agents never modify state directly
---
## 5️⃣ Termination Semantics
The environment must define:
- when an agent is “done”
- when the episode is “done”
Important:
- An agent can terminate before others
- The environment must define what happens to:
    - terminated agents
    - remaining agents
This prevents ambiguity in multi-agent settings.
---
## What the contract explicitly does NOT include
Very important:
The Environment Contract does **not** define:
- rewards logic details
- learning algorithms
- archetypes
- layers
- UI
- storage
- PettingZoo
- RLlib
- PPO / DQN
Those come **later**.
This contract is **domain-agnostic**.

---

### 3) Define the Data Model (State vs Observation)
# Define **State vs Observation Model**
This step exists to answer **one core question**:
> “What does the world actually contain, and what is each agent allowed to see?”
If you don’t separate these properly, **everything breaks later** (rewards, layers, MARL stability).
---
## Core Principle (lock this in)
> **State is the truth of the world.
Observation is an agent’s limited view of that truth.**
They are **not the same** and must **never be mixed**.
---
## 1️⃣ Global State (Environment Truth)
The **state** represents everything that exists in the environment at a given timestep.
### Properties of State
- Complete
- Authoritative
- Internal to the environment
- Never directly visible to agents
- Modified only by the environment’s transition logic
### Examples (conceptual, not domain-specific)
- Resource quantities
- Prices
- Timers
- Hidden variables
- Global flags
- Agent-internal data (even if private)
> If something affects outcomes, it belongs in the **state**.
---
## 2️⃣ Agent Observation (What an agent sees)
An **observation** is a *projection* of the global state.
Each agent receives its **own observation**.
### Properties of Observations
- Per-agent
- Partial (by default)
- Possibly noisy
- Possibly delayed
- Fixed structure across timesteps
Agents **never** see the full state unless explicitly designed to.
---
## 3️⃣ Observation Function (Critical Concept)
For each agent, the environment defines an **observation function**:
> `observation = f(state, agent_id, time)`
This function controls:
- What information is exposed
- What is hidden
- How information is transformed (noise, delay, masking)
This is where:
- information asymmetry
- private knowledge
- hidden variables
    come from.
---
## 4️⃣ Types of Information in Observations
The framework must support **three categories** of information:
### 1. Public Information
- Same for all agents
- Derived from global state
- Example: public scores, time step
### 2. Private Information
- Visible to one agent only
- Example: private inventory, memory, role-specific data
### 3. Hidden Information
- Exists in state
- Visible to **no agents**
- Example: latent events, future shocks
This distinction is essential for advanced environments.
---
## 5️⃣ Stability Guarantees (Very Important)
The environment must guarantee:
- Observation **structure does not change** across timesteps
- Shape and data types are fixed
- Missing information is represented explicitly (not removed)
This prevents agents from “cheating” by inferring structure changes.
---
## 6️⃣ State Mutability Rules
Only the environment may:
- Create state
- Modify state
- Destroy state
Agents:
- Cannot write to state
- Can only influence it via actions
This enforces clean causality.
---
## 7️⃣ Logging vs Observation (Do NOT confuse these)
Important distinction:
- **Logs** can see full state (for analysis)
- **Agents** never see logs
This allows:
- debugging
- dashboards
- explainability
without leaking information to agents.

---

### 4) Define Action Model
This step answers one fundamental question:
> **“What is an agent allowed to do to influence the environment?”**
Everything about power, strategy, and interaction **flows through actions**.
---
## Core Principle (lock this in)
> **Agents do not change the world directly.
They submit actions.
The environment interprets those actions.**
This preserves control, safety, and reproducibility.
---
## 1️⃣ Action as an Intent, Not an Effect
An **action** represents *what an agent intends*, not what actually happens.
Example:
- Action: “request resource”
- Outcome: may fail, succeed, or be delayed
The **environment** decides the outcome.
This separation is critical for:
- conflict resolution
- uncertainty
- soft constraints
---
## 2️⃣ Per-Agent Action Spaces
Each agent has an **action space** defined *before training begins*.
Key rules:
- Action space is **agent-specific**
- Action space is **fixed for the episode**
- Different roles may have different action spaces
Agents must never guess what actions exist.
---
## 3️⃣ Action Space Types (Framework-Level)
The framework must support (at minimum):
### 1. Discrete Actions
- Finite set of choices
- Example: {move, wait, negotiate}
### 2. Continuous Actions
- Real-valued parameters
- Example: price, quantity, allocation %
### 3. Composite Actions
- Structured actions with multiple fields
- Example: (action_type, target_id, amount)
The environment declares which type is used.
---
## 4️⃣ Simultaneity vs Turn-Based Actions
The environment must define **when actions are applied**:
### Simultaneous
- All agents act
- Environment resolves conflicts together
### Turn-Based
- Agents act in sequence
- Order is deterministic or state-driven
This choice affects:
- fairness
- strategy
- learning stability
It must be explicit.
---
## 5️⃣ Invalid Action Handling (Mandatory)
The environment must define **deterministic handling** of invalid actions.
Examples:
- Ignore action
- Clamp values
- Apply penalty
- Replace with no-op
Random handling is **not allowed**.
This ensures:
- reproducibility
- stable training
- debuggability
---
## 6️⃣ Action Constraints (Hard vs Soft)
The framework distinguishes:
### Hard Constraints
- Action is not allowed
- Environment blocks or replaces it
### Soft Constraints
- Action is allowed
- Environment applies cost, delay, or risk
Soft constraints are preferred for advanced environments.
---
## 7️⃣ No Direct State Access Rule
Agents:
- Cannot read full state
- Cannot modify state
- Cannot affect other agents directly
All influence flows through:
> **Action → Environment → State Transition**
This enforces clean causality.

---

### 5) Define Transition / Dynamics Rules
Specify:
- How state updates happen
- Conflict resolution strategy (if two agents do incompatible things)
- Randomness / stochastic events (how randomness is injected + seeded)

---

### 6) Define Reward Model
Specify:
- Individual rewards vs shared rewards vs mixed
- When rewards are computed (per-step vs end-of-episode)
- Penalties, costs, soft constraints
# Define the **Reward Model**
This step answers the question:
> **“Given what happened, how does the environment evaluate it?”**
Rewards are **evaluation**, not control.
They do **not** change the world — they judge it.
---
## Core Principle (lock this in)
> **Rewards do not cause outcomes.
They score outcomes after the fact.**
State transitions already happened in Step 5.
Step 6 only interprets those outcomes.
---
## 1️⃣ Reward as a Function (Conceptual)
Every environment must define a reward function:
> **Reward = g(Previous State, Actions, New State)**
Key points:
- Rewards are derived from:
    - what changed
    - what actions were taken
- Rewards are **computed after** the transition
- Rewards never affect the transition itself
This separation prevents reward hacking via state mutation.
---
## 2️⃣ Per-Agent vs Shared Rewards
The framework must support:
### 1. Individual Rewards
- Each agent gets its own reward
- Encourages selfish optimization
### 2. Shared Rewards
- All agents receive the same reward
- Encourages cooperation
### 3. Mixed Rewards
- Combination of:
    - personal reward
    - group/global reward
This choice defines the **interaction archetype** later.
---
## 3️⃣ Reward Timing
The environment must declare **when rewards are given**:
- Per-step rewards
- Sparse rewards (only at end)
- Delayed rewards (computed later)
Reward timing is explicit and stable.
---
## 4️⃣ Reward Scale & Stability Guarantees
The environment must guarantee:
- Reward ranges are known or bounded
- No unbounded reward explosions
- Consistent scale across episodes
This is critical for:
- learning stability
- fair comparison
- benchmarking
---
## 5️⃣ Costs, Penalties, and Soft Constraints
The reward model must support:
- Costs (negative rewards)
- Risk penalties
- Opportunity costs
Important:
- Penalties are preferred over hard bans
- Actions are usually allowed but “expensive”
This enables emergent strategies.
---
## 6️⃣ Credit Assignment Clarity
In multi-agent settings:
- It must be clear **why** a reward was given
- Attribution logic must be deterministic
This does **not** mean explainability to agents,
but explainability to **humans and logs**.
---
## 7️⃣ Rewards Are Observed, Not Controlled
Agents:
- Receive reward values
- Cannot modify reward logic
- Cannot predict hidden reward components perfectly
Rewards may depend on:
- hidden state
- future outcomes
- stochastic events
That’s allowed and intentional.

---

### 7) Define Termination Model
Specify:
- When an episode ends
- Per-agent termination vs global termination
- Timeouts, failure states, win states
# Define the **Termination Model**
This step answers the question:
> **“When does an episode end, and what does ‘done’ actually mean?”**
In multi-agent systems, termination must be **explicit**, **unambiguous**, and **fair**.
---
## Core Principle (lock this in)
> **Termination is a decision of the environment, not the agents.**
Agents may cause conditions that *lead* to termination,
but only the environment declares it.
---
## 1️⃣ Two Levels of Termination
The framework must support **both**:
### 1. Agent-Level Termination
An individual agent is “done” when:
- it is eliminated
- it exhausts resources
- it reaches a terminal condition
Other agents may continue.
---
### 2. Episode-Level Termination
The entire environment episode ends when:
- a global objective is reached
- a failure condition occurs
- a maximum timestep is reached
- no active agents remain
Both levels must be supported independently.
---
## 2️⃣ Termination Conditions (Explicit)
Termination conditions must be:
- deterministic
- state-based
- explicitly declared
Examples:
- Time limit reached
- Resource pool exhausted
- Threshold crossed
- Stable equilibrium detected
No hidden or implicit termination.
---
## 3️⃣ Post-Termination Behavior (Critical)
The environment must define:
- what happens to terminated agents
- whether they:
    - receive no further observations
    - receive dummy observations
    - receive final rewards
And:
- how remaining agents interact with a terminated agent’s state
    - removed
    - frozen
    - transferred to global state
This avoids undefined behavior.
---
## 4️⃣ Final Rewards on Termination
The termination model must specify:
- whether terminal rewards are issued
- how final scoring works
- whether delayed rewards resolve at termination
This is especially important for:
- sparse rewards
- long-horizon environments
---
## 5️⃣ Termination vs Failure vs Success
The framework must distinguish:
- **Termination** → episode ended
- **Success** → objective achieved
- **Failure** → objective failed
Termination does **not** imply success or failure by default.
This allows:
- neutral endings
- partial outcomes
- comparative evaluation
---
## 6️⃣ Determinism & Reproducibility
Given the same:
- initial state
- actions
- seed
Termination behavior must be identical.
No random episode endings.

---

### 8) Define the “Layer/Dimension” Knobs
# Define the **Layer / Dimension Framework**
This step answers the question:
> **“How do we systematically make environments simple or advanced without breaking the core contract?”**
The answer is **layers** (or dimensions).
---
## Core Principle (lock this in)
> **Layers do not replace the core environment.
They parameterize and modulate it.**
That means:
- The environment contract (Steps 2–7) is **fixed**
- Layers only **constrain, enrich, or transform** behavior within that contract
---
## What a “Layer” Is (Framework Definition)
A **layer** is a cross-cutting dimension that:
- affects multiple environment components simultaneously
- is configurable (on/off or intensity)
- is consistent across archetypes
Layers are **orthogonal axes of complexity**, not features.
---
## What Layers Are Allowed to Affect
Each layer may affect **only** these parts:
- State
- Observation
- Action interpretation
- Transition logic
- Reward logic
- Termination logic
Layers:
- cannot bypass the environment
- cannot give agents direct state access
- cannot violate determinism guarantees
---
## The 7 Framework-Level Layers (Declared, Not Implemented)
At framework level, you only declare their **existence and scope**.
### 1️⃣ Information Structure Layer
Controls:
- what parts of state appear in observations
- public vs private vs hidden information
- noise, masking, delay
---
### 2️⃣ Temporal Structure Layer
Controls:
- whether history matters
- delayed effects
- memory, reputation, trust
---
### 3️⃣ State Hierarchy Layer
Controls:
- organization of state into global / group / agent-local
- shared vs private state partitions
---
### 4️⃣ Interaction Structure Layer
Controls:
- how agent actions couple
- externalities
- shared vs conflicting objectives
---
### 5️⃣ Power & Role Asymmetry Layer
Controls:
- agent roles
- permission differences
- who can affect what
---
### 6️⃣ Soft Constraints & Incentives Layer
Controls:
- costs instead of bans
- risk vs reward trade-offs
- penalties, delays, reputation loss
---
### 7️⃣ Uncertainty & Meta Dynamics Layer
Controls:
- stochastic events
- non-stationarity
- environment adaptation over time
---
## Layer Activation Model (Important)
Each layer has:
- a **baseline** (minimal effect)
- a **configuration** (parameters)
- an **intensity dial** (conceptual)
For any archetype:
- all layers exist
- only some are emphasized
- others remain minimal but valid
This ensures:
- comparability
- extensibility
- controlled complexity
---
## Layers vs Archetypes (Clarify This)
- **Layers** = dimensions of complexity
- **Archetypes** = specific configurations of layers
Archetypes do **not** introduce new mechanics.
They select and tune layers.
---
## What This Step Does NOT Do
We do **not**:
- design layer internals
- assign specific parameters
- bind layers to any domain
- code anything
We only:
- declare the layers
- define what they are allowed to affect

---

### 9) Define Configuration Schema
This step answers the question:
> **“How do we describe an environment instance in a precise, reproducible, machine-readable way?”**
The configuration schema is the **bridge** between:
- paper design
- environment instantiation
- backend / UI / experiments
---
## Core Principle (lock this in)
> **Every environment instance must be fully describable by a single configuration object.**
If it’s not in the config:
- it must be deterministic
- or it must not exist
No hidden magic.
---
## 1️⃣ Purpose of the Configuration Schema
The schema exists to:
- instantiate environments
- reproduce experiments
- compare results
- drive UI forms
- pass configs from frontend → backend → simulator
It is **not** code.
It is **declarative**.
---
## 2️⃣ High-Level Structure of a Config
Every environment config must contain **five sections**.
### 1. Environment Identity
Defines *what* is being instantiated.
Includes:
- environment name / type
- version
- archetype identifier (conceptual)
- seed(s)
Purpose:
- reproducibility
- compatibility checks
---
### 2. Core Environment Parameters
Defines:
- fixed parameters that shape the world
- sizes, limits, constants
Examples (abstract):
- max_steps
- number_of_agents
- resource_caps
These parameters affect:
- state initialization
- transition dynamics
- termination
---
### 3. Layer Configuration
Defines:
- which layers are active
- their intensity / settings
Important:
- All layers exist
- Config decides how strongly they apply
This is where:
- information asymmetry
- uncertainty
- power imbalance
    are configured.
---
### 4. Agent Configuration
Defines:
- agent identities or roles
- observation scopes
- action permissions
- reward attribution (if customized)
This section must support:
- homogeneous agents
- heterogeneous agents
- role-based agents
---
### 5. Instrumentation & Logging
Defines:
- what metrics to collect
- logging frequency
- verbosity
- artifact storage flags
This ensures:
- consistent dashboards
- comparable runs
---
## 3️⃣ Determinism & Completeness Rules
The schema must guarantee:
- Given the same config + seed:
    - the same environment is instantiated
    - the same behavior occurs (modulo stochasticity)
Rules:
- No implicit defaults that affect behavior
- Defaults must be explicit in the schema
- Randomness must be seed-controlled
---
## 4️⃣ Schema Validation Guarantees
Before instantiation:
- config must be validated
- incompatible settings rejected
- missing required fields rejected
This prevents:
- invalid environments
- silent bugs
- impossible archetype combinations
---
## 5️⃣ Schema Is Framework-Level, Not Archetype-Level
Important distinction:
- The **schema shape is fixed**
- Archetypes only populate it differently
This allows:
- UI reuse
- backend reuse
- clean experiment comparison

---

### 10) Define Instrumentation & Logging (Framework-Level)
This step answers the question:
> **“What must every environment expose so we can analyze, compare, debug, and trust results?”**
Instrumentation is **not optional** in a serious system.
If something can’t be measured, it can’t be evaluated.
---
## Core Principle (lock this in)
> **Instrumentation observes the environment without influencing it.**
Logging must:
- never change state
- never leak hidden information to agents
- never affect transitions or rewards
---
## 1️⃣ What Instrumentation Is For
The framework-level goals of instrumentation are:
- debugging environment logic
- understanding agent behavior
- comparing agents fairly
- enabling dashboards
- ensuring reproducibility
This is for **humans and systems**, not agents.
---
## 2️⃣ Mandatory Logging Categories
Every environment must support logging at **four levels**.
---
### 1. Episode-Level Metrics
Collected once per episode.
Examples (abstract):
- total episode reward (per agent and/or global)
- episode length
- termination reason
- success / failure flags
Purpose:
- high-level comparison
- benchmarking
- convergence tracking
---
### 2. Step-Level Metrics
Collected at each timestep.
Examples:
- per-agent rewards
- key state deltas
- action summaries
- constraint violations (if any)
Purpose:
- learning curves
- instability detection
- reward shaping analysis
---
### 3. Agent-Level Metrics
Collected per agent across time.
Examples:
- cumulative reward
- action distribution
- policy entropy (if available)
- role-specific metrics
Purpose:
- agent comparison
- strategy evolution
- identifying dominant or degenerate agents
---
### 4. Event-Level Logs
Sparse, semantic events.
Examples:
- conflicts resolved
- rule changes
- penalties applied
- stochastic shocks
Purpose:
- explainability
- debugging complex interactions
- narrative reconstruction
---
## 3️⃣ Visibility Rules (Critical)
Instrumentation must obey strict visibility rules:
- Logs may access **full global state**
- Agents may access **only their observations**
- Logs are **never** fed back into the environment
This avoids:
- information leakage
- training contamination
- invalid experiments
---
## 4️⃣ Determinism & Traceability
The framework must guarantee:
- logs are deterministic given config + seed
- every logged value can be traced to:
    - state
    - action
    - transition
    - reward
    - termination
This enables:
- replay
- auditing
- reproducible analysis
---
## 5️⃣ Configurable Granularity
Instrumentation must be configurable via the **config schema**:
- enable / disable categories
- control frequency
- control verbosity
This allows:
- lightweight training runs
- heavy analysis runs
without changing environment logic.
---
## 6️⃣ Separation from Storage & UI
Important boundary:
- Instrumentation defines **what is exposed**
- Storage defines **where it goes**
- UI defines **how it is shown**
At framework level:
- you only define *what must be observable*

---

### 11) Define Compatibility Boundary (Adapters Later)
This step answers the question:
> **“How does our environment framework connect to external tools without being owned by them?”**
This is what keeps your project:
- future-proof
- library-agnostic
- cleanly architected
---
## Core Principle (lock this in)
> **The framework owns the semantics.
Adapters own the translation.**
Your environment logic never bends to a library.
Libraries adapt to *you*.
---
## 1️⃣ What an Adapter Boundary Is
An **adapter boundary** is a formal separation between:
- your internal environment framework
- external ecosystems (MARL libraries, trainers, UIs)
The boundary defines:
- what data crosses out
- what data comes back in
- in what structure
- at what time
Nothing else is allowed to leak.
---
## 2️⃣ What the Framework Guarantees at the Boundary
Your framework guarantees it can provide:
- step-based interaction
- per-agent observations
- per-agent actions
- per-agent rewards
- termination signals
- deterministic resets
- full instrumentation hooks
This is the **only promise** you make to the outside world.
---
## 3️⃣ What Adapters Are Allowed to Do
Adapters may:
- reshape data formats
- map concepts (e.g., agent IDs → library agent handles)
- batch or unbatch steps
- expose the environment to trainers
Adapters may **not**:
- modify environment logic
- alter reward semantics
- bypass observation constraints
- inject hidden state into agents
Adapters are translators, not collaborators.
---
## 4️⃣ Types of Adapters (Declared, Not Implemented)
At framework level, you only declare that these adapters **can exist**:
### 1. MARL Adapter
Purpose:
- connect environment to MARL algorithms
Examples:
- PettingZoo-style adapter
- RLlib-style adapter
---
### 2. Evaluation Adapter
Purpose:
- run frozen agents
- benchmark policies
- compare strategies
---
### 3. UI / Backend Adapter
Purpose:
- stream metrics
- accept configs
- control runs
---
### 4. Export / Import Adapter
Purpose:
- save trained agents
- reload policies
- transfer across environments
---
## 5️⃣ Direction of Control (Very Important)
Control always flows:
Framework → Adapter →External Tool
Never the reverse.
External tools:
- call your adapter
- never call your environment directly
This preserves:
- correctness
- auditability
- conceptual ownership
---
## 6️⃣ Stability Promise
Once defined:
- the **framework contract does not change**
- adapters can evolve independently
- new tools can be added without refactoring environments
This is what allows:
- experimentation
- scaling
- long-term maintenance