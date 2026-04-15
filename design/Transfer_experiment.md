# Transfer Experiment Feature

## 1️⃣ What This Is

A research tool that takes a trained agent from one archetype's league and runs it inside a different archetype's environment. Measures how well a policy generalizes beyond its training distribution.

This directly implements the "Population Seeding" and "Agent-as-Module" reuse modes defined in Part 11 of every archetype's design doc.

---

## 2️⃣ Core Research Questions

- Does a cooperative specialist (high role stability, low idle rate) contribute meaningfully in a Mixed resource-sharing environment?
- Does a competitive aggressive agent cause faster system collapse when dropped into a cooperative task queue?
- Does a Mixed tit-for-tat agent adapt to cooperative or competitive environments?
- Is high Elo rating in one environment correlated with above-random performance in others?

---

## 3️⃣ The Observation Dimension Problem

Every trained policy expects a fixed-size observation vector defined at training time and stored in `metadata.json` as `obs_dim`. Each environment produces a different shaped observation vector.

| Archetype | Obs Dim |
|---|---|
| Mixed (Resource Sharing) | 33 |
| Competitive (Head-to-Head) | varies by config |
| Cooperative | 8 + 4T + K(T+1) |

A policy trained in Mixed expects 33 floats. The Cooperative environment might produce 45 floats. They are incompatible by default.

**Resolution: truncate/pad with zero**

- If target obs dim > source policy expected dim: truncate to expected dim
- If target obs dim < source policy expected dim: zero-pad to expected dim
- The mismatch is logged and displayed to the user
- This is intentional and honest — we are testing raw generalization, not engineered compatibility
- The UI clearly states: "Observation spaces differ. Transfer uses truncation/padding to bridge the gap. Results reflect raw policy generalization."

This is the scientifically correct approach for V1. Learned projection matrices or shared obs subsets are deferred to V2.

---

## 4️⃣ Baseline Comparison

Every transfer experiment runs two agents simultaneously:
1. The transferred source policy (truncate/pad applied)
2. A random baseline agent in the same target environment

Results are always shown relative to the random baseline. This answers: "Did the transferred policy do better than chance?"

Metrics shown:
- **Mixed target:** cooperation_rate, shared_pool_final, termination_reason
- **Competitive target:** final_rank, final_score, elimination_step
- **Cooperative target:** completion_ratio, effort_utilization, idle_rate

---

## 5️⃣ UI Design

**Location:** Champion tab on the League page, per archetype. Below the existing "Run Robustness" section.

**Section title:** "Transfer Experiment"

**Form:**
- Target environment selector: Resource Sharing / Head-to-Head / Cooperative Task Arena (excludes source archetype)
- Target config selector: dropdown of available configs for target archetype
- Episodes: number input (default 5, max 20)
- Seed: number input (default 42)
- Run Transfer button

**Results panel (shown after run completes):**
- Source agent: member ID, archetype, strategy label, Elo rating
- Target environment: archetype, config hash
- Obs dim mismatch note: "Source expects Xd, target produces Yd — zero-padded/truncated"
- Results table: metric | transferred agent | random baseline | vs baseline
- Interpretation line: "Transferred agent performed X% above/below random baseline"
- Link: "View full report →" → /research/transfer/[report_id]

---

## 6️⃣ Storage

Transfer reports saved to:
storage/reports/transfer_{src_archetype}{tgt_archetype}{hash}_{timestamp}/

summary.json        — full experiment metadata and results
`summary.json` contains:
- `source_archetype` — mixed / competitive / cooperative
- `source_member_id` — league snapshot ID
- `source_obs_dim` — what the policy expects
- `source_strategy_label` — cluster label if available
- `source_elo` — rating at time of transfer
- `target_archetype` — the environment it was dropped into
- `target_config_hash` — config used
- `target_obs_dim` — what the environment produced
- `obs_mismatch_strategy` — "truncate" or "pad"
- `episodes` — number of episodes run
- `seed` — random seed
- `transferred_results` — per-episode metrics
- `baseline_results` — random agent per-episode metrics
- `transferred_mean` — mean primary metric across episodes
- `baseline_mean` — mean primary metric for random baseline
- `vs_baseline_delta` — transferred_mean - baseline_mean
- `vs_baseline_pct` — percentage above/below baseline

---

## 7️⃣ Backend Design

### New files:

**simulation/transfer/transfer_runner.py**
- `run_transfer_experiment(source_member_id, source_archetype, target_archetype, target_config, episodes, seed)` → results dict
- Loads source policy from correct league registry
- Instantiates target environment from config
- Computes obs mismatch and applies truncate/pad
- Runs N episodes with transferred policy
- Runs N episodes with random baseline
- Returns structured results

**backend/api/routes_transfer.py**
- `POST /api/transfer/run` — starts transfer experiment as background task
- `GET /api/transfer/status/{transfer_id}` — polls status
- `GET /api/transfer/reports` — lists all transfer reports
- `GET /api/transfer/reports/{report_id}` — report detail

### Existing files modified (minimal):
- `backend/main.py` — register transfer router
- `frontend/src/lib/api.ts` — add transfer API functions (additive only)
- `frontend/src/app/league/page.tsx` — add TransferExperiment component to each Champion tab
- `frontend/src/app/research/page.tsx` — add Transfer filter option

---

## 8️⃣ Frontend Design

**New components:**

`frontend/src/components/TransferExperiment.tsx`
- Form: target env selector, config selector, episodes, seed, run button
- Results panel: source/target info, obs mismatch note, results table, interpretation, report link
- Loading state while experiment runs (polls status endpoint)
- Dark theme, teal accent

**New pages:**

`frontend/src/app/research/transfer/[report_id]/page.tsx`
- Full transfer report detail
- Source agent card (archetype, member ID, strategy label, Elo)
- Target environment card (archetype, config)
- Obs mismatch explanation
- Results comparison table: transferred vs baseline, all episodes
- Interpretation summary

---

## 9️⃣ What This Is NOT

- Not a training mechanism — the source policy is never updated
- Not a fair comparison — obs truncation/padding means the agent is operating blind on some dimensions
- Not a deployment tool — results are research artifacts only
- Not guaranteed to produce meaningful results — a policy trained in one env may produce completely random behavior in another, and that itself is a valid finding

---

## 🔟 Explicitly Deferred to V2

- Learned observation projection (train a small projection matrix to map target obs → source obs space)
- Shared observation subset (define common features across all archetypes)
- Multi-agent transfer (inject transferred agent into population, measure influence)
- Fine-tuning on target environment after transfer
- Transfer leaderboard across all archetype pairs


---

## Ambiguities Found & Resolved

During the sanity check phase, the following gaps and underspecifications were identified and resolved. These resolutions are authoritative.

1. **Primary metric undefined per target archetype:** The results table referenced `transferred_mean` as a single scalar but never defined what metric it represents. Resolution: primary metric is archetype-specific — Mixed target → `cooperation_rate` [0,1], Competitive target → normalized rank (1/num_agents for rank 1, descending), Cooperative target → `completion_ratio` [0,1]. Higher is always better across all three.

2. **Background task status stages undefined:** The transfer endpoint was described as a background task with no defined stages. Resolution: status progresses through `pending → running_transfer → running_baseline → saving → done → error`. Frontend polls every 2 seconds and displays current stage.

3. **Source member selector vs champion only:** The design referenced `source_member_id` as a parameter implying any league member could be transferred, but the UI was placed on the Champion tab. Resolution: V1 transfers the current champion only — no member selector in the UI. `source_member_id` is determined automatically from the champion endpoint. Member-level transfer selection deferred to V2.

4. **Transfer reports not distinguishable from other report types:** The Research page filter needed a way to distinguish transfer reports from robustness and eval reports. Resolution: `summary.json` includes `report_type: "transfer"` field. Research page adds "Transfer" as a filter option alongside "Robustness" and "Strategy".

5. **Config dropdown shows all configs regardless of target archetype:** The target config selector would show Mixed, Competitive, and Cooperative configs mixed together. Resolution: config dropdown filters by `identity.environment_type` matching the selected target archetype — same pattern as cooperative benchmark config filtering already implemented in league/page.tsx.