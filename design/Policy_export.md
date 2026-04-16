# Policy Export Feature

## 1️⃣ What This Is

A download feature that packages a trained league agent into a self-contained,
ready-to-use Python artifact. Users can download any league member's policy and
use it in their own Python code without needing to understand the platform
internals.

This implements the "Export Policy" reuse mode defined in Part 11 of every
archetype's design doc and the "Export Policy for Simulation / Integration"
use case from the original Notion vision doc.

---

## 2️⃣ What Gets Downloaded

A `.zip` file containing three files:

### policy.py
A self-contained Python module with:
- The PolicyNetwork class (architecture embedded, no platform imports needed)
- Weights loaded from policy.pt at import time
- A clean `predict(observation)` function that takes a flat list/array and
  returns an action
- Full docstring with: archetype, member ID, Elo rating, strategy label,
  trained steps, config hash, export date, observation schema, action mapping,
  and usage example

### policy.pt
The raw PyTorch weights file — same file already stored in league storage.
Included so users can load it directly with torch.load() if they prefer.

### README.md
Plain English usage instructions covering:
- Requirements (Python 3.11+, torch>=2.0)
- How to use policy.py
- What the observation vector means (field names and ranges)
- What the action output means (type mapping, amount range)
- Reproducibility: config hash + seed to reproduce training

---

## 3️⃣ The predict() Function

Each archetype has a different action space so predict() returns different types:

### Mixed (Resource Sharing Arena)
```python
action_type, action_amount = predict(observation)
# action_type: int — 0=cooperate, 1=extract, 2=defend, 3=conditional
# action_amount: float [0.0, 1.0]
# observation: flat list/array of length obs_dim (default 33)
```

### Competitive (Head-to-Head Strategy)
```python
action_type, action_amount = predict(observation)
# action_type: int — 0=build, 1=attack, 2=defend, 3=gamble
# action_amount: float [0.0, 1.0]
# observation: flat list/array of length obs_dim (default 93)
```

### Cooperative (Cooperative Task Arena)
```python
task_type, effort_amount = predict(observation)
# task_type: int — 0..N-1=work on task type N, N=idle
# effort_amount: float [0.0, 1.0]
# observation: flat list/array of length obs_dim (default 40)
```

---

## 4️⃣ Where the Download Button Lives

### Champion tab (primary location)
Each archetype's Champion tab gets an "Export Champion Policy" button below
the champion info card. Downloads the current champion's policy zip.

### Ratings tab (secondary location)
Each member row in the Ratings table gets a small download icon button.
Downloads that specific member's policy zip.

---

## 5️⃣ Backend Design

### New endpoints (routes_transfer.py or new routes_export.py):

GET /api/export/{archetype}/champion
- Loads champion member_id from league registry
- Loads policy.pt and metadata.json from league storage
- Generates policy.py from template
- Generates README.md from template
- Returns zip file as application/zip response

GET /api/export/{archetype}/members/{member_id}
- Same but for a specific member

### Generation logic (simulation/export/policy_exporter.py):
- generate_policy_py(metadata, archetype) → str
  Renders the policy.py template with embedded architecture and weights path
- generate_readme(metadata, archetype) → str
  Renders the README.md template
- build_export_zip(member_dir, metadata, archetype) → bytes
  Packages policy.py + policy.pt + README.md into a zip

### No new storage — generates on the fly from existing files.

---

## 6️⃣ policy.py Template Structure

```python
"""
[docstring with all metadata]
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── Architecture ──────────────────────────────────────────────────────────────

class _SharedTrunk(nn.Module):
    # exact architecture from SharedPolicyNetwork in ppo_shared.py

class _ActionHead(nn.Module):
    # exact architecture

class _ValueHead(nn.Module):
    # exact architecture

class PolicyNetwork(nn.Module):
    # full network — exact copy of SharedPolicyNetwork

# ── Load weights ──────────────────────────────────────────────────────────────

_POLICY_DIR = Path(__file__).parent
_net = PolicyNetwork(obs_dim={obs_dim}, num_action_types={num_types})
_net.load_state_dict(torch.load(_POLICY_DIR / "policy.pt", map_location="cpu"))
_net.eval()

# ── Public API ────────────────────────────────────────────────────────────────

def predict(observation):
    """
    [per-archetype docstring]
    """
    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action_logits, amount_alpha, amount_beta, _ = _net(obs_tensor)
    action_type = int(action_logits.argmax(dim=-1).item())
    action_amount = float(
        (amount_alpha / (amount_alpha + amount_beta)).squeeze().item()
    )
    return action_type, action_amount
```

---

## 7️⃣ Observation Schema in README

Each archetype's README must document the observation vector fields so users
know what to pass to predict(). Pull from the archetype design docs:

### Mixed observation fields (33d default, 4 agents):
- [0] timestep (normalized)
- [1] shared_pool (normalized)
- [2] own_resources (normalized)
- [3] num_active_agents (normalized)
- [4-7] cooperation_scores per agent (normalized)
- [8-32] action_history (5 steps × 5 agents flattened)

### Competitive observation fields (93d default):
- Document from design/Competitive_archetype.md Part 4

### Cooperative observation fields (40d default, 3 task types, K=3):
- Document from design/Cooperative_archetype.md Part 4

---

## 8️⃣ What This Is NOT

- Not a deployment tool — policy.py is for research and simulation only
- Not a live inference service — batch/offline use only
- Not guaranteed to work in environments with different configs — obs_dim must match
- Not a general ML model — only valid inside multi-agent simulation contexts

---

## 9️⃣ Explicitly Deferred to V2

- ONNX export (works without PyTorch, wider compatibility)
- Weights embedded as base64 in policy.py (single-file, no policy.pt needed)
- Export from Research page report detail
- Batch export of all league members
- Version compatibility checking (warn if torch version differs)


---

## Ambiguities Found & Resolved

1. **policy.pt relative path fragile if files separated:** policy.py uses
   Path(__file__).parent to locate policy.pt. If user moves policy.py without
   policy.pt the import crashes. Resolution: document clearly in README —
   "Keep policy.py and policy.pt in the same directory." Acceptable for V1.

2. **Architecture copy will drift if ppo_shared.py changes:** The embedded
   PolicyNetwork class is a snapshot of the architecture at export time.
   Resolution: architecture is frozen for V1 — SharedPolicyNetwork will not
   change. If architecture changes in future, export format version is bumped.
   Noted in ADR.

3. **Cooperative predict() uses different return variable names:** Design doc
   showed task_type/effort_amount for Cooperative vs action_type/action_amount
   for others. Resolution: standardize all three to return (action_type,
   action_amount). README documents what action_type means per archetype —
   Mixed: 0=cooperate/1=extract/2=defend/3=conditional, Competitive:
   0=build/1=attack/2=defend/3=gamble, Cooperative: 0..N-1=task type/N=idle.

4. **No validation that policy.pt exists before generating zip:** Edge case
   where league member directory exists but policy.pt is missing would crash.
   Resolution: backend export endpoint returns HTTP 404 if policy.pt not found
   before attempting zip generation.

5. **Ratings tab download button scope:** Design mentioned download per row for
   all archetypes. Cooperative ratings tab rows confirmed to have same structure
   as RS and HH. Download button added to all three consistently.