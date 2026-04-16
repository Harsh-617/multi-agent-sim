"""Generate self-contained policy.py + README.md and package them into a zip.

Public API
----------
generate_policy_py(metadata, archetype) -> str
generate_readme(metadata, archetype) -> str
build_export_zip(member_dir, metadata, archetype) -> bytes
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-archetype constants
# ---------------------------------------------------------------------------

_ARCHETYPE_DEFAULTS: dict[str, dict[str, int]] = {
    "mixed":       {"obs_dim": 33, "num_action_types": 4},
    "competitive": {"obs_dim": 93, "num_action_types": 4},
    "cooperative": {"obs_dim": 40, "num_action_types": 4},
}

_ACTION_TYPE_DOCS: dict[str, str] = {
    "mixed":       "0=cooperate, 1=extract, 2=defend, 3=conditional",
    "competitive": "0=build, 1=attack, 2=defend, 3=gamble",
    "cooperative": "0..N-1=work on task type N, N=idle",
}

_OBS_SCHEMAS: dict[str, str] = {
    "mixed": """\
  [0]    timestep (normalized)
  [1]    shared_pool (normalized)
  [2]    own_resources (normalized)
  [3]    num_active_agents (normalized)
  [4-7]  cooperation_scores per agent (normalized)
  [8-32] action_history (5 steps x 5 agents flattened)""",
    "competitive": """\
  See design/Competitive_archetype.md Part 4 for full field documentation.
  Fields include own score, opponent scores, resource levels, and action history.
  Default obs_dim=93.""",
    "cooperative": """\
  See design/Cooperative_archetype.md Part 4 for full field documentation.
  Fields include backlog_norm, queue_norm per task type, effort history.
  Default obs_dim=40 (3 task types, K=3 agents).""",
}

_ACTION_MAPPING_DOCS: dict[str, str] = {
    "mixed": """\
  action_type   int   — 0=cooperate, 1=extract, 2=defend, 3=conditional
  action_amount float — Beta distribution mean, continuous in [0.0, 1.0]""",
    "competitive": """\
  action_type   int   — 0=build, 1=attack, 2=defend, 3=gamble
  action_amount float — Beta distribution mean, continuous in [0.0, 1.0]""",
    "cooperative": """\
  action_type   int   — 0..N-1=work on task type N, N=idle
  action_amount float — effort amount, continuous in [0.0, 1.0]""",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_obs_dim(metadata: dict, archetype: str) -> int:
    return int(metadata.get("obs_dim", _ARCHETYPE_DEFAULTS[archetype]["obs_dim"]))


def _get_num_action_types(metadata: dict, archetype: str) -> int:
    if "num_action_types" in metadata:
        return int(metadata["num_action_types"])
    if "action_mapping" in metadata:
        return len(metadata["action_mapping"])
    return _ARCHETYPE_DEFAULTS[archetype]["num_action_types"]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def generate_policy_py(metadata: dict, archetype: str) -> str:
    """Render a complete self-contained policy.py string.

    The returned string contains:
    - Full module docstring with all metadata and usage example
    - PolicyNetwork class (architecture copied from SharedPolicyNetwork)
    - Weight loading at import time
    - predict(observation) -> (action_type, action_amount) public function
    """
    obs_dim = _get_obs_dim(metadata, archetype)
    num_action_types = _get_num_action_types(metadata, archetype)

    member_id = metadata.get("member_id", "unknown")
    elo = metadata.get("rating", metadata.get("elo", "—"))
    strategy_label = metadata.get("strategy_label", metadata.get("label", "—"))
    trained_steps = metadata.get("training_steps", metadata.get("trained_steps", "—"))
    config_hash = metadata.get("config_hash", "—")
    export_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    obs_schema = _OBS_SCHEMAS.get(archetype, "")
    action_type_doc = _ACTION_TYPE_DOCS.get(archetype, "int")

    return f'''\
"""
Policy Export
=============
Archetype:     {archetype}
Member ID:     {member_id}
Elo Rating:    {elo}
Strategy:      {strategy_label}
Trained Steps: {trained_steps}
Config Hash:   {config_hash}
Export Date:   {export_date}

Observation schema ({obs_dim}d):
{obs_schema}

Action mapping:
  action_type   int   — {action_type_doc}
  action_amount float — continuous in [0.0, 1.0]

Usage example:
    from policy import predict
    obs = [0.0] * {obs_dim}  # replace with a real observation vector
    action_type, action_amount = predict(obs)

WARNING: Keep policy.py and policy.pt in the same directory.
         Moving policy.py without policy.pt will cause an ImportError.
"""

import torch
import torch.nn as nn
from pathlib import Path


# ── Architecture ─────────────────────────────────────────────────────────────


class PolicyNetwork(nn.Module):
    """Actor-critic network: discrete action head + Beta amount head."""

    def __init__(self, obs_dim: int, num_action_types: int = 4, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Discrete action type head
        self.action_type_head = nn.Linear(hidden, num_action_types)
        # Beta distribution params for amount (alpha, beta > 0)
        self.amount_alpha_head = nn.Linear(hidden, 1)
        self.amount_beta_head = nn.Linear(hidden, 1)
        # Value head
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.action_type_head(h)
        alpha = torch.nn.functional.softplus(self.amount_alpha_head(h)) + 1.0
        beta = torch.nn.functional.softplus(self.amount_beta_head(h)) + 1.0
        value = self.value_head(h).squeeze(-1)
        return logits, alpha, beta, value


# ── Load weights ─────────────────────────────────────────────────────────────

_POLICY_DIR = Path(__file__).parent
_net = PolicyNetwork(obs_dim={obs_dim}, num_action_types={num_action_types})
_net.load_state_dict(
    torch.load(_POLICY_DIR / "policy.pt", map_location="cpu", weights_only=True)
)
_net.eval()


# ── Public API ────────────────────────────────────────────────────────────────


def predict(observation):
    """Run inference on a single observation.

    Parameters
    ----------
    observation : list or array-like of length {obs_dim}
        Flat observation vector. See module docstring for field documentation.

    Returns
    -------
    action_type : int
        {action_type_doc}
    action_amount : float
        Continuous value in [0.0, 1.0].
    """
    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action_logits, amount_alpha, amount_beta, _ = _net(obs_tensor)
    action_type = int(action_logits.argmax(dim=-1).item())
    action_amount = float(
        (amount_alpha / (amount_alpha + amount_beta)).squeeze().item()
    )
    return action_type, action_amount
'''


def generate_readme(metadata: dict, archetype: str) -> str:
    """Render README.md for the policy export zip."""
    obs_dim = _get_obs_dim(metadata, archetype)

    member_id = metadata.get("member_id", "unknown")
    elo = metadata.get("rating", metadata.get("elo", "—"))
    strategy_label = metadata.get("strategy_label", metadata.get("label", "—"))
    trained_steps = metadata.get("training_steps", metadata.get("trained_steps", "—"))
    config_hash = metadata.get("config_hash", "—")
    seed = metadata.get("seed", "—")
    export_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    obs_schema = _OBS_SCHEMAS.get(archetype, "")
    action_mapping_doc = _ACTION_MAPPING_DOCS.get(archetype, "")

    return f"""\
# Policy Export — {archetype} / {member_id}

Exported: {export_date}

## Requirements

- Python 3.11+
- torch >= 2.0

```
pip install torch
```

## Usage

```python
from policy import predict

# Replace with your actual observation vector
obs = [0.0] * {obs_dim}

action_type, action_amount = predict(obs)
print(f"action_type: {{action_type}}, action_amount: {{action_amount:.4f}}")
```

## Observation Vector

Length: **{obs_dim}**

```
{obs_schema}
```

## Action Output

```python
action_type, action_amount = predict(observation)
```

```
{action_mapping_doc}
```

All three archetypes use the same `predict()` return signature `(action_type, action_amount)`.
This README documents what `action_type` means for the **{archetype}** archetype.

## Important

> **Keep `policy.py` and `policy.pt` in the same directory.**
> `policy.py` loads weights from `policy.pt` at import time using a path relative
> to its own location. Moving `policy.py` without `policy.pt` will cause an `ImportError`.

## Reproducibility

| Field         | Value            |
|---------------|------------------|
| Config Hash   | `{config_hash}` |
| Seed          | `{seed}` |
| Trained Steps | {trained_steps} |
| Elo Rating    | {elo} |
| Strategy      | {strategy_label} |
| Archetype     | {archetype} |

To reproduce training, use the same config hash and seed with the platform's
training pipeline.

## Limitations

This policy is for **research and simulation use only**:

- Not a deployment artifact — batch/offline use only
- Only valid for observations with `obs_dim={obs_dim}`
- Must be used with the same environment configuration as training
- Not guaranteed to generalise to environments with different configs
"""


def build_export_zip(member_dir: Path, metadata: dict, archetype: str) -> bytes:
    """Package policy.py + policy.pt + README.md into zip bytes.

    Parameters
    ----------
    member_dir : Path
        League member directory containing policy.pt and metadata.json.
    metadata : dict
        Parsed metadata.json content.
    archetype : str
        One of "mixed", "competitive", "cooperative".

    Returns
    -------
    bytes
        Raw zip file content for streaming.

    Raises
    ------
    FileNotFoundError
        If policy.pt is missing from member_dir.
    """
    policy_pt = member_dir / "policy.pt"
    if not policy_pt.exists():
        raise FileNotFoundError(
            f"policy.pt not found in {member_dir}. "
            "The league member exists but has no trained weights."
        )

    policy_py = generate_policy_py(metadata, archetype)
    readme_md = generate_readme(metadata, archetype)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("policy.py", policy_py)
        zf.write(str(policy_pt), "policy.pt")
        zf.writestr("README.md", readme_md)

    return buf.getvalue()
