"""Unit tests for agent policies."""

from pathlib import Path

import pytest

from simulation.agents import (
    ALLOWED_POLICIES,
    create_agent,
    RandomAgent,
    AlwaysCooperateAgent,
    AlwaysExtractAgent,
    TitForTatAgent,
)
from simulation.agents.base import BaseAgent
from simulation.core.seeding import derive_seed
from simulation.envs.mixed.actions import Action, ActionType

# Policies that require external artifacts and should be skipped
# from the generic parametrized test when not available.
_ARTIFACT_POLICIES = {"ppo_shared", "league_snapshot"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42

SAMPLE_OBS = {
    "step": 5,
    "shared_pool": 80.0,
    "own_resources": 15.0,
    "num_active_agents": 3,
    "cooperation_scores": {"agent_1": 0.7, "agent_2": 0.3},
    "action_history": [{"type": "cooperate", "amount": 0.6}],
}


def _make_agent(cls: type[BaseAgent], agent_id: str = "agent_0", seed: int = SEED) -> BaseAgent:
    agent = cls()
    agent.reset(agent_id, seed)
    return agent


# ---------------------------------------------------------------------------
# Valid Action output
# ---------------------------------------------------------------------------

class TestAllAgentsProduceValidActions:
    """Every policy must return an Action with a valid ActionType."""

    @pytest.mark.parametrize("policy_name", sorted(ALLOWED_POLICIES - _ARTIFACT_POLICIES))
    def test_create_and_act(self, policy_name: str) -> None:
        agent = create_agent(policy_name)
        agent.reset("agent_0", SEED)
        action = agent.act(SAMPLE_OBS)

        assert isinstance(action, Action)
        assert isinstance(action.type, ActionType)
        if action.type in (ActionType.COOPERATE, ActionType.EXTRACT):
            assert 0.0 <= action.amount <= 1.0


# ---------------------------------------------------------------------------
# Per-policy behaviour
# ---------------------------------------------------------------------------

class TestAlwaysCooperate:
    def test_always_cooperates(self) -> None:
        agent = _make_agent(AlwaysCooperateAgent)
        for _ in range(10):
            action = agent.act(SAMPLE_OBS)
            assert action.type == ActionType.COOPERATE
            assert action.amount == pytest.approx(0.6)


class TestAlwaysExtract:
    def test_always_extracts(self) -> None:
        agent = _make_agent(AlwaysExtractAgent)
        for _ in range(10):
            action = agent.act(SAMPLE_OBS)
            assert action.type == ActionType.EXTRACT
            assert action.amount == pytest.approx(0.6)


class TestTitForTat:
    def test_cooperates_when_high_coop(self) -> None:
        obs = {**SAMPLE_OBS, "cooperation_scores": {"agent_1": 0.8, "agent_2": 0.6}}
        agent = _make_agent(TitForTatAgent)
        action = agent.act(obs)
        assert action.type == ActionType.COOPERATE
        assert action.amount == pytest.approx(0.6)

    def test_defends_when_low_coop(self) -> None:
        obs = {**SAMPLE_OBS, "cooperation_scores": {"agent_1": 0.2, "agent_2": 0.1}}
        agent = _make_agent(TitForTatAgent)
        action = agent.act(obs)
        assert action.type == ActionType.DEFEND

    def test_cooperates_when_no_scores(self) -> None:
        obs = {**SAMPLE_OBS, "cooperation_scores": {}}
        agent = _make_agent(TitForTatAgent)
        action = agent.act(obs)
        assert action.type == ActionType.COOPERATE


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestRandomAgentDeterminism:
    """Same seed + same observations → identical action sequence."""

    def test_same_seed_same_actions(self) -> None:
        seed = derive_seed(42, 0)

        agent_a = _make_agent(RandomAgent, seed=seed)
        agent_b = _make_agent(RandomAgent, seed=seed)

        for _ in range(20):
            a1 = agent_a.act(SAMPLE_OBS)
            a2 = agent_b.act(SAMPLE_OBS)
            assert a1.type == a2.type
            assert a1.amount == pytest.approx(a2.amount)

    def test_different_seed_different_actions(self) -> None:
        seed_a = derive_seed(42, 0)
        seed_b = derive_seed(42, 1)

        agent_a = _make_agent(RandomAgent, seed=seed_a)
        agent_b = _make_agent(RandomAgent, seed=seed_b)

        # Collect actions — with different seeds they should diverge
        actions_a = [agent_a.act(SAMPLE_OBS).type for _ in range(20)]
        actions_b = [agent_b.act(SAMPLE_OBS).type for _ in range(20)]
        assert actions_a != actions_b


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_policies_registered(self) -> None:
        expected = {"random", "always_cooperate", "always_extract", "tit_for_tat", "ppo_shared", "league_snapshot"}
        assert ALLOWED_POLICIES == expected

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(KeyError):
            create_agent("nonexistent_policy")
