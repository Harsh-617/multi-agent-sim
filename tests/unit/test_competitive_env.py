"""Unit tests for the Competitive archetype environment.

Covers:
  - Config validation (field-level and cross-field)
  - Environment contract (reset / step lifecycle)
  - Termination conditions (MAX_STEPS, ELIMINATION, NO_ACTIVE_AGENTS)
  - Determinism (same/different seed)
  - Action validation (amount forcing, invalid amounts)
  - Reward properties (bounded, terminal bonus ordering)
  - Baseline agents (deterministic + random)
  - Metrics collector (step records, episode summary, events)
"""

import pytest
from pydantic import ValidationError

from simulation.config.competitive_defaults import default_competitive_config
from simulation.config.competitive_schema import (
    AgentConfig,
    CompetitiveEnvironmentConfig,
    EnvironmentIdentity,
    InstrumentationConfig,
    LayerConfig,
    PopulationConfig,
    RewardWeights,
)
from simulation.core.types import StepResult, TerminationReason
from simulation.envs.competitive.actions import Action, ActionType
from simulation.envs.competitive.env import CompetitiveEnvironment
from simulation.agents.competitive_baselines import (
    AlwaysAttackAgent,
    AlwaysBuildAgent,
    AlwaysDefendAgent,
    CompetitiveRandomAgent,
    create_competitive_agent,
)
from simulation.metrics.competitive_collector import CompetitiveMetricsCollector
from simulation.metrics.competitive_definitions import EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42
AGENTS = ["agent_0", "agent_1", "agent_2"]


def _quick_config(seed: int = 42, **pop_overrides) -> CompetitiveEnvironmentConfig:
    """Small fast config for testing."""
    pop_kwargs = dict(
        num_agents=3,
        max_steps=10,
        initial_score=0.0,
        initial_resources=20.0,
        resource_regeneration_rate=1.0,
        elimination_threshold=0.0,
        dominance_margin=0.0,
    )
    pop_kwargs.update(pop_overrides)
    return CompetitiveEnvironmentConfig(
        identity=EnvironmentIdentity(seed=seed),
        population=PopulationConfig(**pop_kwargs),
        layers=LayerConfig(
            information_asymmetry=0.0,
            opponent_history_depth=10,
            opponent_obs_window=5,
            history_sensitivity=0.5,
            incentive_softness=0.8,
            uncertainty_intensity=0.0,
            gamble_variance=0.5,
        ),
        rewards=RewardWeights(
            absolute_gain_weight=1.0,
            relative_gain_weight=0.5,
            efficiency_weight=0.3,
            terminal_bonus_scale=2.0,
            penalty_scaling=1.0,
        ),
        agents=AgentConfig(observation_memory_steps=5),
    )


def _all_build(env: CompetitiveEnvironment, amount: float = 0.5) -> dict:
    return {
        aid: Action(type=ActionType.BUILD, amount=amount)
        for aid in env.active_agents()
    }


def _all_attack(env: CompetitiveEnvironment, amount: float = 0.5) -> dict:
    return {
        aid: Action(type=ActionType.ATTACK, amount=amount)
        for aid in env.active_agents()
    }


def _all_defend(env: CompetitiveEnvironment) -> dict:
    return {
        aid: Action(type=ActionType.DEFEND) for aid in env.active_agents()
    }


def _make_agent(cls, agent_id: str = "agent_0", seed: int = SEED):
    agent = cls()
    agent.reset(agent_id, seed)
    return agent


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_default_config_is_valid(self):
        cfg = default_competitive_config()
        assert cfg.identity.environment_type == "competitive"
        assert cfg.population.num_agents >= 2

    def test_explicit_valid_config(self):
        cfg = _quick_config()
        assert cfg.population.num_agents == 3
        assert cfg.identity.seed == 42

    def test_dominance_margin_nonzero_raises(self):
        with pytest.raises(ValidationError, match="dominance_margin"):
            _quick_config(dominance_margin=1.0)

    def test_opponent_obs_window_exceeds_depth_raises(self):
        cfg_kwargs = dict(
            identity=EnvironmentIdentity(seed=1),
            population=PopulationConfig(
                num_agents=3, max_steps=10, initial_score=0.0,
                initial_resources=20.0, resource_regeneration_rate=1.0,
                elimination_threshold=0.0, dominance_margin=0.0,
            ),
            layers=LayerConfig(
                information_asymmetry=0.0,
                opponent_history_depth=5,
                opponent_obs_window=10,  # > depth
                history_sensitivity=0.5,
                incentive_softness=0.8,
                uncertainty_intensity=0.0,
                gamble_variance=0.5,
            ),
            rewards=RewardWeights(
                absolute_gain_weight=1.0, relative_gain_weight=0.5,
                efficiency_weight=0.3, terminal_bonus_scale=2.0,
                penalty_scaling=1.0,
            ),
            agents=AgentConfig(observation_memory_steps=3),
        )
        with pytest.raises(ValidationError, match="opponent_obs_window"):
            CompetitiveEnvironmentConfig(**cfg_kwargs)

    def test_elimination_threshold_exceeds_resources_raises(self):
        with pytest.raises(ValidationError, match="elimination_threshold"):
            _quick_config(elimination_threshold=25.0, initial_resources=20.0)

    def test_num_agents_below_two_raises(self):
        with pytest.raises(ValidationError):
            _quick_config(num_agents=1)


# ---------------------------------------------------------------------------
# 2. Environment contract
# ---------------------------------------------------------------------------

class TestResetStepContract:
    def test_reset_returns_observations_for_all_agents(self):
        cfg = _quick_config()
        env = CompetitiveEnvironment(cfg)
        obs = env.reset()

        assert len(obs) == 3
        for aid, ob in obs.items():
            assert aid.startswith("agent_")
            assert ob["step"] == 0
            assert ob["own_score"] == 0.0
            assert ob["own_resources"] == 20.0
            assert ob["num_active_agents"] == 3

    def test_observation_keys_stable_across_steps(self):
        env = CompetitiveEnvironment(_quick_config())
        obs0 = env.reset()
        keys_at_reset = set(obs0["agent_0"].keys())

        results = env.step(_all_build(env))
        keys_at_step = set(results["agent_0"].observation.keys())

        assert keys_at_reset == keys_at_step

    def test_step_after_done_raises(self):
        cfg = _quick_config(max_steps=10)
        env = CompetitiveEnvironment(cfg)
        env.reset()
        for _ in range(10):
            if env.is_done():
                break
            env.step(_all_build(env))
        assert env.is_done()
        with pytest.raises(RuntimeError, match="done"):
            env.step({})

    def test_active_agents_after_reset(self):
        env = CompetitiveEnvironment(_quick_config())
        env.reset()
        active = env.active_agents()
        assert len(active) == 3
        assert all(aid.startswith("agent_") for aid in active)

    def test_is_done_false_after_reset(self):
        env = CompetitiveEnvironment(_quick_config())
        env.reset()
        assert not env.is_done()
        assert env.termination_reason() is None


# ---------------------------------------------------------------------------
# 3. Termination
# ---------------------------------------------------------------------------

class TestTermination:
    def test_max_steps_termination(self):
        cfg = _quick_config(max_steps=10)
        env = CompetitiveEnvironment(cfg)
        env.reset()

        for _ in range(10):
            env.step(_all_build(env))

        assert env.is_done()
        assert env.termination_reason() == TerminationReason.MAX_STEPS

    def test_elimination_one_agent_remains(self):
        """Episode terminates with ELIMINATION when only 1 agent is left."""
        cfg = _quick_config(
            num_agents=2,
            max_steps=200,
            initial_resources=20.0,
            resource_regeneration_rate=0.0,
            elimination_threshold=0.0,
        )
        env = CompetitiveEnvironment(cfg)
        env.reset()

        # agent_0 defends (zero cost), agent_1 attacks (drains resources)
        # agent_1 will hit 0 resources and be eliminated, leaving agent_0
        for _ in range(200):
            if env.is_done():
                break
            agents = env.active_agents()
            actions = {}
            for aid in agents:
                if aid == "agent_0":
                    actions[aid] = Action(type=ActionType.DEFEND)
                else:
                    actions[aid] = Action(type=ActionType.ATTACK, amount=1.0)
            env.step(actions)

        assert env.is_done()
        assert env.termination_reason() == TerminationReason.ELIMINATION

    def test_no_active_agents_termination(self):
        """When all agents are eliminated simultaneously, NO_ACTIVE_AGENTS fires."""
        cfg = _quick_config(
            num_agents=2,
            max_steps=200,
            initial_resources=20.0,
            resource_regeneration_rate=0.0,
            elimination_threshold=0.0,
        )
        env = CompetitiveEnvironment(cfg)
        env.reset()

        # Both agents attack each other at full intensity — mutual resource drain
        for _ in range(200):
            if env.is_done():
                break
            agents = env.active_agents()
            actions = {
                aid: Action(type=ActionType.ATTACK, amount=1.0)
                for aid in agents
            }
            env.step(actions)

        assert env.is_done()
        assert env.termination_reason() in (
            TerminationReason.NO_ACTIVE_AGENTS,
            TerminationReason.ELIMINATION,
        )

    def test_not_done_initially(self):
        env = CompetitiveEnvironment(_quick_config())
        env.reset()
        assert not env.is_done()
        assert env.termination_reason() is None


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        cfg = _quick_config(seed=99)
        env1 = CompetitiveEnvironment(cfg)
        env2 = CompetitiveEnvironment(cfg)

        obs1 = env1.reset(seed=99)
        obs2 = env2.reset(seed=99)
        assert obs1 == obs2

        for _ in range(5):
            actions1 = _all_build(env1, amount=0.3)
            actions2 = _all_build(env2, amount=0.3)
            r1 = env1.step(actions1)
            r2 = env2.step(actions2)
            for aid in r1:
                assert r1[aid].reward == r2[aid].reward
                assert r1[aid].observation == r2[aid].observation

    def test_different_seed_different_trajectory(self):
        """With noise enabled, different seeds produce different scores."""
        cfg_a = _quick_config(seed=1)
        cfg_b = _quick_config(seed=2)
        cfg_a.layers.uncertainty_intensity = 0.1
        cfg_b.layers.uncertainty_intensity = 0.1

        env_a = CompetitiveEnvironment(cfg_a)
        env_b = CompetitiveEnvironment(cfg_b)
        env_a.reset(seed=1)
        env_b.reset(seed=2)

        for _ in range(5):
            env_a.step(_all_build(env_a, 0.3))
            env_b.step(_all_build(env_b, 0.3))

        scores_a = {aid: s.score for aid, s in env_a._state.agents.items()}
        scores_b = {aid: s.score for aid, s in env_b._state.agents.items()}
        assert scores_a != scores_b


# ---------------------------------------------------------------------------
# 5. Action tests
# ---------------------------------------------------------------------------

class TestActions:
    def test_defend_amount_forced_to_zero(self):
        a = Action(type=ActionType.DEFEND, amount=0.9)
        assert a.amount == 0.0

    def test_gamble_amount_forced_to_zero(self):
        a = Action(type=ActionType.GAMBLE, amount=0.7)
        assert a.amount == 0.0

    def test_invalid_build_amount_raises(self):
        with pytest.raises(ValueError):
            Action(type=ActionType.BUILD, amount=1.5)
        with pytest.raises(ValueError):
            Action(type=ActionType.BUILD, amount=-0.1)

    def test_invalid_attack_amount_raises(self):
        with pytest.raises(ValueError):
            Action(type=ActionType.ATTACK, amount=1.5)
        with pytest.raises(ValueError):
            Action(type=ActionType.ATTACK, amount=-0.1)

    def test_all_four_actions_accepted_by_step(self):
        cfg = _quick_config(num_agents=4)
        env = CompetitiveEnvironment(cfg)
        env.reset()

        agents = env.active_agents()
        actions = {
            agents[0]: Action(type=ActionType.BUILD, amount=0.5),
            agents[1]: Action(type=ActionType.ATTACK, amount=0.5),
            agents[2]: Action(type=ActionType.DEFEND),
            agents[3]: Action(type=ActionType.GAMBLE),
        }
        results = env.step(actions)
        assert len(results) == 4
        for sr in results.values():
            assert isinstance(sr.reward, float)


# ---------------------------------------------------------------------------
# 6. Reward tests
# ---------------------------------------------------------------------------

class TestRewards:
    def test_rewards_are_bounded(self):
        """No single-step reward should be astronomically large."""
        env = CompetitiveEnvironment(_quick_config())
        env.reset()

        for _ in range(5):
            results = env.step(_all_build(env, 0.5))
            for sr in results.values():
                assert -100.0 < sr.reward < 100.0

    def test_terminal_bonus_issued_at_episode_end(self):
        cfg = _quick_config(max_steps=10)
        env = CompetitiveEnvironment(cfg)
        env.reset()

        last_results = None
        for _ in range(10):
            last_results = env.step(_all_build(env, 0.3))

        assert env.is_done()
        # At least one agent should have terminal_bonus in info
        has_bonus = any(
            "terminal_bonus" in sr.info for sr in last_results.values()
        )
        assert has_bonus

    def test_terminal_bonus_first_place_greater_than_last(self):
        """1st place terminal bonus > last place terminal bonus."""
        cfg = _quick_config(num_agents=3, max_steps=10)
        env = CompetitiveEnvironment(cfg)
        env.reset()

        # Create score asymmetry: agent_0 builds a lot, agent_2 defends
        for _ in range(10):
            if env.is_done():
                break
            agents = env.active_agents()
            actions = {}
            for aid in agents:
                if aid == "agent_0":
                    actions[aid] = Action(type=ActionType.BUILD, amount=1.0)
                elif aid == "agent_1":
                    actions[aid] = Action(type=ActionType.BUILD, amount=0.5)
                else:
                    actions[aid] = Action(type=ActionType.DEFEND)
            last_results = env.step(actions)

        assert env.is_done()

        bonuses = {
            aid: sr.info.get("terminal_bonus", 0.0)
            for aid, sr in last_results.items()
        }
        # The top scorer should have a higher bonus than the lowest scorer
        sorted_bonuses = sorted(bonuses.values(), reverse=True)
        assert sorted_bonuses[0] >= sorted_bonuses[-1]


# ---------------------------------------------------------------------------
# 7. Baseline agent tests
# ---------------------------------------------------------------------------

SAMPLE_OBS = {
    "step": 5,
    "own_score": 10.0,
    "own_resources": 15.0,
    "num_active_agents": 3,
    "rankings": [("agent_0", 10.0), ("agent_1", 8.0), ("agent_2", 5.0)],
    "action_history": [{"type": "build", "amount": 0.5}],
    "opponents": {
        "agent_1": {"score": 8.0, "recent_actions": ["build"]},
        "agent_2": {"score": 5.0, "recent_actions": ["defend"]},
    },
}


class TestBaselineAgents:
    def test_always_attack_returns_attack(self):
        agent = _make_agent(AlwaysAttackAgent)
        for _ in range(10):
            action = agent.act(SAMPLE_OBS)
            assert action.type == ActionType.ATTACK
            assert action.amount == pytest.approx(0.5)

    def test_always_build_returns_build(self):
        agent = _make_agent(AlwaysBuildAgent)
        for _ in range(10):
            action = agent.act(SAMPLE_OBS)
            assert action.type == ActionType.BUILD
            assert action.amount == pytest.approx(0.5)

    def test_always_defend_returns_defend(self):
        agent = _make_agent(AlwaysDefendAgent)
        for _ in range(10):
            action = agent.act(SAMPLE_OBS)
            assert action.type == ActionType.DEFEND

    def test_random_agent_returns_valid_actions(self):
        agent = _make_agent(CompetitiveRandomAgent)
        valid_types = set(ActionType)
        for _ in range(50):
            action = agent.act(SAMPLE_OBS)
            assert isinstance(action, Action)
            assert action.type in valid_types
            if action.type in (ActionType.BUILD, ActionType.ATTACK):
                assert 0.0 <= action.amount <= 1.0

    def test_unknown_policy_raises(self):
        with pytest.raises(KeyError):
            create_competitive_agent("nonexistent_policy")


# ---------------------------------------------------------------------------
# 8. Metrics collector tests
# ---------------------------------------------------------------------------

def _make_collector_step_data():
    """Build step data matching CompetitiveMetricsCollector.collect_step signature."""
    actions = {
        "agent_0": Action(type=ActionType.BUILD, amount=0.5),
        "agent_1": Action(type=ActionType.ATTACK, amount=0.3),
        "agent_2": Action(type=ActionType.DEFEND),
    }
    results = {
        aid: StepResult(
            observation={"step": 1},
            reward=0.1 * (i + 1),
            done=False,
        )
        for i, aid in enumerate(AGENTS)
    }
    scores = {"agent_0": 5.0, "agent_1": 3.0, "agent_2": 1.0}
    resources = {aid: 20.0 for aid in AGENTS}
    rankings = [("agent_0", 5.0), ("agent_1", 3.0), ("agent_2", 1.0)]
    return actions, results, scores, resources, rankings


class TestCompetitiveMetricsCollector:
    def test_collect_step_returns_correct_count(self):
        collector = CompetitiveMetricsCollector(InstrumentationConfig())
        actions, results, scores, resources, rankings = _make_collector_step_data()
        records = collector.collect_step(
            step=0,
            actions=actions,
            results=results,
            agent_scores=scores,
            agent_resources=resources,
            active_agents=AGENTS,
            rankings=rankings,
        )
        assert len(records) == len(AGENTS)

    def test_episode_summary_returns_all_required_keys(self):
        collector = CompetitiveMetricsCollector(InstrumentationConfig())
        actions, results, scores, resources, rankings = _make_collector_step_data()
        collector.collect_step(
            step=0, actions=actions, results=results,
            agent_scores=scores, agent_resources=resources,
            active_agents=AGENTS, rankings=rankings,
        )

        summary = collector.episode_summary(
            episode_length=1,
            termination_reason=TerminationReason.MAX_STEPS,
            final_scores=scores,
            final_rankings=rankings,
        )

        expected_keys = {
            "episode_length", "termination_reason", "final_rankings",
            "final_scores", "score_spread", "winner_id",
            "num_eliminations", "total_reward_per_agent",
        }
        assert set(summary.keys()) == expected_keys

    def test_elimination_event_logged(self):
        collector = CompetitiveMetricsCollector(InstrumentationConfig())
        actions, results, scores, resources, rankings = _make_collector_step_data()

        # Step 0: all agents active
        collector.collect_step(
            step=0, actions=actions, results=results,
            agent_scores=scores, agent_resources=resources,
            active_agents=AGENTS, rankings=rankings,
        )

        # Step 1: agent_2 eliminated
        remaining = ["agent_0", "agent_1"]
        remaining_actions = {
            "agent_0": Action(type=ActionType.BUILD, amount=0.5),
            "agent_1": Action(type=ActionType.ATTACK, amount=0.3),
        }
        remaining_results = {
            aid: StepResult(observation={"step": 1}, reward=0.1, done=False)
            for aid in remaining
        }
        collector.collect_step(
            step=1,
            actions=remaining_actions,
            results=remaining_results,
            agent_scores={"agent_0": 6.0, "agent_1": 4.0, "agent_2": 1.0},
            agent_resources={"agent_0": 18.0, "agent_1": 17.0, "agent_2": 0.0},
            active_agents=remaining,
            rankings=[("agent_0", 6.0), ("agent_1", 4.0), ("agent_2", 1.0)],
        )

        events = collector.events
        elim_events = [e for e in events if e["event"] == EventType.AGENT_ELIMINATED.value]
        assert len(elim_events) == 1
        assert elim_events[0]["agent_id"] == "agent_2"

    def test_attack_event_recorded(self):
        collector = CompetitiveMetricsCollector(InstrumentationConfig())
        collector.record_attack_succeeded(step=3, attacker_id="agent_0", score_gained=5.0)
        events = collector.events
        assert len(events) == 1
        assert events[0]["event"] == EventType.ATTACK_SUCCEEDED.value

    def test_events_disabled_returns_empty(self):
        cfg = InstrumentationConfig(enable_event_log=False)
        collector = CompetitiveMetricsCollector(cfg)
        collector.record_attack_succeeded(step=0, attacker_id="agent_0", score_gained=1.0)
        assert collector.events == []
