"""Unit tests for the Cooperative archetype environment.

Covers:
  - reset returns valid observations for all agents
  - step with all-random actions completes without error
  - step with all-idle actions completes without error
  - task queue grows when agents idle
  - task queue shrinks when agents work
  - specialization score increases with repeated work on same type
  - termination fires at max_steps
  - system collapse termination fires when backlog sustained above threshold
  - reward is bounded [0, 1] every step
  - observation vector is fixed length across all steps
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config.cooperative_defaults import default_cooperative_config
from simulation.config.cooperative_schema import (
    CooperativeEnvironmentConfig,
    EnvironmentIdentity,
    InstrumentationConfig,
    LayerConfig,
    PopulationConfig,
    RewardWeights,
    TaskConfig,
)
from simulation.core.types import TerminationReason
from simulation.envs.cooperative.actions import Action
from simulation.envs.cooperative.env import CooperativeEnvironment
from simulation.agents.cooperative_baselines import (
    AlwaysIdleAgent,
    AlwaysWorkAgent,
    BalancerAgent,
    CooperativeRandomAgent,
    SpecialistAgent,
    create_cooperative_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42


def _quick_config(
    seed: int = SEED,
    num_agents: int = 3,
    max_steps: int = 20,
    num_task_types: int = 2,
    task_arrival_rate: float = 1.0,
    task_difficulty: float = 1.0,
    collapse_threshold: int = 30,
    collapse_sustain_window: int = 5,
    initial_backlog: int = 0,
    enable_early_success: bool = False,
    clearance_sustain_window: int = 5,
    history_window: int = 5,
    specialization_decay: float = 0.2,
    task_arrival_noise: float = 0.0,
    task_difficulty_variance: float = 0.0,
) -> CooperativeEnvironmentConfig:
    """Small fast config for testing."""
    return CooperativeEnvironmentConfig(
        identity=EnvironmentIdentity(seed=seed),
        population=PopulationConfig(
            num_agents=num_agents,
            max_steps=max_steps,
            num_task_types=num_task_types,
            agent_effort_capacity=1.0,
            collapse_sustain_window=collapse_sustain_window,
            enable_early_success=enable_early_success,
            clearance_sustain_window=clearance_sustain_window,
        ),
        layers=LayerConfig(
            observation_noise=0.0,
            history_window=history_window,
            specialization_scale=0.3,
            specialization_decay=specialization_decay,
            task_arrival_noise=task_arrival_noise,
            task_difficulty_variance=task_difficulty_variance,
            free_rider_pressure_scale=1.0,
        ),
        task=TaskConfig(
            task_arrival_rate=[task_arrival_rate] * num_task_types,
            task_difficulty=[task_difficulty] * num_task_types,
            collapse_threshold=collapse_threshold,
            initial_backlog=initial_backlog,
        ),
        rewards=RewardWeights(w_group=0.7, w_individual=0.2, w_efficiency=0.1),
        instrumentation=InstrumentationConfig(),
    )


def _all_idle(env: CooperativeEnvironment) -> dict:
    return {aid: Action(task_type=None) for aid in env.active_agents()}


def _all_work(
    env: CooperativeEnvironment, task_type: int = 0, effort: float = 1.0
) -> dict:
    return {
        aid: Action(task_type=task_type, effort_amount=effort)
        for aid in env.active_agents()
    }


def _random_actions(
    env: CooperativeEnvironment,
    num_task_types: int,
    rng: np.random.Generator,
) -> dict:
    actions = {}
    for aid in env.active_agents():
        choices = num_task_types + 1
        idx = int(rng.integers(choices))
        if idx == num_task_types:
            actions[aid] = Action(task_type=None)
        else:
            effort = float(rng.uniform(0.0, 1.0))
            actions[aid] = Action(task_type=idx, effort_amount=effort)
    return actions


# ---------------------------------------------------------------------------
# 1. Reset contract
# ---------------------------------------------------------------------------

class TestResetContract:
    def test_reset_returns_observations_for_all_agents(self):
        cfg = _quick_config()
        env = CooperativeEnvironment(cfg)
        obs = env.reset()

        assert len(obs) == cfg.population.num_agents
        for aid, ob in obs.items():
            assert aid.startswith("agent_")
            assert "obs_vector" in ob
            assert "step" in ob
            assert ob["step"] == 0

    def test_all_agents_active_after_reset(self):
        cfg = _quick_config(num_agents=4)
        env = CooperativeEnvironment(cfg)
        env.reset()
        assert len(env.active_agents()) == 4

    def test_is_done_false_after_reset(self):
        env = CooperativeEnvironment(_quick_config())
        env.reset()
        assert not env.is_done()
        assert env.termination_reason() is None

    def test_step_before_reset_raises(self):
        env = CooperativeEnvironment(_quick_config())
        with pytest.raises(RuntimeError, match="reset"):
            env.step({})


# ---------------------------------------------------------------------------
# 2. Step contract
# ---------------------------------------------------------------------------

class TestStepContract:
    def test_step_with_random_actions_completes_without_error(self):
        cfg = _quick_config()
        env = CooperativeEnvironment(cfg)
        env.reset()
        rng = np.random.default_rng(SEED)

        for _ in range(5):
            if env.is_done():
                break
            actions = _random_actions(env, cfg.population.num_task_types, rng)
            results = env.step(actions)
            assert len(results) == cfg.population.num_agents

    def test_step_with_all_idle_completes_without_error(self):
        cfg = _quick_config()
        env = CooperativeEnvironment(cfg)
        env.reset()

        results = env.step(_all_idle(env))
        assert len(results) == cfg.population.num_agents
        for sr in results.values():
            assert isinstance(sr.reward, float)
            assert isinstance(sr.done, bool)

    def test_step_returns_step_results_for_all_agents(self):
        cfg = _quick_config()
        env = CooperativeEnvironment(cfg)
        env.reset()

        results = env.step(_all_work(env))
        assert set(results.keys()) == set(env.active_agents())

    def test_step_after_done_raises(self):
        cfg = _quick_config(max_steps=10)
        env = CooperativeEnvironment(cfg)
        env.reset()

        for _ in range(15):
            if env.is_done():
                break
            env.step(_all_idle(env))

        assert env.is_done()
        with pytest.raises(RuntimeError, match="done"):
            env.step({})


# ---------------------------------------------------------------------------
# 3. Task queue dynamics
# ---------------------------------------------------------------------------

class TestTaskQueueDynamics:
    def test_task_queue_grows_when_agents_idle(self):
        """With non-zero arrival rate and all agents idle, queue grows."""
        cfg = _quick_config(
            task_arrival_rate=2.0,
            task_arrival_noise=0.0,  # deterministic arrivals
            initial_backlog=0,
        )
        env = CooperativeEnvironment(cfg)
        env.reset()

        backlog_before = env._state.backlog_level
        env.step(_all_idle(env))
        backlog_after = env._state.backlog_level

        # 2 task types × 2.0 arrival rate = 4 new tasks, 0 completed → grows
        assert backlog_after > backlog_before

    def test_task_queue_shrinks_when_agents_work(self):
        """With initial backlog and zero arrivals, working reduces the queue."""
        cfg = _quick_config(
            num_agents=4,
            task_arrival_rate=0.0001,  # near-zero arrivals
            task_arrival_noise=0.0,
            task_difficulty=1.0,
            initial_backlog=20,
            num_task_types=1,
        )
        env = CooperativeEnvironment(cfg)
        env.reset()

        backlog_before = env._state.backlog_level
        assert backlog_before > 0, "Pre-condition: initial backlog must be non-zero"

        # All 4 agents work on task type 0 with full effort
        env.step(_all_work(env, task_type=0, effort=1.0))
        backlog_after = env._state.backlog_level

        assert backlog_after < backlog_before


# ---------------------------------------------------------------------------
# 4. Specialization
# ---------------------------------------------------------------------------

class TestSpecialization:
    def test_specialization_score_increases_with_repeated_work_on_same_type(self):
        """An agent that repeatedly works on type 0 should see score[0] rise."""
        cfg = _quick_config(
            num_agents=2,
            num_task_types=2,
            specialization_decay=0.3,  # faster convergence for the test
        )
        env = CooperativeEnvironment(cfg)
        env.reset()

        aid = "agent_0"
        score_initial = env._state.agents[aid].specialization_score[0]
        assert score_initial == pytest.approx(0.0)

        for _ in range(10):
            if env.is_done():
                break
            env.step({a: Action(task_type=0, effort_amount=1.0) for a in env.active_agents()})

        score_after = env._state.agents[aid].specialization_score[0]
        assert score_after > score_initial

    def test_specialization_score_stays_in_0_1(self):
        cfg = _quick_config(num_task_types=3, specialization_decay=0.5)
        env = CooperativeEnvironment(cfg)
        env.reset()

        for _ in range(15):
            if env.is_done():
                break
            env.step(_all_work(env, task_type=0))

        for agent in env._state.agents.values():
            for score in agent.specialization_score:
                assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 5. Termination
# ---------------------------------------------------------------------------

class TestTermination:
    def test_max_steps_termination(self):
        cfg = _quick_config(max_steps=10)
        env = CooperativeEnvironment(cfg)
        env.reset()

        for _ in range(10):
            env.step(_all_idle(env))

        assert env.is_done()
        assert env.termination_reason() == TerminationReason.MAX_STEPS

    def test_system_collapse_termination_fires_when_backlog_sustained(self):
        """All-idle agents with high arrival rate triggers system collapse."""
        cfg = _quick_config(
            num_agents=2,
            max_steps=100,
            num_task_types=2,
            task_arrival_rate=5.0,   # 10 tasks/step, no noise
            task_arrival_noise=0.0,
            collapse_threshold=5,    # very small threshold
            collapse_sustain_window=3,  # only 3 consecutive steps needed
            initial_backlog=0,
        )
        env = CooperativeEnvironment(cfg)
        env.reset()

        for _ in range(20):
            if env.is_done():
                break
            env.step(_all_idle(env))

        assert env.is_done()
        assert env.termination_reason() == TerminationReason.SYSTEM_COLLAPSE

    def test_not_done_initially(self):
        env = CooperativeEnvironment(_quick_config())
        env.reset()
        assert not env.is_done()
        assert env.termination_reason() is None


# ---------------------------------------------------------------------------
# 6. Reward properties
# ---------------------------------------------------------------------------

class TestRewardProperties:
    def test_reward_bounded_0_1_with_random_actions(self):
        cfg = _quick_config(max_steps=20)
        env = CooperativeEnvironment(cfg)
        env.reset()
        rng = np.random.default_rng(SEED)

        for _ in range(20):
            if env.is_done():
                break
            actions = _random_actions(env, cfg.population.num_task_types, rng)
            results = env.step(actions)
            for sr in results.values():
                assert 0.0 <= sr.reward <= 1.0, f"Reward out of bounds: {sr.reward}"

    def test_reward_bounded_0_1_with_idle_actions(self):
        cfg = _quick_config()
        env = CooperativeEnvironment(cfg)
        env.reset()

        for _ in range(5):
            if env.is_done():
                break
            results = env.step(_all_idle(env))
            for sr in results.values():
                assert 0.0 <= sr.reward <= 1.0

    def test_reward_components_in_info(self):
        env = CooperativeEnvironment(_quick_config())
        env.reset()
        results = env.step(_all_work(env))

        for sr in results.values():
            comps = sr.info.get("reward_components", {})
            assert "r_group" in comps
            assert "r_individual" in comps
            assert "r_efficiency" in comps


# ---------------------------------------------------------------------------
# 7. Observation vector fixed length
# ---------------------------------------------------------------------------

class TestObservationVectorFixedLength:
    def test_obs_vector_same_length_across_steps(self):
        cfg = _quick_config(num_task_types=3, history_window=4)
        env = CooperativeEnvironment(cfg)
        obs = env.reset()

        expected_len = env.obs_dim()
        # Check at reset
        for ob in obs.values():
            assert len(ob["obs_vector"]) == expected_len

        # Check after several steps
        for step_i in range(8):
            if env.is_done():
                break
            results = env.step(_all_work(env, task_type=step_i % 3))
            for sr in results.values():
                assert len(sr.observation["obs_vector"]) == expected_len, (
                    f"Step {step_i}: expected obs length {expected_len}, "
                    f"got {len(sr.observation['obs_vector'])}"
                )

    def test_obs_dim_formula(self):
        """obs_dim = 6 + 4*T + K*(T+1)."""
        for T, K in [(1, 3), (3, 5), (5, 10)]:
            cfg = _quick_config(num_task_types=T, history_window=K)
            env = CooperativeEnvironment(cfg)
            expected = 8 + 4 * T + K * (T + 1)
            assert env.obs_dim() == expected


# ---------------------------------------------------------------------------
# 8. Baseline agent tests
# ---------------------------------------------------------------------------

SAMPLE_OBS: dict = {
    "obs_vector": np.zeros(20, dtype=np.float32),
    "step": 3,
}


class TestBaselineAgents:
    def test_random_agent_returns_valid_actions(self):
        agent = CooperativeRandomAgent(num_task_types=3)
        agent.reset("agent_0", SEED)
        for _ in range(30):
            act = agent.act(SAMPLE_OBS)
            assert isinstance(act, Action)
            if not act.is_idle:
                assert 0 <= act.task_type <= 2
                assert 0.0 <= act.effort_amount <= 1.0

    def test_always_idle_returns_idle(self):
        agent = AlwaysIdleAgent()
        agent.reset("agent_0", SEED)
        for _ in range(10):
            act = agent.act(SAMPLE_OBS)
            assert act.is_idle
            assert act.effort_amount == pytest.approx(0.0)

    def test_always_work_returns_work_action(self):
        agent = AlwaysWorkAgent(num_task_types=3)
        agent.reset("agent_0", SEED)
        act = agent.act(SAMPLE_OBS)
        assert not act.is_idle
        assert act.effort_amount == pytest.approx(1.0)
        assert 0 <= act.task_type <= 2

    def test_specialist_picks_same_type_every_step(self):
        agent = SpecialistAgent(num_task_types=3)
        agent.reset("agent_0", SEED)
        first_type = agent.act(SAMPLE_OBS).task_type
        for _ in range(20):
            act = agent.act(SAMPLE_OBS)
            assert act.task_type == first_type
            assert act.effort_amount == pytest.approx(1.0)

    def test_balancer_cycles_through_types(self):
        T = 3
        agent = BalancerAgent(num_task_types=T)
        agent.reset("agent_0", SEED)
        types_seen = [agent.act(SAMPLE_OBS).task_type for _ in range(T * 2)]
        # Should cycle: 0, 1, 2, 0, 1, 2
        assert types_seen == list(range(T)) + list(range(T))

    def test_create_cooperative_agent_unknown_raises(self):
        with pytest.raises(KeyError):
            create_cooperative_agent("nonexistent_policy")

    def test_all_policies_run_in_env(self):
        """All five baseline policies complete a short episode without error."""
        for policy in ["random", "always_work", "always_idle", "specialist", "balancer"]:
            cfg = _quick_config(max_steps=10, num_task_types=2)
            env = CooperativeEnvironment(cfg)
            obs = env.reset()
            T = cfg.population.num_task_types
            N = cfg.population.num_agents

            agents = {}
            for i in range(N):
                aid = f"agent_{i}"
                agent = create_cooperative_agent(policy, num_task_types=T)
                agent.reset(aid, SEED + i)
                agents[aid] = agent

            while not env.is_done():
                actions = {
                    aid: agents[aid].act(obs.get(aid))
                    for aid in env.active_agents()
                }
                results = env.step(actions)
                for aid, sr in results.items():
                    obs[aid] = sr.observation
