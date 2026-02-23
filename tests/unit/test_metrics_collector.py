"""Tests for MetricsCollector — verifies output structure and flag handling."""

from __future__ import annotations

import pytest

from simulation.config.schema import InstrumentationConfig
from simulation.core.types import StepResult, TerminationReason
from simulation.envs.mixed.actions import Action, ActionType
from simulation.metrics.collector import MetricsCollector
from simulation.metrics.definitions import STEP_METRIC_KEYS, EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_actions(agent_ids: list[str]) -> dict[str, Action]:
    """Half cooperate, half extract (round up cooperators)."""
    actions = {}
    for i, aid in enumerate(agent_ids):
        if i % 2 == 0:
            actions[aid] = Action(type=ActionType.COOPERATE, amount=0.5)
        else:
            actions[aid] = Action(type=ActionType.EXTRACT, amount=0.3)
    return actions


def _make_results(agent_ids: list[str]) -> dict[str, StepResult]:
    return {
        aid: StepResult(
            observation={"step": 1},
            reward=0.1 * (i + 1),
            done=False,
        )
        for i, aid in enumerate(agent_ids)
    }


def _agent_resources(agent_ids: list[str]) -> dict[str, float]:
    return {aid: 20.0 for aid in agent_ids}


AGENTS = ["agent_0", "agent_1", "agent_2"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCollectorOutputsKeys:
    """Step metric records contain all expected keys."""

    def test_step_metrics_contain_all_keys(self):
        collector = MetricsCollector(InstrumentationConfig())
        actions = _make_actions(AGENTS)
        results = _make_results(AGENTS)
        records = collector.collect_step(
            step=0,
            actions=actions,
            results=results,
            shared_pool=100.0,
            agent_resources=_agent_resources(AGENTS),
            active_agents=AGENTS,
        )

        assert len(records) == len(AGENTS)
        for rec in records:
            for key in STEP_METRIC_KEYS:
                assert key in rec, f"Missing key: {key}"

    def test_step_metrics_values_plausible(self):
        collector = MetricsCollector(InstrumentationConfig())
        actions = _make_actions(AGENTS)
        results = _make_results(AGENTS)
        records = collector.collect_step(
            step=0,
            actions=actions,
            results=results,
            shared_pool=95.0,
            agent_resources=_agent_resources(AGENTS),
            active_agents=AGENTS,
        )

        rec0 = records[0]
        assert rec0["step"] == 0
        assert rec0["agent_id"] == "agent_0"
        assert rec0["shared_pool"] == 95.0
        assert rec0["action_type"] == ActionType.COOPERATE.value
        assert 0.0 <= rec0["coop_ratio"] <= 1.0
        assert 0.0 <= rec0["extraction_ratio"] <= 1.0

    def test_disabled_step_metrics_returns_empty(self):
        cfg = InstrumentationConfig(enable_step_metrics=False)
        collector = MetricsCollector(cfg)
        records = collector.collect_step(
            step=0,
            actions=_make_actions(AGENTS),
            results=_make_results(AGENTS),
            shared_pool=100.0,
            agent_resources=_agent_resources(AGENTS),
            active_agents=AGENTS,
        )
        assert records == []

    def test_step_log_frequency_skips(self):
        cfg = InstrumentationConfig(step_log_frequency=3)
        collector = MetricsCollector(cfg)

        # Step 0 → logged (0 % 3 == 0)
        r0 = collector.collect_step(
            step=0, actions=_make_actions(AGENTS), results=_make_results(AGENTS),
            shared_pool=100.0, agent_resources=_agent_resources(AGENTS),
            active_agents=AGENTS,
        )
        assert len(r0) == 3

        # Step 1 → skipped
        r1 = collector.collect_step(
            step=1, actions=_make_actions(AGENTS), results=_make_results(AGENTS),
            shared_pool=100.0, agent_resources=_agent_resources(AGENTS),
            active_agents=AGENTS,
        )
        assert r1 == []

        # Step 3 → logged
        r3 = collector.collect_step(
            step=3, actions=_make_actions(AGENTS), results=_make_results(AGENTS),
            shared_pool=100.0, agent_resources=_agent_resources(AGENTS),
            active_agents=AGENTS,
        )
        assert len(r3) == 3


class TestCollectorEpisodeSummary:
    """Episode summary accumulates correctly."""

    def test_episode_summary_has_expected_keys(self):
        collector = MetricsCollector(InstrumentationConfig())
        # Feed a couple of steps to accumulate rewards
        for step in range(3):
            collector.collect_step(
                step=step,
                actions=_make_actions(AGENTS),
                results=_make_results(AGENTS),
                shared_pool=100.0 - step,
                agent_resources=_agent_resources(AGENTS),
                active_agents=AGENTS,
            )

        summary = collector.episode_summary(
            episode_length=3,
            termination_reason=TerminationReason.MAX_STEPS,
            final_shared_pool=97.0,
        )

        assert summary["episode_length"] == 3
        assert summary["termination_reason"] == "max_steps"
        assert summary["final_shared_pool"] == 97.0
        assert isinstance(summary["total_reward_per_agent"], dict)
        assert set(summary["total_reward_per_agent"]) == set(AGENTS)

    def test_episode_summary_disabled(self):
        cfg = InstrumentationConfig(enable_episode_metrics=False)
        collector = MetricsCollector(cfg)
        summary = collector.episode_summary(
            episode_length=5,
            termination_reason=TerminationReason.MAX_STEPS,
            final_shared_pool=50.0,
        )
        assert summary == {}


class TestCollectorEvents:
    """Semantic events are captured correctly."""

    def test_collapse_event(self):
        collector = MetricsCollector(InstrumentationConfig())
        collector.record_collapse(step=42, shared_pool=2.5)
        events = collector.events
        assert len(events) == 1
        assert events[0]["event"] == EventType.COLLAPSE.value
        assert events[0]["step"] == 42
        assert events[0]["shared_pool"] == 2.5

    def test_agent_deactivation_event(self):
        collector = MetricsCollector(InstrumentationConfig())
        # First step: all agents active
        collector.collect_step(
            step=0, actions=_make_actions(AGENTS), results=_make_results(AGENTS),
            shared_pool=100.0, agent_resources=_agent_resources(AGENTS),
            active_agents=AGENTS,
        )
        # Second step: agent_1 deactivated
        remaining = ["agent_0", "agent_2"]
        collector.collect_step(
            step=1,
            actions=_make_actions(remaining),
            results=_make_results(remaining),
            shared_pool=90.0,
            agent_resources={"agent_0": 20.0, "agent_1": 0.0, "agent_2": 20.0},
            active_agents=remaining,
        )
        events = collector.events
        assert len(events) == 1
        assert events[0]["event"] == EventType.AGENT_DEACTIVATED.value
        assert events[0]["agent_id"] == "agent_1"

    def test_events_disabled(self):
        cfg = InstrumentationConfig(enable_event_log=False)
        collector = MetricsCollector(cfg)
        collector.record_collapse(step=5, shared_pool=0.0)
        assert collector.events == []
