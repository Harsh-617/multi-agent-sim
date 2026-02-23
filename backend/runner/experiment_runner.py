"""Experiment runner — drives the env loop as an async background task.

Uses MetricsCollector for structured metrics and RunLogger for persistence.
Broadcasts step data to connected WebSocket clients via RunManager.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from simulation.agents import create_agent
from simulation.agents.base import BaseAgent
from simulation.config.schema import MixedEnvironmentConfig
from simulation.core.seeding import derive_seed
from simulation.core.types import TerminationReason
from simulation.envs.mixed import Action, MixedEnvironment
from simulation.metrics.collector import MetricsCollector
from simulation.runner.run_logger import RunLogger

from backend.runner.run_manager import RunManager


async def run_experiment(
    config: MixedEnvironmentConfig,
    run_id: str,
    storage_base: str | Path,
    mgr: RunManager,
    *,
    agent_policy: str = "random",
    agent_kwargs: dict[str, Any] | None = None,
) -> None:
    """Execute a full episode, logging artifacts and broadcasting metrics.

    Designed to be launched as an ``asyncio.Task`` from the API layer.
    Checks ``mgr.stop_requested`` each step for graceful cancellation.
    """
    env = MixedEnvironment(config)
    collector = MetricsCollector(config.instrumentation)
    logger = RunLogger(storage_base, run_id)

    # Persist config snapshot (include agent_policy for registry)
    config_dump = config.model_dump()
    config_dump["_agent_policy"] = agent_policy
    logger.write_config(config_dump)

    # Reset environment and capture initial observations
    initial_obs = env.reset()

    # Instantiate agents with deterministic per-agent seeds
    _agent_kwargs = agent_kwargs or {}
    agents: dict[str, BaseAgent] = {}
    for i, aid in enumerate(env.active_agents()):
        agent = create_agent(agent_policy, **_agent_kwargs)
        agent.reset(aid, derive_seed(config.identity.seed, i))
        agents[aid] = agent

    mgr.running = True
    mgr.run_id = run_id
    mgr.max_steps = config.population.max_steps
    events_logged = 0

    # Track latest observations for each agent
    observations: dict[str, dict] = dict(initial_obs)

    try:
        while not env.is_done():
            if mgr.stop_requested:
                break

            step = env.current_step

            # Build actions from agent policies
            actions: dict[str, Action] = {}
            for aid in env.active_agents():
                obs = observations.get(aid, {})
                actions[aid] = agents[aid].act(obs)

            results = env.step(actions)

            # Update observations for next step
            for aid, sr in results.items():
                observations[aid] = sr.observation

            # Gather state for metrics
            agent_resources = {
                aid: results[aid].observation.get("own_resources", 0.0)
                for aid in results
            }
            shared_pool = next(iter(results.values())).observation.get("shared_pool", 0.0)

            # Collect step metrics
            step_records = collector.collect_step(
                step=step,
                actions=actions,
                results=results,
                shared_pool=shared_pool,
                agent_resources=agent_resources,
                active_agents=env.active_agents(),
            )

            # Check for collapse event
            if env.is_done() and env.termination_reason() == TerminationReason.SYSTEM_COLLAPSE:
                collector.record_collapse(step=env.current_step, shared_pool=shared_pool)

            # Collect new events since last step
            all_events = collector.events
            new_events = all_events[events_logged:]
            events_logged = len(all_events)

            # Persist
            logger.log_step_metrics(step_records)
            logger.log_events(new_events)

            # Broadcast to WS subscribers
            mgr.step = env.current_step
            if step_records:
                await mgr.broadcast({
                    "type": "step",
                    "run_id": run_id,
                    "t": env.current_step,
                    "metrics": step_records,
                    "events": new_events if new_events else None,
                })

            # Pace the loop so WS clients can observe steps in real-time
            await asyncio.sleep(0.05)

        # Episode finished
        reason = env.termination_reason()
        mgr.termination_reason = reason.value if reason else None

        summary = collector.episode_summary(
            episode_length=env.current_step,
            termination_reason=reason,
            final_shared_pool=shared_pool if env.current_step > 0 else config.population.initial_shared_pool,
        )
        logger.write_episode_summary(summary)

        await mgr.broadcast({
            "type": "done",
            "run_id": run_id,
            "termination_reason": reason.value if reason else None,
            "episode_summary": summary,
        })

    finally:
        mgr.running = False
