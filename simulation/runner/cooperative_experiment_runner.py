"""Run a single cooperative episode end-to-end and save artifacts via RunLogger.

Mirrors the competitive experiment runner pattern.
Produces config.json, metrics.jsonl, events.jsonl, and
episode_summary.json under runs_dir/{run_id}/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simulation.agents.cooperative_baselines import create_cooperative_agent
from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.envs.cooperative.env import CooperativeEnvironment
from simulation.metrics.cooperative_collector import CooperativeMetricsCollector
from simulation.runner.run_logger import RunLogger


def run_cooperative_experiment(
    config: CooperativeEnvironmentConfig,
    run_id: str,
    runs_dir: str | Path,
    manager: Any | None = None,
    agent_policy: str = "random",
    agent_kwargs: dict[str, Any] | None = None,
    config_id: str | None = None,
) -> dict[str, Any]:
    """Execute one cooperative episode, log artifacts, return episode summary.

    Parameters
    ----------
    config:
        Fully validated cooperative environment config.
    run_id:
        Unique identifier for this run.
    runs_dir:
        Parent directory where the run folder is created.
    manager:
        Optional run manager (for stop-request checking).
    agent_policy:
        Policy name from COOPERATIVE_POLICY_REGISTRY.
    agent_kwargs:
        Extra keyword arguments forwarded to agent constructor.
    config_id:
        Optional saved config ID (recorded in config snapshot).
    """
    if agent_kwargs is None:
        agent_kwargs = {}

    # 1. Logger
    run_logger = RunLogger(runs_dir, run_id)

    # 2. Config snapshot
    config_dump = config.model_dump()
    config_dump["_agent_policy"] = agent_policy
    if config_id is not None:
        config_dump.setdefault("identity", {})["config_id"] = config_id
    run_logger.write_config(config_dump)

    # 3. Environment
    env = CooperativeEnvironment(config)

    # 4. Metrics collector
    collector = CooperativeMetricsCollector(config.instrumentation)

    # 5. Create agents
    num_agents = config.population.num_agents
    num_task_types = config.population.num_task_types
    agents = {}
    for i in range(num_agents):
        aid = f"agent_{i}"
        agent = create_cooperative_agent(
            agent_policy, num_task_types=num_task_types, **agent_kwargs
        )
        agents[aid] = agent

    # 6. Episode loop
    seed = config.identity.seed
    observations = env.reset(seed)

    for i, (aid, agent) in enumerate(agents.items()):
        agent.reset(aid, seed + i + 1)

    if manager is not None:
        manager.running = True
        manager.run_id = run_id
        manager.max_steps = config.population.max_steps

    events_cursor = 0

    try:
        while not env.is_done():
            if manager is not None and manager.stop_requested:
                break

            step = env.current_step

            if manager is not None:
                manager.step = step

            # Collect actions
            active_ids = env.active_agents()
            actions = {}
            for aid in active_ids:
                if aid in agents:
                    actions[aid] = agents[aid].act(observations.get(aid))

            # Step environment
            results = env.step(actions)

            # Update observations
            for aid, sr in results.items():
                observations[aid] = sr.observation

            # Collect step metrics
            step_records = collector.collect_step(
                step=step,
                actions=actions,
                results=results,
                state=env._state,
            )
            run_logger.log_step_metrics(step_records)

            # Log new events
            all_events = collector.events
            new_events = all_events[events_cursor:]
            run_logger.log_events(new_events)
            events_cursor = len(all_events)

        # 7. Episode summary
        reason = env.termination_reason()
        if manager is not None:
            manager.termination_reason = reason

        summary = collector.episode_summary(
            episode_length=env.current_step,
            termination_reason=reason,
            state=env._state,
        )

        # 8. Save summary
        run_logger.write_episode_summary(summary)

        return summary

    finally:
        if manager is not None:
            manager.running = False
