"""Run a single competitive episode end-to-end and save artifacts via RunLogger.

Mirrors the Mixed-archetype experiment runner pattern but for the Competitive
archetype.  Produces config.json, metrics.jsonl, events.jsonl, and
episode_summary.json under runs_dir/{run_id}/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simulation.agents.competitive_baselines import create_competitive_agent
from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
from simulation.envs.competitive.env import CompetitiveEnvironment
from simulation.metrics.competitive_collector import CompetitiveMetricsCollector
from simulation.runner.run_logger import RunLogger


def run_competitive_experiment(
    config: CompetitiveEnvironmentConfig,
    run_id: str,
    runs_dir: str | Path,
    manager: Any | None = None,
    agent_policy: str = "random",
    agent_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one competitive episode, log artifacts, return episode summary.

    Parameters
    ----------
    config:
        Fully validated competitive environment config.
    run_id:
        Unique identifier for this run (becomes directory name).
    runs_dir:
        Parent directory where the run folder is created.
    manager:
        Optional experiment manager (unused for now, kept for interface parity).
    agent_policy:
        Policy name from COMPETITIVE_POLICY_REGISTRY (applied to all agents).
    agent_kwargs:
        Extra keyword arguments forwarded to agent constructor.
    """
    if agent_kwargs is None:
        agent_kwargs = {}

    # 1. Logger
    run_logger = RunLogger(runs_dir, run_id)

    # 2. Save config snapshot
    run_logger.write_config(config.model_dump())

    # 3. Environment
    env = CompetitiveEnvironment(config)

    # 4. Metrics collector
    collector = CompetitiveMetricsCollector(config.instrumentation)

    # 5. Create agents
    num_agents = config.population.num_agents
    agents = {}
    for i in range(num_agents):
        aid = f"agent_{i}"
        agent = create_competitive_agent(agent_policy, **agent_kwargs)
        agents[aid] = agent

    # 6. Episode loop
    seed = config.identity.seed
    observations = env.reset(seed)

    # Reset agents with their IDs and deterministic seeds
    for i, (aid, agent) in enumerate(agents.items()):
        agent.reset(aid, seed + i + 1)

    events_cursor = 0  # track how many events we've already logged

    while not env.is_done():
        step = env.current_step

        # Collect actions from active agents
        active_ids = env.active_agents()
        actions = {}
        for aid in active_ids:
            if aid in agents:
                actions[aid] = agents[aid].act(observations.get(aid))

        # Step the environment
        results = env.step(actions)

        # Update observations for next step
        for aid, sr in results.items():
            observations[aid] = sr.observation

        # Gather state for metrics
        state = env._state
        agent_scores = {aid: state.agents[aid].score for aid in results}
        agent_resources = {aid: state.agents[aid].resources for aid in results}
        rankings = state.rankings()

        # Collect step metrics
        step_records = collector.collect_step(
            step=step,
            actions=actions,
            results=results,
            agent_scores=agent_scores,
            agent_resources=agent_resources,
            active_agents=env.active_agents(),
            rankings=rankings,
        )
        run_logger.log_step_metrics(step_records)

        # Log any new events
        all_events = collector.events
        new_events = all_events[events_cursor:]
        run_logger.log_events(new_events)
        events_cursor = len(all_events)

    # 7. Episode summary
    final_state = env._state
    final_scores = {aid: s.score for aid, s in final_state.agents.items()}
    final_rankings = final_state.rankings()

    summary = collector.episode_summary(
        episode_length=env.current_step,
        termination_reason=env.termination_reason(),
        final_scores=final_scores,
        final_rankings=final_rankings,
    )

    # 8. Save summary
    run_logger.write_episode_summary(summary)

    # 9. Return summary
    return summary
