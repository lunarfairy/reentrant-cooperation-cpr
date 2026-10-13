from __future__ import annotations

import networkx as nx
import numpy as np

from .model import Parameters, extraction


ABM_METHOD = "connected_er_euler_v1"


def connected_erdos_renyi_adjacency(n: int, degree: int, seed: int) -> tuple[np.ndarray, ...]:
    """Sample ER graphs until connected, as in the CPR Memory reference.

    ``degree`` sets p=degree/n; it is not a fixed node degree. The default
    simulation value 6 gives the reference p=6/n before conditioning on connectivity.
    """
    rng = np.random.default_rng(seed)
    while True:
        graph = nx.erdos_renyi_graph(n, p=degree / n, seed=rng)
        if nx.is_connected(graph):
            break
    return tuple(np.asarray(sorted(graph.neighbors(i)), dtype=np.int32) for i in range(n))


def imitation_probability(resource: float, focal: int, model: int) -> float:
    weight = resource * (1.0 - 2.0 * resource)
    probability = 0.5 + 0.5 * weight * (model - focal)
    if probability < -1e-12 or probability > 1.0 + 1e-12:
        raise ValueError(f"illegal imitation probability {probability} at R={resource}")
    return float(np.clip(probability, 0.0, 1.0))


def expected_strategy_drift(resource: float, cooperation: float) -> float:
    return resource * (1.0 - 2.0 * resource) * cooperation * (1.0 - cooperation)


def _resource_rhs(resources: np.ndarray, xu: float, xd: float, p: Parameters) -> np.ndarray:
    ru, rd = resources
    du = p.a * ru * (1.0 - p.k - ru - extraction(xu, p.e_c_u, p.e_d_u))
    dd = p.a * (
        rd * (1.0 - rd - extraction(xd, p.downstream_e_c, p.downstream_e_d))
        + p.k * ru
    )
    return np.array([du, dd])


def _resource_euler(resources: np.ndarray, xu: float, xd: float, dt: float, p: Parameters) -> np.ndarray:
    updated = resources + dt * _resource_rhs(resources, xu, xd, p)
    return np.clip(updated, 0.0, 1.0)


def _initial_strategies(cooperation: float, n: int, rng: np.random.Generator) -> np.ndarray:
    count = int(round(cooperation * n))
    strategies = np.zeros(n, dtype=np.int8)
    if count:
        strategies[rng.choice(n, size=count, replace=False)] = 1
    return strategies


def _strategy_event(
    strategies: np.ndarray,
    resource: float,
    adjacency: tuple[np.ndarray, ...],
    rng: np.random.Generator,
) -> int:
    focal = int(rng.integers(len(strategies)))
    neighbors = adjacency[focal]
    model_node = int(neighbors[int(rng.integers(len(neighbors)))])
    s_i = int(strategies[focal])
    s_j = int(strategies[model_node])
    if s_i == s_j:
        return 0
    if rng.random() < imitation_probability(resource, s_i, s_j):
        strategies[focal] = s_j
        return s_j - s_i
    return 0


def simulate_abm(
    p: Parameters,
    y0: np.ndarray,
    n: int = 200,
    degree: int = 6,
    t_end: float = 200.0,
    sample_dt: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / n
    steps = int(round(t_end / dt))
    sample_every = int(round(sample_dt / dt))
    if not np.isclose(steps * dt, t_end):
        raise ValueError("t_end must be divisible by 1/n")
    if sample_every < 1 or not np.isclose(sample_every * dt, sample_dt):
        raise ValueError("sample_dt must be a positive multiple of 1/n")

    initial = np.asarray(y0, dtype=float)
    rng = np.random.default_rng(seed)
    graph_seeds = rng.integers(0, np.iinfo(np.int32).max, size=2)
    adjacency_u = connected_erdos_renyi_adjacency(n, degree, int(graph_seeds[0]))
    adjacency_d = connected_erdos_renyi_adjacency(n, degree, int(graph_seeds[1]))
    strategies_u = _initial_strategies(initial[1], n, rng)
    strategies_d = _initial_strategies(initial[3], n, rng)
    resources = np.array([initial[0], initial[2]], dtype=float)

    n_samples = steps // sample_every + 1
    trajectory = np.empty((n_samples, 4), dtype=float)
    times = np.arange(n_samples, dtype=float) * sample_dt
    trajectory[0] = [resources[0], strategies_u.mean(), resources[1], strategies_d.mean()]
    sample_index = 1

    for step in range(1, steps + 1):
        _strategy_event(strategies_u, resources[0], adjacency_u, rng)
        _strategy_event(strategies_d, resources[1], adjacency_d, rng)
        xu = float(strategies_u.mean())
        xd = float(strategies_d.mean())
        resources = _resource_euler(resources, xu, xd, dt, p)
        if step % sample_every == 0:
            trajectory[sample_index] = [resources[0], xu, resources[1], xd]
            sample_index += 1
    return times, trajectory


def simulate_abm_counted(
    p: Parameters,
    y0: np.ndarray,
    n: int = 200,
    degree: int = 6,
    t_end: float = 200.0,
    sample_dt: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / n
    steps = int(round(t_end / dt))
    sample_every = int(round(sample_dt / dt))
    if not np.isclose(steps * dt, t_end):
        raise ValueError("t_end must be divisible by 1/n")
    if sample_every < 1 or not np.isclose(sample_every * dt, sample_dt):
        raise ValueError("sample_dt must be a positive multiple of 1/n")

    initial = np.asarray(y0, dtype=float)
    rng = np.random.default_rng(seed)
    graph_seeds = rng.integers(0, np.iinfo(np.int32).max, size=2)
    adjacency_u = connected_erdos_renyi_adjacency(n, degree, int(graph_seeds[0]))
    adjacency_d = connected_erdos_renyi_adjacency(n, degree, int(graph_seeds[1]))
    strategies_u = _initial_strategies(initial[1], n, rng)
    strategies_d = _initial_strategies(initial[3], n, rng)
    cooperators_u = int(strategies_u.sum())
    cooperators_d = int(strategies_d.sum())
    resources = np.array([initial[0], initial[2]], dtype=float)

    n_samples = steps // sample_every + 1
    trajectory = np.empty((n_samples, 4), dtype=float)
    times = np.arange(n_samples, dtype=float) * sample_dt
    trajectory[0] = [resources[0], cooperators_u / n, resources[1], cooperators_d / n]
    sample_index = 1

    for step in range(1, steps + 1):
        cooperators_u += _strategy_event(strategies_u, resources[0], adjacency_u, rng)
        cooperators_d += _strategy_event(strategies_d, resources[1], adjacency_d, rng)
        xu = cooperators_u / n
        xd = cooperators_d / n
        resources = _resource_euler(resources, xu, xd, dt, p)
        if step % sample_every == 0:
            trajectory[sample_index] = [resources[0], xu, resources[1], xd]
            sample_index += 1
    return times, trajectory


def simulate_abm_ensemble(
    p: Parameters,
    y0: np.ndarray,
    n: int = 200,
    degree: int = 6,
    repeats: int = 50,
    t_end: float = 200.0,
    sample_dt: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    children = np.random.SeedSequence(seed).spawn(repeats)
    runs = []
    times = None
    for child in children:
        run_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        times, trajectory = simulate_abm(
            p=p,
            y0=y0,
            n=n,
            degree=degree,
            t_end=t_end,
            sample_dt=sample_dt,
            seed=run_seed,
        )
        runs.append(trajectory)
    stacked = np.asarray(runs)
    mean = stacked.mean(axis=0)
    standard_error = stacked.std(axis=0, ddof=1) / np.sqrt(repeats)
    return {
        "time": np.asarray(times),
        "runs": stacked,
        "mean": mean,
        "q_low": np.quantile(stacked, 0.025, axis=0),
        "q_high": np.quantile(stacked, 0.975, axis=0),
        "ci_low": mean - 1.96 * standard_error,
        "ci_high": mean + 1.96 * standard_error,
    }
