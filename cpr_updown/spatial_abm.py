"""Well-mixed ODE, connected ER network, and 2D periodic lattice ABM comparison."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .abm import _initial_strategies, connected_erdos_renyi_adjacency
from .cases import _ode_trajectory, figure1_six_cases
from .initial_conditions import generate_shared_initial_conditions
from .model import Parameters, main_branch_equilibrium
from .numerics import integrate_logit_rk4

N_POP = 500
LATTICE_SHAPE = (20, 25)
ER_DEGREE = 6
TIME_END = 100.0
SAMPLE_DT = 0.1
FIG1_REPEATS = 50
FIG1_ROOT_SEED = 82701
FIG2_SEED = 82702
FIG2_REPEATS_PER_IC = 10
FIG2_ABM_ROOT_SEED = 94101
REENTRANT_REPEATS = 30
REENTRANT_T_END = 150.0
REENTRANT_ROOT_SEED = 97301
DEPLETION_THRESHOLD = 1e-4

K_GRID_060 = (0.05, 0.12, 0.16, 0.25, 0.35, 0.50, 0.58, 0.65, 0.80)
K_GRID_071 = (0.205, 0.28, 0.35, 0.42, 0.55, 0.63)


def periodic_square_lattice_adjacency(
    rows: int = LATTICE_SHAPE[0], cols: int = LATTICE_SHAPE[1]
) -> tuple[np.ndarray, ...]:
    """Von Neumann (z=4) periodic square-lattice adjacency for N = rows * cols."""
    adj: list[np.ndarray] = []
    for r in range(rows):
        up = ((r - 1) % rows) * cols
        down = ((r + 1) % rows) * cols
        same = r * cols
        for c in range(cols):
            nbrs = sorted((up + c, down + c, same + ((c - 1) % cols), same + ((c + 1) % cols)))
            adj.append(np.asarray(nbrs, dtype=np.int32))
    return tuple(adj)


def _directed_edges(adjacency: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(adjacency)
    src_list, dst_list, w_list = [], [], []
    for i, nbrs in enumerate(adjacency):
        deg = len(nbrs)
        src_list.append(np.full(deg, i, dtype=np.int32))
        dst_list.append(nbrs)
        w_list.append(np.full(deg, 1.0 / (n * deg), dtype=np.float64))
    return np.concatenate(src_list), np.concatenate(dst_list), np.concatenate(w_list)


def simulate_abm_detailed(
    p: Parameters,
    y0: np.ndarray,
    topology: str,
    n: int = N_POP,
    degree: int = ER_DEGREE,
    t_end: float = TIME_END,
    sample_dt: float = SAMPLE_DT,
    seed: int = 0,
    record_interface: bool = True,
    lattice_shape: tuple[int, int] = LATTICE_SHAPE,
) -> dict[str, Any]:
    """Asynchronous Monte Carlo ABM on connected ER or 2D periodic lattice."""
    dt = 1.0 / n
    steps = int(round(t_end / dt))
    sample_every = int(round(sample_dt / dt))
    if not math.isclose(steps * dt, t_end, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("t_end must be divisible by 1/n")
    if sample_every < 1 or not math.isclose(sample_every * dt, sample_dt, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("sample_dt must be a positive multiple of 1/n")

    initial = np.asarray(y0, dtype=float)
    rng = np.random.default_rng(seed)
    graph_seeds = rng.integers(0, np.iinfo(np.int32).max, size=2)
    if topology == "er":
        adj_u = connected_erdos_renyi_adjacency(n, degree, int(graph_seeds[0]))
        adj_d = connected_erdos_renyi_adjacency(n, degree, int(graph_seeds[1]))
    elif topology == "lattice2d":
        rows, cols = lattice_shape
        if n != rows * cols:
            raise ValueError(f"lattice2d expects n={rows * cols}")
        adj_u = periodic_square_lattice_adjacency(rows, cols)
        adj_d = adj_u
    else:
        raise ValueError(f"unknown topology: {topology}")

    edges_u = _directed_edges(adj_u) if record_interface else None
    edges_d = _directed_edges(adj_d) if record_interface else None

    strat_u = _initial_strategies(float(initial[1]), n, rng)
    strat_d = _initial_strategies(float(initial[3]), n, rng)
    coop_u, coop_d = int(strat_u.sum()), int(strat_d.sum())
    ru, rd = float(initial[0]), float(initial[2])

    nbrs_u = [tuple(int(x) for x in arr) for arr in adj_u]
    degs_u = [len(arr) for arr in nbrs_u]
    nbrs_d = [tuple(int(x) for x in arr) for arr in adj_d]
    degs_d = [len(arr) for arr in nbrs_d]
    su_list, sd_list = strat_u.tolist(), strat_d.tolist()

    n_samples = steps // sample_every + 1
    trajectory = np.empty((n_samples, 4), dtype=np.float64)
    times = np.arange(n_samples, dtype=np.float64) * sample_dt
    trajectory[0] = (ru, coop_u / n, rd, coop_d / n)

    interface = np.empty((n_samples, 2), dtype=np.float64) if record_interface else None
    if record_interface and edges_u is not None and edges_d is not None:
        arr_u = np.asarray(su_list, dtype=np.int8)
        arr_d = np.asarray(sd_list, dtype=np.int8)
        interface[0, 0] = float(np.dot(edges_u[2], arr_u[edges_u[0]] != arr_u[edges_u[1]]))
        interface[0, 1] = float(np.dot(edges_d[2], arr_d[edges_d[0]] != arr_d[edges_d[1]]))

    a_par, k_par = float(p.a), float(p.k)
    ec_u, ed_u = float(p.e_c_u), float(p.e_d_u)
    de_u = ed_u - ec_u
    ec_d, ed_d = float(p.downstream_e_c), float(p.downstream_e_d)
    de_d = ed_d - ec_d
    inv_n = 1.0 / n
    rng_integers, rng_random = rng.integers, rng.random
    sample_idx = 1

    for step in range(1, steps + 1):
        focal_u = int(rng_integers(n))
        model_u = nbrs_u[focal_u][int(rng_integers(degs_u[focal_u]))]
        si_u, sj_u = su_list[focal_u], su_list[model_u]
        if si_u != sj_u:
            prob_u = 0.5 + 0.5 * ru * (1.0 - 2.0 * ru) * (sj_u - si_u)
            prob_u = 0.0 if prob_u < 0.0 else (1.0 if prob_u > 1.0 else prob_u)
            if rng_random() < prob_u:
                su_list[focal_u] = sj_u
                coop_u += sj_u - si_u

        focal_d = int(rng_integers(n))
        model_d = nbrs_d[focal_d][int(rng_integers(degs_d[focal_d]))]
        si_d, sj_d = sd_list[focal_d], sd_list[model_d]
        if si_d != sj_d:
            prob_d = 0.5 + 0.5 * rd * (1.0 - 2.0 * rd) * (sj_d - si_d)
            prob_d = 0.0 if prob_d < 0.0 else (1.0 if prob_d > 1.0 else prob_d)
            if rng_random() < prob_d:
                sd_list[focal_d] = sj_d
                coop_d += sj_d - si_d

        xu, xd = coop_u * inv_n, coop_d * inv_n
        du = a_par * ru * (1.0 - k_par - ru - (ed_u - de_u * xu))
        dd = a_par * (rd * (1.0 - rd - (ed_d - de_d * xd)) + k_par * ru)
        ru_new, rd_new = ru + dt * du, rd + dt * dd
        ru = 0.0 if ru_new < 0.0 else (1.0 if ru_new > 1.0 else ru_new)
        rd = 0.0 if rd_new < 0.0 else (1.0 if rd_new > 1.0 else rd_new)

        if step % sample_every == 0:
            trajectory[sample_idx] = (ru, xu, rd, xd)
            if record_interface and edges_u is not None and edges_d is not None:
                arr_u = np.asarray(su_list, dtype=np.int8)
                arr_d = np.asarray(sd_list, dtype=np.int8)
                interface[sample_idx, 0] = float(np.dot(edges_u[2], arr_u[edges_u[0]] != arr_u[edges_u[1]]))
                interface[sample_idx, 1] = float(np.dot(edges_d[2], arr_d[edges_d[0]] != arr_d[edges_d[1]]))
            sample_idx += 1

    out: dict[str, Any] = {"time": times, "trajectory": trajectory}
    if interface is not None:
        out["interface"] = interface
    return out


def run_spatial_comparison(out: Path, smoke: bool) -> dict[str, Any]:
    """Run Suites A (trajectories/interfaces), B (initial states), and C (coupling scans)."""
    if smoke:
        n, lattice_shape, t_end, sample_dt = 40, (5, 8), 0.5, 0.1
        fig1_repeats, n_ic, ic_repeats = 2, 2, 1
        sweep_repeats, sweep_t_end, sweep_dt = 2, 1.0, 0.5
        k_slices = [("eCd_0p60", 0.60, (0.05, 0.35, 0.80)), ("eCd_0p71_RegimeII", 0.71, (0.205, 0.35, 0.55))]
        avg_samples = 2
    else:
        n, lattice_shape, t_end, sample_dt = N_POP, LATTICE_SHAPE, TIME_END, SAMPLE_DT
        fig1_repeats, n_ic, ic_repeats = FIG1_REPEATS, 50, FIG2_REPEATS_PER_IC
        sweep_repeats, sweep_t_end, sweep_dt = REENTRANT_REPEATS, REENTRANT_T_END, 0.5
        k_slices = [("eCd_0p60", 0.60, K_GRID_060), ("eCd_0p71_RegimeII", 0.71, K_GRID_071)]
        avg_samples = 41  # t in [130, 150] at sample_dt = 0.5

    cases = figure1_six_cases()
    time_grid = np.arange(0.0, t_end + sample_dt / 2.0, sample_dt)
    y0_half = np.full(4, 0.5)

    # Suite A: six representative cases
    for idx, case in enumerate(cases):
        children = np.random.SeedSequence(FIG1_ROOT_SEED + 1009 * idx).spawn(fig1_repeats)
        seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]
        ode = _ode_trajectory(case.parameters, time_grid, y0_half)
        eq_vec, _ = main_branch_equilibrium(
            case.parameters.e_c_u, case.parameters.e_d_u, case.parameters.k,
            case.parameters.downstream_e_c, case.parameters.downstream_e_d,
        )
        er_runs, er_intf, lat_runs, lat_intf = [], [], [], []
        for s in seeds:
            r_er = simulate_abm_detailed(case.parameters, y0_half, "er", n=n, degree=ER_DEGREE,
                                         t_end=t_end, sample_dt=sample_dt, seed=s, lattice_shape=lattice_shape)
            r_lat = simulate_abm_detailed(case.parameters, y0_half, "lattice2d", n=n, degree=ER_DEGREE,
                                          t_end=t_end, sample_dt=sample_dt, seed=s, lattice_shape=lattice_shape)
            er_runs.append(r_er["trajectory"])
            er_intf.append(r_er["interface"])
            lat_runs.append(r_lat["trajectory"])
            lat_intf.append(r_lat["interface"])
        er_runs_a, er_intf_a = np.asarray(er_runs), np.asarray(er_intf)
        lat_runs_a, lat_intf_a = np.asarray(lat_runs), np.asarray(lat_intf)
        np.savez_compressed(
            out / f"suite_a_{case.label}.npz",
            time=time_grid, ode=ode, equilibrium=eq_vec,
            er_runs=er_runs_a, er_mean=er_runs_a.mean(axis=0),
            er_interface=er_intf_a, er_interface_mean=er_intf_a.mean(axis=0),
            lattice_runs=lat_runs_a, lattice_mean=lat_runs_a.mean(axis=0),
            lattice_interface=lat_intf_a, lattice_interface_mean=lat_intf_a.mean(axis=0),
            run_seeds=np.asarray(seeds, dtype=np.uint32),
        )
        print(f"Spatial Suite A {case.label}: {fig1_repeats} runs (ER + 2D), N={n}, t={t_end}", flush=True)

    # Suite B: shared Uniform(0,1)^4 initial states
    ic = generate_shared_initial_conditions(n_ic, FIG2_SEED, lower=0.0, upper=1.0)
    b_dt = 0.5 if not smoke else 0.25
    for idx, case in enumerate(cases):
        _, ode_trajs = integrate_logit_rk4(case.parameters, ic, t_end=t_end, dt=0.02, sample_dt=sample_dt)
        ode_end = ode_trajs[-1]
        case_ss = np.random.SeedSequence(FIG2_ABM_ROOT_SEED + 2027 * idx).spawn(n_ic)
        er_end = np.empty((n_ic, ic_repeats, 4))
        lat_end = np.empty((n_ic, ic_repeats, 4))
        for ic_i, ic_ss in enumerate(case_ss):
            seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in ic_ss.spawn(ic_repeats)]
            for r_i, s in enumerate(seeds):
                er_end[ic_i, r_i] = simulate_abm_detailed(
                    case.parameters, ic[ic_i], "er", n=n, degree=ER_DEGREE,
                    t_end=t_end, sample_dt=b_dt, seed=s, record_interface=False, lattice_shape=lattice_shape,
                )["trajectory"][-1]
                lat_end[ic_i, r_i] = simulate_abm_detailed(
                    case.parameters, ic[ic_i], "lattice2d", n=n, degree=ER_DEGREE,
                    t_end=t_end, sample_dt=b_dt, seed=s, record_interface=False, lattice_shape=lattice_shape,
                )["trajectory"][-1]
        np.savez_compressed(
            out / f"suite_b_ic{n_ic}_{case.label}.npz",
            initial_conditions=ic, ode_endpoints=ode_end, er_endpoints=er_end, lattice_endpoints=lat_end,
        )
        print(f"Spatial Suite B {case.label}: {n_ic} ICs x {ic_repeats} runs (ER + 2D)", flush=True)

    # Suite C: coupling scans
    y0_sweep = np.array([0.50, 0.90, 0.50, 0.90], dtype=float)
    for s_idx, (s_name, e_cd, k_list) in enumerate(k_slices):
        er_means = np.empty((len(k_list), sweep_repeats, 4))
        lat_means = np.empty((len(k_list), sweep_repeats, 4))
        ode_eqs = np.array([main_branch_equilibrium(0.30, 1.5, kv, e_cd, 1.5)[0] for kv in k_list])
        for k_i, kv in enumerate(k_list):
            p = Parameters(a=1.0, f=kv, e_c=0.30, e_d=1.5, e_c_d=e_cd)
            ss = np.random.SeedSequence(REENTRANT_ROOT_SEED + 503 * s_idx + 37 * k_i).spawn(sweep_repeats)
            seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in ss]
            for r_i, s in enumerate(seeds):
                er_means[k_i, r_i] = simulate_abm_detailed(
                    p, y0_sweep, "er", n=n, degree=ER_DEGREE, t_end=sweep_t_end,
                    sample_dt=sweep_dt, seed=s, record_interface=False, lattice_shape=lattice_shape,
                )["trajectory"][-avg_samples:].mean(axis=0)
                lat_means[k_i, r_i] = simulate_abm_detailed(
                    p, y0_sweep, "lattice2d", n=n, degree=ER_DEGREE, t_end=sweep_t_end,
                    sample_dt=sweep_dt, seed=s, record_interface=False, lattice_shape=lattice_shape,
                )["trajectory"][-avg_samples:].mean(axis=0)
        np.savez_compressed(
            out / f"suite_c_{s_name}.npz",
            k_values=np.asarray(k_list, dtype=np.float64), e_c_u=0.30, e_c_d=e_cd,
            ode_equilibrium=ode_eqs, er_late_means=er_means, lattice_late_means=lat_means,
        )
        print(f"Spatial Suite C {s_name}: {len(k_list)} k values x {sweep_repeats} runs", flush=True)

    meta = dict(
        population_per_community=n,
        lattice_shape=list(lattice_shape),
        er_degree_parameter=ER_DEGREE,
        suite_a=dict(repeats=fig1_repeats, time_end=t_end, sample_dt=sample_dt, root_seed=FIG1_ROOT_SEED),
        suite_b=dict(n_initial=n_ic, repeats_per_ic=ic_repeats, ic_seed=FIG2_SEED, abm_root_seed=FIG2_ABM_ROOT_SEED),
        suite_c=dict(repeats=sweep_repeats, time_end=sweep_t_end, root_seed=REENTRANT_ROOT_SEED,
                     slices={name: list(ks) for name, _, ks in k_slices}),
        cases=[dict(label=c.label, parameters=asdict(c.parameters)) for c in cases],
    )
    (out / "spatial_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta
