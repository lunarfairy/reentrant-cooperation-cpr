"""Run the manuscript calculations without plotting or external input files."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import networkx as nx
import numpy as np
import scipy

from cpr_updown.abm import simulate_abm_ensemble
from cpr_updown.cases import figure1_six_cases, _ode_trajectory
from cpr_updown.initial_conditions import generate_shared_initial_conditions
from cpr_updown.numerics import integrate_logit_rk4
from cpr_updown import upstream, equilibrium_grid


def figure1(out: Path, smoke: bool) -> dict:
    n, repeats, t_end = (40, 2, 0.5) if smoke else (500, 50, 100.0)
    root_seed, degree, sample_dt = 82701, 6, 0.1
    initial = np.full(4, 0.5)
    run_seeds = {}
    for index, case in enumerate(figure1_six_cases()):
        seed = root_seed + 1009 * index
        run_seeds[case.label] = [int(child.generate_state(1, dtype=np.uint32)[0])
                                 for child in np.random.SeedSequence(seed).spawn(repeats)]
        result = simulate_abm_ensemble(case.parameters, initial, n=n, degree=degree,
                                      repeats=repeats, t_end=t_end, sample_dt=sample_dt, seed=seed)
        ode = _ode_trajectory(case.parameters, result['time'], initial)
        np.savez_compressed(out / f'figure1_{case.label}.npz', time=result['time'], ode=ode,
                            abm_runs=result['runs'], abm_mean=result['mean'],
                            run_seeds=np.asarray(run_seeds[case.label], dtype=np.uint32))
        np.savetxt(out / f'figure1_{case.label}.csv',
                   np.column_stack((result['time'], ode, result['mean'])), delimiter=',',
                   header='time,ode_Ru,ode_xu,ode_Rd,ode_xd,abm_mean_Ru,abm_mean_xu,abm_mean_Rd,abm_mean_xd', comments='')
        print(f'Figure 1 {case.label}: {repeats} runs, N={n}, t={t_end}', flush=True)
    return dict(population_per_region=n, repeats=repeats, time_end=t_end, root_seed=root_seed,
                run_seeds=run_seeds, degree_parameter=degree, edge_probability=degree/n,
                sample_dt=sample_dt, initial_state=initial.tolist(), resource_step=1/n)


def figure2(out: Path, smoke: bool) -> dict:
    n_initial, t_end = (3, 0.5) if smoke else (50, 100.0)
    seed, dt, sample_dt, threshold = 82702, 0.02, 0.1, 1e-4
    initial = generate_shared_initial_conditions(n_initial, seed, lower=0.0, upper=1.0)
    np.savetxt(out/'initial_conditions.csv', initial, delimiter=',', header='Ru,xu,Rd,xd', comments='', fmt='%.17g')
    counts=[]
    for case in figure1_six_cases():
        time, trajectories = integrate_logit_rk4(case.parameters, initial, t_end=t_end, dt=dt, sample_dt=sample_dt)
        np.savez_compressed(out/f'figure2_{case.label}.npz', time=time, initial=initial, trajectories=trajectories)
        endpoints=trajectories[-1]
        up, down = endpoints[:, 0] < threshold, endpoints[:, 2] < threshold
        counts.append(dict(parameter_set=case.label, n_initial=n_initial,
                           upstream_near_zero=int(up.sum()), downstream_near_zero=int(down.sum()),
                           both_near_zero=int((up & down).sum()), neither_near_zero=int((~up & ~down).sum())))
        np.savetxt(out/f'figure2_{case.label}_endpoints.csv',
                   np.column_stack((np.arange(n_initial),initial,endpoints)),delimiter=',',comments='',
                   header='initial_id,Ru0,xu0,Rd0,xd0,Ru_final,xu_final,Rd_final,xd_final')
        print(f'Figure 2 {case.label}: {n_initial} shared initial states, t={t_end}', flush=True)
    with (out/'finite_time_summary.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(counts[0])); writer.writeheader(); writer.writerows(counts)
    return dict(n_initial=n_initial, time_end=t_end, root_seed=seed, integration_dt=dt,
                sample_dt=sample_dt, distribution='independent Uniform(0, 1)',
                finite_time_resource_threshold=threshold, counts=counts)


def equilibria(out: Path, smoke: bool) -> dict:
    nk = 21 if smoke else upstream.N_K
    values = upstream.upstream_selected_branch(np.linspace(0.0,1.0,nk))
    np.savez_compressed(out/'figure3_upstream.npz', **values)
    original_shape=equilibrium_grid.N_K, equilibrium_grid.N_E_C_D
    if smoke:
        equilibrium_grid.N_K, equilibrium_grid.N_E_C_D = 21, 19
    try:
        for regime in ('I','II','III'):
            grid=equilibrium_grid.stable_equilibrium_grid(regime)
            np.savez_compressed(out/f'figure4_{regime}.npz', **grid)
        shape=[equilibrium_grid.N_E_C_D,equilibrium_grid.N_K]
    finally:
        equilibrium_grid.N_K,equilibrium_grid.N_E_C_D=original_shape
    return dict(upstream_points=nk, downstream_grid_shape=shape,
                upstream_cooperator_extraction=0.3, defector_extraction=1.5,
                downstream_cooperator_extraction_range=[0.4,0.85],
                scope='selected positive-resource branches; depletion cooperation remains path-dependent')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['all','figure1','figure2','equilibria'],default='all')
    parser.add_argument('--profile',choices=['smoke','manuscript'],default='smoke',
                        help='smoke is a short execution check; manuscript uses the original experiment settings')
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=dict(profile=args.profile, variable_order=['Ru','xu','Rd','xd'],
                  versions=dict(numpy=np.__version__,scipy=scipy.__version__,networkx=nx.__version__),
                  cases=[dict(label=c.label,parameters=asdict(c.parameters)) for c in figure1_six_cases()],
                  experiments={})
    for name,function in [('figure1',figure1),('figure2',figure2),('equilibria',equilibria)]:
        if args.experiment in ('all',name):
            out=args.output/name
            out.mkdir(exist_ok=True)
            metadata['experiments'][name]=function(out,args.profile=='smoke')
    (args.output/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('Finished. Run configuration saved in run_metadata.json.')


if __name__=='__main__':
    main()
