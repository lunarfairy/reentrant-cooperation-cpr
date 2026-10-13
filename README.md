# Re-entrant cooperation in coupled common-pool resources

Computational implementation for **Re-entrant Cooperation under Directional Resource Transfer in Coupled Common-Pool Resource Systems**.

This repository contains the coupled upstream-downstream model, deterministic integration, stochastic agent-based simulation, and analytical equilibrium calculations. The `source_data/` directory provides the numerical data underlying Figures 1–4, including saved simulation trajectories and analytical equilibrium arrays. The computational implementation runs without external input data.

## Installation

Tested with Python 3.12.4, NumPy 1.26.4, SciPy 1.13.1 and NetworkX 3.6.1. Use Python 3.11 or newer.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

From the repository directory, run a short execution check:

```sh
python run.py --profile smoke --output outputs/smoke
```

This runs all six parameter cases with reduced population, repetitions and duration, plus smaller equilibrium grids. It is not a replication of the manuscript experiments.

The original manuscript settings are selected explicitly:

```sh
python run.py --profile manuscript --experiment figure1 --output outputs/figure1
python run.py --profile manuscript --experiment figure2 --output outputs/figure2
python run.py --profile manuscript --experiment equilibria --output outputs/equilibria
```

Figure 1 runs 300 stochastic simulations and can take substantial time. Its runner is serial; the original local batch used parallel execution with the same independent run seeds. The output root contains experiment subdirectories and `run_metadata.json`. Use separate output directories for different runs.

| Calculation | Manuscript configuration | Outputs generated on request |
|---|---|---|
| Figure 1 | Six cases; 500 agents per community; 50 runs per case; t = 100; connected Erdős–Rényi networks with p = 6/N; all initial states = 0.5 | ODE and ABM trajectories, ensemble means, seeds, CSV summaries |
| Figure 2 | Six cases sharing 50 independently sampled Uniform(0,1) four-dimensional initial states; t = 100; logit RK4 step 0.02; sampling interval 0.1 | Initial states, trajectories, endpoints and finite-time near-depletion counts |
| Figure 3 | Upstream extraction rates 0.3 and 1.5; 2,001 k values from 0 to 1 | Selected upstream branch arrays |
| Figure 4 | Three coupling regimes; downstream cooperative extraction from 0.4 to 0.85; 451 × 501 cells per regime | Conditional downstream equilibrium grids and boundary flags |

## Source data

The [source data guide](source_data/README_Source_Data.txt) maps files to Figures 1–4. The [manifest](source_data/data_manifest.json) records SHA-256 checksums, CSV columns and array shapes for 32 original data/configuration files. These files are preserved byte-for-byte from the existing figure source data. Download the repository ZIP to obtain both code and data.

## Model and parameters

State arrays use `(Ru, xu, Rd, xd)`: upstream resource and cooperation, followed by downstream resource and cooperation. `Parameters` stores `a`, `f`, `e_c`, `e_d`, and optional downstream extraction rates. Here `k = f/a`; `e_c` and `e_d` correspond to the manuscript's normalized extraction rates, not per-capita extraction rates.

All six cases use `a = 1`, upstream cooperative extraction 0.3, and defector extraction 1.5 in both communities:

| Case | k | Downstream cooperative extraction |
|---|---:|---:|
| MC | 0.05 | 0.60 |
| CC | 0.25 | 0.80 |
| 0C | 0.80 | 0.60 |
| MM | 0.15 | 0.60 |
| CM | 0.25 | 0.60 |
| 0M | 0.80 | 0.20 |

For Figure 1, case index `i` uses `SeedSequence(82701 + 1009*i).spawn(50)`. Each child produces its run seed using `generate_state(1, dtype=uint32)`. The deterministic comparison uses DOP853 with relative tolerance 1e-9 and absolute tolerance 1e-11. Figure 2 uses root seed 82702 and explicitly selects Uniform(0,1); the generic initial-state helper retains its historical default range, which the runner overrides.

In the ABM, a cooperator is encoded as 1 and a defector as 0. Each step attempts one strategy update per community using the old resources, then advances both resources by Euler with step 1/N. Resources are clipped to [0,1], as specified for the microscopic implementation. The deterministic model permits downstream stocks above 1. Imitation probabilities materially outside [0,1] are rejected; only numerical roundoff is clipped.

The equilibrium maps describe selected locally stable positive-resource branches conditional on upstream supply. They are not global basin maps. At upstream depletion, `upstream_selected_branch` reports cooperation as NaN because the model has a continuum of depleted equilibria. The retained general helper `main_branch_equilibrium` returns one representative depleted point with upstream cooperation 1; this does not select the asymptotic cooperation of a trajectory. Critical cells in the downstream grid are marked `K` and contain the continuous limiting values of adjacent branches. Finite-time near-depletion counts use R < 1e-4 and do not establish asymptotic collapse.

## Files and provenance

| File | Purpose |
|---|---|
| `cpr_updown/model.py` | State equations, parameters and equilibrium helpers |
| `cpr_updown/numerics.py` | RK4 and resource/logit-strategy integration |
| `cpr_updown/abm.py` | Connected random networks, update rule and stochastic simulation |
| `cpr_updown/cases.py` | Six original parameter cases and DOP853 comparison |
| `cpr_updown/initial_conditions.py` | Seeded initial-state generation |
| `cpr_updown/upstream.py` | Upstream branch and depletion-manifold representation |
| `cpr_updown/downstream.py` | Downstream branches and threshold crossings |
| `cpr_updown/equilibrium_grid.py` | Conditional equilibrium grids |
| `run.py` | Standalone computation and output entry point |

The three core calculation modules were copied byte-for-byte from the research workspace. Computational definitions in the other modules were extracted unchanged from existing analysis scripts, with plotting imports and machine-specific paths omitted. `run.py` supplies packaging and output orchestration; model equations, formal experiment parameters and random-seed conventions were retained. No input files outside this repository are required.

Release validation covers syntax and imports, existing quick model/ABM checks, short runs of all entry points, and comparisons against original functions and saved data subsets. The full Figure 1 ensemble and full Figure 2 experiment were not rerun for this release.

## Citation and contact

Please cite the associated manuscript by Tianyu Shi, Junhao Lu, Hanqin Qian and Chengyi Tu, using its title above. This repository does not assign a publication DOI. For questions about the implementation, open a GitHub issue. When reporting results, record the repository commit and software versions.

## License

MIT License; see [LICENSE](LICENSE).
