Source Data

Re-entrant Cooperation under Directional Resource Transfer in Coupled Common-Pool Resource Systems

This archive contains the existing numerical source data for Figures 1–4.
All 32 data/configuration files retain their original contents. File checksums,
CSV columns and array shapes are listed in data_manifest.json.

Figure1/: Six parameter cases. CSV files contain time, deterministic trajectories
(ode_*) and agent-based ensemble means (abm_mean_*). NPZ files also contain
the 50 individual simulation runs and their seeds. See config.json for settings.

Figure2/: Six parameter cases with 50 shared initial states sampled independently
from Uniform(0,1), seed 82702. initial_conditions.csv contains these states;
trajectories_*.npz contains trajectories; endpoints_*.csv and summary_t100.csv
contain finite-time outcomes at t=100. R<1e-4 identifies near depletion at that
time and is not an asymptotic-state classification.

Figure3/: Analytical upstream equilibrium branches and configuration. Missing
selected cooperation values on the depleted branch represent the continuum of
possible cooperation states, not a uniquely selected equilibrium fraction.

Figure4/: Arrays and configuration for the downstream positive-resource
equilibrium map conditional on the selected upstream equilibrium branch.

Variables: Ru/xu are upstream resource/cooperation; Rd/xd are downstream
resource/cooperation. The state order in trajectory arrays is Ru, xu, Rd, xd.
Historical file labels C, M and 0 refer to full cooperation, mixed strategies
and resource depletion, respectively. Use the accompanying configurations
and array/column descriptions to interpret each dataset.

CSV and JSON files are readable with standard text/data tools. NPZ files can
be opened with NumPy using numpy.load(path, allow_pickle=False).
Calculation code: https://github.com/lunarfairy/reentrant-cooperation-cpr
