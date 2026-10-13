"""Conditional positive-resource equilibrium grids used in Figure 4."""
from __future__ import annotations
import numpy as np
from .downstream import E_D_D, THETA_U, K_COLLAPSE, K_MAX, driver_for_scenario, candidate_branches


E_C_D_MIN = 0.40


E_C_D_MAX = 0.85


N_K = 501


N_E_C_D = 451


def scenario_specs() -> list[dict[str, float | str]]:
    return [
        {"label": "I", "name": "low loss", "k_min": 0.0, "k_max": THETA_U},
        {
            "label": "II",
            "name": "intermediate loss",
            "k_min": THETA_U,
            "k_max": K_COLLAPSE,
        },
        {
            "label": "III",
            "name": "upstream collapse",
            "k_min": K_COLLAPSE,
            "k_max": K_MAX,
        },
    ]


def _scenario_spec(label: str) -> dict[str, float | str]:
    try:
        return next(spec for spec in scenario_specs() if spec["label"] == label)
    except StopIteration as error:
        raise ValueError(f"Unknown scenario: {label}") from error


def stable_equilibrium_grid(label: str) -> dict[str, np.ndarray]:
    spec = _scenario_spec(label)
    k = np.linspace(float(spec["k_min"]), float(spec["k_max"]), N_K)
    e_c_d = np.linspace(E_C_D_MIN, E_C_D_MAX, N_E_C_D)
    k_mesh, e_c_d_mesh = np.meshgrid(k, e_c_d)
    driver = driver_for_scenario(label, k_mesh)
    branches = candidate_branches(driver, e_c_d=e_c_d_mesh)
    critical_boundary = np.isclose(
        driver, e_c_d_mesh - 0.5, rtol=0.0, atol=1e-12
    ) | np.isclose(driver, E_D_D - 0.5, rtol=0.0, atol=1e-12)

    resource = np.full(k_mesh.shape, np.nan, dtype=float)
    cooperation = np.full(k_mesh.shape, np.nan, dtype=float)
    state = np.full(k_mesh.shape, "N", dtype="<U1")
    selected_count = np.zeros(k_mesh.shape, dtype=np.int8)
    for branch_name in ("C", "M", "D"):
        branch = branches[branch_name]
        selected = np.asarray(branch["stable"]) & np.asarray(branch["feasible"])
        selected_count += selected
        resource[selected] = np.asarray(branch["resource"])[selected]
        cooperation[selected] = np.asarray(branch["cooperation"])[selected]
        state[selected] = branch_name

    if not np.all(selected_count == 1):
        raise AssertionError("Every parameter cell must select exactly one stable branch.")
    state[critical_boundary] = "K"

    return {
        "k": k,
        "e_c_d": e_c_d,
        "k_mesh": k_mesh,
        "e_c_d_mesh": e_c_d_mesh,
        "g": driver,
        "R_d": resource,
        "x_d": cooperation,
        "state": state,
        "critical_boundary": critical_boundary,
    }


def phase_boundary(label: str) -> tuple[np.ndarray, np.ndarray]:
    spec = _scenario_spec(label)
    k = np.linspace(float(spec["k_min"]), float(spec["k_max"]), N_K)
    return k, 0.5 + driver_for_scenario(label, k)
