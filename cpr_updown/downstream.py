"""Downstream analytical branches conditional on upstream supply."""
from __future__ import annotations
import numpy as np


E_C_U = 0.30


E_D_U = 1.50


E_C_D = 0.72


E_D_D = 1.50


THETA_U = 0.50 - E_C_U


K_COLLAPSE = 1.00 - E_C_U


K_MAX = 1.00


def driver_for_scenario(label: str, k: np.ndarray) -> np.ndarray:
    values = np.asarray(k, dtype=float)
    if label == "I":
        return values.copy()
    if label == "II":
        return 2.0 * values * (1.0 - E_C_U - values)
    if label == "III":
        return np.zeros_like(values)
    raise ValueError(f"Unknown scenario: {label}")


def _resource_candidate(driver: np.ndarray, extraction_rate: float) -> np.ndarray:
    intercept = 1.0 - extraction_rate
    return (intercept + np.sqrt(intercept**2 + 2.0 * driver)) / 2.0


def candidate_branches(
    driver: np.ndarray, e_c_d: float = E_C_D
) -> dict[str, dict[str, np.ndarray]]:
    g = np.asarray(driver, dtype=float)
    delta_e = E_D_D - e_c_d
    theta_c = e_c_d - 0.5
    theta_d = E_D_D - 0.5

    resource_c = _resource_candidate(g, e_c_d)
    resource_m = np.full_like(g, 0.5)
    resource_d = _resource_candidate(g, E_D_D)
    cooperation_c = np.ones_like(g)
    cooperation_m = (E_D_D - 0.5 - g) / delta_e
    cooperation_d = np.zeros_like(g)

    feasible_m = (cooperation_m >= 0.0) & (cooperation_m <= 1.0)
    return {
        "C": {
            "resource": resource_c,
            "cooperation": cooperation_c,
            "feasible": np.ones_like(g, dtype=bool),
            "stable": g <= theta_c,
        },
        "M": {
            "resource": resource_m,
            "cooperation": cooperation_m,
            "feasible": feasible_m,
            "stable": feasible_m & (g > theta_c) & (g < theta_d),
        },
        "D": {
            "resource": resource_d,
            "cooperation": cooperation_d,
            "feasible": np.ones_like(g, dtype=bool),
            "stable": g >= theta_d,
        },
    }


def threshold_crossings(label: str, e_c_d: float = E_C_D) -> np.ndarray:
    theta_c = e_c_d - 0.5
    if label == "I":
        roots = np.array([theta_c])
        domain = (0.0, THETA_U)
    elif label == "II":
        discriminant = (1.0 - E_C_U) ** 2 - 2.0 * theta_c
        if discriminant < 0.0:
            return np.array([])
        root = np.sqrt(discriminant)
        roots = np.array(
            [
                ((1.0 - E_C_U) - root) / 2.0,
                ((1.0 - E_C_U) + root) / 2.0,
            ]
        )
        domain = (THETA_U, K_COLLAPSE)
    elif label == "III":
        return np.array([])
    else:
        raise ValueError(f"Unknown scenario: {label}")
    return roots[(roots > domain[0]) & (roots < domain[1])]
