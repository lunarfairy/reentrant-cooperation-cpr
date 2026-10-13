"""Selected upstream equilibrium branches, including the depletion manifold."""
from __future__ import annotations
import numpy as np


E_C_U = 0.30


E_D_U = 1.50


THETA_U = 0.50 - E_C_U


K_COLLAPSE = 1.00 - E_C_U


N_K = 2001


def upstream_selected_branch(k: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.asarray(k)
    if np.iscomplexobj(raw):
        raise ValueError("k must be real, finite and non-negative")
    values = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("k must be finite and non-negative")

    resource = np.zeros_like(values)
    cooperation = np.full_like(values, np.nan)
    scenario = np.full(values.shape, "III", dtype="<U3")
    state = np.full(values.shape, "0", dtype="<U1")
    collapse_manifold = np.ones_like(values, dtype=bool)
    critical_boundary = (values == THETA_U) | (values == K_COLLAPSE)

    mixed = values < THETA_U
    cooperative = (values >= THETA_U) & (values < K_COLLAPSE)
    resource[mixed] = 0.5
    cooperation[mixed] = (E_D_U - 0.5 + values[mixed]) / (E_D_U - E_C_U)
    scenario[mixed] = "I"
    state[mixed] = "M"
    collapse_manifold[mixed] = False
    resource[cooperative] = 1.0 - E_C_U - values[cooperative]
    cooperation[cooperative] = 1.0
    scenario[cooperative] = "II"
    state[cooperative] = "C"
    collapse_manifold[cooperative] = False

    return {
        "k": values,
        "R_u": resource,
        "x_u": cooperation,
        "scenario": scenario,
        "state": state,
        "collapse_manifold": collapse_manifold,
        "critical_boundary": critical_boundary,
    }
