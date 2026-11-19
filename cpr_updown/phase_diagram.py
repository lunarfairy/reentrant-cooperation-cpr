"""Full parameter-plane phase-diagram classification and 4D Jacobian validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .model import E_C_STAR, Parameters, extraction, main_branch_equilibrium, rhs

A_VAL = 1.0
E_CU_BASE = 0.30
E_DU_BASE = 1.50
E_DD_BASE = 1.50
DEFAULT_GRID = 400


def jacobian_4x4(y: np.ndarray, p: Parameters) -> np.ndarray:
    """Exact 4x4 Jacobian of the coupled system in state order (Ru, xu, Rd, xd)."""
    ru, xu, rd, xd = [float(v) for v in y]
    eu = float(extraction(xu, p.e_c_u, p.e_d_u))
    ed = float(extraction(xd, p.downstream_e_c, p.downstream_e_d))
    J = np.zeros((4, 4), dtype=float)
    J[0, 0] = p.a * (1.0 - 2.0 * ru - eu - p.k)
    J[0, 1] = p.a * ru * (p.e_d_u - p.e_c_u)
    J[1, 0] = (1.0 - 4.0 * ru) * xu * (1.0 - xu)
    J[1, 1] = ru * (1.0 - 2.0 * ru) * (1.0 - 2.0 * xu)
    J[2, 0] = p.a * p.k
    J[2, 2] = p.a * (1.0 - 2.0 * rd - ed)
    J[2, 3] = p.a * rd * (p.downstream_e_d - p.downstream_e_c)
    J[3, 2] = (1.0 - 4.0 * rd) * xd * (1.0 - xd)
    J[3, 3] = rd * (1.0 - 2.0 * rd) * (1.0 - 2.0 * xd)
    return J


def classify_asymmetric_plane(
    k: np.ndarray, e_cd: np.ndarray, e_cu: float = E_CU_BASE
) -> tuple[np.ndarray, np.ndarray]:
    """Primary codes (0..5: MC,MM,CC,CM,0C,0M) and multistable codes (0..7) on (k, e_C,d)."""
    theta_u = 0.5 - e_cu
    k_c = 1.0 - e_cu
    primary = np.empty(k.shape, dtype=np.int32)
    full_code = np.empty(k.shape, dtype=np.int32)

    reg1 = k < theta_u
    reg2 = (k >= theta_u) & (k < k_c)
    reg3 = k >= k_c
    g2 = 2.0 * k * (1.0 - e_cu - k)

    primary[reg1 & (e_cd > 0.5 + k)] = 0
    primary[reg1 & (e_cd <= 0.5 + k)] = 1
    primary[reg2 & (e_cd > 0.5 + g2)] = 2
    primary[reg2 & (e_cd <= 0.5 + g2)] = 3
    primary[reg3 & (e_cd >= 0.5)] = 4
    primary[reg3 & (e_cd < 0.5)] = 5

    full_code[primary == 0] = 0
    full_code[(primary == 1) & (e_cd >= 0.5)] = 1
    full_code[(primary == 1) & (e_cd < 0.5)] = 2
    full_code[primary == 2] = 3
    full_code[(primary == 3) & (e_cd >= 0.5)] = 4
    full_code[(primary == 3) & (e_cd < 0.5)] = 5
    full_code[primary == 4] = 6
    full_code[primary == 5] = 7
    return primary, full_code


def classify_symmetric_plane(k: np.ndarray, e_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Primary codes (1..5: MM,CC,CM,0C,0M) and multistable codes (2..7) on (k, e_C)."""
    theta_u = 0.5 - e_c
    k_c = 1.0 - e_c
    primary = np.empty(k.shape, dtype=np.int32)
    full_code = np.empty(k.shape, dtype=np.int32)

    reg1 = (e_c < 0.5) & (k < theta_u)
    reg2 = (k >= np.maximum(0.0, theta_u)) & (k < k_c)
    reg3 = k >= k_c
    g2 = 2.0 * k * (1.0 - e_c - k)

    primary[reg1] = 1
    primary[reg2 & (e_c > 0.5 + g2)] = 2
    primary[reg2 & (e_c <= 0.5 + g2)] = 3
    primary[reg3 & (e_c >= 0.5)] = 4
    primary[reg3 & (e_c < 0.5)] = 5

    full_code[primary == 1] = 2
    full_code[primary == 2] = 3
    full_code[(primary == 3) & (e_c >= 0.5)] = 4
    full_code[(primary == 3) & (e_c < 0.5)] = 5
    full_code[primary == 4] = 6
    full_code[primary == 5] = 7
    return primary, full_code


def _jacobian_batch(
    eq_grid: np.ndarray,
    a_val: float,
    k_grid: np.ndarray,
    e_cu_grid: np.ndarray,
    e_du_val: float,
    e_cd_grid: np.ndarray,
    e_dd_val: float,
) -> np.ndarray:
    ru, xu, rd, xd = eq_grid[..., 0], eq_grid[..., 1], eq_grid[..., 2], eq_grid[..., 3]
    eu = xu * e_cu_grid + (1.0 - xu) * e_du_val
    ed = xd * e_cd_grid + (1.0 - xd) * e_dd_val
    J = np.zeros(ru.shape + (4, 4), dtype=float)
    J[..., 0, 0] = a_val * (1.0 - 2.0 * ru - eu - k_grid)
    J[..., 0, 1] = a_val * ru * (e_du_val - e_cu_grid)
    J[..., 1, 0] = (1.0 - 4.0 * ru) * xu * (1.0 - xu)
    J[..., 1, 1] = ru * (1.0 - 2.0 * ru) * (1.0 - 2.0 * xu)
    J[..., 2, 0] = a_val * k_grid
    J[..., 2, 2] = a_val * (1.0 - 2.0 * rd - ed)
    J[..., 2, 3] = a_val * rd * (e_dd_val - e_cd_grid)
    J[..., 3, 2] = (1.0 - 4.0 * rd) * xd * (1.0 - xd)
    J[..., 3, 3] = rd * (1.0 - 2.0 * rd) * (1.0 - 2.0 * xd)
    return J


def _rhs_residual_batch(
    eq_grid: np.ndarray,
    a_val: float,
    k_grid: np.ndarray,
    e_cu_grid: np.ndarray,
    e_du_val: float,
    e_cd_grid: np.ndarray,
    e_dd_val: float,
) -> np.ndarray:
    ru, xu, rd, xd = eq_grid[..., 0], eq_grid[..., 1], eq_grid[..., 2], eq_grid[..., 3]
    eu = xu * e_cu_grid + (1.0 - xu) * e_du_val
    ed = xd * e_cd_grid + (1.0 - xd) * e_dd_val
    dru = a_val * ru * (1.0 - ru - eu - k_grid)
    dxu = ru * (1.0 - 2.0 * ru) * xu * (1.0 - xu)
    drd = a_val * (rd * (1.0 - rd - ed) + k_grid * ru)
    dxd = rd * (1.0 - 2.0 * rd) * xd * (1.0 - xd)
    return np.maximum(np.maximum(np.abs(dru), np.abs(dxu)), np.maximum(np.abs(drd), np.abs(dxd)))


def _bifurcation_distance_asymmetric(k: np.ndarray, e_cd: np.ndarray, e_cu: float = E_CU_BASE) -> np.ndarray:
    theta_u, k_c = 0.5 - e_cu, 1.0 - e_cu
    gs = np.where(k < theta_u, k, np.where(k < k_c, 2.0 * k * (k_c - k), 0.0))
    return np.minimum(np.minimum(np.abs(k - theta_u), np.abs(k - k_c)), np.abs(e_cd - (0.5 + gs)))


def _bifurcation_distance_symmetric(k: np.ndarray, e_c: np.ndarray) -> np.ndarray:
    theta_u, k_c = 0.5 - e_c, 1.0 - e_c
    gs = np.where((e_c < 0.5) & (k < theta_u), k, np.where(k < k_c, 2.0 * k * (k_c - k), 0.0))
    d_tu = np.where(e_c <= 0.5 + 1e-9, np.abs(k - theta_u), np.inf)
    return np.minimum(np.minimum(d_tu, np.abs(k - k_c)), np.abs(e_c - (0.5 + gs)))


def verify_phase_diagrams(n_grid: int = DEFAULT_GRID) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute asymmetric and symmetric phase grids and verify ODE residuals and 4D Jacobians."""
    code_map = {"MC": 0, "MM": 1, "CC": 2, "CM": 3, "0C": 4, "0M": 5}
    k_vec = np.linspace(0.0, 1.0, n_grid)
    e_vec = np.linspace(0.002, 0.998, n_grid)
    KK, EE = np.meshgrid(k_vec, e_vec)

    prim_a, full_a = classify_asymmetric_plane(KK, EE, E_CU_BASE)
    prim_b, full_b = classify_symmetric_plane(KK, EE)
    dist_a = _bifurcation_distance_asymmetric(KK, EE, E_CU_BASE)
    dist_b = _bifurcation_distance_symmetric(KK, EE)

    eq_a = np.empty((n_grid, n_grid, 4), dtype=float)
    eq_b = np.empty((n_grid, n_grid, 4), dtype=float)
    mismatches_a = mismatches_b = 0
    for i in range(n_grid):
        ev = float(e_vec[i])
        for j in range(n_grid):
            kv = float(k_vec[j])
            ya, ca = main_branch_equilibrium(E_CU_BASE, E_DU_BASE, kv, e_c_d=ev, e_d_d=E_DD_BASE)
            eq_a[i, j] = ya
            if code_map[ca] != int(prim_a[i, j]) and dist_a[i, j] > 1e-9:
                mismatches_a += 1

            yb, cb = main_branch_equilibrium(ev, E_DU_BASE, kv)
            eq_b[i, j] = yb
            if code_map[cb] != int(prim_b[i, j]) and dist_b[i, j] > 1e-9:
                mismatches_b += 1

    if mismatches_a or mismatches_b:
        raise RuntimeError(f"Phase classification mismatch: asymmetric={mismatches_a}, symmetric={mismatches_b}")

    e_cu_const = np.full_like(KK, E_CU_BASE)
    res_a = _rhs_residual_batch(eq_a, A_VAL, KK, e_cu_const, E_DU_BASE, EE, E_DD_BASE)
    res_b = _rhs_residual_batch(eq_b, A_VAL, KK, EE, E_DU_BASE, EE, E_DD_BASE)
    max_res_a, max_res_b = float(np.max(res_a)), float(np.max(res_b))
    if max_res_a >= 1e-11 or max_res_b >= 1e-11:
        raise RuntimeError(f"ODE residual exceeded tolerance: a={max_res_a}, b={max_res_b}")

    eigs_a = np.linalg.eigvals(_jacobian_batch(eq_a, A_VAL, KK, e_cu_const, E_DU_BASE, EE, E_DD_BASE))
    eigs_b = np.linalg.eigvals(_jacobian_batch(eq_b, A_VAL, KK, EE, E_DU_BASE, EE, E_DD_BASE))
    max_re_a = np.max(np.where(np.abs(eigs_a) > 1e-9, np.real(eigs_a), -np.inf), axis=-1)
    max_re_b = np.max(np.where(np.abs(eigs_b) > 1e-9, np.real(eigs_b), -np.inf), axis=-1)

    unstable_a = int(np.sum((dist_a > 1e-9) & (max_re_a >= 0.0)))
    unstable_b = int(np.sum((dist_b > 1e-9) & (max_re_b >= 0.0)))
    if unstable_a or unstable_b:
        raise RuntimeError(f"Unstable interior eigenvalues: asymmetric={unstable_a}, symmetric={unstable_b}")

    # Spot-check ODE convergence to coexisting primary and depletion attractors
    p_cm = Parameters(a=A_VAL, f=0.35, e_c=E_CU_BASE, e_d=E_DU_BASE, e_c_d=0.62, e_d_d=E_DD_BASE)
    xu_crit = min(1.0, (E_DU_BASE - 1.0 + 0.35) / (E_DU_BASE - E_CU_BASE))
    xd_crit = (E_DD_BASE - 1.0) / (E_DD_BASE - 0.62)
    sample_converged = {}
    for label, y0 in (
        ("primary", np.array([0.48, 0.88, 0.48, 0.88])),
        ("upstream_depleted", np.array([1e-5, 0.15 * xu_crit, 0.45, 0.90])),
        ("doubly_depleted", np.array([1e-5, 0.15 * xu_crit, 1e-5, 0.15 * xd_crit])),
    ):
        sol = solve_ivp(lambda t, y: rhs(t, y, p_cm), (0.0, 200.0), y0, rtol=1e-9, atol=1e-11)
        sample_converged[label] = [round(float(v), 5) for v in sol.y[:, -1]]

    summary: dict[str, Any] = {
        "grid_shape": [n_grid, n_grid],
        "points_per_plane": int(n_grid * n_grid),
        "baseline_parameters": {"a": A_VAL, "e_c_u": E_CU_BASE, "e_d_u": E_DU_BASE, "e_d_d": E_DD_BASE},
        "symmetric_dome_apex": {"k_star": round(float((np.sqrt(2.0) - 1.0) / 2.0), 6), "e_c_star": round(float(E_C_STAR), 6)},
        "mismatches": {"asymmetric": mismatches_a, "symmetric": mismatches_b},
        "max_ode_residual": {"asymmetric": float(f"{max_res_a:.3e}"), "symmetric": float(f"{max_res_b:.3e}")},
        "unstable_interior_points": {"asymmetric": unstable_a, "symmetric": unstable_b},
        "sample_multistable_convergence_CM": sample_converged,
    }
    grids = {
        "k_vec": k_vec,
        "e_vec": e_vec,
        "prim_a": prim_a,
        "full_a": full_a,
        "prim_b": prim_b,
        "full_b": full_b,
        "max_ode_res_a": res_a,
        "max_ode_res_b": res_b,
        "max_re_nonzero_eig_a": max_re_a,
        "max_re_nonzero_eig_b": max_re_b,
    }
    return summary, grids


def run_phase_diagram(out: Path, smoke: bool) -> dict[str, Any]:
    """Compute and save full parameter-plane phase-diagram arrays and validation summary."""
    n_grid = 25 if smoke else DEFAULT_GRID
    summary, grids = verify_phase_diagrams(n_grid=n_grid)
    np.savez_compressed(out / "phase_diagram_grids.npz", **grids)
    (out / "phase_diagram_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Phase diagrams: {n_grid}x{n_grid} per plane, 0 mismatches, max residual={summary['max_ode_residual']}", flush=True)
    return summary
