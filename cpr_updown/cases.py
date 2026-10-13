"""Manuscript parameter cases and deterministic comparison trajectories."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from .model import Parameters, rhs


@dataclass(frozen=True)
class HeterogeneousFigureCase:
    label: str
    scenario: str
    parameters: Parameters


def figure1_six_cases() -> list[HeterogeneousFigureCase]:
    return [
        HeterogeneousFigureCase("MC", "I", Parameters(a=1.0, f=0.050, e_c=0.30, e_c_d=0.60)),
        HeterogeneousFigureCase("CC", "II", Parameters(a=1.0, f=0.250, e_c=0.30, e_c_d=0.80)),
        HeterogeneousFigureCase("0C", "III", Parameters(a=1.0, f=0.800, e_c=0.30, e_c_d=0.60)),
        HeterogeneousFigureCase("MM", "I", Parameters(a=1.0, f=0.150, e_c=0.30, e_c_d=0.60)),
        HeterogeneousFigureCase("CM", "II", Parameters(a=1.0, f=0.250, e_c=0.30, e_c_d=0.60)),
        HeterogeneousFigureCase("0M", "III", Parameters(a=1.0, f=0.800, e_c=0.30, e_c_d=0.20)),
    ]


def _ode_trajectory(p: Parameters, time: np.ndarray, y0: np.ndarray) -> np.ndarray:
    solution = solve_ivp(
        lambda t, y: rhs(t, y, p),
        (float(time[0]), float(time[-1])),
        y0,
        t_eval=time,
        rtol=1e-9,
        atol=1e-11,
        method="DOP853",
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T
