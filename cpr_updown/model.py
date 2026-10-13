from __future__ import annotations

from dataclasses import dataclass

import numpy as np


E_C_STAR = 2.0 - np.sqrt(2.0)


@dataclass(frozen=True)
class Parameters:
    a: float
    f: float
    e_c: float
    e_d: float = 1.5
    e_c_d: float | None = None
    e_d_d: float | None = None

    @property
    def k(self) -> float:
        return self.f / self.a

    @property
    def e_c_u(self) -> float:
        return self.e_c

    @property
    def e_d_u(self) -> float:
        return self.e_d

    @property
    def downstream_e_c(self) -> float:
        return self.e_c if self.e_c_d is None else self.e_c_d

    @property
    def downstream_e_d(self) -> float:
        return self.e_d if self.e_d_d is None else self.e_d_d


def extraction(x: np.ndarray | float, e_c: float, e_d: float) -> np.ndarray | float:
    return e_d - (e_d - e_c) * x


def rhs(t: float, y: np.ndarray, p: Parameters) -> np.ndarray:
    del t
    values = np.asarray(y, dtype=float)
    ru, xu, rd, xd = np.moveaxis(values, -1, 0)
    du = p.a * ru * (1.0 - p.k - ru - extraction(xu, p.e_c_u, p.e_d_u))
    dxu = ru * (1.0 - 2.0 * ru) * xu * (1.0 - xu)
    dd = p.a * (
        rd * (1.0 - rd - extraction(xd, p.downstream_e_c, p.downstream_e_d))
        + p.k * ru
    )
    dxd = rd * (1.0 - 2.0 * rd) * xd * (1.0 - xd)
    return np.stack((du, dxu, dd, dxd), axis=-1)


def regime_boundaries(e_c: float) -> dict[str, float]:
    k1 = float(np.round(1.0 - e_c, 15))
    if e_c < 0.5:
        return {"k0": 0.5 - e_c, "k1": k1}
    if e_c < E_C_STAR:
        root = np.sqrt(e_c * e_c - 4.0 * e_c + 2.0)
        return {
            "k_minus": (1.0 - e_c - root) / 2.0,
            "k_plus": (1.0 - e_c + root) / 2.0,
            "k1": k1,
        }
    return {"k1": k1}


def main_branch_equilibrium(
    e_c: float,
    e_d: float,
    k: float,
    e_c_d: float | None = None,
    e_d_d: float | None = None,
) -> tuple[np.ndarray, str]:
    downstream_e_c = e_c if e_c_d is None else e_c_d
    downstream_e_d = e_d if e_d_d is None else e_d_d
    delta_e_u = e_d - e_c
    delta_e_d = downstream_e_d - downstream_e_c
    k1 = 1.0 - e_c

    if e_c < 0.5 and k < 0.5 - e_c:
        ru = 0.5
        xu = (e_d - 0.5 + k) / delta_e_u
        inflow = k * ru
        if 2.0 * inflow < downstream_e_c - 0.5:
            rd = (
                (1.0 - downstream_e_c)
                + np.sqrt((1.0 - downstream_e_c) ** 2 + 4.0 * inflow)
            ) / 2.0
            return np.array([ru, xu, rd, 1.0]), "MC"
        rd = 0.5
        xd = (downstream_e_d - 0.5 - 2.0 * inflow) / delta_e_d
        return np.array([ru, xu, rd, xd]), "MM"

    if k < k1:
        ru = 1.0 - e_c - k
        xu = 1.0
        inflow = k * ru
        if 2.0 * inflow > downstream_e_c - 0.5:
            rd = 0.5
            xd = (downstream_e_d - 0.5 - 2.0 * inflow) / delta_e_d
            return np.array([ru, xu, rd, xd]), "CM"

        rd = (
            (1.0 - downstream_e_c)
            + np.sqrt((1.0 - downstream_e_c) ** 2 + 4.0 * inflow)
        ) / 2.0
        return np.array([ru, xu, rd, 1.0]), "CC"

    if downstream_e_c < 0.5:
        return (
            np.array([0.0, 1.0, 0.5, (downstream_e_d - 0.5) / delta_e_d]),
            "0M",
        )
    return np.array([0.0, 1.0, 1.0 - downstream_e_c, 1.0]), "0C"


def classify_resources(points: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    ru_alive = values[..., 0] > threshold
    rd_alive = values[..., 2] > threshold
    labels = np.empty(ru_alive.shape, dtype=object)
    labels[ru_alive & rd_alive] = "both persist"
    labels[~ru_alive & rd_alive] = "upstream collapse"
    labels[~ru_alive & ~rd_alive] = "both collapse"
    labels[ru_alive & ~rd_alive] = "downstream collapse"
    return labels
