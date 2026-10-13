from __future__ import annotations

import numpy as np
from scipy.special import expit

from .model import Parameters, extraction, rhs


def _bounded_state(y: np.ndarray) -> np.ndarray:
    bounded = np.asarray(y, dtype=float).copy()
    bounded[..., 0] = np.maximum(bounded[..., 0], 0.0)
    bounded[..., 2] = np.maximum(bounded[..., 2], 0.0)
    bounded[..., 1] = np.clip(bounded[..., 1], 0.0, 1.0)
    bounded[..., 3] = np.clip(bounded[..., 3], 0.0, 1.0)
    return bounded


def rk4_step(y: np.ndarray, dt: float, p: Parameters) -> np.ndarray:
    k1 = rhs(0.0, y, p)
    k2 = rhs(0.0, y + 0.5 * dt * k1, p)
    k3 = rhs(0.0, y + 0.5 * dt * k2, p)
    k4 = rhs(0.0, y + dt * k3, p)
    return _bounded_state(y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


def integrate_rk4(
    p: Parameters,
    y0: np.ndarray,
    t_end: float,
    dt: float = 0.02,
    sample_dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    steps = int(round(t_end / dt))
    sample_every = int(round(sample_dt / dt))
    if not np.isclose(steps * dt, t_end):
        raise ValueError("t_end must be divisible by dt")
    if sample_every < 1 or not np.isclose(sample_every * dt, sample_dt):
        raise ValueError("sample_dt must be a positive multiple of dt")

    y = _bounded_state(np.asarray(y0, dtype=float))
    samples = [y.copy()]
    times = [0.0]
    for step in range(1, steps + 1):
        y = rk4_step(y, dt, p)
        if step % sample_every == 0:
            samples.append(y.copy())
            times.append(step * dt)
    return np.asarray(times), np.asarray(samples)


def integrate_logit_rk4(
    p: Parameters,
    y0: np.ndarray,
    t_end: float,
    dt: float = 0.02,
    sample_dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate interior strategy states without roundoff locking at 0 or 1."""
    initial = _bounded_state(np.asarray(y0, dtype=float))
    if np.any((initial[..., (1, 3)] <= 0.0) | (initial[..., (1, 3)] >= 1.0)):
        raise ValueError("logit integration requires strategy initial states strictly inside (0, 1)")

    transformed = initial.copy()
    transformed[..., 1] = np.log(initial[..., 1]) - np.log1p(-initial[..., 1])
    transformed[..., 3] = np.log(initial[..., 3]) - np.log1p(-initial[..., 3])

    steps = int(round(t_end / dt))
    sample_every = int(round(sample_dt / dt))
    if not np.isclose(steps * dt, t_end):
        raise ValueError("t_end must be divisible by dt")
    if sample_every < 1 or not np.isclose(sample_every * dt, sample_dt):
        raise ValueError("sample_dt must be a positive multiple of dt")

    samples = [_from_logit_state(transformed)]
    times = [0.0]
    for step in range(1, steps + 1):
        k1 = _logit_rhs(transformed, p)
        k2 = _logit_rhs(transformed + 0.5 * dt * k1, p)
        k3 = _logit_rhs(transformed + 0.5 * dt * k2, p)
        k4 = _logit_rhs(transformed + dt * k3, p)
        transformed = transformed + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        transformed[..., 0] = np.maximum(transformed[..., 0], 0.0)
        transformed[..., 2] = np.maximum(transformed[..., 2], 0.0)
        if step % sample_every == 0:
            samples.append(_from_logit_state(transformed))
            times.append(step * dt)
    return np.asarray(times), np.asarray(samples)


def _logit_rhs(values: np.ndarray, p: Parameters) -> np.ndarray:
    ru, zu, rd, zd = np.moveaxis(np.asarray(values, dtype=float), -1, 0)
    xu = expit(zu)
    xd = expit(zd)
    du = p.a * ru * (1.0 - p.k - ru - extraction(xu, p.e_c_u, p.e_d_u))
    dzu = ru * (1.0 - 2.0 * ru)
    dd = p.a * (
        rd * (1.0 - rd - extraction(xd, p.downstream_e_c, p.downstream_e_d))
        + p.k * ru
    )
    dzd = rd * (1.0 - 2.0 * rd)
    return np.stack((du, dzu, dd, dzd), axis=-1)


def _from_logit_state(values: np.ndarray) -> np.ndarray:
    state = np.asarray(values, dtype=float).copy()
    state[..., 1] = expit(state[..., 1])
    state[..., 3] = expit(state[..., 3])
    return state


def advance_until(
    p: Parameters,
    y0: np.ndarray,
    t_max: float = 1200.0,
    dt: float = 0.02,
    derivative_tolerance: float = 1e-8,
    check_every: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    y = _bounded_state(np.asarray(y0, dtype=float))
    steps = int(round(t_max / dt))
    check_steps = max(1, int(round(check_every / dt)))
    converged = np.zeros(y.shape[:-1], dtype=bool)
    for step in range(1, steps + 1):
        y = rk4_step(y, dt, p)
        if step % check_steps == 0:
            speed = np.max(np.abs(rhs(0.0, y, p)), axis=-1)
            converged |= speed < derivative_tolerance
            if np.all(converged):
                break
    return y, converged
