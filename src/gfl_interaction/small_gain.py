from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import eigvalsh, solve_continuous_lyapunov
from scipy.optimize import minimize_scalar, root

from .energies import critical_energy, potential_energy
from .model import GFLParameters


@dataclass(frozen=True)
class LocalSmallGainResult:
    current_bandwidth_hz: float
    gamma: float
    peak_frequency_rad_s: float
    peak_frequency_hz: float
    equilibrium_damping: float
    gamma_over_damping: float
    condition_passes: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedRealStorage:
    gamma: float
    matrix: np.ndarray
    residual_max_eigenvalue: float
    positive_definite: bool


@dataclass(frozen=True)
class SwingCertificateAudit:
    current_bandwidth_hz: float
    initial_slow_energy: float
    critical_energy: float
    reserved_critical_energy: float
    initial_over_critical: float
    initial_over_reserved_critical: float
    one_shot_passes_before_fast_penalty: bool
    local_small_gain: LocalSmallGainResult

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fast_residual_state_space(
    params: GFLParameters, current_bandwidth_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearized current-loop residual from PLL speed to slow power mismatch.

    The fast state is ``e=[Delta I_d, Delta dot I_d]`` and the port is
    ``e_dot=Ae+B omega, y=Ce`` in the energy-scaled swing equation.
    """
    kp, ki = params.pll_gains()
    kpc, kic = params.current_gains(current_bandwidth_hz)
    total_l = params.total_inductance_h
    delta_star = params.post_fault_stable_angle_rad

    A = np.array([[0.0, 1.0], [-kic / total_l, -kpc / total_l]], dtype=float)
    B = np.array(
        [[0.0], [params.fault_voltage_v * np.sin(delta_star) / total_l]],
        dtype=float,
    )
    C = np.array(
        [[
            params.omega_grid * params.grid_inductance_h,
            (kp / ki) * params.omega_grid * params.grid_inductance_h,
        ]],
        dtype=float,
    )
    return A, B, C


def equilibrium_damping(params: GFLParameters) -> float:
    kp, ki = params.pll_gains()
    return float(
        (kp / ki)
        * params.fault_voltage_v
        * np.cos(params.post_fault_stable_angle_rad)
        - params.grid_inductance_h * params.current_reference_a
    )


def local_hinf_gain(
    params: GFLParameters,
    current_bandwidth_hz: float,
    omega_bounds: tuple[float, float] = (1e-4, 1e7),
) -> LocalSmallGainResult:
    """Compute the scalar H-infinity norm of ``C(sI-A)^-1B``.

    A bounded log-frequency optimization is used after a coarse scan. The
    system is second order and SISO, so this avoids a control-toolbox
    dependency while retaining a transparent numerical calculation.
    """
    A, B, C = fast_residual_state_space(params, current_bandwidth_hz)
    low, high = omega_bounds
    if not (0.0 < low < high):
        raise ValueError("omega_bounds must satisfy 0 < low < high")

    def magnitude(omega: float) -> float:
        response = C @ np.linalg.solve(1j * omega * np.eye(A.shape[0]) - A, B)
        return float(abs(response[0, 0]))

    grid = np.logspace(np.log10(low), np.log10(high), 1200)
    values = np.array([magnitude(w) for w in grid])
    index = int(np.argmax(values))
    lo_index = max(index - 2, 0)
    hi_index = min(index + 2, grid.size - 1)
    result = minimize_scalar(
        lambda log_w: -magnitude(float(np.exp(log_w))),
        bounds=(float(np.log(grid[lo_index])), float(np.log(grid[hi_index]))),
        method="bounded",
        options={"xatol": 1e-11},
    )
    omega_peak = float(np.exp(result.x))
    gamma = float(-result.fun)
    damping = equilibrium_damping(params)
    ratio = float(np.inf if damping <= 0.0 else gamma / damping)
    return LocalSmallGainResult(
        current_bandwidth_hz=float(current_bandwidth_hz),
        gamma=gamma,
        peak_frequency_rad_s=omega_peak,
        peak_frequency_hz=omega_peak / (2.0 * np.pi),
        equilibrium_damping=damping,
        gamma_over_damping=ratio,
        condition_passes=bool(damping > 0.0 and gamma < damping),
    )


def bounded_real_storage(
    params: GFLParameters,
    current_bandwidth_hz: float,
    gamma: float | None = None,
    gamma_margin: float = 1.05,
) -> BoundedRealStorage:
    """Find a 2x2 storage matrix for the bounded-real inequality.

    The Riccati equality

    ``A.T P + P A + C.T C + gamma^-2 P B B.T P = 0``

    is solved and then checked. ``gamma`` must exceed the computed H-infinity
    norm; by default a five-percent margin is used.
    """
    gain = local_hinf_gain(params, current_bandwidth_hz)
    selected_gamma = gain.gamma * gamma_margin if gamma is None else float(gamma)
    if selected_gamma <= gain.gamma:
        raise ValueError("gamma must be strictly larger than the H-infinity norm")

    A, B, C = fast_residual_state_space(params, current_bandwidth_hz)
    Q = C.T @ C
    P0 = solve_continuous_lyapunov(A.T, -Q)
    x0 = np.array([P0[0, 0], P0[0, 1], P0[1, 1]], dtype=float)

    def equation(x: np.ndarray) -> np.ndarray:
        P = np.array([[x[0], x[1]], [x[1], x[2]]], dtype=float)
        residual = A.T @ P + P @ A + Q + (P @ B @ B.T @ P) / selected_gamma**2
        return np.array([residual[0, 0], residual[0, 1], residual[1, 1]])

    solution = root(equation, x0, method="lm")
    if not solution.success:
        raise RuntimeError(f"bounded-real Riccati solve failed: {solution.message}")
    P = np.array(
        [[solution.x[0], solution.x[1]], [solution.x[1], solution.x[2]]],
        dtype=float,
    )
    residual = A.T @ P + P @ A + Q + (P @ B @ B.T @ P) / selected_gamma**2
    residual_max = float(np.max(eigvalsh(0.5 * (residual + residual.T))))
    positive = bool(np.min(eigvalsh(P)) > 0.0)
    if not positive or residual_max > 1e-7:
        raise RuntimeError(
            "No valid positive-definite bounded-real storage was obtained "
            f"(positive={positive}, residual={residual_max:.3e})."
        )
    return BoundedRealStorage(selected_gamma, P, residual_max, positive)


def swing_certificate_audit(
    params: GFLParameters,
    current_bandwidth_hz: float,
    reserve_factor: float = 0.8,
) -> SwingCertificateAudit:
    """Audit the local small-gain and one-shot slow-energy conditions."""
    if not 0.0 < reserve_factor <= 1.0:
        raise ValueError("reserve_factor must lie in (0, 1]")
    initial = float(potential_energy(params.pre_fault_angle_rad, params))
    critical = critical_energy(params)
    reserved = reserve_factor * critical
    local = local_hinf_gain(params, current_bandwidth_hz)
    return SwingCertificateAudit(
        current_bandwidth_hz=float(current_bandwidth_hz),
        initial_slow_energy=initial,
        critical_energy=critical,
        reserved_critical_energy=reserved,
        initial_over_critical=initial / critical,
        initial_over_reserved_critical=initial / reserved,
        one_shot_passes_before_fast_penalty=bool(initial < reserved),
        local_small_gain=local,
    )
