from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


def _bandwidth_ratio(zeta: float) -> float:
    """Calibration relation used in the original numerical reconstruction."""
    return float(np.sqrt(1 + 2 * zeta**2 + np.sqrt(2 + 4 * zeta**2 + 4 * zeta**4)))


@dataclass(frozen=True)
class GFLParameters:
    """Parameters for the reduced fourth-order GFL model.

    Notes
    -----
    Wu et al. publish the physical plant and PLL gains, but do not publish a
    unique map from the quoted current-loop bandwidth to ``k_pc`` and ``k_ic``.
    This repository therefore keeps that map explicit and configurable.

    The default ``fault_jump_scale=0.2`` is a transparent calibration of the
    boundary-layer initialization that reproduces the paper's qualitative split:
    300 Hz loses synchronism while 500 Hz recovers.
    """

    rated_power_w: float = 7350.0
    line_line_rms_v: float = 400.0
    nominal_frequency_hz: float = 50.0
    grid_inductance_h: float = 2.8e-3
    converter_filter_inductance_h: float = 7.6e-3
    pre_fault_voltage_pu: float = 1.0
    fault_voltage_pu: float = 0.05
    pll_bandwidth_hz: float = 50.0
    pll_damping_ratio: float = 1.2
    current_loop_damping_ratio: float = 1.0
    pll_bandwidth_calibration: float = 1.6
    current_bandwidth_calibration: float = 0.5
    fault_jump_scale: float = 0.2

    @property
    def phase_peak_voltage_v(self) -> float:
        return self.line_line_rms_v * np.sqrt(2.0 / 3.0)

    @property
    def pre_fault_voltage_v(self) -> float:
        return self.pre_fault_voltage_pu * self.phase_peak_voltage_v

    @property
    def fault_voltage_v(self) -> float:
        return self.fault_voltage_pu * self.phase_peak_voltage_v

    @property
    def omega_grid(self) -> float:
        return 2.0 * np.pi * self.nominal_frequency_hz

    @property
    def current_reference_a(self) -> float:
        return 2.0 * self.rated_power_w / (3.0 * self.phase_peak_voltage_v)

    @property
    def total_inductance_h(self) -> float:
        return self.grid_inductance_h + self.converter_filter_inductance_h

    @property
    def pre_fault_angle_rad(self) -> float:
        value = self.omega_grid * self.grid_inductance_h * self.current_reference_a
        return float(np.arcsin(value / self.pre_fault_voltage_v))

    @property
    def post_fault_stable_angle_rad(self) -> float:
        value = self.omega_grid * self.grid_inductance_h * self.current_reference_a
        return float(np.arcsin(value / self.fault_voltage_v))

    @property
    def post_fault_unstable_angle_rad(self) -> float:
        return float(np.pi - self.post_fault_stable_angle_rad)

    def pll_gains(self) -> tuple[float, float]:
        ratio = _bandwidth_ratio(self.pll_damping_ratio)
        omega_n = (
            2.0
            * np.pi
            * self.pll_bandwidth_calibration
            * self.pll_bandwidth_hz
            / ratio
        )
        kp = 2.0 * self.pll_damping_ratio * omega_n / self.phase_peak_voltage_v
        ki = omega_n**2 / self.phase_peak_voltage_v
        return float(kp), float(ki)

    def current_gains(self, current_bandwidth_hz: float) -> tuple[float, float]:
        ratio = _bandwidth_ratio(self.current_loop_damping_ratio)
        omega_n = (
            2.0
            * np.pi
            * self.current_bandwidth_calibration
            * current_bandwidth_hz
            / ratio
        )
        kpc = 2.0 * self.current_loop_damping_ratio * omega_n * self.total_inductance_h
        kic = omega_n**2 * self.total_inductance_h
        return float(kpc), float(kic)


@dataclass
class SimulationResult:
    current_bandwidth_hz: float
    time_s: np.ndarray
    state: np.ndarray
    stable: bool
    terminated_by_event: bool
    stable_angle_rad: float
    unstable_angle_rad: float

    @property
    def delta(self) -> np.ndarray:
        return self.state[0]

    @property
    def omega(self) -> np.ndarray:
        return self.state[1]

    @property
    def delta_i(self) -> np.ndarray:
        return self.state[2]

    @property
    def delta_i_dot(self) -> np.ndarray:
        return self.state[3]


def fourth_order_rhs(
    params: GFLParameters, current_bandwidth_hz: float
) -> Callable[[float, np.ndarray], np.ndarray]:
    kp, ki = params.pll_gains()
    kpc, kic = params.current_gains(current_bandwidth_hz)
    uf = params.fault_voltage_v
    omega_g = params.omega_grid
    lg = params.grid_inductance_h
    total_l = params.total_inductance_h
    i_ref = params.current_reference_a

    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        delta, omega, delta_i, delta_i_dot = x
        denominator = 1.0 - kp * lg * (i_ref + delta_i)
        if abs(denominator) < 1e-9:
            raise FloatingPointError("PLL denominator approached zero.")

        omega_dot = (
            kp
            * (
                -uf * omega * np.cos(delta)
                + (omega_g + omega) * lg * delta_i_dot
            )
            + ki
            * (
                -uf * np.sin(delta)
                + lg * (omega + omega_g) * (i_ref + delta_i)
            )
        ) / denominator

        delta_i_ddot = (
            -kpc * delta_i_dot
            - kic * delta_i
            + uf * omega * np.sin(delta)
        ) / total_l

        return np.array([omega, omega_dot, delta_i_dot, delta_i_ddot], dtype=float)

    return rhs


def initial_state(params: GFLParameters) -> np.ndarray:
    delta0 = params.pre_fault_angle_rad
    voltage_step = params.pre_fault_voltage_v - params.fault_voltage_v
    delta_i_dot0 = (
        params.fault_jump_scale
        * voltage_step
        * np.cos(delta0)
        / params.total_inductance_h
    )
    return np.array([delta0, 0.0, 0.0, delta_i_dot0], dtype=float)


def simulate_fourth_order(
    current_bandwidth_hz: float,
    params: GFLParameters | None = None,
    t_end: float = 1.5,
    max_step: float = 1e-3,
) -> SimulationResult:
    params = params or GFLParameters()
    rhs = fourth_order_rhs(params, current_bandwidth_hz)
    x0 = initial_state(params)
    delta_u = params.post_fault_unstable_angle_rad

    def escape_event(_t: float, x: np.ndarray) -> float:
        # Stop if the trajectory crosses the right UEP, goes far left, or the
        # PLL frequency deviation becomes clearly divergent.
        return min(
            delta_u - x[0],
            x[0] + 2.0 * np.pi,
            2.0 * np.pi * 500.0 - abs(x[1]),
        )

    escape_event.terminal = True  # type: ignore[attr-defined]
    escape_event.direction = -1  # type: ignore[attr-defined]

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        x0,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
        max_step=max_step,
        events=escape_event,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    terminated = sol.t[-1] < t_end - 1e-8
    delta_star = params.post_fault_stable_angle_rad
    stable = bool(
        not terminated
        and abs(sol.y[1, -1]) < 0.1
        and abs(sol.y[0, -1] - delta_star) < 0.1
    )

    return SimulationResult(
        current_bandwidth_hz=float(current_bandwidth_hz),
        time_s=sol.t,
        state=sol.y,
        stable=stable,
        terminated_by_event=terminated,
        stable_angle_rad=delta_star,
        unstable_angle_rad=delta_u,
    )
