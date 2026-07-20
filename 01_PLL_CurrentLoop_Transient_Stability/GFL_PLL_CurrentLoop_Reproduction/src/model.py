"""Equation-level reconstruction of Wu et al. (IEEE TEC, 2024).

The fourth-order model follows Eq. (18) of the paper with states
x = [delta, Delta_omega, Delta_Id, d(Delta_Id)/dt].

Important reproducibility note
------------------------------
The paper's Table I leaves k_pc and k_ic unspecified and does not provide a
unique mapping from the named current-loop bandwidth f_c to those gains.
It also does not explicitly state the post-fault jump of d(Delta_Id)/dt
created by the numerator s in Eq. (14).  This module therefore supports a
transparent calibrated reconstruction in addition to a standard mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


def _closed_loop_bw_ratio(zeta: float) -> float:
    """omega_bw / omega_n for (2*zeta*wn*s+wn^2)/(s^2+2*zeta*wn*s+wn^2)."""
    return float(np.sqrt(1 + 2*zeta**2 + np.sqrt(2 + 4*zeta**2 + 4*zeta**4)))


@dataclass(frozen=True)
class PlantParameters:
    rated_power_w: float = 7.35e3
    ac_line_line_rms_v: float = 400.0
    dc_voltage_v: float = 700.0
    grid_frequency_hz: float = 50.0
    filter_inductance_h: float = 7.6e-3
    grid_inductance_h: float = 2.8e-3
    fault_voltage_pu: float = 0.05

    @property
    def phase_peak_voltage_v(self) -> float:
        return self.ac_line_line_rms_v * np.sqrt(2.0 / 3.0)

    @property
    def omega_grid(self) -> float:
        return 2.0 * np.pi * self.grid_frequency_hz

    @property
    def d_axis_current_ref_a(self) -> float:
        # Peak-value dq convention: P = 3/2 * Vd * Id.
        return 2.0 * self.rated_power_w / (3.0 * self.phase_peak_voltage_v)

    @property
    def total_current_path_inductance_h(self) -> float:
        return self.filter_inductance_h + self.grid_inductance_h

    @property
    def fault_voltage_v(self) -> float:
        return self.fault_voltage_pu * self.phase_peak_voltage_v

    @property
    def initial_delta_rad(self) -> float:
        arg = (
            self.omega_grid
            * self.grid_inductance_h
            * self.d_axis_current_ref_a
            / self.phase_peak_voltage_v
        )
        return float(np.arcsin(np.clip(arg, -1.0, 1.0)))


@dataclass(frozen=True)
class ReconstructionSettings:
    pll_damping_ratio: float = 1.2
    current_loop_damping_ratio: float = 1.0
    # Calibration anchors: at f_PLL=50 Hz, the reduced model has delta_max~2 rad;
    # f_c=300 Hz loses synchronism while f_c=500 Hz resynchronizes.
    pll_bandwidth_scale: float = 1.6
    current_bandwidth_scale: float = 0.5
    fault_current_jump_gain: float = 2.75


def pll_gains(
    f_pll_hz: float,
    plant: PlantParameters,
    settings: ReconstructionSettings,
) -> tuple[float, float]:
    """Return k_p, k_i using a standard 2nd-order bandwidth mapping.

    The explicit scale factor is retained because the paper does not state
    whether f_BW-PLL is natural, crossover, or closed-loop -3 dB bandwidth.
    """
    f_effective = settings.pll_bandwidth_scale * f_pll_hz
    wn = 2.0 * np.pi * f_effective / _closed_loop_bw_ratio(settings.pll_damping_ratio)
    u_nom = plant.phase_peak_voltage_v
    kp = 2.0 * settings.pll_damping_ratio * wn / u_nom
    ki = wn**2 / u_nom
    return float(kp), float(ki)


def current_loop_gains(
    f_current_hz: float,
    plant: PlantParameters,
    settings: ReconstructionSettings,
) -> tuple[float, float]:
    """Infer k_pc, k_ic from f_c for L*s^2+k_pc*s+k_ic.

    This is an inferred mapping because Table I reports '-' for both gains.
    """
    zeta = settings.current_loop_damping_ratio
    f_effective = settings.current_bandwidth_scale * f_current_hz
    wn = 2.0 * np.pi * f_effective / _closed_loop_bw_ratio(zeta)
    inductance = plant.total_current_path_inductance_h
    kpc = 2.0 * zeta * wn * inductance
    kic = wn**2 * inductance
    return float(kpc), float(kic)


def _instability_event(_: float, x: np.ndarray) -> float:
    return float(np.pi - abs(x[0]))


_instability_event.terminal = True
_instability_event.direction = -1


def simulate_second_order(
    f_pll_hz: float,
    t_end_s: float = 2.0,
    plant: PlantParameters | None = None,
    settings: ReconstructionSettings | None = None,
):
    """Simulate Eqs. (6)-(8), starting immediately after the voltage sag."""
    plant = plant or PlantParameters()
    settings = settings or ReconstructionSettings()
    kp, ki = pll_gains(f_pll_hz, plant, settings)
    uf = plant.fault_voltage_v
    lg = plant.grid_inductance_h
    iref = plant.d_axis_current_ref_a
    wg = plant.omega_grid

    def rhs(_: float, x: np.ndarray) -> np.ndarray:
        delta, domega = x
        den = 1.0 - kp * lg * iref
        d2delta = (
            kp * (-uf * domega * np.cos(delta))
            + ki * (-uf * np.sin(delta) + lg * (domega + wg) * iref)
        ) / den
        return np.array([domega, d2delta])

    return solve_ivp(
        rhs,
        (0.0, t_end_s),
        np.array([plant.initial_delta_rad, 0.0]),
        method="Radau",
        max_step=5e-4,
        rtol=2e-7,
        atol=1e-9,
        events=_instability_event,
        dense_output=True,
    )


def simulate_fourth_order(
    f_pll_hz: float,
    f_current_hz: float,
    t_end_s: float = 2.0,
    plant: PlantParameters | None = None,
    settings: ReconstructionSettings | None = None,
):
    """Simulate Eq. (18), starting immediately after the voltage sag."""
    plant = plant or PlantParameters()
    settings = settings or ReconstructionSettings()
    kp, ki = pll_gains(f_pll_hz, plant, settings)
    kpc, kic = current_loop_gains(f_current_hz, plant, settings)

    uf = plant.fault_voltage_v
    u0 = plant.phase_peak_voltage_v
    lg = plant.grid_inductance_h
    inductance = plant.total_current_path_inductance_h
    iref = plant.d_axis_current_ref_a
    wg = plant.omega_grid
    delta0 = plant.initial_delta_rad

    # Eq. (14) has numerator s. A voltage step therefore creates a jump in
    # d(Delta_Id)/dt. The multiplier is calibrated to the current peaks shown
    # in Figs. 10 and 15 because the omitted converter-side voltage transient
    # is not identifiable from the paper's reduced model alone.
    did0 = (
        settings.fault_current_jump_gain
        * (u0 - uf)
        * np.cos(delta0)
        / inductance
    )
    x0 = np.array([delta0, 0.0, 0.0, did0])

    def rhs(_: float, x: np.ndarray) -> np.ndarray:
        delta, domega, did, ddid = x
        den = 1.0 - kp * lg * (iref + did)
        # Guard against the singular surface of Eq. (18).
        if abs(den) < 1e-7:
            den = np.copysign(1e-7, den if den != 0 else 1.0)
        d2delta = (
            kp * (-uf * domega * np.cos(delta) + (wg + domega) * lg * ddid)
            + ki * (-uf * np.sin(delta) + lg * (domega + wg) * (iref + did))
        ) / den
        d2id = (-kpc * ddid - kic * did + uf * domega * np.sin(delta)) / inductance
        return np.array([domega, d2delta, ddid, d2id])

    return solve_ivp(
        rhs,
        (0.0, t_end_s),
        x0,
        method="Radau",
        max_step=5e-4,
        rtol=2e-7,
        atol=1e-9,
        events=_instability_event,
        dense_output=True,
    )


def is_synchronized(solution, t_end_s: float, frequency_tolerance_hz: float = 1.0) -> bool:
    if solution.t[-1] < t_end_s - 1e-6:
        return False
    if np.max(np.abs(solution.y[0])) >= np.pi:
        return False
    return abs(solution.y[1, -1]) / (2.0 * np.pi) <= frequency_tolerance_hz


def sample_solution(solution, t_end_s: float, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    t_last = min(t_end_s, float(solution.t[-1]))
    t = np.linspace(0.0, t_last, n)
    if solution.sol is not None:
        y = solution.sol(t)
    else:
        y = np.vstack([np.interp(t, solution.t, row) for row in solution.y])
    return t, y
