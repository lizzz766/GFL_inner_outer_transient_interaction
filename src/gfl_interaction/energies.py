from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .model import GFLParameters, SimulationResult


@dataclass
class SwingEnergyTrace:
    potential: np.ndarray
    kinetic: np.ndarray
    total: np.ndarray
    critical_energy: float
    damping: np.ndarray


@dataclass
class PIEnergyTrace:
    x_integrator: np.ndarray
    q0: np.ndarray
    residual: np.ndarray
    port_output: np.ndarray
    potential: np.ndarray
    total: np.ndarray
    critical_energy: float
    cumulative_dissipation: np.ndarray
    cumulative_fast_interaction: np.ndarray
    reconstructed_energy: np.ndarray
    max_balance_error: float


def synchronizing_input(params: GFLParameters) -> float:
    return params.omega_grid * params.grid_inductance_h * params.current_reference_a


def potential_energy(delta: np.ndarray | float, params: GFLParameters) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    delta_star = params.post_fault_stable_angle_rad
    p = synchronizing_input(params)
    uf = params.fault_voltage_v
    return uf * (np.cos(delta_star) - np.cos(delta)) - p * (delta - delta_star)


def critical_energy(params: GFLParameters) -> float:
    return float(potential_energy(params.post_fault_unstable_angle_rad, params))


def swing_energy_trace(
    result: SimulationResult, params: GFLParameters
) -> SwingEnergyTrace:
    kp, ki = params.pll_gains()
    inertia = (1.0 - kp * params.grid_inductance_h * params.current_reference_a) / ki
    potential = potential_energy(result.delta, params)
    kinetic = 0.5 * inertia * result.omega**2
    damping = (
        (kp / ki) * params.fault_voltage_v * np.cos(result.delta)
        - params.grid_inductance_h * params.current_reference_a
    )
    return SwingEnergyTrace(
        potential=potential,
        kinetic=kinetic,
        total=potential + kinetic,
        critical_energy=critical_energy(params),
        damping=damping,
    )


def pi_energy_trace(result: SimulationResult, params: GFLParameters) -> PIEnergyTrace:
    """Evaluate the PI-controller energy identity on the fourth-order trajectory.

    The decomposition is

        v_q = q0(delta) + r,
        V_PI_dot = -k_p q0^2 + z r,
        z = x_I - k_p q0.

    This is an exact accounting identity for the reduced fourth-order model,
    not yet a robust certificate: the actual trajectory is used in the two
    integrals.
    """

    kp, ki = params.pll_gains()
    p = synchronizing_input(params)
    uf = params.fault_voltage_v
    lg = params.grid_inductance_h
    omega_g = params.omega_grid
    i_ref = params.current_reference_a

    q0 = p - uf * np.sin(result.delta)
    residual = (
        lg * i_ref * result.omega
        + omega_g * lg * result.delta_i
        + lg * result.omega * result.delta_i
    )
    vq = q0 + residual
    x_integrator = result.omega - kp * vq
    port_output = x_integrator - kp * q0

    potential = potential_energy(result.delta, params)
    total = x_integrator**2 / (2.0 * ki) + potential

    dissipation_integrand = kp * q0**2
    interaction_integrand = port_output * residual
    cumulative_dissipation = np.concatenate(
        ([0.0], cumulative_trapezoid(dissipation_integrand, result.time_s))
    )
    cumulative_interaction = np.concatenate(
        ([0.0], cumulative_trapezoid(interaction_integrand, result.time_s))
    )
    reconstructed = total[0] - cumulative_dissipation + cumulative_interaction
    max_balance_error = float(np.max(np.abs(total - reconstructed)))

    return PIEnergyTrace(
        x_integrator=x_integrator,
        q0=q0,
        residual=residual,
        port_output=port_output,
        potential=potential,
        total=total,
        critical_energy=critical_energy(params),
        cumulative_dissipation=cumulative_dissipation,
        cumulative_fast_interaction=cumulative_interaction,
        reconstructed_energy=reconstructed,
        max_balance_error=max_balance_error,
    )


def first_handoff_time(
    result: SimulationResult,
    pi_trace: PIEnergyTrace,
    fast_relative_tolerance: float = 1e-2,
) -> float | None:
    """Trajectory-based handoff diagnostic.

    Returns the first time for which the fast-state norm has decayed below a
    relative tolerance and the PI energy lies below the UEP energy. This is a
    diagnostic, not a proof, because it uses the simulated trajectory.
    """

    fast_norm = np.sqrt(result.delta_i**2 + result.delta_i_dot**2)
    scale = max(float(fast_norm[0]), 1.0)

    # A handoff is meaningful only if the conditions remain true afterwards.
    # Suffix maxima prevent a transient zero crossing from being misclassified
    # as successful decay.
    future_fast_max = np.maximum.accumulate(fast_norm[::-1])[::-1]
    future_energy_max = np.maximum.accumulate(pi_trace.total[::-1])[::-1]
    valid = (future_fast_max <= fast_relative_tolerance * scale) & (
        future_energy_max < pi_trace.critical_energy
    )
    indices = np.flatnonzero(valid)
    return None if indices.size == 0 else float(result.time_s[indices[0]])
