from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .energies import first_handoff_time, pi_energy_trace, swing_energy_trace
from .model import GFLParameters, SimulationResult, simulate_fourth_order
from .small_gain import swing_certificate_audit


def _case_payload(result: SimulationResult, params: GFLParameters) -> dict[str, object]:
    swing = swing_energy_trace(result, params)
    pi_trace = pi_energy_trace(result, params)
    audit = swing_certificate_audit(params, result.current_bandwidth_hz)
    handoff = first_handoff_time(result, pi_trace)
    return {
        "current_bandwidth_hz": result.current_bandwidth_hz,
        "stable": bool(result.stable),
        "terminated_by_event": bool(result.terminated_by_event),
        "simulation_end_s": float(result.time_s[-1]),
        "maximum_delta_rad": float(np.max(result.delta)),
        "minimum_delta_rad": float(np.min(result.delta)),
        "maximum_abs_omega_rad_s": float(np.max(np.abs(result.omega))),
        "swing_initial_over_critical": float(swing.total[0] / swing.critical_energy),
        "swing_peak_over_critical": float(np.max(swing.total) / swing.critical_energy),
        "minimum_equivalent_damping": float(np.min(swing.damping)),
        "pi_initial_over_critical": float(pi_trace.total[0] / pi_trace.critical_energy),
        "pi_peak_over_critical": float(np.max(pi_trace.total) / pi_trace.critical_energy),
        "pi_balance_error": pi_trace.max_balance_error,
        "cumulative_pi_dissipation_final": float(pi_trace.cumulative_dissipation[-1]),
        "cumulative_fast_interaction_final": float(
            pi_trace.cumulative_fast_interaction[-1]
        ),
        "trajectory_handoff_time_s": handoff,
        "local_small_gain": audit.local_small_gain.to_dict(),
        "one_shot_swing_audit": audit.to_dict(),
    }


def _plot_angles(
    results: list[SimulationResult], params: GFLParameters, output: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for result in results:
        ax.plot(
            result.time_s,
            result.delta,
            label=f"$f_c$={result.current_bandwidth_hz:.0f} Hz",
        )
    ax.axhline(
        params.post_fault_stable_angle_rad,
        linestyle="--",
        linewidth=1.0,
        label="post-fault SEP",
    )
    ax.axhline(
        params.post_fault_unstable_angle_rad,
        linestyle=":",
        linewidth=1.0,
        label="right UEP",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"PLL angle $\delta$ [rad]")
    ax.set_title("Fourth-order current-loop–PLL trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "angle_trajectories.png", dpi=220)
    plt.close(fig)


def _plot_swing_energy(
    results: list[SimulationResult], params: GFLParameters, output: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for result in results:
        trace = swing_energy_trace(result, params)
        ax.plot(
            result.time_s,
            trace.total / trace.critical_energy,
            label=f"$f_c$={result.current_bandwidth_hz:.0f} Hz",
        )
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="UEP energy")
    ax.axhline(0.8, linestyle=":", linewidth=1.0, label="20% reserve")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$V_s/V_{\mathrm{critical}}$")
    ax.set_title("Classical swing-energy audit")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "swing_energy.png", dpi=220)
    plt.close(fig)


def _plot_pi_energy(
    results: list[SimulationResult], params: GFLParameters, output: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for result in results:
        trace = pi_energy_trace(result, params)
        ax.plot(
            result.time_s,
            trace.total / trace.critical_energy,
            label=f"$f_c$={result.current_bandwidth_hz:.0f} Hz",
        )
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="UEP energy")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$V_{PI}/V_{\mathrm{critical}}$")
    ax.set_title("PI-controller energy along the fourth-order trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "pi_energy.png", dpi=220)
    plt.close(fig)


def _plot_pi_balance(
    results: list[SimulationResult], params: GFLParameters, output: Path
) -> None:
    for result in results:
        trace = pi_energy_trace(result, params)
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(
            result.time_s,
            trace.cumulative_dissipation,
            label=r"$k_p\int q_0^2dt$",
        )
        ax.plot(
            result.time_s,
            trace.cumulative_fast_interaction,
            label=r"$\int z r\,dt$",
        )
        ax.plot(
            result.time_s,
            trace.total[0] - trace.total,
            linestyle="--",
            label=r"$V_{PI}(0)-V_{PI}(t)$",
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Accumulated energy term")
        ax.set_title(
            f"PI dissipation versus fast interaction: "
            f"$f_c$={result.current_bandwidth_hz:.0f} Hz"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output / f"pi_balance_fc_{result.current_bandwidth_hz:.0f}.png",
            dpi=220,
        )
        plt.close(fig)


def _plot_small_gain(
    results: list[SimulationResult], params: GFLParameters, output: Path
) -> None:
    bandwidths = [result.current_bandwidth_hz for result in results]
    ratios = [
        swing_certificate_audit(params, bandwidth).local_small_gain.gamma_over_damping
        for bandwidth in bandwidths
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    positions = np.arange(len(bandwidths))
    ax.bar(positions, ratios)
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label=r"$\gamma=D(\delta^\star)$",
    )
    ax.set_xticks(positions, [f"{value:.0f} Hz" for value in bandwidths])
    ax.set_ylabel(r"Local ratio $\gamma/D(\delta^\star)$")
    ax.set_title("Local small gain does not resolve the large-disturbance split")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "local_small_gain.png", dpi=220)
    plt.close(fig)


def run_and_save(
    output_dir: str | Path,
    bandwidths_hz: Iterable[float] = (300.0, 500.0),
    params: GFLParameters | None = None,
) -> dict[str, object]:
    """Run the reconstruction, save figures, and return JSON-ready metrics."""
    params = params or GFLParameters()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bandwidths = [float(value) for value in bandwidths_hz]
    if not bandwidths:
        raise ValueError("At least one current-loop bandwidth is required")

    results = [simulate_fourth_order(value, params=params) for value in bandwidths]
    payload: dict[str, object] = {
        "scope": (
            "Transparent numerical reconstruction; bandwidth-gain and fault-jump "
            "mappings are explicit assumptions rather than authors' source code."
        ),
        "parameters": asdict(params),
        "equilibria": {
            "pre_fault_angle_rad": params.pre_fault_angle_rad,
            "post_fault_stable_angle_rad": params.post_fault_stable_angle_rad,
            "post_fault_unstable_angle_rad": params.post_fault_unstable_angle_rad,
        },
        "cases": [_case_payload(result, params) for result in results],
    }

    _plot_angles(results, params, output)
    _plot_swing_energy(results, params, output)
    _plot_pi_energy(results, params, output)
    _plot_pi_balance(results, params, output)
    _plot_small_gain(results, params, output)

    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return payload
