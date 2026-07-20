from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model import (  # noqa: E402
    PlantParameters,
    ReconstructionSettings,
    simulate_second_order,
    simulate_fourth_order,
    sample_solution,
    is_synchronized,
)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
plant = PlantParameters()
settings = ReconstructionSettings()
f_pll = 50.0
post_fault_end = 1.5
fault_time = 0.5

cases = {
    "2nd order, current neglected": simulate_second_order(f_pll, post_fault_end, plant, settings),
    "4th order, fc=300 Hz": simulate_fourth_order(f_pll, 300.0, post_fault_end, plant, settings),
    "4th order, fc=500 Hz": simulate_fourth_order(f_pll, 500.0, post_fault_end, plant, settings),
}

# Frequency responses
fig, ax = plt.subplots(figsize=(8.2, 4.8))
for label, sol in cases.items():
    t, y = sample_solution(sol, post_fault_end)
    full_t = np.concatenate(([0.0, fault_time], fault_time + t))
    full_f = np.concatenate(([plant.grid_frequency_hz, plant.grid_frequency_hz], plant.grid_frequency_hz + y[1] / (2*np.pi)))
    ax.plot(full_t, full_f, label=label)
ax.axvline(fault_time, linestyle="--", linewidth=1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("PLL frequency [Hz]")
ax.set_xlim(0, fault_time + post_fault_end)
ax.set_ylim(45, 180)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "frequency_comparison.png", dpi=220)
plt.close(fig)

# Current responses for fourth-order cases
fig, ax = plt.subplots(figsize=(8.2, 4.8))
for fc in (300.0, 500.0):
    sol = cases[f"4th order, fc={int(fc)} Hz"]
    t, y = sample_solution(sol, post_fault_end)
    current_pu = (plant.d_axis_current_ref_a + y[2]) / plant.d_axis_current_ref_a
    full_t = np.concatenate(([0.0, fault_time], fault_time + t))
    full_i = np.concatenate(([1.0, 1.0], current_pu))
    ax.plot(full_t, full_i, label=f"fc={int(fc)} Hz")
ax.axvline(fault_time, linestyle="--", linewidth=1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("d-axis current [pu]")
ax.set_xlim(0.45, fault_time + min(post_fault_end, 0.6))
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "current_response.png", dpi=220)
plt.close(fig)

# EAC-like input/output curves
fig, ax = plt.subplots(figsize=(8.2, 4.8))
# Reduced-model constant input and fault output
sol2 = cases["2nd order, current neglected"]
t2, y2 = sample_solution(sol2, post_fault_end)
delta_grid = np.linspace(0, np.pi, 1000)
const_input = plant.omega_grid * plant.grid_inductance_h * plant.d_axis_current_ref_a / plant.phase_peak_voltage_v
fault_output = plant.fault_voltage_pu * np.sin(delta_grid)
ax.plot(delta_grid, np.full_like(delta_grid, const_input), linestyle="--", label="Input, current neglected")
ax.plot(delta_grid, fault_output, label="Fault output Uf sin(delta)")
for fc in (300.0, 500.0):
    sol = cases[f"4th order, fc={int(fc)} Hz"]
    t, y = sample_solution(sol, post_fault_end)
    eq_input = plant.omega_grid * plant.grid_inductance_h * (plant.d_axis_current_ref_a + y[2]) / plant.phase_peak_voltage_v
    # Plot only the first forward swing, as in the paper's EAC construction.
    peak = int(np.argmax(y[0])) + 1
    ax.plot(y[0, :peak], eq_input[:peak], label=f"Dynamic input, fc={int(fc)} Hz")
ax.set_xlabel("Power angle delta [rad]")
ax.set_ylabel("Equivalent input/output [pu]")
ax.set_xlim(0, np.pi)
ax.set_ylim(0, 0.14)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "eac_reconstruction.png", dpi=220)
plt.close(fig)

with open(OUT / "case_summary.txt", "w", encoding="utf-8") as f:
    for label, sol in cases.items():
        status = "STABLE" if is_synchronized(sol, post_fault_end) else "UNSTABLE"
        dmax = float(np.max(sol.y[0]))
        fend = plant.grid_frequency_hz + float(sol.y[1, -1]) / (2*np.pi)
        f.write(f"{label}: {status}, delta_max={dmax:.4f} rad, final_frequency={fend:.4f} Hz, simulated_to={sol.t[-1]:.4f} s\n")
print((OUT / "case_summary.txt").read_text(encoding="utf-8"))
