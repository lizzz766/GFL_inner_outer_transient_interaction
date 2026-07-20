from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# Approximate digitization of the lower blue boundary in Fig. 9.
# This is deliberately stored as data, not presented as a fresh ODE scan.
f_pll = np.array([20, 25, 30, 35, 40, 45, 50, 52, 54, 55, 56, 57], dtype=float)
f_current_boundary = np.array([115, 135, 155, 180, 210, 245, 292, 320, 360, 405, 460, 500], dtype=float)
pd.DataFrame({"f_pll_hz": f_pll, "digitized_boundary_fc_hz": f_current_boundary}).to_csv(
    DATA / "fig9_digitized_boundary.csv", index=False
)

x = np.linspace(20, 60, 400)
conservative = 9.461 * x - 74.2

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.fill_between(f_pll, f_current_boundary, 500, alpha=0.35, label="Digitized attraction region")
ax.plot(f_pll, f_current_boundary, label="Digitized lower boundary")
ax.plot(x, conservative, linewidth=2, label="Published conservative fit: fc=9.461 fPLL-74.2")
ax.set_xlabel("PLL bandwidth [Hz]")
ax.set_ylabel("Current-loop bandwidth [Hz]")
ax.set_xlim(20, 60)
ax.set_ylim(100, 500)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "published_bandwidth_region_reconstruction.png", dpi=220)
plt.close(fig)
