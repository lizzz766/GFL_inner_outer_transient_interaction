# GFL inner–outer transient interaction

Python research code for the current-loop–PLL transient-synchronization example of Wu et al., together with two complementary analyses:

1. **classical swing energy + local fast-dynamic small gain**;
2. **PI-controller energy + fast-dynamic energy accounting** inspired by Zhang et al. (2026).

The repository is intentionally compact and exposes every non-published reconstruction assumption as a parameter.

## What the default case reproduces

With the calibrated reduced-model initialization:

- current-loop bandwidth `300 Hz`: loss of synchronism;
- current-loop bandwidth `500 Hz`: recovery to the post-fault equilibrium;
- the local small-gain ratio is small in both cases, showing that a local test is over-optimistic for the large disturbance;
- the one-shot swing-energy sublevel condition rejects even the stable case;
- the PI energy gives an exact decomposition into PLL dissipation and fast-dynamic interaction along the simulated trajectory.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run

```bash
python scripts/run_all.py
```

Outputs are written to `results/`:

- `summary.json`;
- angle trajectories;
- swing-energy comparison;
- PI-energy balance and cumulative dissipation/interaction plots.

Run tests with:

```bash
pytest
```

## Repository layout

```text
src/gfl_interaction/
  model.py        fourth-order model and simulation
  energies.py     swing and PI energy functions
  small_gain.py   local H-infinity gain and bounded-real storage
  reporting.py    batch run, metrics, and figures
scripts/run_all.py
tests/test_core.py
docs/theory.md
```

## Important scope statement

This is a **transparent numerical reconstruction**, not the authors' original code. The paper does not publish a unique mapping from current-loop bandwidth to `k_pc, k_ic`, nor a unique ODE initialization across the ideal voltage step. The defaults in `GFLParameters` were calibrated to recover the reported 300/500-Hz qualitative split. Change them directly for sensitivity studies.

The PI handoff metric currently uses the actual fourth-order trajectory. It is a useful research diagnostic but not yet a standalone theorem. The intended next step is a finite-time LTV/IQC certificate that upper-bounds fast energy injection and lower-bounds PI dissipation before switching to a post-transient energy certificate.
