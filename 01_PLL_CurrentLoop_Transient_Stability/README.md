# PLL-Current Loop Transient Stability Reproduction

This folder contains the reproduction package for transient synchronization stability of grid-following converters considering nonideal current-loop dynamics.

## Structure

```
src/
  reduced_model.py
  full_current_loop_model.py
scripts/
  run_time_domain.py
  scan_stability_region.py
data/
outputs/
```

## Core idea

The ideal-current model assumes the electrical/current states stay on a quasi-steady manifold:

$$z_f=h_0(x_s)$$

The complete model contains the fast deviation:

$$e=z_f-h_0(x_s).$$

After a large disturbance, $e(0^+)$ can be large because currents, capacitors and controller states cannot jump instantaneously. The accumulated boundary-layer effect modifies PLL dynamics and may change the transient synchronization conclusion.

## Planned reproduction items

- second-order ideal-current model
- full current-loop model
- time-domain comparison
- EAC interpretation
- bandwidth-region analysis
- connection with fast-slow and robustness certificates
