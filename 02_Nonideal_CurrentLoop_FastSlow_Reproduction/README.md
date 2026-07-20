# Nonideal Current Loop Fast-Slow Reproduction

This folder contains the reproduction package for the 12th-order averaged GFL model.

## Contents

- full-order dq averaged model
- Hopf detection
- unstable periodic orbit continuation
- SNPO reproduction
- fast-slow scale interpretation

## Fast-slow formulation

The complete system is represented as

$$
\dot{x}_s=f_s(x_s,z_f),
$$

$$
\dot{z}_f=f_f(x_s,z_f).
$$

The quasi-steady model is obtained by replacing the fast dynamics with the critical manifold:

$$f_f(x_s,z_f)=0.$$

The key mismatch variable is

$$e=z_f-h_0(x_s).$$

Large disturbances can generate a nonzero boundary-layer deviation even when the current loop bandwidth is high, because physical electrical states and controller integrators cannot change instantaneously.

## Planned implementation

```
src/
  gfl_model.py
  continuation.py
scripts/
  run_reproduction.py
  run_continuation.py
outputs/
```

Future extensions:

- composite Lyapunov analysis
- ISS/small-gain conditions
- IQC based model adequacy certificates
