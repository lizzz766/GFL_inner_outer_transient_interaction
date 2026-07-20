# Numerical reproduction overview

## Reproduced numerical landmarks

| Quantity | Reproduction | Paper |
|---|---:|---:|
| Subcritical Hopf in $\kappa_p$ | 0.201139 p.u. | approximately 0.201 p.u. |
| Center-UPO SNPO in $\kappa_p$ | 0.292420 p.u. | 0.292 p.u. |
| Positive critical phase jump at $\kappa_p=0.4$ p.u. | 2.197 rad | approximately 2.3 rad |
| Continued periodic orbits | 143 | branch shown in Figs. 7–8 |

## Main reproduced conclusion

For a $-\pi$ grid phase jump and $\kappa_p=0.4$ p.u.:

- the ideal-current second-order model settles to the next $2\pi$-equivalent SEP;
- the 12th-order model crosses the divergence threshold because the current/LCL boundary-layer transient strongly perturbs the PLL.

## Continuation method

The code does not use MATCONT. It implements:

1. finite-difference Jacobian and Hopf detection;
2. a Hopf-eigenvector periodic-orbit initial guess;
3. collocation using `scipy.integrate.solve_bvp`;
4. pseudo-arclength continuation with an auxiliary integral state.

This recovers the fold of periodic orbits at $\kappa_p\approx0.2924$ p.u.

## Accuracy caveat

For current-loop bandwidth sweeps, both current-controller gains are scaled together while preserving $\beta_i/\beta_p$, following the paper's stated tuning rule. This is an approximate mapping from desired bandwidth to gains, so the high-bandwidth small-signal threshold is qualitative rather than an exact reproduction of Fig. 12.
