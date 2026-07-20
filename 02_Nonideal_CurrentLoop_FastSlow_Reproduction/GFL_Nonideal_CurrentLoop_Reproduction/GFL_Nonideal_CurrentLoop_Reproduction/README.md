
# Reproduction: Transient Synchronization Stability of GFLCs Considering a Nonideal Current Loop

This repository reproduces the central numerical claims of:

C. C. Liu, J. Yang, C. K. Tse, and M. Huang,  
“Transient Synchronization Stability of Grid-Following Converters Considering Nonideal Current Loop,”  
IEEE Transactions on Power Electronics, 38(11), 13757–13769, 2023.  
DOI: 10.1109/TPEL.2023.3303447.

## What is reproduced

1. **The complete 12th-order averaged model**, using the paper’s equations (1)–(14) and Table I.
2. **The phase-jump conclusion reversal**:
   - the ideal-current second-order model converges;
   - the full model diverges for the same post-jump PLL state.
3. **The local subcritical Hopf point** near
   \[
   \kappa_p \approx 0.201\ {\rm p.u.}
   \]
4. **The optimistic critical-phase-jump margin of the second-order model**.
5. **The center unstable-periodic-orbit branch**, using:
   - a Hopf eigenvector starter,
   - collocation with `scipy.integrate.solve_bvp`,
   - pseudo-arclength continuation.
6. **The saddle-node bifurcation of periodic orbits (SNPO)** near
   \[
   \kappa_p \approx 0.292\ {\rm p.u.},
   \]
   agreeing with the paper’s reported value.
7. **A fast–slow spectral diagnostic**, including the frozen fast block and the Schur-complement slow Jacobian.

Run:

```bash
python scripts/run_reproduction.py
```

## Important parameter convention

The paper states several gain settings in per unit. For example, its mismatch example uses

\[
\kappa_p=0.4\ {\rm p.u.},
\]

which means the physical gain is

\[
\kappa_p=0.4\times 0.413.
\]

Using `0.4` directly as the physical gain incorrectly predicts convergence in both models.

---

# Fast–slow interpretation

Partition the full state into

\[
x_s=\begin{bmatrix}y_\omega&\delta_L\end{bmatrix}^{\!\top},
\qquad
z_f=
\begin{bmatrix}
i_{rd}&i_{rq}&v_{cd}&v_{cq}&i_{gd}&i_{gq}&
y_{id}&y_{iq}&v_{od}&v_{oq}
\end{bmatrix}^{\!\top}.
\]

The full model has the form

\[
\dot x_s=f_s(x_s,z_f),\qquad
\dot z_f=f_f(x_s,z_f).
\]

When the electrical and current-control states are assumed infinitely fast, solve

\[
f_f(x_s,z_f)=0
\]

for the critical manifold

\[
z_f=h_0(x_s).
\]

Substituting it into the slow dynamics gives exactly the paper’s ideal-current second-order model:

\[
\dot x_s=f_s(x_s,h_0(x_s)).
\]

Therefore, the second-order model is not merely an empirical approximation: it is the **quasi-steady slow flow on the fast equilibrium manifold**.

## Why a large phase jump defeats a bandwidth-only argument

Let

\[
e=z_f-h_0(x_s).
\]

Immediately after a phase jump, the physical inductor currents, capacitor voltages, PI integrators, and delayed converter voltages cannot jump, whereas the new quasi-steady value \(h_0(x_s^+)\) changes substantially. Hence

\[
e(0^+)=z_f(0^-)-h_0(x_s^+)
\]

can be large even when the current loop is much faster than the PLL.

The initial fast boundary layer then produces a finite kick to the PLL:

\[
\Delta x_{s,\rm bl}
\approx
\int_0^{t_{\rm bl}}
\left[
f_s(x_s,z_f)-f_s(x_s,h_0(x_s))
\right]dt.
\]

A reduced trajectory may remain inside its basin while this kick moves the full state across the stable manifold of a UPO.

## Geometric meaning of the paper’s UPO

For the reduced two-state system, a UPO created by a homoclinic or subcritical-Hopf bifurcation bounds the stable equilibrium’s basin. Under uniform normal hyperbolicity, Fenichel theory suggests persistence of the slow manifold and nearby hyperbolic invariant sets for sufficiently small scale ratio.

The paper shows what happens when that approximation is not uniform:

- the current/LCL states deform the periodic-orbit branch;
- an **internal UPO branch** approaches the SEP;
- the internal and external UPO branches collide in an **SNPO**;
- the basin can collapse much earlier than predicted by the second-order model.

Thus the conclusion reversal is a **global invariant-manifold error**, not merely a local eigenvalue error.

## Spectral crossover and Shilnikov behavior

At the relevant unstable equilibrium, the full-order system can have complex leading stable eigenvalues associated with electrical/current-loop dynamics. They then participate in the homoclinic geometry and raise the effective center-manifold dimension above two. This is why the full system can display Shilnikov-type multiple periodic orbits that the planar slow model cannot represent.

Increasing current-loop bandwidth pushes these modes left and may recover a more two-dimensional, Type-1 saddle. But increasing gain indefinitely is not a valid singular limit because the PWM delay and LCL dynamics remain fixed. The paper’s high-bandwidth Hopf instability is the failure of the **fast boundary-layer stability condition**.

## Fastness versus decoupling

The paper finds that full voltage feedforward \(k=1\) makes the full and reduced basins nearly coincide more effectively than merely increasing current-loop bandwidth.

In fast–slow language:

- higher bandwidth increases contraction of the fast subsystem;
- feedforward cancels a forcing/coupling channel and makes the quasi-steady manifold more nearly invariant.

Therefore model validity depends on both

\[
\text{fast decay rate}
\quad\text{and}\quad
\text{slow–fast coupling strength},
\]

not on bandwidth ratio alone.

## Toward a certificate

A natural next step is to construct slow and fast Lyapunov functions satisfying

\[
\dot V_s\le-a\|x_s\|^2+b\|x_s\|\|e\|,
\]

\[
\dot V_f\le-c\|e\|^2+d\|x_s\|\|e\|.
\]

For \(W=V_s+\rho V_f\),

\[
\dot W<0
\]

is guaranteed if some \(\rho>0\) satisfies

\[
(b+\rho d)^2<4\rho ac.
\]

This would convert the paper’s bifurcation map into a conservative but certifiable fast–slow coupling condition. A second, complementary condition is a boundary-layer kick bound ensuring that the post-disturbance state remains inside the UPO’s stable-manifold boundary.

## Reproduction limits

- The code reproduces the Hopf-to-SNPO center-UPO branch.
- It does not resolve every small Shilnikov “wiggle” or continue all the way to an infinite-period homoclinic orbit; that requires tighter tolerances and substantially more continuation steps.
- The switched PWM experiment is not reproduced. The implemented model is the paper’s 12th-order averaged full-order model.
