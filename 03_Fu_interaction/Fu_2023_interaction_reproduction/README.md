
# Fu et al. (2023) GFL-VSC interaction reproduction

## 1. Reproduced content

This package reconstructs the main mechanism in:

> X. Fu et al., “Synchronization Stability of Grid-Following VSC Considering Interactions of Inner Current Loop and Parallel-Connected Converters,” IEEE Transactions on Smart Grid, 2023.

It contains two levels of models.

### A. Interaction-level reduced model

The model follows paper equation (11):

\[
M_i\ddot{\delta}_i
=
A_i+B_i-V_g\sin\delta_i
+\Delta A_i+\Delta B_i
+
(A_{i1}+B_{i1}-D_i\cos\delta_i)\dot{\delta}_i.
\]

The extended Lyapunov function follows equations (16)–(18):

\[
W_i
=
\frac12M_i\Delta\omega_i^2
-(A_i+B_i)\delta_i
-V_g\cos\delta_i
-\beta_i(A_i+B_i-V_g\sin\delta_i)\Delta\omega_i+\alpha_i.
\]

The shaded/hatched non-dissipative region in the figures is defined by a robust interpretation:

\[
\max_{|\Delta A_i+\Delta B_i|\le \bar u_i}\dot W_i>0.
\]

This is slightly more conservative than selecting one particular dynamic-interaction waveform, but it makes the uncertainty interpretation explicit.

### B. Full dq average model

The full model assembles paper equations (2)–(7):

- local SRF-PLL;
- current PI controller;
- output-filter current states;
- local dq/common DQ coordinate transformations;
- algebraic multi-converter network coupling.

The q-axis voltage and PLL frequency form an algebraic feedback loop. It is solved as a small linear system at every ODE evaluation.

## 2. Main numerical findings

Run:

```bash
cd src
python run_reproduction.py
```

The generated examples show:

1. **Inner-loop/PLL coupling**
   - With ideal voltage feedforward, the current disturbance \(\Delta A_1\) remains small and the converter resynchronizes.
   - Without feedforward, current transients inject a much larger \(\Delta A_1\), and the projected trajectory crosses the unstable boundary.

2. **Static device coupling**
   - For two identical converters, the neighbor produces approximately \(B_1=0.75A_1\), matching the paper.
   - The 50%-voltage, 0.04-s fault escapes; the 70%-voltage, 0.02-s fault returns.

3. **Dynamic device coupling**
   - Parameter mismatch makes \(\delta_{12}(t)\) time-varying.
   - Consequently, \(\Delta B_1\neq0\), and converter 1 sees a history-dependent forcing from converter 2.

4. **Projected path dependence**
   - Two trajectories start at exactly the same \((\delta_1,\Delta\omega_1)\).
   - Their hidden interaction states have opposite signs.
   - One trajectory escapes while the other returns to the SEP.
   - Therefore, the two-dimensional phase plane is not an autonomous state description once current-loop and neighbor states are omitted.

## 3. Meaning of the non-dissipative region

The hatched region does **not** mean “all states are unstable.”

It means that the chosen energy function can increase there:

\[
\dot W_i>0.
\]

If an outer annulus still has \(\dot W_i\le0\), trajectories may remain bounded and form a limit cycle or a bounded oscillation. This is the physical meaning of Fu et al.’s “non-asymptotic synchronization.” When the dynamic interaction disappears, the same bounded set may become a genuine region of attraction.

## 4. Relation to Wu et al. (2024)

### Fu et al.

- Main question: how do inner-loop and parallel-converter interactions alter synchronization energy and damping?
- Reduced state: \((\delta_i,\Delta\omega_i)\).
- Hidden effects: summarized as \(\Delta A_i+\Delta B_i\).
- Main tool: extended invariance principle and an extended Lyapunov function.
- Strength: multi-converter interaction, non-dissipative regions, limit cycles, cascading synchronization.
- Limitation: it does not derive a tight bound on \(\Delta A_i\) from current-loop bandwidth; the interaction bound is imposed or measured.

### Wu et al.

- Main question: when is the fast current loop sufficiently fast to be ignored?
- State: explicit fourth-order current-loop–PLL model.
- Main tool: improved equal-area criterion, numerical region of attraction, current-loop/PLL bandwidth boundary.
- Strength: explicit fast/slow dynamics and a practical model-reduction threshold.
- Limitation: essentially a single-converter problem; it does not capture \(\Delta B_i\) from asynchronous neighboring devices.

### Unified interpretation

Wu supplies a **dynamic generator/bound for \(\Delta A_i\)**.  
Fu supplies a **networked energy framework for \(\Delta A_i+\Delta B_i\)**.

A natural combined model is

\[
\dot x_i=f_i(x_i)+G_i r_i,\qquad
r_i=\Delta A_i(z_i,x_i)+\Delta B_i(x_i,x_j,z_j),
\]

where \(x_i=(\delta_i,\Delta\omega_i)\) is slow and \(z_i\) contains current-loop states.

The reduced phase plane is valid only when:

1. the fast boundary layer is uniformly stable;
2. its tracking residual is small enough;
3. the neighbor-induced residual is small enough;
4. the residual does not destroy the outer dissipative barrier.

This is the point where singular perturbation, ISS/small-gain, and robust/IQC analysis can be connected.

## 5. Reproduction limitations

The paper does not list the filter inductance \(L_f\) separately in Table I. This package uses \(L_f=1\) mH as a transparent reconstruction setting.

The paper’s “dynamic interaction in the range of \(\pm\gamma A_i\)” does not give a unique state-dependent waveform. The contour plots here use the worst-case derivative over the stated bound, so the hatched region is a robust non-dissipative set rather than a pixel-identical reconstruction of the original figure.

The exact cascading threshold in Fig. 12 also depends on implementation details of the current controller and protection/tripping logic. The package reproduces the interaction channel and path dependence, while avoiding claims of pixel-exact agreement where the paper does not provide enough implementation detail.
