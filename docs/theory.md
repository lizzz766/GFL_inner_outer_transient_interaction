# Theory map

## 1. Fourth-order reconstruction

The state is

\[
x=(\delta,\Delta\omega,\Delta I_d,\Delta\dot I_d).
\]

The implementation follows Wu et al.'s fourth-order model. The current-loop bandwidth-to-PI-gain map and the discontinuity initialization are not uniquely specified in the paper, so both are explicit reconstruction settings.

## 2. Classical swing energy + local fast small gain

The quasi-steady PLL model admits

\[
V_s=\frac12M\omega^2+U_f(\cos\delta^\star-\cos\delta)-p(\delta-\delta^\star).
\]

The local fast residual model is

\[
\dot e=Ae+B\omega,\qquad y=Ce,
\]

and the code computes

\[
\gamma=\|C(sI-A)^{-1}B\|_\infty.
\]

This exposes the two-sided failure discussed in the project:

- local \(\gamma<D(\delta^\star)\) looks safe even for the 300 Hz unstable case;
- the one-shot energy condition rejects the stable 500 Hz case because the initial post-fault energy already exceeds the UEP level.

## 3. PI-controller energy + fast-dynamic interaction

Let

\[
q_0(\delta)=\omega_gL_gI_{ref}-U_f\sin\delta,
\]

and decompose the full PLL q-axis voltage as

\[
v_q=q_0+r,
\]

with

\[
r=L_gI_{ref}\omega+\omega_gL_g\Delta I_d+L_g\omega\Delta I_d.
\]

The PLL integrator coordinate is

\[
x_I=\omega-k_pv_q.
\]

The tailored PI energy is

\[
V_{PI}=\frac{x_I^2}{2k_i}+U_f(\cos\delta^\star-\cos\delta)-p(\delta-\delta^\star),
\]

and satisfies the exact accounting identity

\[
\dot V_{PI}=-k_pq_0^2+(x_I-k_pq_0)r.
\]

The code plots both cumulative terms. It also reports the first simulated handoff time at which the fast state has decayed and \(V_{PI}<V_{critical}\). That handoff is currently a trajectory diagnostic, not a robust certificate. A rigorous next step is to replace the actual interaction integral by an LTV/IQC upper bound and replace the actual PI dissipation by a certified lower bound.
