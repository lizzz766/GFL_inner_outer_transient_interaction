
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


@dataclass
class FuParameters:
    """Parameters from Table I of Fu et al. (2023)."""
    Vg: float = 311.0
    omega0: float = 100.0 * np.pi
    Ri: float = 0.1
    Li: float = 1e-3
    Rg: float = 0.1
    Lg: float = 3e-3
    kp_pll: float = 0.3
    ki_pll: float = 30.0
    kp_acc: float = 5.0
    ki_acc: float = 50.0
    id_ref: float = 100.0
    iq_ref: float = -10.0
    # The paper does not list Lf separately in Table I. 1 mH is used here
    # as a transparent reconstruction setting.
    Lf: float = 1e-3

    @property
    def A(self) -> float:
        return (
            (self.Ri + self.Rg) * self.iq_ref
            + (self.Li + self.Lg) * self.omega0 * self.id_ref
        )

    @property
    def A1(self) -> float:
        return (self.Li + self.Lg) * self.id_ref

    @property
    def B1_identical_neighbor(self) -> float:
        return self.Lg * self.id_ref

    @property
    def D(self) -> float:
        return self.kp_pll * self.Vg / self.ki_pll


@dataclass
class FuReducedModel:
    """Equation (11) and the extended Lyapunov function (16)-(18)."""
    p: FuParameters = field(default_factory=FuParameters)
    static_scale: float = 1.5
    dynamic_bound_scale: float = 0.3
    b1: Optional[float] = None
    beta_fraction: float = 0.65

    def quantities(self):
        p = self.p
        B1 = p.B1_identical_neighbor if self.b1 is None else self.b1
        Pm = self.static_scale * p.A
        if abs(Pm) >= p.Vg:
            raise ValueError("No equilibrium exists because |A+B| >= Vg.")
        delta_sep = np.arcsin(Pm / p.Vg)
        delta_uep = np.pi - delta_sep
        At = p.A1 + B1
        M = (1.0 - p.kp_pll * At) / p.ki_pll
        D = p.D
        numerator = 4.0 * M * (D * np.cos(delta_sep) - At)
        denominator = (
            4.0 * M * p.Vg * np.cos(delta_sep)
            + (D * np.cos(delta_sep) - At) ** 2
        )
        beta_max = numerator / denominator
        beta = self.beta_fraction * beta_max
        return {
            "Pm": Pm,
            "delta_sep": delta_sep,
            "delta_uep": delta_uep,
            "At": At,
            "M": M,
            "D": D,
            "beta_max": beta_max,
            "beta": beta,
            "u_max": self.dynamic_bound_scale * p.A,
        }

    def energy(self, delta, omega):
        q = self.quantities()
        Pm = q["Pm"]
        beta = q["beta"]
        C = Pm - self.p.Vg * np.sin(delta)
        alpha = Pm * q["delta_sep"] + self.p.Vg * np.cos(q["delta_sep"])
        return (
            0.5 * q["M"] * omega**2
            - Pm * delta
            - self.p.Vg * np.cos(delta)
            - beta * C * omega
            + alpha
        )

    def worst_case_wdot(self, delta, omega):
        """
        Worst-case derivative over |Delta A + Delta B| <= u_max.
        This is a robust interpretation of the paper's bounded dynamic interaction.
        """
        q = self.quantities()
        C = q["Pm"] - self.p.Vg * np.sin(delta)
        a = q["At"] - q["D"] * np.cos(delta)
        beta = q["beta"]
        base = (
            (a + beta * self.p.Vg * np.cos(delta)) * omega**2
            - beta / q["M"] * (C + a * omega) * C
        )
        return base + q["u_max"] * np.abs(omega - beta * C / q["M"])

    def rhs_with_hidden_residual(self, t, x, tau_a=0.03, tau_b=0.08):
        """
        A minimal hidden-state realization of Delta A and Delta B.
        x = [delta, omega, rA, rB].
        It is used only to demonstrate projection/path dependence.
        """
        delta, omega, r_a, r_b = x
        q = self.quantities()
        a = q["At"] - q["D"] * np.cos(delta)
        delta_dot = omega
        omega_dot = (
            q["Pm"] - self.p.Vg * np.sin(delta) + a * omega + r_a + r_b
        ) / q["M"]
        r_a_dot = (-r_a + 0.05 * self.p.A * np.tanh(omega / 20.0)) / tau_a
        r_b_dot = -r_b / tau_b
        return np.array([delta_dot, omega_dot, r_a_dot, r_b_dot])

    def simulate_hidden_path(self, delta0, omega0, r_a0, r_b0, t_end=2.0):
        def event_escape(t, x):
            return np.pi - abs(x[0])
        event_escape.terminal = True
        event_escape.direction = -1
        return solve_ivp(
            self.rhs_with_hidden_residual,
            (0.0, t_end),
            np.array([delta0, omega0, r_a0, r_b0], dtype=float),
            method="RK45",
            rtol=1e-8,
            atol=1e-9,
            max_step=1e-3,
            events=event_escape,
        )


@dataclass
class FullMultiVSC:
    """
    Full dq average model assembled from Fu et al. equations (2)-(7).

    Each converter has states:
    [x_PLL, delta, x_acc_d, x_acc_q, i_d, i_q].

    The q-axis network/PLL algebraic loop is solved exactly at every RHS call.
    """
    n: int
    Vg0: float = 311.0
    omega0: float = 100.0 * np.pi
    Rg: float = 0.1
    Lg: float = 3e-3
    Ri: np.ndarray = field(init=False)
    Li: np.ndarray = field(init=False)
    Lf: np.ndarray = field(init=False)
    kp_pll: np.ndarray = field(init=False)
    ki_pll: np.ndarray = field(init=False)
    kp_acc: np.ndarray = field(init=False)
    ki_acc: np.ndarray = field(init=False)
    id_ref: np.ndarray = field(init=False)
    iq_ref: np.ndarray = field(init=False)
    voltage_feedforward: np.ndarray = field(init=False)

    def __post_init__(self):
        self.Ri = np.full(self.n, 0.1)
        self.Li = np.full(self.n, 1e-3)
        self.Lf = np.full(self.n, 1e-3)
        self.kp_pll = np.full(self.n, 0.3)
        self.ki_pll = np.full(self.n, 30.0)
        self.kp_acc = np.full(self.n, 5.0)
        self.ki_acc = np.full(self.n, 50.0)
        self.id_ref = np.full(self.n, 100.0)
        self.iq_ref = np.full(self.n, -10.0)
        self.voltage_feedforward = np.ones(self.n)

    def network_voltage(self, x, Vg):
        xi_pll = x[0::6]
        delta = x[1::6]
        i_d = x[4::6]
        i_q = x[5::6]

        c = (self.Ri + self.Rg) * i_q - Vg * np.sin(delta)
        B = np.zeros((self.n, self.n))

        for i in range(self.n):
            B[i, i] = (self.Li[i] + self.Lg) * i_d[i]
            for j in range(self.n):
                if i == j:
                    continue
                dij = delta[i] - delta[j]
                c[i] += (
                    self.Rg * i_q[j] * np.cos(dij)
                    - self.Rg * i_d[j] * np.sin(dij)
                )
                B[i, j] = self.Lg * (
                    i_d[j] * np.cos(dij) + i_q[j] * np.sin(dij)
                )

        lhs = np.eye(self.n) - B * self.kp_pll[None, :]
        rhs = c + B @ (self.omega0 + self.ki_pll * xi_pll)
        v_q = np.linalg.solve(lhs, rhs)
        omega_abs = self.omega0 + self.kp_pll * v_q + self.ki_pll * xi_pll

        v_d = np.empty(self.n)
        for i in range(self.n):
            v_d[i] = (
                Vg * np.cos(delta[i])
                + (self.Ri[i] + self.Rg) * i_d[i]
                - (self.Li[i] + self.Lg) * omega_abs[i] * i_q[i]
            )
            for j in range(self.n):
                if i == j:
                    continue
                dij = delta[i] - delta[j]
                a = self.Rg * i_d[j] - self.Lg * omega_abs[j] * i_q[j]
                b = self.Rg * i_q[j] + self.Lg * omega_abs[j] * i_d[j]
                v_d[i] += a * np.cos(dij) + b * np.sin(dij)

        return v_d, v_q, omega_abs

    def equilibrium(self):
        def residual(delta):
            x = np.zeros(6 * self.n)
            x[1::6] = delta
            x[4::6] = self.id_ref
            x[5::6] = self.iq_ref
            return self.network_voltage(x, self.Vg0)[1]

        guess = np.full(self.n, 0.42 if self.n == 1 else 0.78)
        sol = root(residual, guess)
        if not sol.success:
            raise RuntimeError(f"Equilibrium solve failed: {sol.message}")

        x = np.zeros(6 * self.n)
        x[1::6] = sol.x
        x[4::6] = self.id_ref
        x[5::6] = self.iq_ref
        v_d, v_q, _ = self.network_voltage(x, self.Vg0)

        # At equilibrium, the PI output compensates the non-feedforward voltage.
        x[2::6] = (1.0 - self.voltage_feedforward) * v_d / self.ki_acc
        x[3::6] = (1.0 - self.voltage_feedforward) * v_q / self.ki_acc
        return x

    def rhs(self, Vg):
        def f(t, x):
            v_d, v_q, omega_abs = self.network_voltage(x, Vg)
            dx = np.zeros_like(x)
            dx[0::6] = v_q
            dx[1::6] = omega_abs - self.omega0

            for i in range(self.n):
                i_d = x[6 * i + 4]
                i_q = x[6 * i + 5]
                error = np.array(
                    [self.id_ref[i] - i_d, self.iq_ref[i] - i_q]
                )
                dx[6 * i + 2 : 6 * i + 4] = error
                v = np.array([v_d[i], v_q[i]])
                x_acc = x[6 * i + 2 : 6 * i + 4]

                # Cross-decoupling cancels the rotating-frame plant term.
                dx[6 * i + 4 : 6 * i + 6] = (
                    self.kp_acc[i] * error
                    + self.ki_acc[i] * x_acc
                    - (1.0 - self.voltage_feedforward[i]) * v
                ) / self.Lf[i]

            return dx
        return f

    def simulate_fault(self, depth, duration, fault_time=0.1, t_end=0.8):
        y = self.equilibrium()
        t_parts = []
        y_parts = []

        def event_escape(t, x):
            return np.pi - np.max(np.abs(x[1::6]))
        event_escape.terminal = True
        event_escape.direction = -1

        segments = [
            (0.0, fault_time, self.Vg0),
            (fault_time, fault_time + duration, depth * self.Vg0),
            (fault_time + duration, t_end, self.Vg0),
        ]

        escaped = False
        for a, b, Vg in segments:
            if b <= a or escaped:
                continue
            sol = solve_ivp(
                self.rhs(Vg),
                (a, b),
                y,
                method="BDF",
                rtol=2e-6,
                atol=1e-8,
                max_step=1e-3,
                events=event_escape,
            )
            t_parts.append(sol.t)
            y_parts.append(sol.y)
            y = sol.y[:, -1]
            escaped = bool(sol.t_events and len(sol.t_events[0]) > 0)

        t = np.concatenate(t_parts)
        Y = np.concatenate(y_parts, axis=1)
        omega = np.zeros((len(t), self.n))
        delta_A = np.zeros((len(t), self.n))
        delta_B = np.zeros((len(t), self.n))

        delta_eq = self.equilibrium()[1::6]
        for k, tk in enumerate(t):
            if fault_time <= tk < fault_time + duration:
                Vg = depth * self.Vg0
            else:
                Vg = self.Vg0
            _, _, omega_abs = self.network_voltage(Y[:, k], Vg)
            omega[k, :] = omega_abs - self.omega0

            i_d = Y[4::6, k]
            i_q = Y[5::6, k]
            delta = Y[1::6, k]
            for i in range(self.n):
                delta_A[k, i] = (
                    (self.Ri[i] + self.Rg) * (i_q[i] - self.iq_ref[i])
                    + (self.Li[i] + self.Lg)
                    * (omega_abs[i] * i_d[i] - self.omega0 * self.id_ref[i])
                )
                for j in range(self.n):
                    if i == j:
                        continue
                    dij = delta[i] - delta[j]
                    dije = delta_eq[i] - delta_eq[j]
                    coeff_c = self.Rg * self.iq_ref[j] + self.Lg * self.omega0 * self.id_ref[j]
                    coeff_s = self.Rg * self.id_ref[j] - self.Lg * self.omega0 * self.iq_ref[j]
                    delta_B[k, i] += (
                        coeff_c * (np.cos(dij) - np.cos(dije))
                        - coeff_s * (np.sin(dij) - np.sin(dije))
                    )

        return {
            "t": t,
            "x": Y,
            "omega": omega,
            "delta_A": delta_A,
            "delta_B": delta_B,
            "escaped": escaped,
        }
