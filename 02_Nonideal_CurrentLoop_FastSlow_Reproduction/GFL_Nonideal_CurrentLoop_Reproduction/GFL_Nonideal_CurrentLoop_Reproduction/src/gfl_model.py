
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

KP_BASE = 0.413
KI_BASE = 7.786
BETA_P_BASE = 1.648
BETA_I_BASE = 51.779
FCL_BASE_HZ = 126.29

@dataclass(frozen=True)
class Params:
    Vsm: float = 311.0
    Lr: float = 2e-3
    Ls: float = 5e-3
    rr: float = 0.1
    rs: float = 0.1
    Cr: float = 60e-6
    rc: float = 1.0
    Lg: float = 0.6e-3
    rg: float = 0.05
    kp: float = KP_BASE
    ki: float = KI_BASE
    beta_p: float = BETA_P_BASE
    beta_i: float = BETA_I_BASE
    w0: float = 100*np.pi
    Tctr: float = 0.25e-3
    igd_ref: float = 30.0
    igq_ref: float = 0.0
    kff: float = 0.0

    @property
    def Td(self) -> float:
        return 1.5*self.Tctr

def with_kp_pu(p: Params, kp_pu: float) -> Params:
    return replace(p, kp=kp_pu*KP_BASE)

def with_current_bandwidth(p: Params, fcl_hz: float) -> Params:
    # Paper keeps beta_i/beta_p fixed while tuning current-loop bandwidth.
    scale = fcl_hz/FCL_BASE_HZ
    return replace(p, beta_p=BETA_P_BASE*scale, beta_i=BETA_I_BASE*scale)

def rhs_full(t: float, x: np.ndarray, p: Params) -> np.ndarray:
    yw, delta, ird, irq, vcd, vcq, igd, igq, yid, yiq, vod, voq = x
    vsd = p.Vsm*np.cos(delta)
    vsq = -p.Vsm*np.sin(delta)
    vrd = vcd + p.rc*(ird-igd)
    vrq = vcq + p.rc*(irq-igq)

    fac = p.Ls*p.Lg/(p.Ls+p.Lg)
    vgq = fac*(vsq/p.Ls + vrq/p.Lg + (p.rs/p.Ls-p.rg/p.Lg)*igq)
    vgd = fac*(vsd/p.Ls + vrd/p.Lg + (p.rs/p.Ls-p.rg/p.Lg)*igd)

    wL = p.kp*vgq + p.ki*yw + p.w0
    vord = p.beta_p*(p.igd_ref-igd) + p.beta_i*yid + p.kff*vgd
    vorq = p.beta_p*(p.igq_ref-igq) + p.beta_i*yiq + p.kff*vgq

    Lt = p.Ls+p.Lg
    rt = p.rs+p.rg
    return np.array([
        vgq,
        wL-p.w0,
        (vod-vrd-p.rr*ird+wL*p.Lr*irq)/p.Lr,
        (voq-vrq-wL*p.Lr*ird-p.rr*irq)/p.Lr,
        (ird-igd+wL*p.Cr*vcq)/p.Cr,
        (irq-igq-wL*p.Cr*vcd)/p.Cr,
        (vrd-vsd-rt*igd+wL*Lt*igq)/Lt,
        (vrq-vsq-wL*Lt*igd-rt*igq)/Lt,
        p.igd_ref-igd,
        p.igq_ref-igq,
        (vord-vod)/p.Td,
        (vorq-voq)/p.Td,
    ], dtype=float)


def rhs_full_batch(Y: np.ndarray, p: Params) -> np.ndarray:
    """Vectorized version of rhs_full. Y has shape (12, n_points)."""
    yw, delta, ird, irq, vcd, vcq, igd, igq, yid, yiq, vod, voq = Y
    vsd = p.Vsm*np.cos(delta)
    vsq = -p.Vsm*np.sin(delta)
    vrd = vcd + p.rc*(ird-igd)
    vrq = vcq + p.rc*(irq-igq)
    fac = p.Ls*p.Lg/(p.Ls+p.Lg)
    vgq = fac*(vsq/p.Ls + vrq/p.Lg + (p.rs/p.Ls-p.rg/p.Lg)*igq)
    vgd = fac*(vsd/p.Ls + vrd/p.Lg + (p.rs/p.Ls-p.rg/p.Lg)*igd)
    wL = p.kp*vgq + p.ki*yw + p.w0
    vord = p.beta_p*(p.igd_ref-igd) + p.beta_i*yid + p.kff*vgd
    vorq = p.beta_p*(p.igq_ref-igq) + p.beta_i*yiq + p.kff*vgq
    Lt = p.Ls+p.Lg
    rt = p.rs+p.rg
    return np.vstack([
        vgq,
        wL-p.w0,
        (vod-vrd-p.rr*ird+wL*p.Lr*irq)/p.Lr,
        (voq-vrq-wL*p.Lr*ird-p.rr*irq)/p.Lr,
        (ird-igd+wL*p.Cr*vcq)/p.Cr,
        (irq-igq-wL*p.Cr*vcd)/p.Cr,
        (vrd-vsd-rt*igd+wL*Lt*igq)/Lt,
        (vrq-vsq-wL*Lt*igd-rt*igq)/Lt,
        p.igd_ref-igd,
        p.igq_ref-igq,
        (vord-vod)/p.Td,
        (vorq-voq)/p.Td,
    ])

def rhs_reduced(t: float, x: np.ndarray, p: Params) -> np.ndarray:
    yw, delta = x
    den = 1-p.kp*p.Ls*p.igd_ref
    vgq = (-p.Vsm*np.sin(delta)
           + p.igd_ref*p.Ls*(p.ki*yw+p.w0)
           + p.rs*p.igq_ref)/den
    wL = p.kp*vgq+p.ki*yw+p.w0
    return np.array([vgq, wL-p.w0])

def equilibrium_angle(p: Params, stable: bool = True) -> float:
    arg = (p.w0*p.Ls*p.igd_ref+p.rs*p.igq_ref)/p.Vsm
    d = np.arcsin(arg)
    return d if stable else np.pi-d

def equilibrium_full(p: Params, stable: bool = True) -> np.ndarray:
    d = equilibrium_angle(p, stable)
    w, V = p.w0, p.Vsm
    igd, igq = p.igd_ref, p.igq_ref
    vsd, vsq = V*np.cos(d), -V*np.sin(d)
    Lt, rt = p.Ls+p.Lg, p.rs+p.rg
    vrd = vsd+rt*igd-w*Lt*igq
    vrq = vsq+w*Lt*igd+rt*igq
    a = w*p.Cr*p.rc
    vcd, vcq = np.linalg.solve(np.array([[1.,-a],[a,1.]]), np.array([vrd,vrq]))
    ird = igd-w*p.Cr*vcq
    irq = igq+w*p.Cr*vcd
    vod = vrd+p.rr*ird-w*p.Lr*irq
    voq = vrq+w*p.Lr*ird+p.rr*irq
    return np.array([
        0., d, ird, irq, vcd, vcq, igd, igq,
        (vod-p.kff*vrd)/p.beta_i,
        (voq-p.kff*vrq)/p.beta_i,
        vod, voq
    ])

def jacobian_fd(fun, x: np.ndarray, rel_step: float = 1e-6) -> np.ndarray:
    n = x.size
    J = np.empty((n,n))
    for j in range(n):
        h = rel_step*max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:,j] = (fun(xp)-fun(xm))/(2*h)
    return J

def dominant_real_full(p: Params, stable: bool = True) -> float:
    xe = equilibrium_full(p, stable)
    ev = np.linalg.eigvals(jacobian_fd(lambda z: rhs_full(0,z,p), xe))
    return float(np.max(ev.real))

def find_hopf_kp_pu(p: Params, bracket=(0.18,0.23)) -> float:
    return brentq(lambda q: dominant_real_full(with_kp_pu(p,q)), *bracket)

def simulate_full_phase_jump(
    p: Params, jump: float, t_end: float = 3.0,
    divergence_angle: float = 6*np.pi
):
    xe = equilibrium_full(p, True)
    x0 = xe.copy()
    x0[1] += jump
    def event(t, x):
        return divergence_angle-abs(x[1]-xe[1])
    event.terminal = True
    event.direction = -1
    return solve_ivp(
        lambda t,x: rhs_full(t,x,p), (0,t_end), x0,
        method="BDF", rtol=2e-6, atol=2e-7, max_step=1.5e-3,
        events=event
    )

def simulate_reduced_phase_jump(p: Params, jump: float, t_end: float = 3.0):
    xeq = np.array([0., equilibrium_angle(p,True)])
    x0 = xeq.copy()
    x0[1] += jump
    return solve_ivp(
        lambda t,x: rhs_reduced(t,x,p), (0,t_end), x0,
        method="DOP853", rtol=1e-9, atol=1e-10, max_step=2e-3
    )

def is_stable_full(p: Params, jump: float, t_end: float = 3.0) -> bool:
    sol = simulate_full_phase_jump(p,jump,t_end)
    if sol.t[-1] < t_end-1e-7:
        return False
    xe = equilibrium_full(p,True)
    dwrap = (sol.y[1,-1]-xe[1]+np.pi)%(2*np.pi)-np.pi
    err = np.r_[sol.y[0,-1], dwrap, sol.y[2:,-1]-xe[2:]]
    speed = np.linalg.norm(rhs_full(sol.t[-1],sol.y[:,-1],p))
    return bool(np.linalg.norm(err)<0.06 and speed<2.0)

def is_stable_reduced(p: Params, jump: float, t_end: float = 5.0) -> bool:
    sol = simulate_reduced_phase_jump(p,jump,t_end)
    deq = equilibrium_angle(p,True)
    dwrap = (sol.y[1,-1]-deq+np.pi)%(2*np.pi)-np.pi
    return bool(np.hypot(sol.y[0,-1], dwrap)<1e-3)

def critical_phase_jump(p: Params, full=True, sign=1, iterations=10) -> float:
    lo, hi = 0.0, np.pi
    test = is_stable_full if full else is_stable_reduced
    for _ in range(iterations):
        mid = 0.5*(lo+hi)
        if test(p, sign*mid):
            lo = mid
        else:
            hi = mid
    return lo
