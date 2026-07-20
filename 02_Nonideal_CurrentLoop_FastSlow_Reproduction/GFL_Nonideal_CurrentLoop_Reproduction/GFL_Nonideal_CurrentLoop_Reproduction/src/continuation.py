
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_bvp, trapezoid
from scipy.interpolate import CubicSpline
from scipy.linalg import eig
from gfl_model import (
    Params, with_kp_pu, equilibrium_full, rhs_full,
    jacobian_fd, find_hopf_kp_pu, rhs_full_batch
)

STATE_SCALE = np.array([1,1,100,100,400,400,100,100,10,10,400,400.])[:,None]
PARAM_SCALE = np.array([0.2,0.3])

def hopf_eigenvector(p: Params, kh: float):
    ph = with_kp_pu(p,kh)
    xe = equilibrium_full(ph)
    J = jacobian_fd(lambda z: rhs_full(0,z,ph), xe)
    vals, vecs = eig(J)
    idx = np.argmin(np.abs(vals.real)+np.abs(np.abs(vals.imag)-39.0))
    lam, v = vals[idx], vecs[:,idx]
    v = v*np.exp(-1j*np.angle(v[1]))
    scale = v.real[1]
    vr, vi = v.real/scale, v.imag/scale
    return xe, lam, vr, vi

def initial_periodic_orbit(p: Params, amplitude=0.01):
    kh = find_hopf_kp_pu(p)
    xe, lam, vr, vi = hopf_eigenvector(p,kh)
    vi_norm = vi/np.linalg.norm(vi)
    tau = np.linspace(0,1,101)
    Y = xe[:,None] + amplitude*(
        vr[:,None]*np.cos(2*np.pi*tau)[None,:]
        - vi[:,None]*np.sin(2*np.pi*tau)[None,:]
    )
    pars0 = np.array([2*np.pi/abs(lam.imag),kh+1e-3])
    def fun(t,Y,pars):
        T,kpp = pars
        pp = with_kp_pu(p,kpp)
        return T*rhs_full_batch(Y,pp)
    def bc(ya,yb,pars):
        return np.r_[
            ya-yb,
            np.dot(ya-xe,vi_norm),
            ya[1]-xe[1]-amplitude
        ]
    res = solve_bvp(fun,bc,tau,Y,p=pars0,tol=2e-5,max_nodes=4000)
    if res.status != 0:
        raise RuntimeError("Initial periodic-orbit solve failed: "+res.message)
    return kh, xe, vi_norm, res

def amplitude_continue(p, xe, vi_norm, res0, amplitudes):
    def fun(t,Y,pars):
        T,kpp = pars
        pp = with_kp_pu(p,kpp)
        return T*rhs_full_batch(Y,pp)
    results=[res0]
    for amp in amplitudes:
        mesh=np.linspace(0,1,161)
        Yg=results[-1].sol(mesh)[:12]
        pg=results[-1].p.copy()
        def bc(ya,yb,pars,amp=amp):
            return np.r_[
                ya-yb,
                np.dot(ya-xe,vi_norm),
                ya[1]-xe[1]-amp
            ]
        rr=solve_bvp(fun,bc,mesh,Yg,p=pg,tol=5e-5,max_nodes=6000)
        if rr.status != 0:
            break
        results.append(rr)
    return results

def pseudo_arclength_step(p, xe, vi_norm, r0, r1, ds=0.07):
    mesh=np.linspace(0,1,141)
    u0,u1=r0.sol(mesh)[:12],r1.sol(mesh)[:12]
    p0,p1=r0.p.copy(),r1.p.copy()
    dun=(u1-u0)/STATE_SCALE
    dpn=(p1-p0)/PARAM_SCALE
    nrm=np.sqrt(trapezoid(np.sum(dun*dun,axis=0),mesh)+dpn@dpn)
    tun,tpn=dun/nrm,dpn/nrm
    u1spl=CubicSpline(mesh,u1,axis=1)
    tspl=CubicSpline(mesh,tun,axis=1)

    def fun(t,Y,pars):
        T,kpp=pars
        pp=with_kp_pu(p,kpp)
        U=Y[:12]
        out=np.empty_like(Y)
        for j in range(U.shape[1]):
            out[:12,j]=T*rhs_full(0,U[:,j],pp)
        out[12]=np.sum(((U-u1spl(t))/STATE_SCALE)*tspl(t),axis=0)
        return out
    def bc(ya,yb,pars):
        return np.r_[
            ya[:12]-yb[:12],
            np.dot(ya[:12]-xe,vi_norm),
            ya[12],
            yb[12]+np.dot((pars-p1)/PARAM_SCALE,tpn)-ds
        ]
    Yg=np.vstack([u1+ds*(STATE_SCALE*tun),np.zeros(mesh.size)])
    pg=p1+ds*PARAM_SCALE*tpn
    return solve_bvp(
        fun,bc,mesh,Yg,p=pg,tol=1.2e-4,max_nodes=8000
    )

def trace_internal_branch_through_snpo(p: Params):
    kh,xe,vi_norm,rinit=initial_periodic_orbit(p,0.01)
    amps=np.r_[np.linspace(.02,.20,10),np.linspace(.23,.35,13),[.352,.354,.356,.358,.360]]
    branch=amplitude_continue(p,xe,vi_norm,rinit,amps)
    if len(branch)<4:
        raise RuntimeError("Amplitude continuation failed too early.")
    # Pseudo-arclength continuation; enough to pass the SNPO near kp=0.292.
    for _ in range(105):
        rr=pseudo_arclength_step(p,xe,vi_norm,branch[-2],branch[-1],ds=0.095)
        if rr.status != 0:
            break
        branch.append(rr)
        kps=np.array([r.p[1] for r in branch])
        # Stop after clearly crossing the fold.
        if kps.max()>0.291 and len(kps)>np.argmax(kps)+12:
            break
    return kh,xe,branch
