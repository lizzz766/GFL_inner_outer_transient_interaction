
from pathlib import Path
import sys, json, csv
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from gfl_model import *
from continuation import trace_internal_branch_through_snpo

OUT=ROOT/"outputs"
OUT.mkdir(exist_ok=True)
p0=Params()

# 1. Phase-jump mismatch, Fig. 3-type result.
p_mis=with_kp_pu(p0,0.4)
sf=simulate_full_phase_jump(p_mis,np.pi,t_end=1.2)
sr=simulate_reduced_phase_jump(p_mis,np.pi,t_end=1.2)
deq=equilibrium_angle(p_mis)
plt.figure(figsize=(7.5,4.5))
plt.plot(sr.t,sr.y[1]-deq,label="second-order, ideal current loop")
plt.plot(sf.t,sf.y[1]-deq,label="12th-order, nonideal current loop")
plt.axhline(2*np.pi,linewidth=0.8)
plt.xlabel("Time (s)")
plt.ylabel(r"$\delta_L-\delta_{\rm SEP}$ (rad)")
plt.title(r"Phase jump $\Delta\theta_s=-\pi$: reduced model stable, full model diverges")
plt.legend()
plt.tight_layout()
plt.savefig(OUT/"phase_jump_mismatch.png",dpi=180)
plt.close()

plt.figure(figsize=(7.5,4.5))
plt.plot(sf.t,sf.y[6],label=r"$i_{gd}$")
plt.axhline(p0.igd_ref,linewidth=0.8,label="reference")
plt.xlabel("Time (s)")
plt.ylabel("Grid-side d-axis current (A)")
plt.title("Current transient omitted by the second-order model")
plt.legend()
plt.tight_layout()
plt.savefig(OUT/"phase_jump_current_transient.png",dpi=180)
plt.close()

# 2. Local Hopf boundary.
kp_grid=np.linspace(0.10,0.45,141)
dom=np.array([dominant_real_full(with_kp_pu(p0,q)) for q in kp_grid])
kh=find_hopf_kp_pu(p0)
plt.figure(figsize=(7.5,4.5))
plt.plot(kp_grid,dom,label="dominant real part")
plt.axhline(0,linewidth=0.8)
plt.axvline(kh,linewidth=0.8,label=f"Hopf at {kh:.4f} p.u.")
plt.xlabel(r"PLL proportional gain $\kappa_p$ (p.u.)")
plt.ylabel(r"$\max\operatorname{Re}\lambda$ (s$^{-1}$)")
plt.title("Full-order equilibrium loses stability through Hopf bifurcation")
plt.legend()
plt.tight_layout()
plt.savefig(OUT/"hopf_boundary.png",dpi=180)
plt.close()

# 3. Critical positive phase jump, Fig. 9-type result.
kp_crit=np.array([0.24,0.28,0.32,0.36,0.40,0.46,0.52])
full_crit=[]
red_crit=[]
for q in kp_crit:
    pp=with_kp_pu(p0,float(q))
    full_crit.append(critical_phase_jump(pp,full=True,sign=1,iterations=9))
    red_crit.append(critical_phase_jump(pp,full=False,sign=1,iterations=11))
full_crit=np.array(full_crit)
red_crit=np.array(red_crit)
with open(OUT/"critical_phase_jump.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["kp_pu","full_order_rad","second_order_rad"])
    w.writerows(zip(kp_crit,full_crit,red_crit))
plt.figure(figsize=(7.5,4.5))
plt.plot(kp_crit,full_crit,marker="o",label="12th-order model")
plt.plot(kp_crit,red_crit,marker="s",label="second-order model")
plt.xlabel(r"PLL proportional gain $\kappa_p$ (p.u.)")
plt.ylabel("Critical positive phase jump (rad)")
plt.title("The ideal-current model gives an optimistic transient margin")
plt.legend()
plt.tight_layout()
plt.savefig(OUT/"critical_phase_jump.png",dpi=180)
plt.close()

# 4. Periodic-orbit continuation and SNPO.
kh,xe,branch=trace_internal_branch_through_snpo(p0)
period=np.array([r.p[0] for r in branch])
kpp=np.array([r.p[1] for r in branch])
isn=int(np.argmax(kpp))
np.savetxt(
    OUT/"upo_branch.csv",
    np.c_[kpp,period],
    delimiter=",",header="kp_pu,period_s",comments=""
)
plt.figure(figsize=(7.5,4.5))
plt.plot(kpp,period,label="continued center UPO branch")
plt.scatter([kpp[isn]],[period[isn]],label=f"SNPO ≈ {kpp[isn]:.4f} p.u.")
plt.scatter([kh],[2*np.pi/39.0],label=f"subcritical Hopf ≈ {kh:.4f} p.u.")
plt.xlabel(r"PLL proportional gain $\kappa_p$ (p.u.)")
plt.ylabel("Periodic-orbit period (s)")
plt.title("Numerical continuation reproduces the internal UPO and its fold")
plt.legend()
plt.tight_layout()
plt.savefig(OUT/"upo_branch_snpo.png",dpi=180)
plt.close()

orbit=branch[isn].sol(np.linspace(0,1,500))[:12]
plt.figure(figsize=(6.5,5.0))
plt.plot(orbit[1],orbit[0])
plt.scatter([xe[1]],[xe[0]],label="SEP")
plt.xlabel(r"$\delta_L$ (rad)")
plt.ylabel(r"$y_\omega$ (V·s)")
plt.title("Projection of the unstable periodic orbit near the SNPO")
plt.legend()
plt.tight_layout()
plt.savefig(OUT/"upo_projection_near_snpo.png",dpi=180)
plt.close()

# 5. Fast-slow diagnostics: Schur reduction and current-bandwidth sweep.
fcls=np.array([100.,126.29,150.,200.,244.52,300.,400.,700.,900.,1000.])
rows=[]
for fcl in fcls:
    pp=with_current_bandwidth(with_kp_pu(p0,0.4),fcl)
    xe=equilibrium_full(pp)
    J=jacobian_fd(lambda z: rhs_full(0,z,pp),xe)
    slow=[0,1]; fast=list(range(2,12))
    Jss=J[np.ix_(slow,slow)]
    Jsf=J[np.ix_(slow,fast)]
    Jfs=J[np.ix_(fast,slow)]
    Jff=J[np.ix_(fast,fast)]
    Jsch=Jss-Jsf@np.linalg.solve(Jff,Jfs)
    fullmax=np.max(np.linalg.eigvals(J).real)
    fastmax=np.max(np.linalg.eigvals(Jff).real)
    schmax=np.max(np.linalg.eigvals(Jsch).real)
    rows.append([fcl,fullmax,fastmax,schmax])
rows=np.asarray(rows)
np.savetxt(
    OUT/"fast_slow_spectrum.csv",rows,delimiter=",",
    header="fcl_hz,full_max_real,fast_block_max_real,schur_slow_max_real",comments=""
)
plt.figure(figsize=(7.5,4.5))
plt.plot(rows[:,0],rows[:,1],marker="o",label="full system")
plt.plot(rows[:,0],rows[:,2],marker="s",label="frozen fast block")
plt.plot(rows[:,0],rows[:,3],marker="^",label="Schur-complement slow model")
plt.axhline(0,linewidth=0.8)
plt.xlabel("Current-loop bandwidth setting (Hz)")
plt.ylabel(r"Dominant real part (s$^{-1}$)")
plt.title("Fastness is not monotonic: high gain meets delay/LCL dynamics")
plt.legend()
plt.tight_layout()
plt.savefig(OUT/"fast_slow_spectrum.png",dpi=180)
plt.close()

summary={
    "paper_default_hopf_kp_pu":float(kh),
    "continued_snpo_kp_pu":float(kpp[isn]),
    "continued_snpo_period_s":float(period[isn]),
    "phase_jump_case":{
        "kp_pu":0.4,
        "jump_rad":float(np.pi),
        "full_termination_time_s":float(sf.t[-1]),
        "full_terminal_delta_rad":float(sf.y[1,-1]),
        "reduced_terminal_delta_rad":float(sr.y[1,-1])
    },
    "critical_jump_at_kp_0p4_rad":float(full_crit[np.where(kp_crit==0.40)[0][0]]),
    "notes":[
        "The 12th-order averaged model uses equations (1)-(14) from the paper.",
        "Current-loop tuning scales beta_p and beta_i together, retaining beta_i/beta_p.",
        "The UPO branch is obtained with collocation plus pseudo-arclength continuation implemented in SciPy.",
        "Continuation reproduces the subcritical Hopf and the SNPO. It does not attempt the full Shilnikov wiggle structure close to the homoclinic limit."
    ]
}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
