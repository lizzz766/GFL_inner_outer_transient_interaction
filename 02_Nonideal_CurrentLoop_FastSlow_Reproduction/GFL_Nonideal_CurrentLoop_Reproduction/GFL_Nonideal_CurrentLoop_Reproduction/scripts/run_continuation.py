
from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from gfl_model import *
from continuation import trace_internal_branch_through_snpo

OUT=ROOT/"outputs"
p0=Params()

kh,xe,branch=trace_internal_branch_through_snpo(p0)
period=np.array([r.p[0] for r in branch])
kpp=np.array([r.p[1] for r in branch])
isn=int(np.argmax(kpp))
np.savetxt(
    OUT/"upo_branch.csv",np.c_[kpp,period],delimiter=",",
    header="kp_pu,period_s",comments=""
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

fcls=np.array([100.,126.29,150.,200.,244.52,300.,400.,700.,900.,1000.])
rows=[]
for fcl in fcls:
    pp=with_current_bandwidth(with_kp_pu(p0,0.4),fcl)
    xe2=equilibrium_full(pp)
    J=jacobian_fd(lambda z: rhs_full(0,z,pp),xe2)
    slow=[0,1]; fast=list(range(2,12))
    Jss=J[np.ix_(slow,slow)]
    Jsf=J[np.ix_(slow,fast)]
    Jfs=J[np.ix_(fast,slow)]
    Jff=J[np.ix_(fast,fast)]
    Jsch=Jss-Jsf@np.linalg.solve(Jff,Jfs)
    rows.append([
        fcl,
        np.max(np.linalg.eigvals(J).real),
        np.max(np.linalg.eigvals(Jff).real),
        np.max(np.linalg.eigvals(Jsch).real)
    ])
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

summary_path=OUT/"summary.json"
old=json.loads(summary_path.read_text()) if summary_path.exists() else {}
old.update({
    "paper_default_hopf_kp_pu":float(kh),
    "continued_snpo_kp_pu":float(kpp[isn]),
    "continued_snpo_period_s":float(period[isn]),
    "number_of_continued_periodic_orbits":int(len(branch))
})
summary_path.write_text(json.dumps(old,indent=2),encoding="utf-8")
print(json.dumps({
    "hopf_kp_pu":kh,
    "snpo_kp_pu":kpp[isn],
    "snpo_period_s":period[isn],
    "branch_points":len(branch)
},indent=2))
