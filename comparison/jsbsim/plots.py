"""Engineering plots generated from comparison time histories and modes."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
SURFACE,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e4e3df"
C_JSB,C_REPO="#2a78d6","#eb6834"
UNITS={"u":("u",1.,"m/s"),"v":("v",1.,"m/s"),"w":("w",1.,"m/s"),
 "p":("roll rate p",np.rad2deg(1),"deg/s"),"q":("pitch rate q",np.rad2deg(1),"deg/s"),
 "r":("yaw rate r",np.rad2deg(1),"deg/s"),"phi":("bank angle",np.rad2deg(1),"deg"),
 "theta":("pitch angle",np.rad2deg(1),"deg"),"psi":("heading",np.rad2deg(1),"deg"),
 "alpha":("angle of attack",np.rad2deg(1),"deg"),"beta":("sideslip",np.rad2deg(1),"deg"),
 "altitude":("altitude",1.,"m")}
CASE_SIGNALS={"elevator_doublet":["u","alpha","q","altitude"],"aileron_pulse":["p","phi","r","beta"],
 "rudder_pulse":["r","beta","phi","psi"]}
def _style(ax):
 ax.set_facecolor(SURFACE); ax.grid(True,color=GRID,lw=.7)
 for s in ("top","right"): ax.spines[s].set_visible(False)
 for s in ("left","bottom"): ax.spines[s].set_color(INK2)
 ax.tick_params(colors=INK2,labelsize=8)
def time_history(case_key:str,title:str,jc:dict,rc:dict,path:Path)->None:
 sigs=CASE_SIGNALS[case_key]; fig,axes=plt.subplots(2,len(sigs),figsize=(4.2*len(sigs),6.2),
  facecolor=SURFACE,gridspec_kw={"height_ratios":[2.2,1]})
 for k,name in enumerate(sigs):
  label,scale,unit=UNITS[name]; yj=(jc[name]-jc[name][0])*scale; yr=(rc[name]-rc[name][0])*scale
  a,b=axes[0,k],axes[1,k]; a.plot(jc["t"],yj,color=C_JSB,lw=1.8,label="JSBSim")
  a.plot(rc["t"],yr,color=C_REPO,lw=1.6,ls=(0,(5,2)),label="this repo")
  a.set_title(f"{label} (change from trim)",fontsize=9,color=INK,loc="left"); a.set_ylabel(unit,fontsize=8,color=INK2)
  b.plot(rc["t"],yr-yj,color=INK2,lw=1.2); b.axhline(0,color=GRID,lw=1)
  b.set_ylabel(f"repo - JSBSim [{unit}]",fontsize=8,color=INK2); b.set_xlabel("time [s]",fontsize=8,color=INK2)
  _style(a); _style(b)
  if k==0:a.legend(frameon=False,fontsize=8,loc="best")
 fig.suptitle(title,x=.01,ha="left",fontsize=11,color=INK); fig.tight_layout(rect=(0,0,1,.96))
 fig.savefig(path,dpi=150,facecolor=SURFACE); plt.close(fig)
def mode_plane(stock:dict,noadot:dict,path:Path)->None:
 panels=[("Fast modes",("short_period","dutch_roll","roll_subsidence")),("Phugoid",("phugoid",)),("Spiral",("spiral",))]
 fig,axes=plt.subplots(1,3,figsize=(14,4.6),facecolor=SURFACE)
 for ax,(title,keys) in zip(axes,panels):
  for src,marker,size,label,color in (("jsbsim","o",70,"JSBSim",C_JSB),("repo","x",60,"this repo",C_REPO)):
   modes=stock[f"modes_{src}"]; pts=[(modes[k]["re"],modes[k]["im"]) for k in keys]
   ax.scatter([p[0] for p in pts]*2,[p[1] for p in pts]+[-p[1] for p in pts],s=size,marker=marker,
    color=color,facecolors="none" if marker=="o" else None,lw=1.8,label=label,zorder=3)
  modes=noadot["modes_jsbsim"]; pts=[(modes[k]["re"],modes[k]["im"]) for k in keys]
  ax.scatter([p[0] for p in pts]*2,[p[1] for p in pts]+[-p[1] for p in pts],s=26,marker="s",color=INK2,label="JSBSim, alpha-dot off",zorder=2)
  ax.axvline(0,color=GRID,lw=1); ax.axhline(0,color=GRID,lw=1); ax.set_title(title,fontsize=10,color=INK,loc="left")
  ax.set_xlabel("real part [1/s]",fontsize=8,color=INK2); ax.set_ylabel("imaginary part [rad/s]",fontsize=8,color=INK2); _style(ax)
 axes[0].legend(frameon=False,fontsize=8,loc="best")
 fig.suptitle("Eigenvalues of linearized models (Cessna 172, 55 m/s, 1000 m)",x=.01,ha="left",fontsize=11,color=INK)
 fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(path,dpi=150,facecolor=SURFACE); plt.close(fig)
