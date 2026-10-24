"""Thin SI-unit wrapper around a JSBSim executive; API values are SI."""
from __future__ import annotations
import numpy as np
from make_model import new_fdm
FT=0.3048; SLUG=14.593902937206; LBF=4.4482216152605; SLUGFT2=1.3558179483314; LBFFT=LBF*FT
STATE_KEYS=("u","v","w","p","q","r","phi","theta","psi")

class JsbPlane:
    """JSBSim C172 with direct surface angles and a fixed-thrust force."""
    def __init__(self,dt_s:float=.01,no_alpha_dot:bool=False):
        self.dt=dt_s; self.fdm=new_fdm(dt_s,no_alpha_dot); self.fdm.disable_output()
        self._get=self.fdm.get_property_value; self._set=self.fdm.set_property_value
    def set_controls(self,aileron:float,elevator:float,rudder:float,thrust_N:float)->None:
        self._set("fcs/aileron-cmd-norm",aileron); self._set("fcs/elevator-cmd-norm",elevator)
        self._set("fcs/rudder-cmd-norm",rudder); self._set("external_reactions/thrust/magnitude",thrust_N/LBF)
    def set_state(self,x:np.ndarray,altitude_m:float,controls:tuple[float,float,float,float])->None:
        u,v,w,p,q,r,phi,theta,psi=x; s=self._set
        s("ic/h-sl-ft",altitude_m/FT); s("ic/u-fps",u/FT); s("ic/v-fps",v/FT); s("ic/w-fps",w/FT)
        s("ic/p-rad_sec",p); s("ic/q-rad_sec",q); s("ic/r-rad_sec",r)
        s("ic/phi-rad",phi); s("ic/theta-rad",theta); s("ic/psi-true-rad",psi)
        self.set_controls(*controls)
        if not self.fdm.run_ic(): raise RuntimeError("JSBSim run_ic failed")
        # run_ic may reset FCS properties; reapply direct commands and refresh IC.
        self.set_controls(*controls)
        if not self.fdm.run_ic(): raise RuntimeError("JSBSim run_ic failed after control initialization")
    def state(self)->np.ndarray:
        g=self._get
        return np.array([g("velocities/u-fps")*FT,g("velocities/v-fps")*FT,g("velocities/w-fps")*FT,
            g("velocities/p-rad_sec"),g("velocities/q-rad_sec"),g("velocities/r-rad_sec"),
            g("attitude/phi-rad"),g("attitude/theta-rad"),g("attitude/psi-rad")])
    def derivatives(self)->np.ndarray:
        """[udot vdot wdot pdot qdot rdot phidot thetadot psidot] in SI."""
        g=self._get
        return np.array([g("accelerations/udot-ft_sec2")*FT,g("accelerations/vdot-ft_sec2")*FT,
            g("accelerations/wdot-ft_sec2")*FT,g("accelerations/pdot-rad_sec2"),g("accelerations/qdot-rad_sec2"),
            g("accelerations/rdot-rad_sec2"),g("velocities/phidot-rad_sec"),g("velocities/thetadot-rad_sec"),
            g("velocities/psidot-rad_sec")])
    def altitude_m(self)->float: return self._get("position/h-sl-ft")*FT
    def alpha_beta(self)->tuple[float,float]: return self._get("aero/alpha-rad"),self._get("aero/beta-rad")
    def surfaces(self)->np.ndarray:
        g=self._get
        return np.array([g("fcs/effective-aileron-pos"),g("fcs/elevator-pos-rad"),g("fcs/rudder-pos-rad")])
    def ned_velocity(self)->np.ndarray:
        g=self._get
        return np.array([g("velocities/v-north-fps"),g("velocities/v-east-fps"),g("velocities/v-down-fps")])*FT
    def applied_load(self)->tuple[np.ndarray,np.ndarray]:
        """Non-gravity body-axis force and CG moment: aerodynamics plus external thrust."""
        g=self._get
        f=np.array([g("forces/fbx-total-lbs"),g("forces/fby-total-lbs"),g("forces/fbz-total-lbs")])*LBF
        m=np.array([g("moments/l-total-lbsft"),g("moments/m-total-lbsft"),g("moments/n-total-lbsft")])*LBFFT
        return f,m
    def aero_load_body(self)->tuple[np.ndarray,np.ndarray]:
        g=self._get
        return (np.array([g("forces/fbx-aero-lbs"),g("forces/fby-aero-lbs"),g("forces/fbz-aero-lbs")])*LBF,
                np.array([g("moments/l-aero-lbsft"),g("moments/m-aero-lbsft"),g("moments/n-aero-lbsft")])*LBFFT)
    def aero_coefficient_inputs(self)->dict:
        g=self._get; qS=g("aero/qbar-psf")*g("metrics/Sw-sqft")
        if qS<=0: raise RuntimeError("JSBSim dynamic pressure-area product must be positive")
        b,c=g("metrics/bw-ft"),g("metrics/cbarw-ft")
        return {"CL":g("forces/fwz-aero-lbs")/qS,"CD":g("forces/fwx-aero-lbs")/qS,"CY":g("forces/fwy-aero-lbs")/qS,
                "Cl":g("moments/l-aero-lbsft")/(qS*b),"Cm":g("moments/m-aero-lbsft")/(qS*c),
                "Cn":g("moments/n-aero-lbsft")/(qS*b)}
    def mass_properties(self)->dict:
        g=self._get; ixx,iyy,izz=(g(f"inertia/{a}-slugs_ft2") for a in ("ixx","iyy","izz"))
        ixy,ixz,iyz=(g(f"inertia/{a}-slugs_ft2") for a in ("ixy","ixz","iyz"))
        I=np.array([[ixx,ixy,ixz],[ixy,iyy,iyz],[ixz,iyz,izz]])*SLUGFT2
        return {"mass_kg":g("inertia/mass-slugs")*SLUG,"inertia_kg_m2":I,
            "wing_area_m2":g("metrics/Sw-sqft")*FT**2,"wing_span_m":g("metrics/bw-ft")*FT,
            "mean_chord_m":g("metrics/cbarw-ft")*FT}
    def environment(self)->dict:
        g=self._get; weight=np.array([g("forces/fbx-weight-lbs"),g("forces/fby-weight-lbs"),g("forces/fbz-weight-lbs")])*LBF
        mass=g("inertia/mass-slugs")*SLUG
        return {"density_kg_m3":g("atmosphere/rho-slugs_ft3")*SLUG/FT**3,
                "speed_of_sound_m_s":g("atmosphere/a-fps")*FT,"gravity_m_s2":float(np.linalg.norm(weight)/mass)}
    def step(self)->None:
        if not self.fdm.run(): raise RuntimeError("JSBSim run() failed")

def trim_jsbsim(J:JsbPlane,V:float,altitude_m:float,*,tol:float=1e-8)->dict:
    """Newton trim on JSBSim's own longitudinal accelerations."""
    def residual(z):
        alpha,de,thrust=z; x=np.array([V*np.cos(alpha),0.,V*np.sin(alpha),0,0,0,0.,alpha,0.])
        J.set_state(x,altitude_m,(0.,de,0.,thrust)); d=J.derivatives()
        return np.array([d[0],d[2],d[4]])
    z=np.array([.04,-.02,400.])
    for it in range(40):
        r=residual(z)
        if np.max(np.abs(r))<tol: break
        steps=np.array([1e-6,1e-6,1e-3]); eye=np.eye(3)
        jac=np.column_stack([(residual(z+eye[k]*steps[k])-residual(z-eye[k]*steps[k]))/(2*steps[k]) for k in range(3)])
        z=z-np.linalg.solve(jac,r)
        if not np.all(np.isfinite(z)): raise RuntimeError("JSBSim trim iterate became non-finite")
    else: raise RuntimeError("JSBSim trim did not converge")
    err=float(np.max(np.abs(residual(z))))
    return {"alpha_rad":float(z[0]),"elevator_rad":float(z[1]),"thrust_N":float(z[2]),"iterations":it,"residual":err}
