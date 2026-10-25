"""Three-level comparison with JSBSim: rigid-body EOM, trim/modes, and time response."""
from __future__ import annotations
import numpy as np
import models  # noqa: F401; ensures repo source tree is importable
from jsb import JsbPlane,trim_jsbsim
from models import OMEGA_EARTH,T_MAX_N,WGS84_A
from aircraft6dof import AircraftState,ControlInput,Simulator,TrimCondition,linearize_trim
from aircraft6dof.trim import trim
from aircraft6dof.equations import rigid_body_derivative
from aircraft6dof.linearize import euler_rates
from aircraft6dof.mathutils import dcm_body_to_ned_from_quat,quat_from_euler321
from aircraft6dof.state import VehicleGeometry

V_TRIM,ALT=55.,1000.
def _random_state(rng):
 x=np.array([55+rng.normal()*8,rng.normal()*4,rng.normal()*4,*(rng.normal(size=3)*.2),
             rng.uniform(-.8,.8),rng.uniform(-.4,.4),rng.uniform(-3,3)])
 return x,(rng.normal()*.05,rng.normal()*.05,rng.normal()*.05,200.)

def jsbsim_ecef_angular_acceleration(
    omega_ecef_body: np.ndarray,
    omega_ecef_relative_body: np.ndarray,
    moment_body: np.ndarray,
    inertia_body: np.ndarray,
) -> np.ndarray:
    """Return JSBSim's body-axis derivative of ECEF-relative PQR.

    JSBSim integrates PQR relative to ECEF, but applies Euler's rigid-body
    equation using the total inertial angular velocity. Its implementation
    (FGAccelerations::CalculatePQRdot) first computes the inertial angular
    acceleration and then adds the rotating-frame transport term.
    """
    omega_ecef_body = np.asarray(omega_ecef_body, dtype=float)
    omega_rel = np.asarray(omega_ecef_relative_body, dtype=float)
    moment_body = np.asarray(moment_body, dtype=float)
    inertia_body = np.asarray(inertia_body, dtype=float)
    omega_inertial_body = omega_rel + omega_ecef_body
    omega_inertial_dot_body = np.linalg.solve(
        inertia_body,
        moment_body - np.cross(omega_inertial_body, inertia_body @ omega_inertial_body),
    )
    return omega_inertial_dot_body + np.cross(omega_inertial_body, omega_ecef_body)


def eom_check(J:JsbPlane,n:int=40,seed:int=1)->dict:
 """Compare equations while explicitly reconciling ECEF and inertial frames."""
 rng=np.random.default_rng(seed); raw=[]; earth_corrected=[]; corrected=[]; kin_euler=[]; kin_ned=[]
 angular_raw=[]; angular_corrected=[]
 for _ in range(n):
  x,controls=_random_state(rng); J.set_state(x,ALT,controls); mp=J.mass_properties()
  geo=VehicleGeometry(mp["mass_kg"],mp["inertia_kg_m2"],mp["wing_area_m2"],mp["wing_span_m"],mp["mean_chord_m"])
  force,moment=J.applied_load(); q=quat_from_euler321(*x[6:9])
  s=AircraftState(np.array([0.,0.,-ALT]),x[:3],x[3:6],q); g=J.environment()["gravity_m_s2"]
  d=rigid_body_derivative(s,force,moment,np.array([0.,0.,g]),geo)
  ours=np.concatenate([d.velocity_body_m_s,d.omega_body_rad_s]); jd=J.derivatives()
  C=dcm_body_to_ned_from_quat(q)
  # At the equator, the ECEF angular velocity in the local NED frame is
  # [Omega, 0, 0]. Transform that rate to body axes before building PQR_i.
  omega_e_ned=OMEGA_EARTH*np.array([1.,0.,0.])
  omega_e_body=C.T@omega_e_ned
  r_ned=np.array([0.,0.,-(WGS84_A+ALT)]); v_ned=C@x[:3]
  rotation=C.T@(-np.cross(omega_e_ned,np.cross(omega_e_ned,r_ned))-2*np.cross(omega_e_ned,v_ned))
  earth_d=ours.copy(); earth_d[:3]+=rotation
  corrected_d=earth_d.copy()
  # JSBSim's PQR are ECEF-relative. Correct the repository's inertial-Euler
  # derivative using the full inertial angular rate and the transport term,
  # not just the translational rotating-Earth correction.
  corrected_d[3:]=jsbsim_ecef_angular_acceleration(
      omega_e_body, x[3:6], moment, geo.inertia_kg_m2
  )
  raw.append(np.abs(ours-jd[:6]))
  earth_corrected.append(np.abs(earth_d-jd[:6]))
  corrected.append(np.abs(corrected_d-jd[:6]))
  angular_raw.append(np.abs(ours[3:6]-jd[3:6]))
  angular_corrected.append(np.abs(corrected_d[3:6]-jd[3:6]))
  kin_euler.append(np.abs(euler_rates(x[6],x[7],*x[3:6])-jd[6:9]))
  kin_ned.append(np.abs(v_ned-J.ned_velocity()))
 mx=lambda a:np.max(a,axis=0).tolist()
 return {"samples":n,"channels_linear_m_s2":["udot","vdot","wdot"],"channels_angular_rad_s2":["pdot","qdot","rdot"],
   "max_abs_error_raw":mx(raw),
   "max_abs_error_after_earth_rotation":mx(earth_corrected),
   "max_abs_error_after_frame_corrections":mx(corrected),
   "max_abs_angular_error_raw_rad_s2":mx(angular_raw),
   "max_abs_angular_error_after_frame_correction_rad_s2":mx(angular_corrected),
   "max_abs_euler_rate_error_rad_s":mx(kin_euler),"max_abs_ned_velocity_error_m_s":mx(kin_ned)}

STATE8=("u","v","w","p","q","r","phi","theta"); OUR_IDX=[3,4,5,6,7,8,9,10]
INPUTS=("aileron","elevator","rudder","thrust")
def jsbsim_jacobian(J:JsbPlane,tr:dict)->tuple[np.ndarray,np.ndarray]:
 """Finite-difference the selected JSBSim state and control derivatives."""
 alpha=tr["alpha_rad"]; x0=np.array([V_TRIM*np.cos(alpha),0,V_TRIM*np.sin(alpha),0,0,0,0,alpha])
 u0=np.array([0.,tr["elevator_rad"],0.,tr["thrust_N"]])
 def f(x,u):
  J.set_state(np.append(x,0.),ALT,tuple(u)); return J.derivatives()[:8]
 A=np.empty((8,8))
 for j in range(8):
  h=1e-3 if j<3 else 1e-5; delta=np.zeros(8); delta[j]=h
  A[:,j]=(f(x0+delta,u0)-f(x0-delta,u0))/(2*h)
 B=np.empty((8,4))
 for j in range(4):
  h=1e-4 if j<3 else .5; delta=np.zeros(4); delta[j]=h
  B[:,j]=(f(x0,u0+delta)-f(x0,u0-delta))/(2*h)
 return A,B

def mode_table(A8:np.ndarray)->dict:
 from aircraft6dof.linearize import LinearModel
 from aircraft6dof.modes import lateral_modes,longitudinal_modes
 lm=LinearModel(A=A8,B=np.zeros((8,1)),x0=np.zeros(8),u0=np.zeros(1),xdot0=np.zeros(8),
                state_names=STATE8,input_names=("dummy",))
 return {**longitudinal_modes(lm.subsystem(("u","w","q","theta"),("dummy",))),
         **lateral_modes(lm.subsystem(("v","p","r","phi"),("dummy",)))}
def mode_dict(modes):
 return {k:{"re":m.eigenvalue.real,"im":m.eigenvalue.imag,"wn_rad_s":m.wn_rad_s,"zeta":m.zeta,
  "period_s":m.period_s,"time_constant_s":m.time_constant_s,"time_to_double_s":m.time_to_double_s} for k,m in modes.items()}

def trim_and_modes(J:JsbPlane,aircraft,env)->dict:
 jt=trim_jsbsim(J,V_TRIM,ALT); rt=trim(aircraft,env,TrimCondition(V_TRIM,ALT)); lin=linearize_trim(aircraft,env,rt)
 Aj,Bj=jsbsim_jacobian(J,jt); Ar=lin.A[np.ix_(OUR_IDX,OUR_IDX)]; Br=lin.B[OUR_IDX][:,[0,1,2,3]].copy(); Br[:,3]/=T_MAX_N
 mj,mr=mode_dict(mode_table(Aj)),mode_dict(mode_table(Ar))
 return {"trim":{"jsbsim":{"alpha_deg":np.rad2deg(jt["alpha_rad"]),"elevator_deg":np.rad2deg(jt["elevator_rad"]),
  "thrust_N":jt["thrust_N"],"residual":jt["residual"]},
  "repo":{"alpha_deg":np.rad2deg(rt.alpha_rad),"elevator_deg":np.rad2deg(rt.controls.elevator),
  "thrust_N":rt.controls.throttle*T_MAX_N,"residual":rt.residual_norm}},
  "A_jsbsim":Aj.tolist(),"A_repo":Ar.tolist(),"B_jsbsim":Bj.tolist(),"B_repo":Br.tolist(),
  "state_names":STATE8,"input_names":INPUTS,"modes_jsbsim":mj,"modes_repo":mr,
  "_trim_jsbsim":jt,"_trim_repo":rt}

def doublet(t,t0,width,amp):
 return amp if t0<=t<t0+width else (-amp if t0+width<=t<t0+2*width else 0.)
def pulse(t,t0,width,amp): return amp if t0<=t<t0+width else 0.
CASES={
 "elevator_doublet":{"label":"Elevator doublet (+/-1 deg, 1 s each), 40 s","duration_s":40.,
  "du":lambda t:np.array([0.,doublet(t,1.,1.,np.deg2rad(1.)),0.])},
 "aileron_pulse":{"label":"Aileron pulse (+1 deg, 1.5 s) with rudder held, 20 s","duration_s":20.,
  "du":lambda t:np.array([pulse(t,1.,1.5,np.deg2rad(1.)),0.,0.])},
 "rudder_pulse":{"label":"Rudder pulse (+1 deg, 1 s), 20 s","duration_s":20.,
  "du":lambda t:np.array([0.,0.,pulse(t,1.,1.,np.deg2rad(1.))])}}
SIGNALS=("u","v","w","p","q","r","phi","theta","psi","alpha","beta","altitude")
def _derived(t,x,altitude):
 speed=np.sqrt(x[:,0]**2+x[:,1]**2+x[:,2]**2)
 out={k:x[:,i] for i,k in enumerate(("u","v","w","p","q","r","phi","theta","psi"))}
 out["psi"]=np.unwrap(x[:,8]); out["alpha"]=np.arctan2(x[:,2],x[:,0])
 out["beta"]=np.arcsin(np.clip(x[:,1]/speed,-1,1)); out["altitude"]=altitude; out["t"]=t
 return out
def run_jsbsim_case(J:JsbPlane,tr:dict,du,duration_s:float,dt:float)->dict:
 a=tr["alpha_rad"]; x0=np.array([V_TRIM*np.cos(a),0,V_TRIM*np.sin(a),0,0,0,0,a,0.])
 base=np.array([0.,tr["elevator_rad"],0.]); J.set_state(x0,ALT,(*base,tr["thrust_N"]))
 n=int(round(duration_s/dt)); X=np.empty((n+1,9)); H=np.empty(n+1); X[0],H[0]=J.state(),J.altitude_m()
 for i in range(n):
  J.set_controls(*(base+du(i*dt)),tr["thrust_N"]); J.step(); X[i+1],H[i+1]=J.state(),J.altitude_m()
 return _derived(np.linspace(0,n*dt,n+1),X,H)
def run_repo_case(aircraft,env,repo_trim,du,duration_s:float,dt:float)->dict:
 base=np.array([repo_trim.controls.aileron,repo_trim.controls.elevator,repo_trim.controls.rudder])
 def controls(t):
  c=base+du(t); return ControlInput(c[0],c[1],c[2],repo_trim.controls.throttle)
 h=Simulator(aircraft).run(repo_trim.state,controls,lambda _:env,duration_s=duration_s,dt_s=dt)
 x=np.column_stack([h.state[:,3:9],h.euler_rad])
 return _derived(h.time_s,x,-h.state[:,2])
PRIMARY={"elevator_doublet":("u","alpha","q","theta","altitude"),
 "aileron_pulse":("p","phi","r","beta","v"),"rudder_pulse":("r","beta","phi","p","v")}
def compare_case(key:str,jc:dict,rc:dict)->dict:
 primary,secondary={},{}
 for sig in SIGNALS:
  pj=jc[sig]-jc[sig][0]; pr=rc[sig]-rc[sig][0]; peak=float(np.max(np.abs(pj))); diff=pj-pr
  row={"jsbsim_peak_change":peak,"max_abs_diff":float(np.max(np.abs(diff))),
       "rms_diff":float(np.sqrt(np.mean(diff**2)))}
  if sig in PRIMARY[key]:
   if peak<=0: raise ValueError(f"JSBSim reference signal {sig!r} has zero peak change")
   row["max_diff_pct_of_peak"]=100*row["max_abs_diff"]/peak
   row["rms_diff_pct_of_peak"]=100*row["rms_diff"]/peak; primary[sig]=row
  else: secondary[sig]=row
 return {"primary":primary,"secondary":secondary}
