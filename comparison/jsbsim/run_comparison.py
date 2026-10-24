"""Generate the external JSBSim comparison report, numeric summary and figures."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import compare as C
from identify_aero import collect,fit,reconstruction_error
from jsb import JsbPlane
from models import build_environment,build_repo_model
from plots import mode_plane,time_history
RESULTS=Path(__file__).resolve().parent/"results"
DT=.01
def pipeline(no_alpha_dot:bool,with_cases:bool=True)->dict:
 J=JsbPlane(DT,no_alpha_dot=no_alpha_dot); jt=C.trim_jsbsim(J,C.V_TRIM,C.ALT)
 data=collect(J,C.V_TRIM,C.ALT,jt["alpha_rad"],(0.,jt["elevator_rad"],0.),n=400)
 mp=J.mass_properties(); coeffs,quality=fit(data,C.V_TRIM,mp["wing_span_m"],mp["mean_chord_m"])
 aircraft=build_repo_model(J,coeffs); env=build_environment(J,C.ALT)
 out={"mass_properties":{k:(v.tolist() if hasattr(v,"tolist") else v) for k,v in mp.items()},
  "environment":{"density_kg_m3":env.density_kg_m3,"gravity_down_m_s2":float(env.gravity_ned_m_s2[2])},
  "fitted_coefficients":coeffs,"fit_quality":quality,
  "reconstruction":reconstruction_error(coeffs,data,aircraft.parameters.geometry,quality["diagnostics"]["CD_abs_beta"])}
 l2=C.trim_and_modes(J,aircraft,env); out["level2"]={k:v for k,v in l2.items() if not k.startswith("_")}; out["_series"]={}
 if with_cases:
  out["level3"]={}
  for key,case in C.CASES.items():
   jc=C.run_jsbsim_case(J,l2["_trim_jsbsim"],case["du"],case["duration_s"],DT)
   rc=C.run_repo_case(aircraft,env,l2["_trim_repo"],case["du"],case["duration_s"],DT)
   out["level3"][key]={"label":case["label"],"errors":C.compare_case(key,jc,rc)}
   out["_series"][key]=(case["label"],jc,rc)
 return out
def _json_safe(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,(np.floating,np.integer)):return value.item()
 raise TypeError(f"not JSON serializable: {type(value).__name__}")
def main()->None:
 t0=time.time(); RESULTS.mkdir(parents=True,exist_ok=True)
 print("Level 1: equations of motion with JSBSim's own loads ..."); level1=C.eom_check(JsbPlane(DT),n=40)
 print("Levels 2 and 3: stock Cessna 172 aerodynamic database ..."); stock=pipeline(False)
 print("Levels 2 and 3: diagnostic copy with alpha-dot terms disabled ..."); noadot=pipeline(True)
 for key,(label,jc,rc) in stock["_series"].items():time_history(key,f"{label}: JSBSim vs this repo",jc,rc,RESULTS/f"time_{key}.png")
 mode_plane(stock["level2"],noadot["level2"],RESULTS/"modes_s_plane.png")
 summary={"condition":{"airspeed_m_s":C.V_TRIM,"altitude_m":C.ALT,"latitude_deg":0.,"heading":"north",
  "jsbsim_model":"c172x (symmetric payload, fixed external thrust, direct surface angles)"},
  "level1_equations_of_motion":level1,"stock":{k:v for k,v in stock.items() if not k.startswith("_")},
  "alpha_dot_terms_off":{k:v for k,v in noadot.items() if not k.startswith("_")}}
 (RESULTS/"summary.json").write_text(json.dumps(summary,indent=2,default=_json_safe)+"\n",encoding="utf-8")
 import report
 report.write(summary,RESULTS/"REPORT.md")
 print(f"done in {time.time()-t0:.0f} s -> {RESULTS}")
if __name__=="__main__":main()
