"""Engineering data export and visualization for simulation histories."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aero import aerodynamic_loads
from .atmosphere import standard_atmosphere
from .mathutils import dcm_body_to_ned_from_quat


def _series(history, controls, environment, aircraft):
    t, X = history.time_s, history.state
    geom, aero, propulsion = aircraft.parameters.geometry, aircraft.parameters.aero, aircraft.parameters.propulsion
    rows = []
    for i, (ti, x) in enumerate(zip(t, X)):
        pos, vb, om, quat = x[0:3], x[3:6], x[6:9], x[9:13]
        C = dcm_body_to_ned_from_quat(quat); vn = C @ vb
        env = environment(float(ti)) if callable(environment) else environment
        u = controls(float(ti)) if callable(controls) else controls
        wind, gust = np.asarray(env.wind_ned_m_s), np.asarray(env.gust_ned_m_s)
        vrel_n = vn - wind - gust; vrel_b = C.T @ vrel_n; V = float(np.linalg.norm(vrel_b))
        alpha = float(np.arctan2(vrel_b[2], vrel_b[0])) if V > 1e-9 else 0.0
        beta = float(np.arcsin(np.clip(vrel_b[1] / V, -1, 1))) if V > 1e-9 else 0.0
        qbar = 0.5 * env.density_kg_m3 * V * V
        mach = V / env.speed_of_sound_m_s if env.speed_of_sound_m_s else 0.0
        fb, mb, coeff = aerodynamic_loads(env.density_kg_m3, vrel_b, alpha, beta, *om, geom,
                                           np.array([u.aileron, u.elevator, u.rudder]), aero)
        fp, mp = propulsion.force_and_moment(u.throttle, V)
        atm = standard_atmosphere(max(0.0, float(-pos[2])))
        rows.append({
            "time_s": float(ti), "north_m": pos[0], "east_m": pos[1], "down_m": pos[2], "altitude_m": -pos[2],
            "u_m_s": vb[0], "v_m_s": vb[1], "w_m_s": vb[2], "vn_m_s": vn[0], "ve_m_s": vn[1], "vd_m_s": vn[2],
            "speed_m_s": V, "p_rad_s": om[0], "q_rad_s": om[1], "r_rad_s": om[2],
            "roll_rad": history.euler_rad[i,0], "pitch_rad": history.euler_rad[i,1], "yaw_rad": history.euler_rad[i,2],
            "q0": quat[0], "q1": quat[1], "q2": quat[2], "q3": quat[3], "alpha_rad": alpha, "beta_rad": beta,
            "dynamic_pressure_Pa": qbar, "mach": mach, **{k: coeff[k] for k in ("CL","CD","CY","Cl","Cm","Cn")},
            "Fx_aero_N": fb[0], "Fy_aero_N": fb[1], "Fz_aero_N": fb[2],
            "Mx_aero_Nm": mb[0], "My_aero_Nm": mb[1], "Mz_aero_Nm": mb[2], "thrust_N": fp[0], "thrust_moment_Nm": mp[1],
            "aileron_rad": u.aileron, "elevator_rad": u.elevator, "rudder_rad": u.rudder, "throttle": u.throttle,
            "wind_n_m_s": wind[0], "wind_e_m_s": wind[1], "wind_d_m_s": wind[2], "gust_n_m_s": gust[0], "gust_e_m_s": gust[1], "gust_d_m_s": gust[2],
            "temperature_K": atm.temperature_K, "pressure_Pa": atm.pressure_Pa, "density_kg_m3": atm.density_kg_m3, "speed_of_sound_m_s": atm.speed_of_sound_m_s,
        })
    return rows


def _plot(path, t, series, labels, title, ylabel):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for y, label in zip(series, labels): ax.plot(t, y, label=label, linewidth=1.4)
    ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
    if len(labels) > 1: ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot3d(path, north, east, altitude):
    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(east, north, altitude, linewidth=1.6); ax.set_title("3D Flight Trajectory")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_zlabel("Altitude (m)")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def export_simulation(history, output_dir: Path, controls, environment, aircraft):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = _series(history, controls, environment, aircraft); t = history.time_s
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    with (output_dir / "simulation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    _plot(output_dir/"flight_history.png",t,[a["north_m"],a["east_m"],a["altitude_m"]],["North","East","Altitude"],"Flight History","Position (m)")
    _plot3d(output_dir/"trajectory_3d.png",a["north_m"],a["east_m"],a["altitude_m"])
    _plot(output_dir/"attitude.png",t,[np.rad2deg(a["roll_rad"]),np.rad2deg(a["pitch_rad"]),np.rad2deg(a["yaw_rad"])],["Roll","Pitch","Yaw"],"Aircraft Attitude","Angle (deg)")
    _plot(output_dir/"position_ned.png",t,[a["north_m"],a["east_m"],a["down_m"]],["North","East","Down"],"NED Position","Position (m)")
    _plot(output_dir/"velocity.png",t,[a["u_m_s"],a["v_m_s"],a["w_m_s"],a["speed_m_s"]],["u","v","w","Airspeed"],"Body Velocity","Velocity (m/s)")
    _plot(output_dir/"angular_rates.png",t,[np.rad2deg(a["p_rad_s"]),np.rad2deg(a["q_rad_s"]),np.rad2deg(a["r_rad_s"])],["p","q","r"],"Body Angular Rates","Rate (deg/s)")
    _plot(output_dir/"aerodynamic_angles.png",t,[np.rad2deg(a["alpha_rad"]),np.rad2deg(a["beta_rad"])],["Angle of attack","Sideslip"],"Aerodynamic Angles","Angle (deg)")
    _plot(output_dir/"aerodynamic_forces.png",t,[a["Fx_aero_N"],a["Fy_aero_N"],a["Fz_aero_N"]],["Fx","Fy","Fz"],"Aerodynamic Forces","Force (N)")
    _plot(output_dir/"aerodynamic_moments.png",t,[a["Mx_aero_Nm"],a["My_aero_Nm"],a["Mz_aero_Nm"]],["Mx","My","Mz"],"Aerodynamic Moments","Moment (N·m)")
    _plot(output_dir/"control_inputs.png",t,[np.rad2deg(a["aileron_rad"]),np.rad2deg(a["elevator_rad"]),np.rad2deg(a["rudder_rad"]),100*a["throttle"]],["Aileron","Elevator","Rudder","Throttle"],"Control Inputs","Command (deg / %)")
    _plot(output_dir/"atmospheric_state.png",t,[a["temperature_K"],a["pressure_Pa"],a["density_kg_m3"]],["Temperature","Pressure","Density"],"Atmospheric State","Value (SI units)")
    _plot(output_dir/"wind_and_gust.png",t,[a["wind_n_m_s"],a["wind_e_m_s"],a["wind_d_m_s"],a["gust_n_m_s"],a["gust_e_m_s"],a["gust_d_m_s"]],["Wind N","Wind E","Wind D","Gust N","Gust E","Gust D"],"Wind and Gust","Velocity (m/s)")
    _plot(output_dir/"propulsion.png",t,[a["thrust_N"]],["Thrust"],"Propulsion","Thrust (N)")
    _plot(output_dir/"dynamic_pressure.png",t,[a["dynamic_pressure_Pa"]],["Dynamic pressure"],"Dynamic Pressure","Pressure (Pa)")
    _plot(output_dir/"mach_number.png",t,[a["mach"]],["Mach"],"Mach Number","Mach")
    ground = np.hypot(a["vn_m_s"], a["ve_m_s"]); gamma = np.rad2deg(np.arctan2(-a["vd_m_s"], np.maximum(ground,1e-9)))
    _plot(output_dir/"flight_path.png",t,[gamma],["Flight-path angle"],"Flight Path","Angle (deg)")
    summary = {"duration_s":float(t[-1]),"time_step_s":float(np.median(np.diff(t))),"samples":int(len(t)),
               "initial":{"position_ned_m":history.state[0,:3].tolist(),"speed_m_s":float(a["speed_m_s"][0])},
               "final":{"position_ned_m":history.state[-1,:3].tolist(),"speed_m_s":float(a["speed_m_s"][-1]),"altitude_m":float(a["altitude_m"][-1])},
               "extrema":{"max_airspeed_m_s":float(a["speed_m_s"].max()),"min_airspeed_m_s":float(a["speed_m_s"].min()),"max_altitude_m":float(a["altitude_m"].max()),"min_altitude_m":float(a["altitude_m"].min()),"max_mach":float(a["mach"].max()),"max_dynamic_pressure_Pa":float(a["dynamic_pressure_Pa"].max()),"max_alpha_deg":float(np.rad2deg(a["alpha_rad"]).max()),"min_alpha_deg":float(np.rad2deg(a["alpha_rad"]).min()),"max_sideslip_deg":float(np.abs(np.rad2deg(a["beta_rad"])).max()),"max_thrust_N":float(a["thrust_N"].max())},
               "model":{"mass_kg":float(aircraft.parameters.geometry.mass_kg),"wing_area_m2":float(aircraft.parameters.geometry.wing_area_m2)},
               "outputs":[p.name for p in sorted(output_dir.glob("*.png"))]+["simulation.csv","simulation_summary.json"]}
    with (output_dir/"simulation_summary.json").open("w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
