"""Project-level entry point for the Aircraft 6-DOF simulator."""

from pathlib import Path

import numpy as np

from aircraft6dof import AircraftModel, AircraftState, ControlInput, Environment, Simulator, VehicleGeometry
from aircraft6dof.aero import AeroCoefficients
from aircraft6dof.atmosphere import standard_atmosphere
from aircraft6dof.equations import AircraftParameters
from aircraft6dof.propulsion import Propulsion
from aircraft6dof.reporting import export_simulation
from aircraft6dof.wind import OneMinusCosineGust

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def build_aircraft() -> AircraftModel:
    geometry = VehicleGeometry(
        mass_kg=1200.0,
        inertia_kg_m2=np.diag([1800.0, 2500.0, 4200.0]),
        wing_area_m2=16.2,
        wing_span_m=10.9,
        mean_chord_m=1.49,
        cg_to_ref_m=np.array([0.15, 0.0, 0.0]),
    )
    aero = AeroCoefficients(
        CL0=0.25, CL_alpha=5.2, CL_de=0.55, CL_q=7.0,
        CD0=0.032, CD_alpha2=0.32, CD_de2=0.02,
        CY_beta=-0.75, CY_da=0.02, CY_dr=0.18,
        Cl_beta=-0.12, Cl_p=-0.50, Cl_r=0.18, Cl_da=0.16, Cl_dr=0.03,
        Cm0=0.03, Cm_alpha=-1.05, Cm_q=-10.0, Cm_de=-1.15,
        Cn_beta=0.18, Cn_p=-0.06, Cn_r=-0.20, Cn_da=0.02, Cn_dr=-0.10,
    )
    propulsion = Propulsion(
        max_thrust_n=4200.0,
        thrust_velocity_factor=0.25,
        thrust_arm_m=np.array([-1.2, 0.0, 0.0]),
    )
    return AircraftModel(AircraftParameters(geometry=geometry, aero=aero, propulsion=propulsion))


def initial_state() -> AircraftState:
    return AircraftState(
        position_ned_m=np.array([0.0, 0.0, -1000.0]),
        velocity_body_m_s=np.array([55.0, 0.0, 0.0]),
        omega_body_rad_s=np.zeros(3),
        quaternion_bn=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def controls(t: float) -> ControlInput:
    return ControlInput(
        aileron=np.deg2rad(3.0 if 18.0 <= t < 26.0 else 0.0),
        elevator=np.deg2rad(-1.5 if 8.0 <= t < 16.0 else 0.0),
        rudder=np.deg2rad(1.0 if 18.0 <= t < 26.0 else 0.0),
        throttle=0.72 if t < 12.0 else 0.82,
    )


def environment(t: float) -> Environment:
    atm = standard_atmosphere(1000.0)
    gust = OneMinusCosineGust(
        amplitude_m_s=np.array([0.0, 4.0, 2.0]), start_s=10.0, duration_s=12.0
    )
    return Environment(
        wind_ned_m_s=np.array([5.0, 2.0, 0.0]),
        gust_ned_m_s=gust.value(t),
        density_kg_m3=atm.density_kg_m3,
        speed_of_sound_m_s=atm.speed_of_sound_m_s,
        gravity_ned_m_s2=np.array([9.806, 0.0, 0.0]),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    simulator = Simulator(build_aircraft())
    history = simulator.run(
        initial_state=initial_state(),
        duration_s=40.0,
        dt_s=0.02,
        controls=controls,
        environment=environment,
    )
    export_simulation(history, OUTPUT_DIR, controls=controls, environment=environment, aircraft=simulator.aircraft)
    print("Aircraft 6-DOF simulation complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Samples: {len(history.time)}")


if __name__ == "__main__":
    main()
