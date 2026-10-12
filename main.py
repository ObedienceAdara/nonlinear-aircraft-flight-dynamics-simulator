"""Project-level entry point for the nonlinear aircraft 6-DOF simulator."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aircraft6dof import (
    AircraftModel,
    AircraftState,
    ControlInput,
    Environment,
    SimulationDivergenceError,
    Simulator,
    TrimCondition,
    VehicleGeometry,
)
from aircraft6dof.actuators import ActuatorChannel, ActuatorSet
from aircraft6dof.aero import AeroCoefficients
from aircraft6dof.atmosphere import standard_atmosphere
from aircraft6dof.equations import AircraftParameters
from aircraft6dof.propulsion import Propulsion
from aircraft6dof.reporting import export_simulation
from aircraft6dof.wind import OneMinusCosineGust
from aircraft6dof.trim import trim

OUTPUT_DIR = ROOT / "outputs"
REFERENCE_ALTITUDE_M = 1000.0


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
        CL0=0.25,
        CL_alpha=5.2,
        CL_de=0.55,
        CL_q=7.0,
        CD0=0.032,
        CD_alpha2=0.32,
        CD_de2=0.02,
        CY_beta=-0.75,
        CY_da=0.02,
        CY_dr=0.18,
        Cl_beta=-0.12,
        Cl_p=-0.50,
        Cl_r=0.18,
        Cl_da=0.16,
        Cl_dr=0.03,
        Cm0=0.03,
        Cm_alpha=-1.05,
        Cm_q=-10.0,
        Cm_de=-1.15,
        Cn_beta=0.18,
        Cn_p=-0.06,
        Cn_r=-0.20,
        # Positive Cn_da is proverse yaw in this sign convention; the dedicated adverse-yaw
        # study intentionally overrides this parameter so the opposite-sign phenomenon can be isolated.
        Cn_da=0.02,
        Cn_dr=-0.10,
    )
    propulsion = Propulsion(
        max_thrust_N=4200.0,
        thrust_velocity_factor=0.25,
        thrust_arm_m=np.array([-1.2, 0.0, 0.0]),
    )
    return AircraftModel(AircraftParameters(geometry=geometry, aero=aero, propulsion=propulsion))


def build_actuators(initial_controls: ControlInput | None = None) -> ActuatorSet:
    """Build provisional actuator channels initialized at the baseline trim.

    Time constants and limits are demonstration assumptions, not identified
    hardware data. Supply initial_controls so surfaces begin at trim deflections
    instead of introducing a startup actuator transient.
    """
    baseline = ControlInput() if initial_controls is None else initial_controls
    return ActuatorSet(
        aileron=ActuatorChannel(
            time_constant_s=0.15,
            rate_limit_rad_s=np.deg2rad(80.0),
            position_limit_rad=np.deg2rad(25.0),
            position_rad=baseline.aileron,
        ),
        elevator=ActuatorChannel(
            time_constant_s=0.20,
            rate_limit_rad_s=np.deg2rad(50.0),
            position_limit_rad=np.deg2rad(25.0),
            position_rad=baseline.elevator,
        ),
        rudder=ActuatorChannel(
            time_constant_s=0.20,
            rate_limit_rad_s=np.deg2rad(40.0),
            position_limit_rad=np.deg2rad(30.0),
            position_rad=baseline.rudder,
        ),
    )


def trim_initial_condition(aircraft: AircraftModel, env: Environment):
    """Solve the default straight, level 55 m/s trim at the reference altitude."""
    return trim(
        aircraft,
        env,
        TrimCondition(airspeed_m_s=55.0, altitude_m=REFERENCE_ALTITUDE_M),
    )


def initial_state(
    aircraft: AircraftModel | None = None,
    env: Environment | None = None,
) -> AircraftState:
    """Return a trim-consistent initial state for the default demo condition."""
    model = build_aircraft() if aircraft is None else aircraft
    initial_environment = environment(0.0) if env is None else env
    return trim_initial_condition(model, initial_environment).state


def controls(t: float, baseline: ControlInput | None = None) -> ControlInput:
    """Scheduled control increments about a baseline (trim for the main run).

    With no baseline, retain the original demonstration schedule for callers
    that use this function independently.
    """
    base = ControlInput(throttle=0.72) if baseline is None else baseline
    return ControlInput(
        aileron=base.aileron + np.deg2rad(3.0 if 18.0 <= t < 26.0 else 0.0),
        elevator=base.elevator + np.deg2rad(-1.5 if 8.0 <= t < 16.0 else 0.0),
        rudder=base.rudder + np.deg2rad(1.0 if 18.0 <= t < 26.0 else 0.0),
        throttle=min(1.0, base.throttle + (0.10 if t >= 12.0 else 0.0)),
    )


def environment(t: float) -> Environment:
    # The demo deliberately holds the 1,000 m reference atmosphere constant.
    # This is a fixed-environment experiment, not an altitude-varying ISA model.
    atm = standard_atmosphere(REFERENCE_ALTITUDE_M)
    gust = OneMinusCosineGust(
        amplitude_m_s=np.array([0.0, 4.0, 2.0]),
        start_s=10.0,
        duration_s=12.0,
    )
    return Environment(
        wind_ned_m_s=np.array([5.0, 2.0, 0.0]),
        gust_ned_m_s=gust.value(t),
        density_kg_m3=atm.density_kg_m3,
        speed_of_sound_m_s=atm.speed_of_sound_m_s,
        gravity_ned_m_s2=np.array([0.0, 0.0, 9.806]),
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aircraft = build_aircraft()
    env0 = environment(0.0)
    initial_condition = trim_initial_condition(aircraft, env0)
    control_schedule = lambda t: controls(t, initial_condition.controls)
    actuators = build_actuators(initial_condition.controls)
    try:
        history = Simulator(aircraft).run(
            initial_condition.state,
            control_schedule,
            environment,
            duration_s=40.0,
            dt_s=0.02,
            actuators=actuators,
        )
        export_simulation(
            history,
            OUTPUT_DIR,
            controls=control_schedule,
            environment=environment,
            aircraft=aircraft,
        )
    except SimulationDivergenceError as exc:
        print(exc.report(), file=sys.stderr)
        return 2

    print("Aircraft 6-DOF simulation complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Samples: {len(history.time_s)}")
    print(f"Initial condition: numerical trim at {initial_condition.condition.airspeed_m_s:.1f} m/s, "
          f"{initial_condition.condition.altitude_m:.0f} m.")
    print("Atmosphere: fixed 1,000 m reference conditions throughout the demonstration.")
    print("Generated simulation.csv, simulation_summary.json, and engineering plots.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
