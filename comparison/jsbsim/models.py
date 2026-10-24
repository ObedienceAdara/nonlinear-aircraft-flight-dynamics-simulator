"""Construct the repository model and environment matching JSBSim."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aircraft6dof import AircraftModel, Environment, VehicleGeometry  # noqa: E402
from aircraft6dof.aero import AeroCoefficients  # noqa: E402
from aircraft6dof.equations import AircraftParameters  # noqa: E402
from aircraft6dof.propulsion import Propulsion  # noqa: E402

OMEGA_EARTH = 7.2921150e-5
WGS84_A = 6378137.0
T_MAX_N = 3000.0


def build_repo_model(jsb, aero_coeffs: dict) -> AircraftModel:
    """Use JSBSim mass/inertia/planform and coefficients fitted to its loads."""
    mp = jsb.mass_properties()
    geometry = VehicleGeometry(
        mass_kg=mp["mass_kg"],
        inertia_kg_m2=mp["inertia_kg_m2"],
        wing_area_m2=mp["wing_area_m2"],
        wing_span_m=mp["wing_span_m"],
        mean_chord_m=mp["mean_chord_m"],
        cg_to_ref_m=np.zeros(3),
    )
    propulsion = Propulsion(max_thrust_N=T_MAX_N, thrust_velocity_factor=0.0, thrust_arm_m=np.zeros(3))
    return AircraftModel(AircraftParameters(geometry=geometry, aero=AeroCoefficients(**aero_coeffs), propulsion=propulsion))


def build_environment(jsb, altitude_m: float) -> Environment:
    """Still air using JSBSim density and effective gravity.

    JSBSim integrates in an Earth-fixed rotating frame. At the equator, with
    north-heading straight flight, Coriolis acceleration is zero and the
    leading centripetal term is represented by g_eff = g - Omega^2 * (a + h).
    """
    properties = jsb.environment()
    effective_g = properties["gravity_m_s2"] - OMEGA_EARTH**2 * (WGS84_A + altitude_m)
    return Environment(
        density_kg_m3=properties["density_kg_m3"],
        speed_of_sound_m_s=properties["speed_of_sound_m_s"],
        gravity_ned_m_s2=np.array([0.0, 0.0, effective_g]),
    )
