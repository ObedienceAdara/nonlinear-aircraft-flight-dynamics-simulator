"""Demonstrate the Chapter 03 flight-environment pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from aircraft6dof.atmosphere import standard_atmosphere
from aircraft6dof.airdata import relative_velocity_body, compute_air_data
from aircraft6dof.aerodynamics import AerodynamicCoefficients, aerodynamic_loads


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "chapter_03_results.csv"


def main() -> None:
    # Example aircraft state: moderate forward flight with a small attitude.
    velocity_body = np.array([70.0, 4.0, 6.0])  # m/s
    phi, theta, psi = np.deg2rad([5.0, 7.0, 20.0])

    # Steady NED wind. This makes air-relative velocity different from
    # ground-referenced velocity and exercises the frame transformations.
    wind_ned = np.array([8.0, -3.0, 0.5])

    atmosphere = standard_atmosphere(1500.0)
    v_rel_body = relative_velocity_body(
        velocity_body, wind_ned, phi, theta, psi
    )
    airdata = compute_air_data(v_rel_body, atmosphere)

    coeffs = AerodynamicCoefficients(
        CD=0.035,
        CL=0.45,
        CY=-0.02,
        Cl=0.015,
        Cm=-0.035,
        Cn=0.01,
    )

    loads = aerodynamic_loads(
        airdata.dynamic_pressure_pa,
        wing_area_m2=16.2,
        span_m=11.0,
        chord_m=1.5,
        alpha_rad=airdata.angle_of_attack_rad,
        beta_rad=airdata.sideslip_rad,
        coefficients=coeffs,
    )

    values = [
        ("altitude_m", atmosphere.altitude_m),
        ("temperature_K", atmosphere.temperature_K),
        ("density_kg_m3", atmosphere.density_kg_m3),
        ("speed_of_sound_m_s", atmosphere.speed_of_sound_m_s),
        ("true_airspeed_m_s", airdata.true_airspeed_m_s),
        ("alpha_deg", np.rad2deg(airdata.angle_of_attack_rad)),
        ("beta_deg", np.rad2deg(airdata.sideslip_rad)),
        ("dynamic_pressure_pa", airdata.dynamic_pressure_pa),
        ("mach", airdata.mach),
        ("X_body_N", loads.force_body_N[0]),
        ("Y_body_N", loads.force_body_N[1]),
        ("Z_body_N", loads.force_body_N[2]),
        ("L_roll_Nm", loads.moment_body_Nm[0]),
        ("M_pitch_Nm", loads.moment_body_Nm[1]),
        ("N_yaw_Nm", loads.moment_body_Nm[2]),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value"])
        writer.writerows(values)

    print("Chapter 03 flight-environment demonstration complete.")
    for name, value in values:
        print(f"{name:24s}: {value:.6g}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
