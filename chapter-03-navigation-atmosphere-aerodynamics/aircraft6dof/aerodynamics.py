"""Wind-axis aerodynamic loads and conversion to body axes."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AerodynamicCoefficients:
    """Dimensionless aerodynamic force and moment coefficients."""

    CD: float
    CL: float
    CY: float
    Cl: float
    Cm: float
    Cn: float


@dataclass(frozen=True)
class AerodynamicLoads:
    force_body_N: np.ndarray
    moment_body_Nm: np.ndarray
    force_wind_N: np.ndarray
    moment_body_from_coefficients_Nm: np.ndarray


def aerodynamic_loads(
    dynamic_pressure_pa: float,
    wing_area_m2: float,
    span_m: float,
    chord_m: float,
    alpha_rad: float,
    beta_rad: float,
    coefficients: AerodynamicCoefficients,
) -> AerodynamicLoads:
    """Compute aerodynamic forces and moments.

    Wind-axis convention:
        x_w points into the relative wind,
        z_w points approximately downward through the lift-producing sign
        convention, so the load vector is [-D, Y, -L].

    Rotation first accounts for alpha then beta; the explicit matrix keeps
    coordinate/sign assumptions visible for review.
    """
    q = float(dynamic_pressure_pa)
    S, b, c = map(float, (wing_area_m2, span_m, chord_m))

    if q < 0 or S <= 0 or b <= 0 or c <= 0:
        raise ValueError("Dynamic pressure must be non-negative and geometry positive.")

    force_wind = q * S * np.array(
        [-coefficients.CD, coefficients.CY, -coefficients.CL],
        dtype=float,
    )

    # Wind-to-body DCM for the stated body/wind convention.
    sa, ca = np.sin(alpha_rad), np.cos(alpha_rad)
    sb, cb = np.sin(beta_rad), np.cos(beta_rad)

    c_bw = np.array([
        [ca * cb, -ca * sb, -sa],
        [sb,       cb,       0.0],
        [sa * cb, -sa * sb,  ca],
    ])

    force_body = c_bw @ force_wind

    moment_body = q * S * np.array(
        [b * coefficients.Cl, c * coefficients.Cm, b * coefficients.Cn],
        dtype=float,
    )

    return AerodynamicLoads(
        force_body_N=force_body,
        moment_body_Nm=moment_body,
        force_wind_N=force_wind,
        moment_body_from_coefficients_Nm=moment_body,
    )
