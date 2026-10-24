"""Fit the repository's compact aerodynamic coefficient model to JSBSim locally.

The fitted derivatives make this a simulator-implementation comparison, not an
independent validation of the repository's aerodynamic data. Fit residuals are
reported to expose effects the simpler model cannot represent.
"""
from __future__ import annotations

import numpy as np

DEG = np.pi / 180.0


def sample_ranges(alpha0: float, controls0: tuple[float, float, float]) -> dict:
    da0, de0, dr0 = controls0
    return {
        "alpha": (alpha0 - 2 * DEG, alpha0 + 2 * DEG),
        "beta": (-3 * DEG, 3 * DEG),
        "p": (-0.25, 0.25), "q": (-0.12, 0.12), "r": (-0.12, 0.12),
        "de": (de0 - 3 * DEG, de0 + 3 * DEG),
        "da": (da0 - 3 * DEG, da0 + 3 * DEG),
        "dr": (dr0 - 3 * DEG, dr0 + 3 * DEG),
    }


def collect(jsb, airspeed_m_s: float, altitude_m: float, alpha0: float,
            controls0: tuple[float, float, float] = (0.0, 0.0, 0.0),
            n: int = 400, seed: int = 0) -> dict:
    """Collect reproducible local samples of JSBSim aerodynamic coefficients and loads."""
    rng = np.random.default_rng(seed)
    ranges = sample_ranges(alpha0, controls0)
    rows = []
    for _ in range(n):
        values = {key: rng.uniform(*bounds) for key, bounds in ranges.items()}
        x = np.array([
            airspeed_m_s * np.cos(values["alpha"]) * np.cos(values["beta"]),
            airspeed_m_s * np.sin(values["beta"]),
            airspeed_m_s * np.sin(values["alpha"]) * np.cos(values["beta"]),
            values["p"], values["q"], values["r"], 0.0, values["alpha"], 0.0,
        ])
        jsb.set_state(x, altitude_m, (values["da"], values["de"], values["dr"], 0.0))
        alpha, beta = jsb.alpha_beta()
        surfaces = jsb.surfaces()
        force, moment = jsb.aero_load_body()
        rows.append({
            **values, "alpha": alpha, "beta": beta,
            "da": surfaces[0], "de": surfaces[1], "dr": surfaces[2],
            **jsb.aero_coefficient_inputs(),
            "u": x[0], "vv": x[1], "w": x[2],
            "fx": force[0], "fy": force[1], "fz": force[2],
            "mx": moment[0], "my": moment[1], "mz": moment[2],
        })
    data = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    data["rho"] = jsb.environment()["density_kg_m3"]
    return data


def fit(data: dict, airspeed_m_s: float, span_m: float, chord_m: float) -> tuple[dict, dict]:
    """Least-squares fit for the coefficient terms present in aircraft6dof.aero."""
    alpha, beta, de, da, dr = (data[key] for key in ("alpha", "beta", "de", "da", "dr"))
    p_hat = data["p"] * span_m / (2 * airspeed_m_s)
    q_hat = data["q"] * chord_m / (2 * airspeed_m_s)
    r_hat = data["r"] * span_m / (2 * airspeed_m_s)
    ones = np.ones_like(alpha)
    plan = {
        "CL": ([ones, alpha, de, q_hat], ["CL0", "CL_alpha", "CL_de", "CL_q"]),
        "CD": ([ones, alpha, de**2, np.abs(beta)], ["CD0", "CD_alpha", "CD_de2", "~CD_abs_beta"]),
        "CY": ([beta, da, dr, p_hat, r_hat], ["CY_beta", "CY_da", "CY_dr", "CY_p", "CY_r"]),
        "Cl": ([beta, p_hat, r_hat, da, dr], ["Cl_beta", "Cl_p", "Cl_r", "Cl_da", "Cl_dr"]),
        "Cm": ([ones, alpha, q_hat, de], ["Cm0", "Cm_alpha", "Cm_q", "Cm_de"]),
        "Cn": ([beta, p_hat, r_hat, da, dr], ["Cn_beta", "Cn_p", "Cn_r", "Cn_da", "Cn_dr"]),
    }
    coefficients, quality, diagnostics = {}, {}, {}
    for channel, (columns, names) in plan.items():
        X = np.column_stack(columns)
        y = data[channel]
        solution, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ solution
        spread = float(np.std(y))
        for name, value in zip(names, map(float, solution)):
            (diagnostics if name.startswith("~") else coefficients)[name.lstrip("~")] = value
        quality[channel] = {
            "rms_residual": float(np.sqrt(np.mean(residual**2))),
            "max_residual": float(np.max(np.abs(residual))),
            "spread_of_coefficient": spread,
            "r_squared": float(1.0 - np.var(residual) / np.var(y)) if spread > 0 else float("nan"),
        }
    quality["diagnostics"] = diagnostics
    return coefficients, quality


def reconstruction_error(coeffs: dict, data: dict, geometry, cd_abs_beta: float = 0.0) -> dict:
    """Check fitted loads by rebuilding JSBSim sampled body-axis loads with repo equations."""
    from aircraft6dof.aero import AeroCoefficients, aerodynamic_loads

    aero = AeroCoefficients(**coeffs)
    force_errors, moment_errors, force_errors_without_beta_drag = [], [], []
    for i in range(len(data["alpha"])):
        velocity = np.array([data["u"][i], data["vv"][i], data["w"][i]])
        force, moment, _ = aerodynamic_loads(
            data["rho"], velocity, data["alpha"][i], data["beta"][i],
            data["p"][i], data["q"][i], data["r"][i], geometry,
            np.array([data["da"][i], data["de"][i], data["dr"][i]]), aero,
        )
        jsb_force = np.array([data["fx"][i], data["fy"][i], data["fz"][i]])
        jsb_moment = np.array([data["mx"][i], data["my"][i], data["mz"][i]])
        force_errors.append(force - jsb_force)
        alpha, beta = data["alpha"][i], data["beta"][i]
        qS = 0.5 * data["rho"] * float(velocity @ velocity) * geometry.wing_area_m2
        extra_drag = cd_abs_beta * abs(beta) * qS
        beta_drag_wind = extra_drag * np.array([np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)])
        force_errors_without_beta_drag.append(force - (jsb_force + beta_drag_wind))
        moment_errors.append(moment - jsb_moment)

    force_errors = np.asarray(force_errors)
    moment_errors = np.asarray(moment_errors)
    clean_force_errors = np.asarray(force_errors_without_beta_drag)
    return {
        "force_rms_N": np.sqrt(np.mean(force_errors**2, axis=0)).tolist(),
        "force_max_N": np.max(np.abs(force_errors), axis=0).tolist(),
        "force_rms_without_sideslip_drag_N": np.sqrt(np.mean(clean_force_errors**2, axis=0)).tolist(),
        "moment_rms_Nm": np.sqrt(np.mean(moment_errors**2, axis=0)).tolist(),
        "moment_max_Nm": np.max(np.abs(moment_errors), axis=0).tolist(),
        "force_span_N": np.std(np.array([data["fx"], data["fy"], data["fz"]]), axis=1).tolist(),
        "moment_span_Nm": np.std(np.array([data["mx"], data["my"], data["mz"]]), axis=1).tolist(),
    }
