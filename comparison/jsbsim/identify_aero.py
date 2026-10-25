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
            "alpha_dot": jsb.alpha_dot_rad_s(),
            "da": surfaces[0], "de": surfaces[1], "dr": surfaces[2],
            **jsb.aero_coefficient_inputs(),
            "u": x[0], "vv": x[1], "w": x[2],
            "fx": force[0], "fy": force[1], "fz": force[2],
            "mx": moment[0], "my": moment[1], "mz": moment[2],
        })
    data = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    data["rho"] = jsb.environment()["density_kg_m3"]
    return data


def _fit_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute residual metrics against the values actually predicted."""
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    centered = np.asarray(observed, dtype=float) - float(np.mean(observed))
    total_ss = float(centered @ centered)
    residual_ss = float(residual @ residual)
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0.0 else float("nan")
    spread = float(np.std(observed))
    return {
        "rms_residual": float(np.sqrt(np.mean(residual**2))),
        "max_residual": float(np.max(np.abs(residual))),
        "spread_of_coefficient": spread,
        "r_squared": r_squared,
    }


def fit(data: dict, airspeed_m_s: float, span_m: float, chord_m: float,
        fixed_coefficients: dict | None = None) -> tuple[dict, dict]:
    """Identify supported derivatives without hiding unmodelled terms in R².

    CL and Cm are estimated with alpha-dot as a nuisance predictor. The
    alpha-dot coefficient is diagnostic-only because the compact model does not
    apply it at runtime; therefore applied-model R² is recomputed after
    dropping that predictor. The joint fit helps expose rate-derivative
    confounding without claiming that the complete extended model is deployed.

    Drag is fitted using all implemented terms, including CD_alpha2 and
    CD_abs_beta. A separate ablation score quantifies the effect of removing
    the implemented absolute-sideslip drag term.
    """
    if not np.isfinite(airspeed_m_s) or airspeed_m_s <= 0.0:
        raise ValueError("airspeed_m_s must be finite and positive")
    if not np.isfinite(span_m) or span_m <= 0.0:
        raise ValueError("span_m must be finite and positive")
    if not np.isfinite(chord_m) or chord_m <= 0.0:
        raise ValueError("chord_m must be finite and positive")
    required = ("alpha", "beta", "de", "da", "dr", "p", "q", "r", "alpha_dot",
                "CL", "CD", "CY", "Cl", "Cm", "Cn")
    missing_data = sorted(set(required) - set(data))
    if missing_data:
        raise ValueError(f"sample data missing required channels: {missing_data}")
    data_vectors = {key: np.asarray(data[key], dtype=float) for key in required}
    sample_count = len(data_vectors["alpha"])
    if sample_count == 0:
        raise ValueError("at least one sample is required for aerodynamic identification")
    for key, values in data_vectors.items():
        if values.ndim != 1 or len(values) != sample_count:
            raise ValueError(f"sample channel {key!r} must be a 1-D array of length {sample_count}")
        if not np.isfinite(values).all():
            raise ValueError(f"sample channel {key!r} contains non-finite values")
    alpha, beta, de, da, dr = (data_vectors[key] for key in ("alpha", "beta", "de", "da", "dr"))
    p_hat = np.asarray(data["p"], dtype=float) * span_m / (2 * airspeed_m_s)
    q_hat = np.asarray(data["q"], dtype=float) * chord_m / (2 * airspeed_m_s)
    r_hat = np.asarray(data["r"], dtype=float) * span_m / (2 * airspeed_m_s)
    alpha_dot_hat = np.asarray(data["alpha_dot"], dtype=float) * chord_m / (2 * airspeed_m_s)
    ones = np.ones_like(alpha)
    plan = {
        "CL": ([ones, alpha, de, q_hat], ["CL0", "CL_alpha", "CL_de", "CL_q"]),
        "CD": ([ones, alpha, alpha**2, de**2, np.abs(beta)],
               ["CD0", "CD_alpha", "CD_alpha2", "CD_de2", "CD_abs_beta"]),
        "CY": ([beta, da, dr, p_hat, r_hat],
               ["CY_beta", "CY_da", "CY_dr", "CY_p", "CY_r"]),
        "Cl": ([beta, p_hat, r_hat, da, dr],
               ["Cl_beta", "Cl_p", "Cl_r", "Cl_da", "Cl_dr"]),
        "Cm": ([ones, alpha, q_hat, de],
               ["Cm0", "Cm_alpha", "Cm_q", "Cm_de"]),
        "Cn": ([beta, p_hat, r_hat, da, dr],
               ["Cn_beta", "Cn_p", "Cn_r", "Cn_da", "Cn_dr"]),
    }
    coefficients, quality, diagnostics = {}, {}, {}
    if fixed_coefficients is not None:
        missing = sorted(set(name for _, names in plan.values() for name in names) - set(fixed_coefficients))
        if missing:
            raise ValueError(f"fixed_coefficients missing fitted terms: {missing}")
        nonfinite = sorted(
            name for name in (name for _, names in plan.values() for name in names)
            if not np.isfinite(float(fixed_coefficients[name]))
        )
        if nonfinite:
            raise ValueError(f"fixed_coefficients contains non-finite terms: {nonfinite}")
    for channel, (columns, names) in plan.items():
        X = np.column_stack(columns)
        y = np.asarray(data[channel], dtype=float)
        model_solution, *_ = np.linalg.lstsq(X, y, rcond=None)
        selected_solution = model_solution
        extended_metrics = None
        if channel in ("CL", "Cm"):
            # Include the JSBSim alpha-dot rate as a nuisance predictor when
            # identifying the supported CL/Cm terms. The returned vector below
            # omits only the final alpha-dot coefficient; the run-time model
            # does not apply that additional force/moment term.
            X_extended = np.column_stack([X, alpha_dot_hat])
            extended_solution, *_ = np.linalg.lstsq(X_extended, y, rcond=None)
            # Use the joint fit for the supported coefficient estimates so
            # q-dependent aerodynamic effects are not silently absorbed into
            # q derivatives when alpha-dot is omitted as a predictor. The
            # alpha-dot coefficient itself remains diagnostic-only: the compact
            # model does not apply it at run time. Compare deployed-model R²
            # below with the alpha-dot column excluded.
            selected_solution = extended_solution[:-1]
            q_index = names.index(f"{channel}_q")
            diagnostics[f"{channel}_adot"] = float(extended_solution[-1])
            diagnostics[f"{channel}_q_only_fit"] = float(model_solution[q_index])
            diagnostics[f"{channel}_q_alpha_dot_adjusted"] = float(extended_solution[q_index])
            extended_metrics = _fit_metrics(y, X_extended @ extended_solution)
        effective_solution = (
            selected_solution if fixed_coefficients is None
            else np.asarray([fixed_coefficients[name] for name in names], dtype=float)
        )
        for name, value in zip(names, map(float, effective_solution)):
            coefficients[name] = value

        # The score below is the performance of the actual coefficients that
        # will be passed into AeroCoefficients, not the extended diagnostic.
        actual_metrics = _fit_metrics(y, X @ effective_solution)
        best_supported_metrics = _fit_metrics(y, X @ model_solution)
        quality[channel] = {
            **actual_metrics,
            "r_squared_best_supported_terms": best_supported_metrics["r_squared"],
        }
        if extended_metrics is not None:
            quality[channel]["r_squared_with_alpha_dot_diagnostic"] = (
                extended_metrics["r_squared"]
            )

        if channel == "CD":
            # Quantify the effect of removing the now-implemented beta-drag
            # term. The reported CD R² above uses all coefficients the model
            # actually applies, including CD_abs_beta.
            X_without_beta = X[:, :-1]
            without_beta_solution, *_ = np.linalg.lstsq(X_without_beta, y, rcond=None)
            without_beta_metrics = _fit_metrics(y, X_without_beta @ without_beta_solution)
            diagnostics["CD_r_squared_without_abs_beta"] = without_beta_metrics["r_squared"]
            diagnostics["CD_rms_residual_without_abs_beta"] = without_beta_metrics["rms_residual"]
            diagnostics["CD_max_residual_without_abs_beta"] = without_beta_metrics["max_residual"]

    quality["diagnostics"] = diagnostics
    return coefficients, quality

def reconstruction_error(coeffs: dict, data: dict, geometry) -> dict:
    """Rebuild JSBSim sampled loads and quantify the implemented beta-drag term."""
    from aircraft6dof.aero import AeroCoefficients, aerodynamic_loads

    aero = AeroCoefficients(**coeffs)
    aero_without_beta_drag = AeroCoefficients(**{**coeffs, "CD_abs_beta": 0.0})
    force_errors, moment_errors, force_errors_without_beta_drag = [], [], []
    for i in range(len(data["alpha"])):
        velocity = np.array([data["u"][i], data["vv"][i], data["w"][i]])
        args = (
            data["rho"], velocity, data["alpha"][i], data["beta"][i],
            data["p"][i], data["q"][i], data["r"][i], geometry,
            np.array([data["da"][i], data["de"][i], data["dr"][i]]),
        )
        force, moment, _ = aerodynamic_loads(*args, aero)
        force_without_beta, _, _ = aerodynamic_loads(*args, aero_without_beta_drag)
        jsb_force = np.array([data["fx"][i], data["fy"][i], data["fz"][i]])
        jsb_moment = np.array([data["mx"][i], data["my"][i], data["mz"][i]])
        force_errors.append(force - jsb_force)
        force_errors_without_beta_drag.append(force_without_beta - jsb_force)
        moment_errors.append(moment - jsb_moment)

    force_errors = np.asarray(force_errors)
    force_errors_without_beta_drag = np.asarray(force_errors_without_beta_drag)
    moment_errors = np.asarray(moment_errors)
    return {
        "force_rms_N": np.sqrt(np.mean(force_errors**2, axis=0)).tolist(),
        "force_max_N": np.max(np.abs(force_errors), axis=0).tolist(),
        "force_rms_without_abs_sideslip_drag_N": np.sqrt(
            np.mean(force_errors_without_beta_drag**2, axis=0)
        ).tolist(),
        "moment_rms_Nm": np.sqrt(np.mean(moment_errors**2, axis=0)).tolist(),
        "moment_max_Nm": np.max(np.abs(moment_errors), axis=0).tolist(),
        "force_span_N": np.std(np.array([data["fx"], data["fy"], data["fz"]]), axis=1).tolist(),
        "moment_span_Nm": np.std(np.array([data["mx"], data["my"], data["mz"]]), axis=1).tolist(),
    }
