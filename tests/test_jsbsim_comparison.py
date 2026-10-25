"""Fast JSBSim comparison tests; module skips only when optional JSBSim is unavailable."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jsbsim", reason="JSBSim comparison requires the .[compare] extra")
CMP = Path(__file__).resolve().parents[1] / "comparison" / "jsbsim"
if str(CMP) not in sys.path:
    sys.path.insert(0, str(CMP))
import compare as C  # noqa: E402
from identify_aero import collect, fit  # noqa: E402
import make_model as model_builder  # noqa: E402
from jsb import JsbPlane  # noqa: E402
from models import build_environment, build_repo_model  # noqa: E402


@pytest.fixture(scope="module")
def plane():
    return JsbPlane(.01)


@pytest.fixture(scope="module")
def fitted(plane):
    jt = C.trim_jsbsim(plane, C.V_TRIM, C.ALT)
    data = collect(plane, C.V_TRIM, C.ALT, jt["alpha_rad"],
                   (0., jt["elevator_rad"], 0.), n=200)
    mp = plane.mass_properties()
    coeffs, quality = fit(data, C.V_TRIM, mp["wing_span_m"], mp["mean_chord_m"])
    return build_repo_model(plane, coeffs), build_environment(plane, C.ALT), quality, coeffs, data


def test_equations_of_motion_match_jsbsim(plane):
    result = C.eom_check(plane, n=15)
    assert max(result["max_abs_error_after_earth_rotation"][:3]) < 1e-8
    assert max(result["max_abs_error_after_frame_corrections"][3:]) < 1e-10
    assert max(result["max_abs_euler_rate_error_rad_s"]) < 1e-9
    assert max(result["max_abs_ned_velocity_error_m_s"]) < 1e-9


def test_aero_fit_is_good_where_model_can_represent_it(fitted):
    _, _, quality, coeffs, data = fitted
    for channel in ("CL", "CY", "Cl", "Cm", "Cn"):
        assert quality[channel]["r_squared"] > .99, channel
    assert quality["diagnostics"]["CD_abs_beta"] == pytest.approx(.17, abs=.01)


def test_trim_and_modes_match_jsbsim(plane, fitted):
    aircraft, env, _, _, _ = fitted
    result = C.trim_and_modes(plane, aircraft, env)
    jsb, repo = result["trim"]["jsbsim"], result["trim"]["repo"]
    assert repo["alpha_deg"] == pytest.approx(jsb["alpha_deg"], abs=.05)
    assert repo["elevator_deg"] == pytest.approx(jsb["elevator_deg"], abs=.5)
    assert repo["thrust_N"] == pytest.approx(jsb["thrust_N"], rel=.05)
    for name in ("short_period", "dutch_roll"):
        assert result["modes_repo"][name]["wn_rad_s"] == pytest.approx(
            result["modes_jsbsim"][name]["wn_rad_s"], rel=.02
        ), name
        assert result["modes_repo"][name]["zeta"] == pytest.approx(
            result["modes_jsbsim"][name]["zeta"], rel=.03
        ), name
    assert result["modes_repo"]["roll_subsidence"]["re"] == pytest.approx(
        result["modes_jsbsim"]["roll_subsidence"]["re"], rel=.02
    )


def test_aileron_pulse_response_matches(plane, fitted):
    aircraft, env, _, _, _ = fitted
    result = C.trim_and_modes(plane, aircraft, env)
    case = C.CASES["aileron_pulse"]
    jsb_case = C.run_jsbsim_case(plane, result["_trim_jsbsim"], case["du"], 12., .01)
    repo_case = C.run_repo_case(aircraft, env, result["_trim_repo"], case["du"], 12., .01)
    errors = C.compare_case("aileron_pulse", jsb_case, repo_case)["primary"]
    for signal in ("phi", "r", "beta", "v"):
        assert errors[signal]["rms_diff_pct_of_peak"] < 3., signal


def test_report_writer_includes_validation_boundary(tmp_path):
    from report import write
    summary = json.loads((CMP / "results" / "summary.json").read_text())
    output = tmp_path / "REPORT.md"
    write(summary, output)
    text = output.read_text()
    assert "not independent validation against a real aircraft" in text
    assert "does not independently validate either aerodynamic model" in text

def test_collected_alpha_dot_is_finite_and_varies(fitted):
    _, _, _, _, data = fitted
    assert "alpha_dot" in data
    assert np.isfinite(data["alpha_dot"]).all()
    assert np.std(data["alpha_dot"]) > 1e-8

def test_pitch_rate_derivatives_are_adjusted_for_alpha_dot(fitted):
    _, _, quality, coeffs, _ = fitted
    diagnostic = quality["diagnostics"]
    assert coeffs["CL_q"] == pytest.approx(diagnostic["CL_q_alpha_dot_adjusted"])
    assert coeffs["Cm_q"] == pytest.approx(diagnostic["Cm_q_alpha_dot_adjusted"])
    assert abs(coeffs["CL_q"] - diagnostic["CL_q_only_fit"]) > 0.2
    assert abs(coeffs["Cm_q"] - diagnostic["Cm_q_only_fit"]) > 0.5

def test_alpha_dot_terms_are_diagnostic_not_deployed_coefficients(fitted):
    _, _, quality, coeffs, _ = fitted
    assert "CL_adot" in quality["diagnostics"]
    assert "Cm_adot" in quality["diagnostics"]
    assert "CL_adot" not in coeffs
    assert "Cm_adot" not in coeffs
    assert np.isfinite(quality["diagnostics"]["CL_adot"])
    assert np.isfinite(quality["diagnostics"]["Cm_adot"])

def test_drag_quadratic_alpha_coefficient_is_fitted(fitted):
    _, _, _, coeffs, _ = fitted
    assert "CD_alpha2" in coeffs
    assert np.isfinite(coeffs["CD_alpha2"])
    assert coeffs["CD_alpha2"] != pytest.approx(0.32, abs=1e-6)

def test_drag_r_squared_uses_only_supported_terms(fitted):
    _, _, quality, coeffs, data = fitted
    alpha, de, observed = data["alpha"], data["de"], data["CD"]
    predicted = (
        coeffs["CD0"] + coeffs["CD_alpha"] * alpha
        + coeffs["CD_alpha2"] * alpha**2 + coeffs["CD_de2"] * de**2
    )
    r_squared = 1.0 - np.sum((observed - predicted) ** 2) / np.sum(
        (observed - np.mean(observed)) ** 2
    )
    assert quality["CD"]["r_squared"] == pytest.approx(r_squared, abs=1e-12)

def test_sideslip_drag_diagnostic_does_not_inflate_applied_model_r_squared(fitted):
    _, _, quality, coeffs, _ = fitted
    diagnostics = quality["diagnostics"]
    assert "CD_abs_beta" in diagnostics
    assert "CD_r_squared_with_abs_beta" in diagnostics
    assert diagnostics["CD_r_squared_with_abs_beta"] >= quality["CD"]["r_squared"] - 1e-10
    assert "CD_abs_beta" not in coeffs

def test_fit_can_score_a_fixed_coefficient_set(fitted, plane):
    _, _, _, coeffs, data = fitted
    mp = plane.mass_properties()
    fixed, quality = fit(
        data, C.V_TRIM, mp["wing_span_m"], mp["mean_chord_m"],
        fixed_coefficients=coeffs,
    )
    assert fixed == pytest.approx(coeffs)
    alpha = data["alpha"]
    de = data["de"]
    observed = data["CL"]
    predicted = (fixed["CL0"] + fixed["CL_alpha"] * alpha
                 + fixed["CL_de"] * de
                 + fixed["CL_q"] * data["q"] * mp["mean_chord_m"] / (2 * C.V_TRIM))
    expected_r2 = 1.0 - np.sum((observed - predicted) ** 2) / np.sum(
        (observed - np.mean(observed)) ** 2
    )
    assert quality["CL"]["r_squared"] == pytest.approx(expected_r2, abs=1e-12)

def test_fit_rejects_incomplete_fixed_coefficient_set(fitted, plane):
    _, _, _, _, data = fitted
    mp = plane.mass_properties()
    with pytest.raises(ValueError, match="fixed_coefficients missing fitted terms"):
        fit(data, C.V_TRIM, mp["wing_span_m"], mp["mean_chord_m"], fixed_coefficients={})

def test_loaded_cg_is_weighted_from_empty_aircraft_and_payload():
    import xml.etree.ElementTree as ET

    root = ET.parse(model_builder.source_aircraft_file()).getroot()
    x_in, y_in, z_in = model_builder._loaded_cg_in(root)
    assert x_in == pytest.approx(45.3523, abs=0.02)
    assert y_in == pytest.approx(0.0, abs=1e-12)
    assert z_in == pytest.approx(32.4930, abs=0.02)

def test_generated_thrust_location_matches_loaded_cg():
    import xml.etree.ElementTree as ET

    build_root = model_builder.build_model()
    xml_path = build_root / "aircraft" / model_builder.MODEL_NAME / (
        model_builder.MODEL_NAME + ".xml"
    )
    root = ET.parse(xml_path).getroot()
    thrust = root.find("external_reactions/force[@name='thrust']/location")
    assert thrust is not None
    expected = model_builder._loaded_cg_in(root)
    actual = tuple(float(thrust.find(axis).text) for axis in "xyz")
    assert actual == pytest.approx(expected, abs=1e-9)
    assert actual != pytest.approx((41.0, 0.0, 36.5), abs=0.1)

def test_loaded_cg_remains_laterally_centered():
    import xml.etree.ElementTree as ET

    root = ET.parse(model_builder.source_aircraft_file()).getroot()
    assert model_builder._loaded_cg_in(root)[1] == pytest.approx(0.0, abs=1e-12)
    lateral_payload_moment = sum(weight * y for _, weight, _, y, _ in model_builder.PAYLOAD)
    assert lateral_payload_moment == pytest.approx(0.0, abs=1e-12)

def test_angular_frame_correction_removes_ecef_inertial_residual(plane):
    result = C.eom_check(plane, n=15, seed=7)
    raw = result["max_abs_angular_error_raw_rad_s2"]
    corrected = result["max_abs_angular_error_after_frame_correction_rad_s2"]
    assert max(raw) > 1e-5
    assert max(corrected) < 1e-10
    assert max(corrected) < max(raw) * 1e-4

def test_angular_frame_transform_reduces_to_euler_equation_without_earth_rate():
    omega = np.array([0.13, -0.08, 0.04])
    moment = np.array([13.0, -9.0, 4.0])
    inertia = np.array([[1200.0, 0.0, 12.0], [0.0, 1800.0, 0.0], [12.0, 0.0, 2500.0]])
    expected = np.linalg.solve(inertia, moment - np.cross(omega, inertia @ omega))
    actual = C.jsbsim_ecef_angular_acceleration(np.zeros(3), omega, moment, inertia)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

def test_all_deployed_fitted_coefficients_are_finite(fitted):
    _, _, _, coeffs, _ = fitted
    assert coeffs
    assert all(np.isfinite(value) for value in coeffs.values())

def test_aerodynamic_identification_is_reproducible(fitted, plane):
    _, _, _, coeffs, data = fitted
    mp = plane.mass_properties()
    repeat, _ = fit(data, C.V_TRIM, mp["wing_span_m"], mp["mean_chord_m"])
    assert repeat == pytest.approx(coeffs, rel=1e-12, abs=1e-12)

def test_phugoid_frequency_is_checked_after_alpha_dot_adjustment(plane, fitted):
    aircraft, env, _, _, _ = fitted
    result = C.trim_and_modes(plane, aircraft, env)
    jsb_mode = result["modes_jsbsim"]["phugoid"]
    repo_mode = result["modes_repo"]["phugoid"]
    assert repo_mode["wn_rad_s"] == pytest.approx(jsb_mode["wn_rad_s"], rel=0.06)

def test_phugoid_damping_is_checked_after_alpha_dot_adjustment(plane, fitted):
    aircraft, env, _, _, _ = fitted
    result = C.trim_and_modes(plane, aircraft, env)
    jsb_mode = result["modes_jsbsim"]["phugoid"]
    repo_mode = result["modes_repo"]["phugoid"]
    assert repo_mode["zeta"] == pytest.approx(jsb_mode["zeta"], rel=0.15)

def test_alpha_dot_ab_keeps_repository_coefficients_fixed(plane):
    from run_comparison import pipeline

    stock = pipeline(False, with_cases=False)
    no_adot = pipeline(True, with_cases=False,
                       common_coefficients=stock["fitted_coefficients"])
    assert no_adot["fitted_coefficients"] == pytest.approx(stock["fitted_coefficients"])
    assert no_adot["level2"]["modes_repo"]["phugoid"]["wn_rad_s"] == pytest.approx(
        stock["level2"]["modes_repo"]["phugoid"]["wn_rad_s"], abs=1e-12
    )
    assert "independent_refit_diagnostic" in no_adot
    assert no_adot["independent_refit_diagnostic"]["note"].find("shared stock-fit") < 0

def test_alpha_dot_disabled_diagnostic_changes_reference_not_repo_model(plane):
    from run_comparison import pipeline

    stock = pipeline(False, with_cases=False)
    diagnostic = pipeline(True, with_cases=False,
                          common_coefficients=stock["fitted_coefficients"])
    assert diagnostic["fitted_coefficients"] == pytest.approx(stock["fitted_coefficients"])
    assert diagnostic["level2"]["modes_jsbsim"]["short_period"]["zeta"] != pytest.approx(
        stock["level2"]["modes_jsbsim"]["short_period"]["zeta"], abs=1e-4
    )

def test_elevator_doublet_pitch_rate_error_has_absolute_scale(plane, fitted):
    aircraft, env, _, _, _ = fitted
    result = C.trim_and_modes(plane, aircraft, env)
    case = C.CASES["elevator_doublet"]
    jsb_case = C.run_jsbsim_case(plane, result["_trim_jsbsim"], case["du"], 12.0, 0.01)
    repo_case = C.run_repo_case(aircraft, env, result["_trim_repo"], case["du"], 12.0, 0.01)
    row = C.compare_case("elevator_doublet", jsb_case, repo_case)["primary"]["q"]
    assert 0.0 < row["max_abs_diff"] < 0.02
    assert 0.0 < row["rms_diff"] < row["max_abs_diff"]

def test_response_percentages_are_normalized_by_peak_change():
    jc = {"p": np.array([0.0, 1.0, 0.0])}
    rc = {"p": np.array([0.0, 0.8, 0.0])}
    result = C.compare_case("aileron_pulse", jc, rc)["primary"]["p"]
    assert result["jsbsim_peak_change"] == pytest.approx(1.0)
    assert result["max_diff_pct_of_peak"] == pytest.approx(20.0)
    assert result["max_abs_diff"] == pytest.approx(0.2)

def test_report_writer_exposes_absolute_response_scales(tmp_path):
    from report import write

    summary = json.loads((CMP / "results" / "summary.json").read_text())
    output = tmp_path / "REPORT.md"
    write(summary, output)
    report = output.read_text()
    assert "Maximum absolute error" in report
    assert "RMS absolute error" in report
    assert "body-axis forward velocity" in report
    assert "peak change" in report

def test_report_writer_labels_unsupported_fit_terms(tmp_path):
    from report import write

    summary = json.loads((CMP / "results" / "summary.json").read_text())
    output = tmp_path / "REPORT.md"
    write(summary, output)
    report = output.read_text()
    assert "diagnostic only" in report
    assert "does not independently validate either aerodynamic model" in report
    assert "CD_alpha2" in report

def test_loaded_cg_satisfies_weighted_moment_equilibrium():
    import xml.etree.ElementTree as ET

    root = ET.parse(model_builder.source_aircraft_file()).getroot()
    mass = root.find("mass_balance")
    empty_weight = float(mass.find("emptywt").text)
    empty_cg_node = mass.find("location[@name='CG']")
    empty_cg = np.array([float(empty_cg_node.find(axis).text) for axis in "xyz"])
    loaded_cg = np.array(model_builder._loaded_cg_in(root))
    payload_weight = sum(item[1] for item in model_builder.PAYLOAD)
    total_weight = empty_weight + payload_weight
    weighted_station = empty_weight * empty_cg
    for _, weight, x, y, z in model_builder.PAYLOAD:
        weighted_station += weight * np.array([x, y, z])
    np.testing.assert_allclose(total_weight * loaded_cg, weighted_station, atol=1e-8)

def test_loaded_cg_rejects_missing_empty_weight():
    import xml.etree.ElementTree as ET

    root = ET.parse(model_builder.source_aircraft_file()).getroot()
    root.find("mass_balance").remove(root.find("mass_balance").find("emptywt"))
    with pytest.raises(RuntimeError, match="empty weight"):
        model_builder._loaded_cg_in(root)
