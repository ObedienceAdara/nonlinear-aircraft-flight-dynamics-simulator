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
    aircraft, env, _ = fitted
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
