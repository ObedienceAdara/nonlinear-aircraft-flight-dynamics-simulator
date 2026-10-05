"""Fast checks on the adverse yaw study (short runs, no plots or video)."""

import sys
from pathlib import Path

import numpy as np
import pytest

STUDY = Path(__file__).resolve().parents[1] / "studies" / "adverse_yaw"
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

import study_setup as S  # noqa: E402

SHORT_S = 6.0


@pytest.fixture(scope="module")
def cases():
    a = S.run_case("A", 0.0, duration_s=SHORT_S)
    b = S.run_case("B", S.BEST_K_DEFAULT, duration_s=SHORT_S)
    return a, b


def test_trim_is_a_steady_state():
    aircraft = S.build_aircraft()
    env = S.make_environment()
    trim = S.trim_level_flight(aircraft, env)
    d = aircraft.derivative(S.initial_state(trim),
                            S.make_controls(trim, 0.0)(0.0), env)
    assert abs(d.velocity_body_m_s[0]) < 1e-6
    assert abs(d.velocity_body_m_s[2]) < 1e-6
    assert abs(d.omega_body_rad_s[1]) < 1e-6


def test_aileron_only_yaws_against_the_roll(cases):
    a, _ = cases
    # right aileron -> positive roll rate, but the nose first goes left (r < 0)
    assert a.rates_deg[:, 0].max() > 5.0
    assert a.rates_deg[:, 2].min() < -1.0
    # and sideslip builds up
    assert np.abs(a.beta_deg).max() > 2.0


def test_rudder_interconnect_cuts_sideslip(cases):
    a, b = cases
    assert np.abs(b.beta_deg).max() < 0.5 * np.abs(a.beta_deg).max()


def test_repo_default_cn_da_is_proverse():
    """Documents why the study overrides Cn_da: the repo default has no adverse yaw."""
    assert S.repo_main.build_aircraft().parameters.aero.Cn_da > 0.0
    assert S.CN_DA_ADVERSE < 0.0


def test_quaternion_stays_unit_length(cases):
    for c in cases:
        assert np.allclose(np.linalg.norm(c.state[:, 9:13], axis=1), 1.0, atol=1e-9)


def test_study_gravity_points_down():
    g = S.make_environment().gravity_ned_m_s2
    assert g[0] == 0.0 and g[1] == 0.0 and g[2] > 9.0
