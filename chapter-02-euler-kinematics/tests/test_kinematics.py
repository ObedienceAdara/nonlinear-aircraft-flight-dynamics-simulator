import numpy as np
import pytest

from aircraft6dof.frames import body_to_navigation_dcm, navigation_to_body_dcm
from aircraft6dof.kinematics import euler_angle_rates, euler_rate_matrix


def test_zero_body_rates_give_zero_euler_rates():
    result = euler_angle_rates(0.4, 0.3, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(result, np.zeros(3))


def test_pure_roll_at_zero_pitch():
    p = 0.2
    result = euler_angle_rates(0.7, 0.0, p, 0.0, 0.0)
    np.testing.assert_allclose(result, [p, 0.0, 0.0])


def test_matrix_form_matches_direct_equations():
    phi, theta = 0.35, -0.25
    rates = np.array([0.12, -0.07, 0.09])
    direct = euler_angle_rates(phi, theta, *rates)
    matrix = euler_rate_matrix(phi, theta)
    np.testing.assert_allclose(direct, matrix @ rates)


def test_pitch_singularity_is_rejected():
    with pytest.raises(ValueError, match="singularity"):
        euler_angle_rates(0.0, np.pi / 2, 0.0, 1.0, 0.0)


def test_dcm_is_orthonormal():
    c = body_to_navigation_dcm(0.2, -0.3, 0.7)
    np.testing.assert_allclose(c.T @ c, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(c), 1.0, atol=1e-12)


def test_dcm_round_trip():
    c_bn = body_to_navigation_dcm(0.2, -0.3, 0.7)
    c_nb = navigation_to_body_dcm(0.2, -0.3, 0.7)

    vector_body = np.array([12.0, -4.0, 2.5])
    vector_ned = c_bn @ vector_body
    reconstructed = c_nb @ vector_ned

    np.testing.assert_allclose(reconstructed, vector_body, atol=1e-12)
