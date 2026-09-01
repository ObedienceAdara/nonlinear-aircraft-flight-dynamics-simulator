import numpy as np

from aircraft6dof.frames import body_to_navigation_dcm, navigation_to_body_dcm


def test_dcm_round_trip():
    c_bn = body_to_navigation_dcm(0.2, -0.1, 0.7)
    c_nb = navigation_to_body_dcm(0.2, -0.1, 0.7)
    vector = np.array([4.0, -2.0, 1.0])
    np.testing.assert_allclose(c_nb @ (c_bn @ vector), vector, atol=1e-12)


def test_dcm_is_rotation_matrix():
    c = body_to_navigation_dcm(0.2, -0.1, 0.7)
    np.testing.assert_allclose(c.T @ c, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(c), 1.0, atol=1e-12)
