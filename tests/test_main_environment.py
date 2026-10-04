import numpy as np

from main import environment


def test_main_environment_gravity_points_down():
    gravity = environment(0.0).gravity_ned_m_s2

    np.testing.assert_allclose(
        gravity,
        np.array([0.0, 0.0, 9.806]),
        atol=1e-12,
    )
