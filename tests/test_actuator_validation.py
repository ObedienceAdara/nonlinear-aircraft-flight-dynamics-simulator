import pytest
from aircraft6dof.actuators import ActuatorChannel


def test_rejects_non_positive_time_constant():
    with pytest.raises(ValueError):
        ActuatorChannel(time_constant_s=0.0, rate_limit_rad_s=1.0, position_limit_rad=0.4)


def test_rejects_non_positive_limits():
    with pytest.raises(ValueError):
        ActuatorChannel(time_constant_s=0.1, rate_limit_rad_s=-1.0, position_limit_rad=0.4)
    with pytest.raises(ValueError):
        ActuatorChannel(time_constant_s=0.1, rate_limit_rad_s=1.0, position_limit_rad=0.0)


def test_rejects_non_positive_dt():
    ch = ActuatorChannel(time_constant_s=0.1, rate_limit_rad_s=1.0, position_limit_rad=0.4)
    with pytest.raises(ValueError):
        ch.step(0.1, 0.0)


def test_step_respects_position_limit():
    ch = ActuatorChannel(time_constant_s=0.05, rate_limit_rad_s=100.0, position_limit_rad=0.4)
    for _ in range(500):
        ch.step(2.0, 0.01)
    assert ch.position_rad == pytest.approx(0.4, abs=1e-6)
