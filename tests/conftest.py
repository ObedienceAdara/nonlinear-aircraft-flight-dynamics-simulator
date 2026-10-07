"""Shared fixtures: the repo's generic aircraft in still air at 1000 m."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main as repo_main  # noqa: E402
from aircraft6dof import Environment  # noqa: E402
from aircraft6dof.atmosphere import standard_atmosphere  # noqa: E402


@pytest.fixture(scope="session")
def aircraft():
    return repo_main.build_aircraft()


@pytest.fixture(scope="session")
def still_air():
    atm = standard_atmosphere(1000.0)
    return Environment(
        density_kg_m3=atm.density_kg_m3,
        speed_of_sound_m_s=atm.speed_of_sound_m_s,
        gravity_ned_m_s2=np.array([0.0, 0.0, 9.80665]),
    )