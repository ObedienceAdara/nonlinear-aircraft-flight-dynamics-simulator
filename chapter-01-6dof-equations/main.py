"""Run the Chapter 01 6-DOF rigid-body dynamics demonstration."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aircraft6dof.dynamics import equations_of_motion
from aircraft6dof.integrators import rk4_step


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "chapter_01_results.csv"


def main() -> None:
    mass = 1_200.0  # kg

    # A realistic rigid body can have products of inertia. Chapter 01 keeps
    # the example diagonal so the equation structure stays transparent.
    inertia = np.diag(
        [
            1_800.0,  # Ix, kg m^2
            2_100.0,  # Iy, kg m^2
            3_300.0,  # Iz, kg m^2
        ]
    )

    # State = [u, v, w, p, q, r]
    state = np.array(
        [
            65.0,  # m/s
            2.0,   # m/s
            1.0,   # m/s
            0.02,  # rad/s
            0.01,  # rad/s
            0.03,  # rad/s
        ],
        dtype=float,
    )

    # Constant external body loads.
    # These are intentionally simple: Chapter 01 demonstrates equations of
    # motion, not a full aerodynamic or propulsion model.
    force_body = np.array([500.0, 50.0, -120.0], dtype=float)  # N
    moment_body = np.array([80.0, 140.0, 60.0], dtype=float)   # N m

    duration = 20.0
    dt = 0.01

    derivative = lambda x: equations_of_motion(
        x,
        mass=mass,
        inertia=inertia,
        force_body=force_body,
        moment_body=moment_body,
    )

    steps = int(round(duration / dt))
    times = np.linspace(0.0, steps * dt, steps + 1)
    states = np.empty((steps + 1, 6), dtype=float)
    states[0] = state

    for i in range(steps):
        state = rk4_step(state, dt, derivative)
        states[i + 1] = state

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "u_mps", "v_mps", "w_mps", "p_rad_s", "q_rad_s", "r_rad_s"])
        writer.writerows(np.column_stack((times, states)))

    print("Chapter 01 simulation complete.")
    print(f"Samples: {len(times)}")
    print(f"CSV: {OUTPUT_CSV}")
    print()
    print("Initial state:")
    print(states[0])
    print()
    print("Final state:")
    print(states[-1])


if __name__ == "__main__":
    main()
