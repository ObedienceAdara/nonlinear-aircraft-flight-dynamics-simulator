"""Run the Chapter 02 Euler attitude-kinematics demonstrations."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aircraft6dof.kinematics import euler_angle_rates
from aircraft6dof.integrators import rk4_step


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "chapter_02_results.csv"


def simulate(initial_angles, body_rates, duration=20.0, dt=0.01):
    """Propagate Euler angles under constant body angular rates."""
    angles = np.asarray(initial_angles, dtype=float)
    rates = np.asarray(body_rates, dtype=float)

    if angles.shape != (3,) or rates.shape != (3,):
        raise ValueError("initial_angles and body_rates must each have shape (3,).")

    steps = int(round(duration / dt))
    times = np.linspace(0.0, steps * dt, steps + 1)
    history = np.empty((steps + 1, 6))
    history[0, :3] = angles
    history[0, 3:] = rates

    def derivative(x):
        phi, theta, psi = x
        return euler_angle_rates(phi, theta, *rates)

    for i in range(steps):
        angles = rk4_step(angles, dt, derivative)
        history[i + 1, :3] = angles
        history[i + 1, 3:] = rates

    return times, history


def main() -> None:
    cases = {
        "pure_roll": (np.deg2rad([0.0, 0.0, 0.0]), np.array([0.05, 0.0, 0.0])),
        "coupled_motion": (
            np.deg2rad([5.0, 10.0, -15.0]),
            np.array([0.03, 0.02, 0.04]),
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case", "time_s", "phi_rad", "theta_rad", "psi_rad",
            "p_rad_s", "q_rad_s", "r_rad_s"
        ])

        for name, (initial, rates) in cases.items():
            times, history = simulate(initial, rates)
            for t, row in zip(times, history):
                writer.writerow([name, t, *row])

            print(f"{name}:")
            print(f"  initial attitude (deg): {np.rad2deg(history[0, :3])}")
            print(f"  final attitude   (deg): {np.rad2deg(history[-1, :3])}")

    print(f"Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
