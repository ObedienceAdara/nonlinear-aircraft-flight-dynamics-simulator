"""Plot Chapter 02 Euler-angle simulation results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "outputs" / "chapter_02_results.csv"
OUTPUT_PNG = ROOT / "outputs" / "chapter_02_euler_kinematics.png"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError("Run 'python main.py' before plotting.")

    by_case = {}
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            by_case.setdefault(row["case"], []).append(row)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for case, rows in by_case.items():
        t = np.array([float(r["time_s"]) for r in rows])
        angles = np.rad2deg(np.array([
            [float(r["phi_rad"]), float(r["theta_rad"]), float(r["psi_rad"])]
            for r in rows
        ]))
        rates = np.rad2deg(np.array([
            [float(r["p_rad_s"]), float(r["q_rad_s"]), float(r["r_rad_s"])]
            for r in rows
        ]))

        axes[0].plot(t, angles[:, 0], label=f"{case}: phi")
        axes[0].plot(t, angles[:, 1], label=f"{case}: theta")
        axes[0].plot(t, angles[:, 2], label=f"{case}: psi")
        axes[1].plot(t, rates[:, 0], label=f"{case}: p")
        axes[1].plot(t, rates[:, 1], label=f"{case}: q")
        axes[1].plot(t, rates[:, 2], label=f"{case}: r")

    axes[0].set_ylabel("Euler angle (deg)")
    axes[0].set_title("Chapter 02 — Euler Attitude Kinematics")
    axes[0].grid(True)
    axes[0].legend(ncols=2)

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Body rate (deg/s)")
    axes[1].grid(True)
    axes[1].legend(ncols=2)

    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
