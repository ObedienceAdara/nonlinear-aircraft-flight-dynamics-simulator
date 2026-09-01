"""Visualize Chapter 03 air-data and aerodynamic loads."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "outputs" / "chapter_03_results.csv"
OUTPUT_PNG = ROOT / "outputs" / "chapter_03_flight_environment.png"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError("Run 'python main.py' before plotting.")

    values = {}
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[row["quantity"]] = float(row["value"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    axes[0].bar(
        ["TAS (m/s)", "q∞ / 1000 (Pa/1000)", "Mach", "α (deg)", "β (deg)"],
        [
            values["true_airspeed_m_s"],
            values["dynamic_pressure_pa"] / 1000.0,
            values["mach"],
            values["alpha_deg"],
            values["beta_deg"],
        ],
    )
    axes[0].set_title("Chapter 03 — Air Data")
    axes[0].grid(axis="y")

    axes[1].bar(
        ["X (N)", "Y (N)", "Z (N)", "L (N·m)", "M (N·m)", "N (N·m)"],
        [
            values["X_body_N"], values["Y_body_N"], values["Z_body_N"],
            values["L_roll_Nm"], values["M_pitch_Nm"], values["N_yaw_Nm"],
        ],
    )
    axes[1].set_title("Chapter 03 — Aerodynamic Loads")
    axes[1].grid(axis="y")

    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
