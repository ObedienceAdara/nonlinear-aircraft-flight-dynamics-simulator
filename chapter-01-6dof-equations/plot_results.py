"""Plot Chapter 01 simulation results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "outputs" / "chapter_01_results.csv"
OUTPUT_PNG = ROOT / "outputs" / "chapter_01_dynamics.png"


def load_results(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run 'python main.py' first."
        )

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    header = rows[0]
    data = np.asarray(rows[1:], dtype=float)

    if header != ["time_s", "u_mps", "v_mps", "w_mps", "p_rad_s", "q_rad_s", "r_rad_s"]:
        raise ValueError("Unexpected CSV header.")
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError("Unexpected CSV shape.")

    return data[:, 0], data[:, 1:]


def main() -> None:
    time, states = load_results(INPUT_CSV)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(time, states[:, 0], label="u")
    axes[0].plot(time, states[:, 1], label="v")
    axes[0].plot(time, states[:, 2], label="w")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Chapter 01 — Body-Frame Translational Dynamics")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time, states[:, 3], label="p")
    axes[1].plot(time, states[:, 4], label="q")
    axes[1].plot(time, states[:, 5], label="r")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular rate (rad/s)")
    axes[1].set_title("Chapter 01 — Body-Frame Rotational Dynamics")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.show()

    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
