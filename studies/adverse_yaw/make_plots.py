"""Static engineering plots for the adverse yaw study."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import study_setup as S

RED = "#c62828"     # no rudder
GREEN = "#2e7d32"   # rudder coordinated
INK = "#111111"
GRID = "#dddddd"


def _style(ax, ylabel=None):
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    for s in ax.spines.values():
        s.set_color("#888888")


def _roll_window(ax):
    ax.axvspan(S.T_ROLL_ON_S, S.T_ROLL_OFF_S, color="#000000", alpha=0.06, lw=0)


def plot_main(A, B, path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    t = A.t

    ax = axes[0]
    ax.plot(t, np.degrees(A.act[:, 0]), color=INK, lw=1.6, label="aileron (both cases, identical)")
    ax.plot(t, np.degrees(A.act[:, 2]), color=RED, lw=1.6, label="rudder, case A (none)")
    ax.plot(t, np.degrees(B.act[:, 2]), color=GREEN, lw=1.6, label=f"rudder, case B (k = {B.k_ari:.2f})")
    ax.set_title("Control surface deflections (actual, after actuator lag)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="center right")
    _style(ax, "deg")

    ax = axes[1]
    ax.plot(t, A.rates_deg[:, 2], color=RED, lw=1.8, label="A: no rudder")
    ax.plot(t, B.rates_deg[:, 2], color=GREEN, lw=1.8, label="B: rudder coordinated")
    ax.axhline(0, color="#666666", lw=0.8)
    iA = int(np.argmin(A.rates_deg[:, 2]))
    ax.annotate(f"adverse yaw: nose swings the WRONG way\n{A.rates_deg[iA, 2]:.1f} deg/s at t = {t[iA]:.1f} s",
                xy=(t[iA], A.rates_deg[iA, 2]), xytext=(t[iA] + 1.6, A.rates_deg[iA, 2] - 0.2),
                fontsize=8.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_title("Yaw rate r   (positive = nose right, the direction of the turn)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    _style(ax, "deg/s")

    ax = axes[2]
    ax.plot(t, A.beta_deg, color=RED, lw=1.8, label="A: no rudder")
    ax.plot(t, B.beta_deg, color=GREEN, lw=1.8, label="B: rudder coordinated")
    ax.axhline(0, color="#666666", lw=0.8)
    iA, iB = int(np.argmax(np.abs(A.beta_deg))), int(np.argmax(np.abs(B.beta_deg)))
    ax.annotate(f"peak sideslip {A.beta_deg[iA]:+.1f} deg", xy=(t[iA], A.beta_deg[iA]),
                xytext=(t[iA] + 1.4, A.beta_deg[iA] - 0.3), fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate(f"{B.beta_deg[iB]:+.1f} deg", xy=(t[iB], B.beta_deg[iB]),
                xytext=(t[iB] + 1.2, B.beta_deg[iB] + 0.9), fontsize=8.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_title("Sideslip angle beta   (positive = airflow from the right of the nose)", fontsize=10, loc="left")
    ax.set_xlabel("time (s)", fontsize=9)
    _style(ax, "deg")

    for ax in axes:
        _roll_window(ax)
    fig.suptitle("Adverse yaw: same right-aileron pulse, with and without rudder\n(shaded band = aileron applied)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roll_response(A, B, path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
    t = A.t

    ax = axes[0]
    ax.plot(t, A.rates_deg[:, 0], color=RED, lw=1.8, label="A: no rudder")
    ax.plot(t, B.rates_deg[:, 0], color=GREEN, lw=1.8, label="B: rudder coordinated")
    ax.set_title("Roll rate p", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    _style(ax, "deg/s")

    ax = axes[1]
    ax.plot(t, A.euler_deg[:, 0], color=RED, lw=1.8, label=f"A: max bank {A.euler_deg[:, 0].max():.1f} deg")
    ax.plot(t, B.euler_deg[:, 0], color=GREEN, lw=1.8, label=f"B: max bank {B.euler_deg[:, 0].max():.1f} deg")
    ax.set_title("Bank angle phi: the sideslip costs roll authority", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="center right")
    _style(ax, "deg")

    ax = axes[2]
    ax.plot(t, A.cl_terms["aileron  Cl_da*da"], color=INK, lw=1.6, label="aileron roll moment (same in A and B)")
    ax.plot(t, A.cl_terms["dihedral  Cl_beta*beta"], color=RED, lw=1.8, label="dihedral effect Cl_beta*beta, A")
    ax.plot(t, B.cl_terms["dihedral  Cl_beta*beta"], color=GREEN, lw=1.8, label="dihedral effect Cl_beta*beta, B")
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_title("Why: sideslip produces a rolling moment that opposes the roll", fontsize=10, loc="left")
    ax.set_xlabel("time (s)", fontsize=9)
    ax.legend(fontsize=8)
    _style(ax, "moment coefficient Cl")

    for ax in axes:
        _roll_window(ax)
    fig.suptitle("Roll response", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_yaw_moment_breakdown(A, B, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    styles = {
        "aileron  Cn_da*da": ("#1f77b4", "-"),
        "roll rate  Cn_p*p^": ("#9467bd", "-"),
        "sideslip  Cn_beta*beta": ("#ff7f0e", "-"),
        "yaw rate  Cn_r*r^": ("#8c564b", "-"),
        "rudder  Cn_dr*dr": (GREEN, "-"),
        "total Cn": (INK, "-"),
    }
    for ax, c, title in ((axes[0], A, "A: no rudder"), (axes[1], B, f"B: rudder coordinated (k = {B.k_ari:.2f})")):
        m = c.t <= 7.0
        for label, series in c.cn_terms.items():
            col, ls = styles[label]
            ax.plot(c.t[m], series[m], color=col, ls=ls, lw=2.6 if label == "total Cn" else 1.4, label=label)
        ax.axhline(0, color="#666666", lw=0.8)
        _roll_window(ax)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xlabel("time (s)", fontsize=9)
        _style(ax)
    axes[0].set_ylabel("yaw moment coefficient contribution Cn", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Where the yaw moment comes from: aileron + roll rate push the nose the wrong way first; "
                 "sideslip then pushes back", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ground_track(A, B, path):
    fig = plt.figure(figsize=(9, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.7, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0])
    for c, col in ((A, RED), (B, GREEN)):
        N, E = c.state[:, 0], c.state[:, 1]
        ax.plot(E, N, color=col, lw=1.6, label=c.name)
        step = int(round(1.0 / S.DT_S))
        ax.plot(E[::step], N[::step], "o", color=col, ms=3.5)
    ax.set_aspect("equal")
    ax.set_xlabel("East (m)", fontsize=9)
    ax.set_ylabel("North (m)", fontsize=9)
    ax.set_title("Ground track, top view (dot = position each second)", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    _style(ax)

    ax = fig.add_subplot(gs[1])
    for c, col in ((A, RED), (B, GREEN)):
        ax.plot(c.t, -c.state[:, 2], color=col, lw=1.6, label=f"{c.name}: altitude")
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("altitude (m)", fontsize=9)
    ax.set_title("Altitude: no pitch correction was applied, so the banked turn descends", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    _style(ax)
    fig.suptitle("Where the aircraft actually went", fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gain_sweep(rows, k_best, path):
    k = np.array([r["k"] for r in rows])
    peak = np.array([r["peak_abs_beta"] for r in rows])
    bmax = np.array([r["max_beta"] for r in rows])
    bmin = np.array([r["min_beta"] for r in rows])
    dip = np.array([r["yaw_dip"] for r in rows])
    bank = np.array([r["max_bank"] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax = axes[0]
    ax.plot(k, bmax, color="#ff7f0e", lw=1.4, label="max sideslip (nose lags the turn)")
    ax.plot(k, bmin, color="#1f77b4", lw=1.4, label="min sideslip (nose leads the turn)")
    ax.plot(k, peak, color=INK, lw=2.4, label="peak |sideslip|")
    ax.axvline(k_best, color=GREEN, ls="--", lw=1.4)
    ax.annotate(f"best k = {k_best:.2f}\npeak |beta| = {peak[np.argmin(peak)]:.2f} deg",
                xy=(k_best, peak.min()), xytext=(k_best + 0.15, peak.min() + 2.2), fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.text(0.03, 0.9, "too little rudder:\nadverse yaw", transform=ax.transAxes, fontsize=8.5, color=RED, va="top")
    ax.text(0.97, 0.9, "too much rudder:\nover-coordinated", transform=ax.transAxes, fontsize=8.5, color="#1f77b4",
            va="top", ha="right")
    ax.set_title("Sideslip vs. rudder coordination gain (rudder = -k x aileron)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.03, 0.42))
    _style(ax, "deg")

    ax = axes[1]
    ax.plot(k, dip, color=RED, lw=2)
    ax.axvline(k_best, color=GREEN, ls="--", lw=1.4)
    ax.set_title("Adverse yaw: size of the wrong-way yaw-rate dip", fontsize=10, loc="left")
    _style(ax, "deg/s")

    ax = axes[2]
    ax.plot(k, bank, color=INK, lw=2)
    ax.axvline(k_best, color=GREEN, ls="--", lw=1.4)
    ax.set_title("Max bank angle from the identical aileron pulse", fontsize=10, loc="left")
    ax.set_xlabel("rudder coordination gain k", fontsize=9)
    _style(ax, "deg")
    fig.suptitle("Finding the right amount of rudder", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_slip_ball(A, B, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax = axes[0]
    ax.plot(A.t, A.n_y_g, color=RED, lw=1.8, label="A: no rudder")
    ax.plot(B.t, B.n_y_g, color=GREEN, lw=1.8, label="B: rudder coordinated")
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_title("Lateral specific force (in g): what a slip ball / inclinometer reads", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    _roll_window(ax)
    _style(ax, "g")
    ax = axes[1]
    ax.plot(A.t, -A.n_y_g, color=RED, lw=1.8)
    ax.plot(B.t, -B.n_y_g, color=GREEN, lw=1.8)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_title("Ball displacement (positive = ball rolls right)", fontsize=10, loc="left")
    ax.set_xlabel("time (s)", fontsize=9)
    _roll_window(ax)
    _style(ax, "~ball offset")
    fig.suptitle("The slip ball is the pilot's adverse-yaw detector", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_modes(modes, A, path):
    """Lateral-directional poles from the Jacobian, and a check that the nonlinear
    simulation rings at the period the linearization predicts."""
    ev = np.array(modes["eigenvalues"])
    dr = modes["dutch_roll"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.5]})

    ax = axes[0]
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.8)
    lim = 1.3 * abs(dr["wn_rad_s"])
    for z in (0.1, 0.234, 0.5, 0.707):
        a = np.arccos(z)
        ax.plot([0, -lim * np.cos(a)], [0, lim * np.sin(a)], color="#cccccc", lw=0.8, ls=":")
        ax.plot([0, -lim * np.cos(a)], [0, -lim * np.sin(a)], color="#cccccc", lw=0.8, ls=":")
    ax.plot(ev.real, ev.imag, "x", color=RED, ms=11, mew=2.5)
    ax.annotate(f"Dutch roll\nzeta = {dr['zeta']:.3f}\nperiod {dr['period_s']:.2f} s",
                xy=(dr["lambda"].real, dr["lambda"].imag), xytext=(dr["lambda"].real - 4.6, dr["lambda"].imag - 0.2),
                fontsize=9, arrowprops=dict(arrowstyle="->"))
    real = [e for e in ev if abs(e.imag) < 1e-6]
    for e in real:
        name = "roll subsidence" if e.real < -1 else ("spiral (unstable)" if e.real > 0 else "spiral")
        ax.annotate(name, xy=(e.real, 0), xytext=(e.real, 0.7 if e.real < 0 else -0.9), fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-lim * 2.2, lim * 0.35)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("real part (1/s)", fontsize=9)
    ax.set_ylabel("imaginary part (rad/s)", fontsize=9)
    ax.set_title("Lateral-directional poles at trim (dotted: constant damping ratio)", fontsize=10, loc="left")
    _style(ax)

    ax = axes[1]
    r = A.rates_deg[:, 2]
    ax.plot(A.t, r, color=RED, lw=1.8, label="case A yaw rate (nonlinear simulation)")
    ax.axhline(0, color="#666666", lw=0.8)
    pk = np.where((r[1:-1] > r[:-2]) & (r[1:-1] >= r[2:]))[0] + 1
    pk = pk[A.t[pk] > 3.0]
    t0 = A.t[pk[0]]
    for k in range(0, 6):
        ax.axvline(t0 + k * dr["period_s"], color="#555555", lw=1.0, ls="--",
                   label=f"linearization: Dutch roll period {dr['period_s']:.2f} s" if k == 0 else None)
    ax.plot(A.t[pk], r[pk], "o", color=RED)
    ax.set_xlim(0, 10)
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("deg/s", fontsize=9)
    meas = np.diff(A.t[pk])
    ax.set_title(f"Simulation rings at the predicted period (measured first interval {meas[0]:.2f} s)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    _style(ax)
    fig.suptitle("Simulation vs. its own linearization: same Dutch roll period", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(A, B, sweep_rows, k_best, out: Path, modes=None):
    out.mkdir(parents=True, exist_ok=True)
    plot_main(A, B, out / "01_adverse_yaw_main.png")
    plot_roll_response(A, B, out / "02_roll_response_and_lost_authority.png")
    plot_yaw_moment_breakdown(A, B, out / "03_yaw_moment_breakdown.png")
    plot_ground_track(A, B, out / "04_ground_track_and_altitude.png")
    plot_gain_sweep(sweep_rows, k_best, out / "05_rudder_gain_sweep.png")
    plot_slip_ball(A, B, out / "06_slip_ball_lateral_accel.png")
    if modes is not None:
        plot_modes(modes, A, out / "07_lateral_modes_check.png")
