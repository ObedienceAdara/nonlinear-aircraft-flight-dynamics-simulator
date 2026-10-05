"""Side-by-side 3D animation: identical right-aileron pulse, with and without
rudder coordination. Same aircraft mesh style as the earlier gimbal-lock work,
now driven by the repo's own RK4 6-DOF integration instead of a kinematic demo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import study_setup as S
from aircraft6dof.mathutils import dcm_body_to_ned_from_quat

RED = "#c62828"
GREEN = "#2e7d32"
B_AXES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])   # body (fwd,right,down) -> plot (fwd,right,up)


def aircraft_faces():
    faces = []

    def box(cx, cy, cz, dx, dy, dz):
        x0, x1 = cx - dx / 2, cx + dx / 2
        y0, y1 = cy - dy / 2, cy + dy / 2
        z0, z1 = cz - dz / 2, cz + dz / 2
        v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
             [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
        idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
        return [[v[i] for i in f] for f in idx]

    faces += box(0, 0, 0, 1.6, 0.22, 0.22)
    faces += box(-0.05, 0, 0, 0.85, 3.0, 0.05)
    faces += box(-0.82, 0, 0, 0.35, 1.25, 0.045)
    faces += box(-0.82, 0, 0.38, 0.35, 0.045, 0.55)
    tip = [1.3, 0, 0]
    base = [[0.8, -0.11, -0.11], [0.8, 0.11, -0.11], [0.8, 0.11, 0.11], [0.8, -0.11, 0.11]]
    for i in range(4):
        faces.append([base[i], base[(i + 1) % 4], tip])
    return faces


FACES_LOCAL = [np.array(f) for f in aircraft_faces()]


def draw_aircraft(ax, R, offset, color):
    faces = [(f @ R.T) + offset for f in FACES_LOCAL]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor="black", linewidths=0.5))


def style_3d(ax, title, color):
    r = 2.0
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_facecolor("white")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 1))
        a._axinfo["grid"]["color"] = (0, 0, 0, 0.08)
    ax.tick_params(colors="black", labelsize=6)
    ax.set_title(title, color=color, fontsize=11, fontweight="bold")


def make_video(A, B, path="adverse_yaw_3d.mp4", fps=30):
    stride = max(1, int(round(1 / (fps * S.DT_S))))
    idxs = np.arange(0, len(A.t), stride)

    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = fig.add_subplot(2, 1, 2)

    ax3.set_facecolor("white")
    ax3.grid(color="0.85", lw=0.6)
    ax3.set_xlim(0, A.t[-1])
    ylim = 1.15 * max(np.abs(A.rates_deg[:, 2]).max(), np.abs(A.beta_deg).max(),
                       np.abs(B.rates_deg[:, 2]).max(), np.abs(B.beta_deg).max())
    ax3.set_ylim(-ylim, ylim)
    ax3.axhline(0, color="0.5", lw=0.8)
    ax3.axvspan(S.T_ROLL_ON_S, S.T_ROLL_OFF_S, color="black", alpha=0.06, lw=0)
    ax3.set_xlabel("time (s)")
    l_rA, = ax3.plot([], [], color=RED, lw=1.8, ls="-", label="yaw rate r, A")
    l_bA, = ax3.plot([], [], color=RED, lw=1.4, ls="--", label="sideslip beta, A")
    l_rB, = ax3.plot([], [], color=GREEN, lw=1.8, ls="-", label="yaw rate r, B")
    l_bB, = ax3.plot([], [], color=GREEN, lw=1.4, ls="--", label="sideslip beta, B")
    cursor = ax3.axvline(0, color="black", lw=1.0, alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8, ncol=2)
    ax3.set_title("Yaw rate (deg/s, solid) and sideslip (deg, dashed)", fontsize=10, loc="left")

    warn_txt = fig.text(0.5, 0.965, "", ha="center", fontsize=12, fontweight="bold", color="black")

    def render(i):
        idx = idxs[i]
        t = A.t[idx]

        ax1.cla(); ax2.cla()
        for ax, c, col, title in ((ax1, A, RED, "A: aileron only"), (ax2, B, GREEN, "B: rudder coordinated")):
            R = B_AXES @ dcm_body_to_ned_from_quat(c.state[idx, 9:13])
            draw_aircraft(ax, R, np.zeros(3), "0.85" if c is A else "0.88")
            beta = np.radians(c.beta_deg[idx])
            nose = R @ np.array([1, 0, 0])
            flow = R @ np.array([-np.cos(beta), -np.sin(beta), 0]) * 1.3
            ax.quiver(*(nose * 1.05), *flow, color=col, lw=1.6, arrow_length_ratio=0.25)
            style_3d(ax, f"{title}\nyaw r={c.rates_deg[idx,2]:+5.1f} deg/s  beta={c.beta_deg[idx]:+4.1f} deg", col)
            ax.view_init(elev=18, azim=200)

        l_rA.set_data(A.t[:idx], A.rates_deg[:idx, 2])
        l_bA.set_data(A.t[:idx], A.beta_deg[:idx])
        l_rB.set_data(B.t[:idx], B.rates_deg[:idx, 2])
        l_bB.set_data(B.t[:idx], B.beta_deg[:idx])
        cursor.set_xdata([t, t])

        if S.T_ROLL_ON_S <= t <= S.T_ROLL_OFF_S + 2.0 and A.rates_deg[idx, 2] < -0.3:
            warn_txt.set_text(f"ADVERSE YAW — nose yawing LEFT while rolling RIGHT  ({A.rates_deg[idx,2]:+.1f} deg/s)")
            warn_txt.set_color(RED)
        else:
            warn_txt.set_text("Same aileron input, same instant — rudder is the only difference")
            warn_txt.set_color("black")
        return []

    anim = FuncAnimation(fig, render, frames=len(idxs))
    anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=4000), dpi=140)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    A = S.run_case("A: no rudder", 0.0)
    B = S.run_case("B: rudder coordinated", 0.6)
    make_video(A, B)
