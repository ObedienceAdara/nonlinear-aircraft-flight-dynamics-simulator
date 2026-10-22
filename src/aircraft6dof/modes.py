"""Classify the eigenvalues of a linearized model into flight-dynamics modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linearize import LinearModel


@dataclass(frozen=True)
class Mode:
    name: str
    eigenvalue: complex
    wn_rad_s: float            # |lambda|
    zeta: float                # damping ratio (complex modes), else nan
    period_s: float            # 2 pi / Im(lambda) (complex modes), else nan
    time_constant_s: float     # -1/Re(lambda) for stable real modes, else nan
    time_to_double_s: float    # ln2/Re(lambda) for unstable modes, else nan

    @property
    def is_oscillatory(self) -> bool:
        return abs(self.eigenvalue.imag) > 1e-9

    @property
    def stable(self) -> bool:
        return self.eigenvalue.real < 0.0

    def describe(self) -> str:
        lam = self.eigenvalue
        if self.is_oscillatory:
            return (f"{self.name}: {lam.real:+.4f} {lam.imag:+.4f}j 1/s  "
                    f"wn={self.wn_rad_s:.3f} rad/s  zeta={self.zeta:.4f}  period={self.period_s:.3f} s")
        tail = (f"time constant={self.time_constant_s:.3f} s" if self.stable
                else f"time to double={self.time_to_double_s:.2f} s")
        return f"{self.name}: {lam.real:+.5f} 1/s  {tail}"


def _make_mode(name: str, lam: complex) -> Mode:
    lam = complex(lam)
    nan = float("nan")
    if abs(lam.imag) > 1e-9:
        lam = complex(lam.real, abs(lam.imag))      # report the upper-half-plane root
        wn = abs(lam)
        return Mode(name, lam, wn, -lam.real / wn, 2 * np.pi / lam.imag, nan, nan)
    lam = complex(lam.real, 0.0)
    if lam.real < 0.0:
        return Mode(name, lam, abs(lam), nan, nan, -1.0 / lam.real, nan)
    if lam.real > 0.0:
        return Mode(name, lam, abs(lam), nan, nan, nan, float(np.log(2.0) / lam.real))
    return Mode(name, lam, 0.0, nan, nan, nan, nan)


def _split(eigs: np.ndarray, tol: float = 1e-9) -> tuple[list[complex], list[float]]:
    """Upper-half-plane complex roots and the real roots."""
    complex_roots = [e for e in eigs if e.imag > tol]
    real_roots = [float(e.real) for e in eigs if abs(e.imag) <= tol]
    return complex_roots, real_roots


def longitudinal_modes(model: LinearModel) -> dict[str, Mode]:
    """Short period and phugoid from the [u, w, q, theta] block.

    Needs two complex pairs: the faster is the short period, the slower the
    phugoid. If either mode is overdamped the roots are returned as
    ``real_1``, ``real_2``... instead of guessing.
    """
    sub = model.longitudinal if model.A.shape[0] == 12 else model
    cplx, reals = _split(np.linalg.eigvals(sub.A))
    if len(cplx) == 2:
        fast, slow = sorted(cplx, key=abs, reverse=True)
        return {"short_period": _make_mode("short period", fast), "phugoid": _make_mode("phugoid", slow)}
    out = {f"complex_{i + 1}": _make_mode(f"complex {i + 1}", c) for i, c in enumerate(cplx)}
    out.update({f"real_{i + 1}": _make_mode(f"real {i + 1}", r) for i, r in enumerate(sorted(reals))})
    return out


def lateral_modes(model: LinearModel) -> dict[str, Mode]:
    """Dutch roll, roll subsidence and spiral from the [v, p, r, phi] block.

    Expects one complex pair (Dutch roll) and two real roots: the fastest is
    roll subsidence, the slowest is the spiral mode.
    """
    sub = model.lateral if model.A.shape[0] == 12 else model
    cplx, reals = _split(np.linalg.eigvals(sub.A))
    if len(cplx) == 1 and len(reals) == 2:
        roll, spiral = sorted(reals, key=abs, reverse=True)
        return {
            "dutch_roll": _make_mode("Dutch roll", cplx[0]),
            "roll_subsidence": _make_mode("roll subsidence", roll),
            "spiral": _make_mode("spiral", spiral),
        }
    out = {f"complex_{i + 1}": _make_mode(f"complex {i + 1}", c) for i, c in enumerate(cplx)}
    out.update({f"real_{i + 1}": _make_mode(f"real {i + 1}", r) for i, r in enumerate(sorted(reals))})
    return out


def all_modes(model: LinearModel) -> dict[str, Mode]:
    return {**longitudinal_modes(model), **lateral_modes(model)}