"""1D model of a sandwich lateral-flow assay: capillary flow, binding, and capture at the test and control lines.

Following the structure of Qian & Bau (Anal Biochem 322:89, 2003, doi:10.1016/j.ab.2003.07.011):
- The liquid front advances by Lucas-Washburn wicking, x_f = sqrt(D_w t); behind it the liquid moves with the
  uniform velocity u = dx_f/dt (u stays at its end value once the front reaches the absorbent pad).
- Mobile species (concentrations in nM): analyte A (supplied continuously by the sample), gold-labelled
  reporter P (released from the conjugate pad), and reporter-analyte complex PA.
  A + P <-> PA everywhere in the liquid.
- Test line, immobilised capture antibody R (sandwich):  R + PA <-> RPA,  R + A <-> RA,  RA + P <-> RPA,
  plus weak non-specific sticking of reporter, R + P -> RP (what makes blank strips not perfectly blank).
- Control line, anti-species antibody C:  C + P <-> CP,  C + PA <-> CPA.
Visible signal = captured reporter: RPA + RP at the test line, CP + CPA at the control line.

At very high analyte concentration, free analyte saturates both R and P before sandwiches can form, so the
test line fades again: the high-dose hook effect emerges from the kinetics without being imposed.

Transport is first-order upwind (method of lines). Each binding reaction is advanced over a time step with
the exact solution of second-order kinetics, so the scheme is stable at any concentration; the slow
dissociation is added explicitly. Parameter values are literature-typical, not those of a specific assay.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Strip:
    length: float = 30.0          # mm, sample pad to absorbent pad
    dx: float = 0.1               # mm
    d_wick: float = 6.0           # Lucas-Washburn coefficient (mm^2/s): front at 30 mm after 150 s
    conj: tuple = (2.0, 6.0)      # conjugate pad (mm)
    test: tuple = (17.5, 18.5)    # test line (mm)
    control: tuple = (22.5, 23.5)  # control line (mm)
    p0: float = 20.0              # reporter in the conjugate pad (nM)
    r_tot: float = 500.0          # capture-antibody capacity at the test line (nM equivalent)
    c_tot: float = 500.0          # anti-species capacity at the control line (nM equivalent)
    k_on: float = 1e-3            # nM^-1 s^-1  (1e6 M^-1 s^-1)
    k_off: float = 1e-4           # s^-1
    k_nsb: float = 2e-6           # non-specific reporter capture at the test line (nM^-1 s^-1)


@dataclass
class AssayResult:
    t: np.ndarray                 # (n_t,) s
    test: np.ndarray              # (n_t,) captured reporter at the test line (nM, line average)
    control: np.ndarray           # (n_t,) captured reporter at the control line
    front: np.ndarray             # (n_t,) liquid-front position (mm)
    x: np.ndarray                 # (n_x,) cell centres (mm)
    gold: np.ndarray              # (n_t, n_x) total reporter (bound + mobile) per cell, for rendering
    strip: Strip = field(default_factory=Strip)


def _bind(x, y, z, k, dt):
    """Advance x + y -> z over dt with the exact solution of dz/dt = k x y (arrays, element-wise)."""
    d = x - y
    small = np.abs(d) < 1e-9 * (np.abs(x) + np.abs(y) + 1e-30)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.exp(-k * d * dt)
        dz = np.where(small, x * x * k * dt / (1 + x * k * dt), x * y * (1 - e) / (x - y * e))
    dz = np.nan_to_num(np.clip(dz, 0, np.minimum(x, y)))
    return x - dz, y - dz, z + dz


def run(analyte: float, strip: Strip = Strip(), t_end: float = 600.0, record_every: float = 5.0) -> AssayResult:
    """Simulate a test with analyte concentration `analyte` (nM) in the sample."""
    s = strip
    n = int(round(s.length / s.dx))
    x = (np.arange(n) + 0.5) * s.dx
    a, p, pa = np.zeros(n), np.zeros(n), np.zeros(n)
    p[(x >= s.conj[0]) & (x < s.conj[1])] = s.p0
    tl = (x >= s.test[0]) & (x < s.test[1])
    cl = (x >= s.control[0]) & (x < s.control[1])
    r, ra, rpa, rp = np.where(tl, s.r_tot, 0.0), np.zeros(n), np.zeros(n), np.zeros(n)
    c, cp, cpa = np.where(cl, s.c_tot, 0.0), np.zeros(n), np.zeros(n)

    t, front = 0.0, 1.0
    rec_t, rec_test, rec_ctrl, rec_front, rec_gold = [], [], [], [], []
    next_rec = 0.0
    while t <= t_end + 1e-9:
        if t >= next_rec - 1e-9:
            rec_t.append(t)
            rec_test.append((rpa + rp)[tl].mean())
            rec_ctrl.append((cp + cpa)[cl].mean())
            rec_front.append(min(front, s.length))
            rec_gold.append(p + pa + rpa + rp + cp + cpa)
            next_rec += record_every
        u = s.d_wick / (2 * min(front, s.length))
        dt = min(0.4 * s.dx / u, 0.5)
        wet = x <= front
        # --- advection (upwind), only in wetted cells; sample enters at x = 0
        for q, inflow in ((a, analyte), (p, 0.0), (pa, 0.0)):
            up = np.concatenate([[inflow], q[:-1]])
            flux_in = u * up * wet
            flux_out = u * q * np.concatenate([wet[1:], [True]])       # outflow into the absorbent pad
            q += dt / s.dx * (flux_in - flux_out) * wet
        # --- reactions (operator splitting, exact second-order steps + explicit dissociation)
        a, p, pa = _bind(a, p, pa, s.k_on, dt)
        r, pa, rpa = _bind(r, pa, rpa, s.k_on, dt)
        r, a, ra = _bind(r, a, ra, s.k_on, dt)
        ra, p, rpa = _bind(ra, p, rpa, s.k_on, dt)
        r, p, rp = _bind(r, p, rp, s.k_nsb, dt)
        c, p, cp = _bind(c, p, cp, s.k_on, dt)
        c, pa, cpa = _bind(c, pa, cpa, s.k_on, dt)
        off = s.k_off * dt
        d1, d2, d3, d4 = off * pa, off * rpa, off * cp, off * cpa
        pa, a, p = pa - d1, a + d1, p + d1
        rpa, r, pa = rpa - d2, r + d2, pa + d2
        cp, c, p = cp - d3, c + d3, p + d3
        cpa, c, pa = cpa - d4, c + d4, pa + d4
        t += dt
        front = np.sqrt(s.d_wick * t) if front < s.length else s.length
        front = max(front, 1.0)
    return AssayResult(np.array(rec_t), np.array(rec_test), np.array(rec_ctrl), np.array(rec_front), x,
                       np.array(rec_gold), s)
