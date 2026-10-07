"""Strip readers and the limit of detection (LoD) in analyte-concentration units.

Readers turn a photo into a score; a test is called positive when the score exceeds the limit of blank
(LoB, the 95th percentile of blank scores). LoD95 is the concentration detected with 95 % probability,
from a probit fit of detection rate against log concentration (in the spirit of CLSI EP17).
"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

from . import image as im


def _profile(photo: np.ndarray) -> np.ndarray:
    """Line-darkness profile along the window: inverted green channel, averaged over the window's height."""
    g = photo[im.WIN_Y0 + 4: im.WIN_Y1 - 4, im.WIN_X0: im.WIN_X1, 1].astype(float)
    prof = 255 - g.mean(axis=0)
    return prof - np.median(prof)


def _px(mm: float) -> int:
    return int(round((mm - im.WIN_MM[0]) / (im.WIN_MM[1] - im.WIN_MM[0]) * (im.WIN_X1 - im.WIN_X0)))


def tc_ratio(photo: np.ndarray, search: int = 8, half: int = 8) -> float:
    """Classic reader: background-subtracted peak areas at the nominal T and C positions, T / C."""
    prof = _profile(photo)
    area = {}
    for name, mm in (("T", 18.0), ("C", 23.0)):
        c = _px(mm)
        best = max(range(c - search, c + search + 1),
                   key=lambda k: prof[max(k - half, 0): k + half].sum())
        area[name] = prof[max(best - half, 0): best + half].sum()
    return float(area["T"] / max(area["C"], 1.0))


def matched_filter(photo: np.ndarray, search: int = 12, width: float = 6.0) -> float:
    """Matched filter: best correlation of the profile with a line-shaped template near T, divided by noise."""
    prof = _profile(photo)
    k = np.arange(-15, 16)
    tmpl = np.exp(-0.5 * (k / width) ** 2)
    tmpl = (tmpl - tmpl.mean()) / np.linalg.norm(tmpl - tmpl.mean())
    c = _px(18.0)
    quiet = np.r_[prof[: _px(15.5)], prof[_px(20.5): _px(21.5)]]
    noise = quiet.std() + 1e-6
    vals = [float(np.dot(prof[i - 15: i + 16], tmpl)) for i in range(c - search, c + search + 1)
            if i - 15 >= 0 and i + 16 <= prof.size]
    return max(vals) / noise


READERS = {"T/C ratio": tc_ratio, "matched filter": matched_filter}


def _phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def lod95(conc: np.ndarray, detected: np.ndarray) -> float:
    """LoD95 from a probit fit P(detect) = Phi((log10 c - mu) / sigma), maximum likelihood on a grid."""
    lc = np.log10(conc)
    best, arg = -np.inf, (np.nan, np.nan)
    for mu in np.linspace(lc.min() - 1, lc.max() + 1, 241):
        for sig in np.linspace(0.05, 1.5, 59):
            p = np.clip([_phi((v - mu) / sig) for v in lc], 1e-9, 1 - 1e-9)
            ll = float(np.sum(detected * np.log(p) + (1 - detected) * np.log(1 - p)))
            if ll > best:
                best, arg = ll, (mu, sig)
    return float(10 ** (arg[0] + 1.645 * arg[1]))
