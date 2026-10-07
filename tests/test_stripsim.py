"""Checks of the assay physics, the optics and the readers."""
import numpy as np
import pytest

from stripsim import LAB, Strip, lod95, matched_filter, reflectance, render, run, tc_ratio


def test_reporter_is_conserved_before_it_reaches_the_absorbent_pad():
    s = Strip()
    r = run(5.0, s, t_end=60.0)
    total = r.gold.sum(axis=1) * s.dx
    assert np.allclose(total, s.p0 * (s.conj[1] - s.conj[0]), rtol=1e-6)


def test_low_dose_response_is_linear():
    lo, hi = run(0.02).test[-1] - run(0.0).test[-1], run(0.2).test[-1] - run(0.0).test[-1]
    assert hi / lo == pytest.approx(10.0, rel=0.1)


def test_high_dose_hook_effect_emerges():
    peak = max(run(c).test[-1] for c in (30.0, 100.0, 300.0))
    assert run(1e5).test[-1] < 0.2 * peak


def test_front_follows_lucas_washburn():
    s = Strip()
    r = run(0.0, s, t_end=100.0, record_every=25.0)
    expect = np.clip(np.sqrt(s.d_wick * r.t), 1.0, s.length)
    assert np.allclose(r.front[1:], expect[1:], atol=0.2)


def test_membrane_optics():
    white = reflectance(np.zeros(1), wet=False)[0]
    assert np.all(white > 0.85)
    dark = reflectance(np.array([50.0]))[0]
    assert dark[1] < dark[0] and dark[1] < dark[2]          # gold absorbs green most: line looks red-purple


def test_readers_separate_blank_from_positive():
    rng = np.random.default_rng(0)
    blank, pos = run(0.0), run(5.0)
    pb = render(blank.gold[-1], blank.x, blank.front[-1], LAB, rng)
    pp = render(pos.gold[-1], pos.x, pos.front[-1], LAB, rng)
    assert tc_ratio(pp) > tc_ratio(pb) + 0.1
    assert matched_filter(pp) > 5 * max(matched_filter(pb), 1.0)


def test_lod_fit_recovers_a_known_probit():
    rng = np.random.default_rng(1)
    conc = np.repeat(np.logspace(-2, 1, 13), 40)
    p = 0.5 * (1 + np.vectorize(__import__("math").erf)((np.log10(conc) + 1.0) / (0.3 * 2 ** 0.5)))
    det = (rng.random(conc.size) < p).astype(float)
    assert lod95(conc, det) == pytest.approx(10 ** (-1.0 + 1.645 * 0.3), rel=0.35)
