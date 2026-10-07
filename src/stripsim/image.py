"""From captured gold to a phone photo: membrane optics, camera pipeline and cassette rendering.

Optics. The nitrocellulose membrane is a diffuse scatterer; its reflectance follows Kubelka-Munk,
    R_inf = 1 + K/S - sqrt((K/S)^2 + 2 K/S),   K/S = (K/S)_membrane + kappa * eps_c * gold,
evaluated at three effective wavelengths for the camera's R, G, B channels (610, 540, 465 nm). Gold
nanoparticles absorb through a plasmon band, modelled as a Lorentzian at 525 nm (FWHM 80 nm), so lines look
red-purple. Wet membrane is slightly less reflective than dry.

Camera. The scene is lit by an illuminant with a colour temperature, the camera's white balance corrects it
imperfectly, and the image gets exposure error, vignetting, Poisson-Gaussian sensor noise, sRGB gamma,
defocus blur, a small rotation and offset, and JPEG compression. Every nuisance is drawn from a `Conditions`
distribution with a seed, so images are reproducible.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CHANNELS_NM = np.array([610.0, 540.0, 465.0])
EPS = 1.0 / (1.0 + ((CHANNELS_NM - 525.0) / 40.0) ** 2)     # relative plasmon extinction per channel
KAPPA = 0.018                                                # K/S per nM of captured gold (at the peak)
KS_DRY, KS_WET = 0.008, 0.03

# image geometry (pixels)
W, H = 480, 170
WIN_X0, WIN_X1, WIN_Y0, WIN_Y1 = 150, 390, 72, 100           # result window
WIN_MM = (12.0, 28.0)                                        # strip span visible in the window


def reflectance(gold: np.ndarray, wet: np.ndarray | bool = True) -> np.ndarray:
    """Membrane RGB reflectance (..., 3) for captured gold (nM) via Kubelka-Munk."""
    ks = np.where(np.asarray(wet)[..., None], KS_WET, KS_DRY) + KAPPA * np.asarray(gold)[..., None] * EPS
    return 1 + ks - np.sqrt(ks ** 2 + 2 * ks)


def _kelvin_rgb(k: float) -> np.ndarray:
    """Rough linear RGB of a blackbody-like illuminant, normalised to green = 1."""
    lam = CHANNELS_NM * 1e-9
    planck = 1 / (lam ** 5 * (np.exp(1.4388e-2 / (lam * k)) - 1))
    return planck / planck[1]


@dataclass(frozen=True)
class Conditions:
    """Ranges of imaging nuisances (each photo draws uniformly within them)."""

    name: str
    kelvin: tuple = (5000, 5000)
    wb_error: float = 0.0         # residual white-balance error (fraction per channel)
    ev: float = 0.0               # exposure error, +/- EV
    photons: float = 20000.0      # photo-electrons at full scale (sets shot noise)
    read_noise: float = 0.002     # fraction of full scale
    blur: tuple = (0.0, 0.0)      # Gaussian blur sigma (px)
    rot: float = 0.0              # +/- degrees
    shift: float = 0.0            # +/- px
    jpeg: tuple = (95, 95)        # quality range
    shadow: float = 0.0           # max brightness drop across the image


LAB = Conditions("lab")
PHONE = Conditions("phone", kelvin=(2800, 6500), wb_error=0.08, ev=0.7, photons=6000, read_noise=0.004,
                   blur=(0.0, 1.5), rot=2.5, shift=5, jpeg=(60, 95), shadow=0.15)
HARSH = Conditions("harsh", kelvin=(2500, 7500), wb_error=0.15, ev=1.2, photons=1500, read_noise=0.008,
                   blur=(0.5, 3.0), rot=5.0, shift=10, jpeg=(30, 70), shadow=0.35)


def render(gold_x: np.ndarray, x_mm: np.ndarray, front: float, cond: Conditions = LAB,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """Photo (H, W, 3) uint8 of a cassette whose membrane carries gold profile gold_x along x_mm."""
    rng = rng or np.random.default_rng()
    # --- scene in linear reflectance
    scene = np.empty((H, W, 3))
    scene[:] = [0.32, 0.30, 0.27]                                   # table
    yy, xx = np.mgrid[0:H, 0:W]
    body = (np.abs(xx - W / 2) < 225) & (np.abs(yy - H / 2) < 70)
    scene[body] = [0.80, 0.80, 0.78]                                # white plastic cassette
    well = (xx - 85) ** 2 + (yy - 86) ** 2 < 24 ** 2
    scene[well] = [0.55, 0.54, 0.52]
    cols = np.arange(WIN_X0, WIN_X1)
    mm = WIN_MM[0] + (cols - WIN_X0 + 0.5) / (WIN_X1 - WIN_X0) * (WIN_MM[1] - WIN_MM[0])
    g = np.interp(mm, x_mm, gold_x)
    refl = reflectance(g, mm <= front)
    scene[WIN_Y0:WIN_Y1, WIN_X0:WIN_X1] = refl[None, :, :]
    scene[WIN_Y0 - 2:WIN_Y0, WIN_X0:WIN_X1] = 0.5                   # window bevel
    scene[WIN_Y1:WIN_Y1 + 2, WIN_X0:WIN_X1] = 0.6
    # --- illumination, white balance, exposure, vignetting and shadow
    light = _kelvin_rgb(rng.uniform(*cond.kelvin))
    wb = 1 / light * (1 + cond.wb_error * rng.uniform(-1, 1, 3))
    gain = 2.0 ** rng.uniform(-cond.ev, cond.ev) * 0.95
    vign = 1 - 0.25 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    sh = 1 - cond.shadow * rng.uniform() * (xx / W if rng.uniform() < 0.5 else 1 - xx / W)
    lin = np.clip(scene * light * wb * gain * (vign * sh)[..., None], 0, 1)
    # --- sensor noise (Poisson-Gaussian), then sRGB gamma
    e = rng.poisson(lin * cond.photons) / cond.photons + cond.read_noise * rng.standard_normal(lin.shape)
    e = np.clip(e, 0, 1)
    srgb = np.where(e <= 0.0031308, 12.92 * e, 1.055 * e ** (1 / 2.4) - 0.055)
    img = Image.fromarray((srgb * 255 + 0.5).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for label, x0 in (("C", 23.0), ("T", 18.0)):
        px = WIN_X0 + (x0 - WIN_MM[0]) / (WIN_MM[1] - WIN_MM[0]) * (WIN_X1 - WIN_X0)
        d.text((px - 3, WIN_Y0 - 16), label, fill=(60, 60, 60))
    # --- optics of the phone: blur, pose, compression
    b = rng.uniform(*cond.blur)
    if b > 0:
        img = img.filter(ImageFilter.GaussianBlur(b))
    if cond.rot or cond.shift:
        img = img.rotate(rng.uniform(-cond.rot, cond.rot), resample=Image.BILINEAR,
                         translate=tuple(rng.uniform(-cond.shift, cond.shift, 2)), fillcolor=(82, 77, 69))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(rng.integers(cond.jpeg[0], cond.jpeg[1] + 1)))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
