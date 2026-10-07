"""Dose-response, rendered strips, LoD per reader and condition, development GIF."""
import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from stripsim import HARSH, LAB, PHONE, READERS, Strip, lod95, render, run

rng = np.random.default_rng(2026)
Path("assets").mkdir(exist_ok=True)
CONC = np.array([0.03, 0.06, 0.1, 0.2, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0])
REPS, BLANKS = 30, 60
CONDS = (LAB, PHONE, HARSH)


def jittered():
    base = Strip()
    return dataclasses.replace(base, p0=base.p0 * rng.uniform(0.9, 1.1), r_tot=base.r_tot * rng.uniform(0.9, 1.1),
                               k_nsb=base.k_nsb * rng.uniform(0.7, 1.3))


# ---- 1. dose-response and the hook effect
dose = np.logspace(-3, 5, 33)
res = [run(c) for c in dose]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.semilogx(dose, [r.test[-1] for r in res], color="#8e2d6b", lw=2, label="test line")
ax.semilogx(dose, [r.control[-1] for r in res], color="#555555", lw=2, ls="--", label="control line")
ax.set_xlabel("analyte in sample (nM)", fontsize=9)
ax.set_ylabel("captured gold reporter (nM)", fontsize=9)
peak = dose[int(np.argmax([r.test[-1] for r in res]))]
ax.axvline(peak, color="grey", lw=0.8, ls=":")
ax.text(peak * 1.3, 5, "hook effect\n(signal falls again)", fontsize=8, color="grey")
ax.set_title("Dose-response from binding kinetics", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("assets/hook.png", dpi=120)
print(f"test-line peak at {peak:.0f} nM; hook: signal at 1e5 nM is "
      f"{res[-1].test[-1] / max(r.test[-1] for r in res) * 100:.0f} % of the peak")

# ---- 2. photo grid
show = [0.0, 0.3, 1.0, 10.0, 100.0, 1e4]
rows = []
for cond in CONDS:
    row = []
    for c in show:
        r = run(c)
        row.append(render(r.gold[-1], r.x, r.front[-1], cond, rng)[30:140, 120:420])
    rows.append(np.hstack(row))
grid = np.vstack(rows)
fig, ax = plt.subplots(figsize=(12, 3.4))
ax.imshow(grid)
ax.set_xticks([150 + 300 * i for i in range(len(show))], [f"{c:g} nM" for c in show], fontsize=8)
ax.set_yticks([55 + 110 * i for i in range(len(CONDS))], [c.name for c in CONDS], fontsize=8)
ax.set_title("Rendered strips: concentration (columns) x imaging condition (rows); note the hook at 10^4 nM",
             fontsize=9)
fig.tight_layout()
fig.savefig("assets/strips.png", dpi=110)

# ---- 3. LoD per reader and condition
scores = {"ideal photometer": {"blank": [], "pos": []}}
for cond in CONDS:
    for rd in READERS:
        scores[(cond.name, rd)] = {"blank": [], "pos": []}
levels = [0.0] * BLANKS + [c for c in CONC for _ in range(REPS)]
for c in levels:
    r = run(c, jittered())
    key = "blank" if c == 0 else "pos"
    scores["ideal photometer"][key].append((c, r.test[-1]))
    for cond in CONDS:
        photo = render(r.gold[-1], r.x, r.front[-1], cond, rng)
        for rd, fn in READERS.items():
            scores[(cond.name, rd)][key].append((c, fn(photo)))

lods = {}
for k, v in scores.items():
    lob = np.percentile([s for _, s in v["blank"]], 95)
    conc = np.array([c for c, _ in v["pos"]])
    det = np.array([s > lob for _, s in v["pos"]], float)
    lods[k] = lod95(conc, det)
print(f"\n{'reader':40s} LoD95 (nM)")
for k, v in lods.items():
    print(f"  {str(k):38s} {v:7.3f}   x{v / lods['ideal photometer']:.1f} of ideal")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
labels, vals, cols = [], [], []
palette = {"lab": "#4c9a2a", "phone": "#e0a100", "harsh": "#c0392b"}
for cond in CONDS:
    for rd in READERS:
        labels.append(f"{cond.name}\n{rd}")
        vals.append(lods[(cond.name, rd)])
        cols.append(palette[cond.name])
ax.bar(range(len(vals)), vals, color=cols)
ax.axhline(lods["ideal photometer"], color="k", ls="--", lw=1)
ax.text(len(vals) - 0.5, lods["ideal photometer"] * 1.05, "chemistry-limited (ideal photometer)", fontsize=7,
        ha="right")
ax.set_yscale("log")
ax.set_xticks(range(len(vals)), labels, fontsize=7)
ax.set_ylabel("LoD95 (nM analyte)", fontsize=9)
ax.set_title("How much the photo inflates the limit of detection", fontsize=10)
fig.tight_layout()
fig.savefig("assets/lod.png", dpi=120)

# ---- 4. development GIF at 2 nM (lab light)
r = run(2.0, record_every=5.0)
frames = [Image.fromarray(render(r.gold[i], r.x, r.front[i], LAB, np.random.default_rng(i))[20:150, 110:430])
          for i in range(0, len(r.t), 2)]
frames[0].save("assets/develop.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)
print("wrote assets/hook.png, strips.png, lod.png, develop.gif")
