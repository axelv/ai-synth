"""Does the cheap closed-form EQ work on a HALF-fitted patch, or only on a converged one?

The delivered pipeline runs the 26-band EQ last, on a converged CMA-ES patch. The
staircase question is whether the same closed-form warm start (fit_eq_full.oracle_start,
one render plus one least-squares solve) also rescues an early patch. If it does, the
big spectral win is available in seconds rather than at the end of the fit.

Nothing is written outside .scratch. Run:
PYTHONPATH=scripts uv run python .scratch/app-design-flow/research/bench_staircase.py
"""

from __future__ import annotations

import json
import time

import numpy as np

import eq_stage
import fit_eq_full as F
import stage2
import synth
from bend2 import bend_curve

MAXFEV = 150
score = F.FullScore()
notes = stage2.load_notes()

renderer = synth.PadRenderer(n_voices=24, dsp=synth.DSP_SAW)
renderer.set_notes(notes)
renderer.set_bend(bend_curve(int(stage2.DUR * stage2.SR) + stage2.SR))

FLAT = np.zeros(eq_stage.N_BANDS)


def flat_render_of(x: np.ndarray) -> np.ndarray:
    p = synth.denorm(x)
    p.update(eq_stage.gain_dict(FLAT))
    renderer.set_params(p)
    return renderer.render(stage2.DUR).mean(0)


delivered = synth.pad_normalized(np.array(json.load(open("out/patch.json"))["normalized"]))
cma_prefix = np.array(json.load(open(".scratch/app-design-flow/research/bench_cma.json"))["xbest"])

cases = {
    "seed (no fitting at all)": stage2.seeded_start(),
    "CMA prefix, 8 min / 353 renders": cma_prefix,
    "delivered patch, EQ flat": delivered,
}

R = {}
for name, x in cases.items():
    t = time.perf_counter()
    flat = flat_render_of(x)
    t_render = time.perf_counter() - t

    t = time.perf_counter()
    l_flat, lvl, cos = score(flat)
    t_score = time.perf_counter() - t

    t = time.perf_counter()
    g0 = F.oracle_start(flat, score)
    t_oracle = time.perf_counter() - t
    l_oracle, _, cos_o = score(eq_stage.eq_window(flat, g0))

    t = time.perf_counter()
    g, info = F.fit(flat, score, 3e-5, g0, maxfev=MAXFEV)
    t_powell = time.perf_counter() - t

    R[name] = {
        "loss_flat_level_corrected": l_flat,
        "cos_flat": cos,
        "loss_after_oracle_eq": l_oracle,
        "cos_after_oracle_eq": cos_o,
        "loss_after_powell_%d" % MAXFEV: info["loss"],
        "t_render_s": t_render, "t_score_s": t_score,
        "t_oracle_s": t_oracle, "t_powell_s": t_powell,
        "powell_evals": info["evals"],
        "max_abs_gain_db": float(np.abs(g).max()),
    }
    print(f"\n{name}")
    print(f"  flat, level-corrected   loss {l_flat:.4f}  cos {cos:.4f}"
          f"   (render {t_render:.2f} s + score {t_score:.2f} s)")
    print(f"  + closed-form EQ oracle loss {l_oracle:.4f}  cos {cos_o:.4f}"
          f"   (+{t_oracle:.2f} s)")
    print(f"  + Powell {info['evals']} evals   loss {info['loss']:.4f}"
          f"   (+{t_powell:.0f} s)")

json.dump(R, open(".scratch/app-design-flow/research/bench_staircase.json", "w"), indent=1)
print("\nwrote .scratch/app-design-flow/research/bench_staircase.json")
