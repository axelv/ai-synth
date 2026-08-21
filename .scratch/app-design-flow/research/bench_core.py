"""Timing bench for the existing pipeline. Reads only; writes nothing outside .scratch.

Run: PYTHONPATH=scripts uv run python .scratch/app-design-flow/research/bench_core.py
"""

from __future__ import annotations

import json
import time

T0 = time.perf_counter()
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
T_HEAVY_IMPORTS = time.perf_counter() - T0

T0 = time.perf_counter()
import synth  # noqa: E402
import stage2  # noqa: E402
from bend2 import bend_curve  # noqa: E402
T_PROJECT_IMPORTS = time.perf_counter() - T0

R = {}
R["import_librosa_numpy_torch_s"] = T_HEAVY_IMPORTS
R["import_project_s"] = T_PROJECT_IMPORTS


def timeit(fn, n=1):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, ts


# ---- faust compile / engine construction ----
_, ts = timeit(lambda: synth.PadRenderer(n_voices=24, dsp=synth.DSP), 3)
R["padrenderer_construct_s"] = ts
print("PadRenderer construct (faust compile + engine):", [round(t, 3) for t in ts])

# ---- notes ----
t = time.perf_counter()
notes = stage2.load_notes()
R["load_notes_s"] = time.perf_counter() - t
R["n_notes"] = len(notes)
print(f"load_notes: {R['load_notes_s']*1000:.1f} ms, {len(notes)} notes")

# ---- objective construction (target load + resample + stft setup) ----
t = time.perf_counter()
obj = stage2.Objective(notes)
R["objective_construct_s"] = time.perf_counter() - t
print(f"Objective construct (incl. its own PadRenderer + 48->44.1k resample): "
      f"{R['objective_construct_s']:.3f} s")

# ---- bend curve ----
_, ts = timeit(lambda: bend_curve(int(stage2.DUR * stage2.SR) + stage2.SR), 3)
R["bend_curve_s"] = ts
print("bend_curve:", [round(t * 1000, 1) for t in ts], "ms")

# ---- one full-clip render ----
x_seed = stage2.seeded_start()
x_final = np.array(json.load(open("out/patch.json"))["normalized"])

obj.renderer.set_params(synth.denorm(x_seed))
_, ts = timeit(lambda: obj.renderer.render(stage2.DUR), 6)
R["render_full_clip_s"] = ts
print("render 17.904 s clip, 29 notes, 24 voices:", [round(t, 3) for t in ts])

aud = obj.renderer.render(stage2.DUR)

# ---- one loss evaluation on already-rendered audio ----
_, ts = timeit(lambda: obj.loss_of(aud), 6)
R["loss_of_s"] = ts
print("loss_of (mono MRSTFT + env):", [round(t, 3) for t in ts])

_, ts = timeit(lambda: obj.loss_parts(aud), 3)
R["loss_parts_s"] = ts
print("loss_parts (mono + side):", [round(t, 3) for t in ts])

# ---- full objective call (render + score) ----
_, ts = timeit(lambda: obj(x_seed), 4)
R["objective_call_s"] = ts
print("objective __call__ (render + loss):", [round(t, 3) for t in ts])

# ---- short render: what a single held note costs (preview / audition path) ----
r2 = synth.PadRenderer(n_voices=24, dsp=synth.DSP)
r2.set_notes([(60, 100, 0.0, 1.5)])
r2.set_params(synth.denorm(x_seed))
r2.set_bend(None)
_, ts = timeit(lambda: r2.render(2.0), 5)
R["render_one_note_2s_s"] = ts
print("render one note, 2 s:", [round(t, 3) for t in ts])

# ---- losses of the landmark patches, so "how bad is early" has numbers ----
def loss_at(x):
    return obj.loss_of(obj.render(x))


landmarks = {
    "norm_defaults": synth.norm_defaults(),
    "seeded_start": x_seed,
    "delivered_patch": synth.pad_normalized(x_final),
}
# the delivered patch with its 26 EQ bands zeroed = the pre-EQ state of the pipeline
x_noeq = synth.pad_normalized(x_final).copy()
for i in range(synth.N_EQ):
    x_noeq[synth.PARAM_INDEX[f"eq{i}"]] = synth.normalize_one(
        synth.PARAMS[synth.PARAM_INDEX[f"eq{i}"]], 0.0)
landmarks["delivered_patch_eq_flat"] = x_noeq

R["landmark_loss"] = {}
for k, v in landmarks.items():
    t = time.perf_counter()
    l = loss_at(v)
    R["landmark_loss"][k] = l
    print(f"loss[{k}] = {l:.4f}   ({time.perf_counter()-t:.2f}s)")

# ---- calibration controls, so a loss number can be read ----
y, _ = librosa.load("data/original.wav", sr=stage2.SR, mono=True)
R["landmark_loss"]["target_vs_itself"] = obj.loss_of(np.vstack([y, y]))
rng = np.random.default_rng(0)
noise = rng.standard_normal(len(y)).astype(np.float32) * float(np.sqrt((y ** 2).mean()))
R["landmark_loss"]["white_noise_same_rms"] = obj.loss_of(np.vstack([noise, noise]))
R["landmark_loss"]["silence"] = obj.loss_of(np.vstack([y, y]) * 1e-6)
print("control losses:", {k: round(v, 4) for k, v in R["landmark_loss"].items()})

json.dump(R, open(".scratch/app-design-flow/research/bench_core.json", "w"), indent=1)
print("\nwrote .scratch/app-design-flow/research/bench_core.json")
