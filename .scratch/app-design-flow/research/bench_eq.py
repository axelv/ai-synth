"""Time the 26-band EQ fit: setup, per-candidate cost, and a bounded Powell prefix.

fit_eq_full.main() writes into out/, so nothing here calls it; the pieces are timed
directly. Run:
PYTHONPATH=scripts uv run python .scratch/app-design-flow/research/bench_eq.py [maxfev]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import eq_stage
import fit_eq_full as F

MAXFEV = int(sys.argv[1]) if len(sys.argv) > 1 else 300
R = {"maxfev": MAXFEV}

t = time.perf_counter()
score = F.FullScore()
R["fullscore_construct_s"] = time.perf_counter() - t
print(f"FullScore construct: {R['fullscore_construct_s']:.2f} s")

t = time.perf_counter()
flat = F.flat_render("out/patch.json")
R["flat_render_s"] = time.perf_counter() - t
print(f"flat_render (one 17.9 s poly render, EQ at 0 dB): {R['flat_render_s']:.2f} s")

t = time.perf_counter()
base_loss, base_lvl, base_cos = score(flat)
R["score_once_s"] = time.perf_counter() - t
R["flat_loss"] = base_loss
R["flat_cos"] = base_cos
print(f"score(flat): loss {base_loss:.6f}  cos {base_cos:.4f}  "
      f"({R['score_once_s']:.2f} s)")

g_test = np.zeros(eq_stage.N_BANDS)
ts = []
for k in range(4):
    g_test[0] = 0.1 * (k + 1)      # change the gains so nothing is cached away
    t = time.perf_counter()
    y = eq_stage.eq_window(flat, g_test)
    ts.append(time.perf_counter() - t)
R["eq_window_s"] = ts
print("eq_window (18 s through the cascade):", [round(v, 3) for v in ts])

t = time.perf_counter()
x0 = F.oracle_start(flat, score)
R["oracle_start_s"] = time.perf_counter() - t
l0, _, _ = score(eq_stage.eq_window(flat, x0))
R["oracle_loss"] = l0
print(f"oracle_start (closed form, no search): loss {l0:.6f} "
      f"({R['oracle_start_s']:.2f} s)")

# bounded Powell prefix at the lambda the delivered patch used
lam = 3e-5
t = time.perf_counter()
g, info = F.fit(flat, score, lam, x0, maxfev=MAXFEV)
dt = time.perf_counter() - t
R["fit_prefix"] = {"lam": lam, "wall_s": dt, "evals": info["evals"],
                   "loss": info["loss"], "per_eval_s": dt / max(info["evals"], 1)}
print(f"Powell prefix lam={lam:.0e}, maxfev={MAXFEV}: {info['evals']} evals in "
      f"{dt:.1f} s ({dt/max(info['evals'],1):.3f} s/eval), loss {info['loss']:.6f}")
print(f"  (recorded full fit at this lambda: 1200 evals, loss 1.382292)")

json.dump(R, open(".scratch/app-design-flow/research/bench_eq.json", "w"), indent=1)
print("wrote .scratch/app-design-flow/research/bench_eq.json")
