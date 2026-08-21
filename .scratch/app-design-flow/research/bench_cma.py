"""Bounded prefix of a stage-2 CMA-ES fit, recording loss against wall clock.

A full fit is ~7500 renders. This runs a fixed wall-clock budget from the same seed
stage2.main() uses and records every evaluation, so the shape of the curve is measured
even though the tail is not.

Run: PYTHONPATH=scripts uv run python .scratch/app-design-flow/research/bench_cma.py [budget_s]
"""

from __future__ import annotations

import json
import sys
import time

import cma
import numpy as np

import stage2
import synth

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 420.0

t_start = time.perf_counter()
notes = stage2.load_notes()
obj = stage2.Objective(notes)
t_setup = time.perf_counter() - t_start
print(f"setup (load notes + objective + faust): {t_setup:.2f} s")

x0 = stage2.seeded_start()
trace = []          # (wallclock_since_fit_start, loss, best_so_far)
gen_marks = []      # (gen, wallclock, best_so_far, renders)

t_fit = time.perf_counter()
best = float("inf")

idx = [synth.PARAM_INDEX[n] for n in stage2.CORE]
base = x0.copy()


def sub(z):
    global best
    x = base.copy()
    x[idx] = np.clip(z, 0.0, 1.0)
    l = obj(x)
    best = min(best, l)
    trace.append((time.perf_counter() - t_fit, l, best))
    return l


l_seed = sub(np.asarray(base[idx]))
print(f"seed loss {l_seed:.4f}")

es = cma.CMAEvolutionStrategy(
    base[idx].tolist(), 0.22,
    {"bounds": [0, 1], "popsize": 16, "seed": 100, "verbose": -9, "maxiter": 10000},
)
g = 0
while (time.perf_counter() - t_fit) < BUDGET:
    sols = es.ask()
    vals = [sub(np.asarray(s)) for s in sols]
    es.tell(sols, vals)
    g += 1
    el = time.perf_counter() - t_fit
    gen_marks.append((g, el, float(es.result.fbest), obj.calls))
    print(f"  gen {g:3d}  best {es.result.fbest:.4f}  ({obj.calls} renders, {el:.0f}s)")

xbest = base.copy()
xbest[idx] = np.clip(np.asarray(es.result.xbest), 0.0, 1.0)

out = {
    "budget_s": BUDGET,
    "setup_s": t_setup,
    "gens": g,
    "renders": obj.calls,
    "elapsed_s": time.perf_counter() - t_fit,
    "seed_loss": l_seed,
    "best_loss": float(es.result.fbest),
    "trace": trace,
    "gen_marks": gen_marks,
    "xbest": xbest.tolist(),
}
json.dump(out, open(".scratch/app-design-flow/research/bench_cma.json", "w"), indent=1)
print(f"\n{g} generations, {obj.calls} renders, "
      f"{(time.perf_counter()-t_fit)/max(obj.calls,1):.3f} s per render")
print("wrote .scratch/app-design-flow/research/bench_cma.json")
