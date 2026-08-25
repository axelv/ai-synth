"""How far from the true patch can a fit start and still get back to it?

The self-recovery bench established that the search, not the reachable set, is what
fails: CMA-ES from a generic start plateaus around 1.3 on targets it could reach, while
a polish started AT truth sits still at the 0.0006 floor. That leaves the question this
script answers, which is the one an initialiser has to be designed against: how accurate
does a starting point have to be before the existing local polish finishes the job.

Two measurements, cheap first.

`--profile` walks away from truth in random directions and records the loss. One render
per sample, so it costs minutes. It maps the SHAPE around truth: how fast the loss rises
and where it stops rising, which is where truth stops being distinguishable at all.

`--probe` starts a local polish from truth plus a perturbation and asks whether it comes
back. That is the ATTRACTION radius, and it is the number an initialiser is judged
against. It costs a fit per sample, so the radii are chosen from the profile first.

Distances are per-coordinate rms in the normalised box, the same unit the bench reports,
so they read against its chance baseline of 0.333. A perturbation of rms r is drawn as a
random direction scaled so the rms is exactly r before clipping to the box; clipping only
shrinks it, so the realised distance is recorded rather than assumed.

Run:  PYTHONPATH=scripts uv run python scripts/basin.py --only t00 --profile
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import synth
from selfrecover import TIER_A, load_target
from stage2 import CORE, Objective, load_notes, run_cma

RADII = (0.01, 0.02, 0.05, 0.10, 0.20, 0.33)   # 0.33 is the chance baseline


def perturb(truth: np.ndarray, r: float, rng: np.random.Generator) -> np.ndarray:
    """A random direction at per-coordinate rms `r`, clipped into the box."""
    g = rng.standard_normal(len(truth))
    u = g / np.linalg.norm(g)
    return np.clip(truth + r * np.sqrt(len(truth)) * u, 0.0, 1.0)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def profile(meta: dict, args) -> dict:
    """Loss against distance from truth. One render per sample."""
    obj = Objective(load_notes(), target_path=meta["wav"])
    truth = np.array(meta["normalized"], dtype=float)
    floor = obj.loss_of(obj.render(truth))
    rng = np.random.default_rng(args.seed)
    rows = []
    for r in args.radii:
        for k in range(args.dirs):
            x = perturb(truth, r, rng)
            rows.append({"r": r, "k": k, "realised": rms(x, truth),
                         "loss": obj.loss_of(obj.render(x))})
        got = [z["loss"] for z in rows if z["r"] == r]
        print(f"[{meta['id']}] rms {r:.2f}: loss {np.mean(got):.4f} "
              f"(min {min(got):.4f}, max {max(got):.4f})  floor {floor:.4f}")
    res = {"id": meta["id"], "floor": floor, "dirs": args.dirs, "rows": rows}
    with open(os.path.join(args.targets, f"{meta['id']}_profile{args.tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def probe(meta: dict, args) -> dict:
    """Local polish from a perturbed start: does it return to truth?

    sigma tracks the perturbation rather than being fixed, because the question is what
    an initialiser of KNOWN accuracy buys: a search told its start is good to rms r
    should look at that scale. A fixed sigma would answer a different question at every
    radius.
    """
    obj = Objective(load_notes(), target_path=meta["wav"])
    truth = np.array(meta["normalized"], dtype=float)
    floor = obj.loss_of(obj.render(truth))
    free_all = [p.name for p in synth.PARAMS]
    rng = np.random.default_rng(args.seed)
    rows = []
    for r in args.radii:
        for k in range(args.dirs):
            x0 = perturb(truth, r, rng)
            sig = max(r, 0.02)
            obj.calls = 0
            xc, _ = run_cma(obj, x0, CORE, 2, sig, 100 + k,
                            f"{meta['id']}r{r}k{k}core", args.plateau)
            xf, lf = run_cma(obj, xc, free_all, args.full_gens, sig * 0.7, 200 + k,
                             f"{meta['id']}r{r}k{k}full", args.plateau)
            if obj.best[1] is not None and obj.best[0] < lf:
                xf, lf = obj.best[1], obj.best[0]
            row = {"r": r, "k": k, "start_rms": rms(x0, truth),
                   "start_loss": obj.loss_of(obj.render(x0)),
                   "final_rms": rms(xf, truth), "final_loss": lf,
                   "renders": obj.calls,
                   # The endpoint vector, not just its distance. It is the sharpest
                   # adversary the bake-off has: same distance from truth as a random
                   # near point, far lower loss, which is the whole reason the search
                   # ends up there instead of at the answer.
                   "final_x": [float(v) for v in xf]}
            rows.append(row)
            print(f"[{meta['id']}] rms {r:.2f} dir {k}: "
                  f"{row['start_rms']:.3f} -> {row['final_rms']:.3f}  "
                  f"loss {row['start_loss']:.4f} -> {lf:.4f} (floor {floor:.4f})")
    res = {"id": meta["id"], "floor": floor, "full_gens": args.full_gens, "rows": rows}
    with open(os.path.join(args.targets, f"{meta['id']}_basin{args.tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


# "Returned" needs a criterion stated up front rather than read off the numbers. rms 0.02
# is where a parameter is close enough that a human would call the slider the same, and
# 10x the floor is well below anything the bench's failures reached.
RETURN_RMS = 0.02
RETURN_LOSS = 10.0


def summarise(targets: str, tag: str = "") -> None:
    prof = [json.load(open(p))
            for p in sorted(glob.glob(os.path.join(targets, f"*_profile{tag}.json")))]
    if prof:
        radii = sorted({z["r"] for r in prof for z in r["rows"]})
        print("\nloss against distance from truth (mean over targets and directions)")
        print("rms     realised    loss     x floor")
        for r in radii:
            v = [z["loss"] for t in prof for z in t["rows"] if z["r"] == r]
            w = [z["realised"] for t in prof for z in t["rows"] if z["r"] == r]
            f = float(np.mean([t["floor"] for t in prof]))
            print(f"{r:.4f}  {np.mean(w):.4f}  {np.mean(v):8.4f}  {np.mean(v) / f:8.0f}")

    bas = [json.load(open(p))
           for p in sorted(glob.glob(os.path.join(targets, f"*_basin{tag}.json")))]
    if bas:
        radii = sorted({z["r"] for b in bas for z in b["rows"]})
        print(f"\nreturn to truth after a local polish "
              f"(rms < {RETURN_RMS} and loss < {RETURN_LOSS}x floor)")
        print("rms    n   returned  mean final rms  mean final loss")
        for r in radii:
            rows = [(z, b["floor"]) for b in bas for z in b["rows"] if z["r"] == r]
            back = sum(1 for z, f in rows
                       if z["final_rms"] < RETURN_RMS and z["final_loss"] < RETURN_LOSS * f)
            print(f"{r:.2f}  {len(rows):3d}  {back:3d}/{len(rows):<3d}   "
                  f"{np.mean([z['final_rms'] for z, _ in rows]):.3f}"
                  f"           {np.mean([z['final_loss'] for z, _ in rows]):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=TIER_A)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--profile", action="store_true", help="loss against distance, 1 render per sample")
    ap.add_argument("--probe", action="store_true", help="polish from a perturbed start")
    ap.add_argument("--radii", type=float, nargs="+", default=list(RADII))
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--full-gens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="", help="suffix for the output files, to keep runs apart")
    ap.add_argument("--no-plateau", dest="plateau", action="store_const", const=None,
                    default=0.99)
    args = ap.parse_args()

    metas = [load_target(p)
             for p in sorted(glob.glob(os.path.join(args.targets, "t[0-9][0-9].json")))]
    if args.only:
        metas = [m for m in metas if m["id"] in args.only]
    for m in metas:
        if args.profile:
            profile(m, args)
        if args.probe:
            probe(m, args)
    summarise(args.targets, args.tag)


if __name__ == "__main__":
    main()
