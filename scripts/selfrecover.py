"""Self-recovery bench: hand CMA-ES a patch the synth itself rendered, ask for it back.

This is the experiment that tells us which half of stage 2 is the bottleneck. A Tier A
target is reachable by construction, so the reachable-set excuse does not apply. If the
optimiser cannot recover a vector the synth produced, the optimiser is the problem and
the whole fitting strategy needs rethinking. If it recovers cleanly, then the residual
gap on `data/original.wav` is the reachable set, which the repo has inferred three times
and never directly measured.

Tier A validates machinery only. `data/original.wav` stays the sole acceptance target:
an in-domain synthetic fit says nothing about whether the synth can express a real pad.

Two numbers matter more than the loss:

- The floor. Truth scored against its own PCM_24 file is 0.0008 here, not 0, so any
  final loss is read against that and not against zero.
- Parameter distance. A low loss with a distant parameter vector is the informative
  failure: it means the loss cannot tell the patches apart, which is a statement about
  the objective rather than about the search. Reported per parameter and per group,
  because several parameters are unidentifiable whenever the stage that uses them is
  off (dlyTime under dlyWet=0, chRate under chDepth=0, lfoRate under lfoAmt=0), and a
  single L2 over all 55 dims hides that inside the noise those dead dimensions make.

Run:  PYTHONPATH=scripts uv run python scripts/selfrecover.py --only t00
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import librosa
import numpy as np

import synth
from metrics import report
from stage2 import CORE, FX, SR, Objective, load_notes, run_cma, seeded_start

TIER_A = "out/tierA"
EQ = [f"eq{i}" for i in range(synth.N_EQ)]


def load_target(path: str) -> dict:
    meta = json.load(open(path))
    names = [p.name for p in synth.PARAMS]
    if meta["param_names"] != names:
        raise SystemExit(
            f"{path} was written against a different PARAMS: {meta['n_params']} params "
            f"then, {len(names)} now. Regenerate the targets with selfgen.py. Comparing "
            "a stored truth vector against a changed PARAMS lines truth up with the "
            "wrong sliders and the bench reports a recovery failure that is really a "
            "bookkeeping one."
        )
    return meta


def group_error(err: dict[str, float], names: list[str]) -> float:
    """RMS normalised error over a named group of parameters."""
    v = np.array([err[n] for n in names])
    return float(np.sqrt((v ** 2).mean()))


def recover(meta: dict, args) -> dict:
    notes = load_notes()
    obj = Objective(notes, target_path=meta["wav"], loss=args.loss)
    truth = np.array(meta["normalized"], dtype=float)

    # The floor for THIS target: the same vector that made the file, scored against the
    # file. Non-zero only because the target went through PCM_24 on the way to disk.
    # Deliberately NOT through obj(): __call__ records the best vector it has seen, and
    # scoring truth through it hands the answer to the search. That mistake reports a
    # perfect recovery from 98 renders, which is how it was found.
    # Whatever the fit descends: the candidate loss when one is named, otherwise the
    # incumbent. The floor and the start have to be on the same scale as the search.
    scored = obj.alt_of if obj.alt is not None else obj.loss_of
    floor = scored(obj.render(truth))
    # `truth` is the control that separates a search failure from a loss failure: start
    # AT the answer with a small sigma and see whether the optimiser stays. If it walks
    # off and finds something below the floor, the objective's minimum is not at the
    # true parameters and no amount of search fixes that.
    starts = {"seeded": seeded_start, "defaults": synth.norm_defaults,
              "truth": lambda: truth.copy()}
    x0 = starts[args.start]()
    start = scored(obj.render(x0))
    print(f"[{meta['id']}] floor {floor:.4f}  start {start:.4f}")

    t0 = time.time()
    best_x, best_l = x0, start
    free_all = [p.name for p in synth.PARAMS]
    for rs in range(args.restarts):
        sig_c = args.sigma if args.sigma else 0.22 + 0.08 * rs
        xc, _ = run_cma(obj, best_x if rs == 0 else x0, CORE, args.core_gens, sig_c,
                        100 + rs, f"{meta['id']}core{rs}", args.plateau)
        sig_f = args.sigma * 0.7 if args.sigma else 0.14 + 0.06 * rs
        xf, lf = run_cma(obj, xc, free_all, args.full_gens, sig_f, 200 + rs,
                         f"{meta['id']}full{rs}", args.plateau)
        if lf < best_l:
            best_x, best_l = xf, lf
    if obj.best[1] is not None and obj.best[0] < best_l:
        best_x, best_l = obj.best[1], obj.best[0]
    wall = time.time() - t0

    # Score from the file on disk, not from the array: PCM_24 is what every other
    # number in this repo is measured through, and a mismatch here would mean the
    # reported loss is not the one the wav reproduces.
    audio = obj.render(best_x)
    # Only the plain seeded runs under the incumbent own the bare name; controls and
    # candidate-loss runs get a suffix so summarise() cannot mix them into the table.
    tag = "" if args.start == "seeded" else f"_{args.start}"
    if args.loss:
        tag += f"_{args.loss}"
    wav = os.path.join(args.targets, f"{meta['id']}{tag}_recover.wav")
    synth.write_render(wav, audio)
    from_disk, _ = librosa.load(wav, sr=SR, mono=False)
    tgt, _ = librosa.load(meta["wav"], sr=SR, mono=False)

    err = {p.name: float(abs(best_x[i] - truth[i])) for i, p in enumerate(synth.PARAMS)}
    rng_ctrl = np.random.default_rng(0)
    res = {
        "id": meta["id"],
        "floor": floor,
        "start_loss": start,
        "final_loss": best_l,
        # Always the incumbent figure too, whatever the fit descended, so a run under a
        # candidate loss stays comparable with every number already in the repo.
        "loss_name": args.loss or "incumbent",
        "loss_from_disk": scored(from_disk),
        "incumbent_from_disk": obj.loss_of(from_disk),
        "renders": obj.calls,
        "wall_s": wall,
        "param_l2": float(np.linalg.norm(best_x - truth)),
        "param_rms": float(np.sqrt(((best_x - truth) ** 2).mean())),
        "core_rms": group_error(err, CORE),
        "fx_rms": group_error(err, FX),
        "eq_rms": group_error(err, EQ),
        # Controls for the parameter distance. Three of them, because the obvious one is
        # wrong and was believed for weeks: 1/3 is E|u-v|, the mean ABSOLUTE error between
        # two uniforms, while the number reported here is an RMS, whose uniform value is
        # sqrt(1/6) = 0.408. Measured against the actual truth vectors it comes out at
        # 0.376. The one that matters most is the last: setting every knob to the middle
        # of its range scores 0.242, and the fit scores 0.328, so the fit is 36% WORSE
        # than the most trivial strategy available. Report all three or the reader will
        # compare against whichever flatters the run.
        "chance_rms": float(np.sqrt(((rng_ctrl.random((4000, len(truth))) - truth) ** 2)
                                    .mean(axis=1)).mean()),
        "chance_rms_analytic": float(np.sqrt(1.0 / 6.0)),
        "midpoint_rms": float(np.sqrt(((0.5 - truth) ** 2).mean())),
        "param_err": err,
        # Per-generation best. Whether the curve was still descending at the budget is
        # the difference between "the optimiser cannot do this" and "it ran out of
        # renders", and the two conclusions point at different work.
        "history": [float(v) for v in obj.history],
        "metrics": report(tgt, audio),
        "normalized": best_x.tolist(),
        "params": synth.denorm(best_x),
        "truth_params": meta["params"],
        "budget": {"core_gens": args.core_gens, "full_gens": args.full_gens,
                   "restarts": args.restarts, "start": args.start,
                   "plateau": args.plateau},
    }
    with open(os.path.join(args.targets, f"{meta['id']}{tag}_recover.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[{meta['id']}] final {best_l:.4f} (floor {floor:.4f})  "
          f"param rms {res['param_rms']:.3f}  core {res['core_rms']:.3f}  "
          f"fx {res['fx_rms']:.3f}  eq {res['eq_rms']:.3f}  "
          f"{obj.calls} renders, {wall / 60:.0f} min")
    return res


# A parameter has to move the loss by more than the PCM_24 floor (about 0.001) before a
# recovery error in it means anything. 0.01 is an order of magnitude above that, and an
# order of magnitude below the 0.16 the whole EQ stage bought on the real target.
LIVE = 0.01


def sensitivity(meta: dict, args) -> dict:
    """How much the loss moves when each parameter moves alone, at the truth vector.

    Without this the per-parameter error is unreadable. Several parameters are dead in
    any given patch: dlyTime and dlyFb do nothing at dlyWet=0, chRate nothing at
    chDepth=0, lfoRate nothing at lfoAmt=0. Missing a coordinate the loss cannot see is
    not a search failure, and averaging those coordinates into one L2 buries whatever
    the live ones are saying. Which parameters are dead depends on the patch, so it is
    measured per target rather than argued from the DSP.
    """
    obj = Objective(load_notes(), target_path=meta["wav"])
    truth = np.array(meta["normalized"], dtype=float)
    base = obj.loss_of(obj.render(truth))
    out: dict[str, float] = {}
    for i, p in enumerate(synth.PARAMS):
        rises = []
        for step in (-args.delta, args.delta):
            x = truth.copy()
            x[i] = float(np.clip(truth[i] + step, 0.0, 1.0))
            if x[i] != truth[i]:
                rises.append(obj.loss_of(obj.render(x)) - base)
        # The larger of the two one-sided rises: the most generous reading of "the loss
        # can see this parameter", so a dead verdict is not an artefact of the step
        # falling off the edge of the box.
        out[p.name] = float(max(rises)) if rises else 0.0
    live = [n for n, v in out.items() if v >= LIVE]
    res = {"id": meta["id"], "base": base, "delta": args.delta,
           "live": live, "n_live": len(live), "sensitivity": out}
    with open(os.path.join(args.targets, f"{meta['id']}_sens.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[{meta['id']}] {len(live)}/{len(out)} parameters move the loss by >= {LIVE} "
          f"at +-{args.delta} normalised")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=TIER_A)
    ap.add_argument("--only", action="append", default=[], help="target id, repeatable")
    # Defaults mirror stage2.main so a bench run is the same shape as a real fit.
    ap.add_argument("--core-gens", type=int, default=90)
    ap.add_argument("--full-gens", type=int, default=140)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--no-plateau", dest="plateau", action="store_const", const=None,
                    default=0.99, help="run the full generation budget without the "
                                       "1%%-over-20-generations early stop")
    ap.add_argument("--start", choices=("seeded", "defaults", "truth"), default="seeded")
    ap.add_argument("--loss", default=None,
                    help="candidate loss name from losses.LOSSES; default is the incumbent")
    ap.add_argument("--sensitivity", action="store_true",
                    help="measure per-parameter loss sensitivity at truth instead of fitting")
    ap.add_argument("--delta", type=float, default=0.05,
                    help="normalised step for --sensitivity")
    ap.add_argument("--summarise", action="store_true",
                    help="skip the fits, just re-collect the per-target results")
    args = ap.parse_args()

    # Exactly the target files. A looser glob picks up the _recover and _sens results
    # this script writes into the same directory, which have no param_names and blow up
    # the contract check with a KeyError.
    paths = sorted(glob.glob(os.path.join(args.targets, "t[0-9][0-9].json")))
    metas = [load_target(p) for p in paths]
    if args.only:
        metas = [m for m in metas if m["id"] in args.only]
    if not metas:
        raise SystemExit(f"no targets in {args.targets}; run selfgen.py first")

    if args.sensitivity:
        for m in metas:
            sensitivity(m, args)
        return
    if not args.summarise:
        for m in metas:
            recover(m, args)
    summarise(args.targets)


def summarise(targets: str) -> list[dict]:
    """Collect whatever per-target results exist into one table.

    Separate from the run so the bench can be driven one target per process (the
    renders are CPU-bound and single-threaded, so that is how it gets parallelised)
    and still produce a single summary afterwards.
    """
    # Only the seeded runs: the `truth` and `defaults` starts are controls, not entries
    # in the bench table, and averaging a control in would flatter the result.
    results = [json.load(open(p))
               for p in sorted(glob.glob(os.path.join(targets, "t[0-9][0-9]_recover.json")))]
    if not results:
        raise SystemExit(f"no results in {targets}")
    with open(os.path.join(targets, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nid    floor   start   final   param_rms  core   fx     eq     renders")
    for r in results:
        print(f"{r['id']}  {r['floor']:.4f}  {r['start_loss']:.3f}  {r['final_loss']:.4f}  "
              f"{r['param_rms']:.3f}      {r['core_rms']:.3f}  "
              f"{r['fx_rms']:.3f}  {r['eq_rms']:.3f}  {r['renders']}")
    live = {}
    for r in results:
        sp = os.path.join(targets, f"{r['id']}_sens.json")
        if os.path.exists(sp):
            live[r["id"]] = json.load(open(sp))["live"]
    if live:
        print("\nrestricted to parameters the loss can actually see at truth:")
        print("id    live  param_rms  dead_rms")
        for r in results:
            names = live.get(r["id"])
            if not names:
                continue
            dead = [n for n in r["param_err"] if n not in names]
            print(f"{r['id']}  {len(names):4d}  {group_error(r['param_err'], names):.3f}"
                  f"      {group_error(r['param_err'], dead) if dead else float('nan'):.3f}")

    n = len(results)
    print(f"\nmean of {n}: final {np.mean([r['final_loss'] for r in results]):.4f}  "
          f"floor {np.mean([r['floor'] for r in results]):.4f}  "
          f"param_rms {np.mean([r['param_rms'] for r in results]):.3f} "
          f"(random {np.mean([r.get('chance_rms', float('nan')) for r in results]):.3f}, "
          f"all-knobs-mid {np.mean([r.get('midpoint_rms', float('nan')) for r in results]):.3f})  core {np.mean([r['core_rms'] for r in results]):.3f}  "
          f"fx {np.mean([r['fx_rms'] for r in results]):.3f}  "
          f"eq {np.mean([r['eq_rms'] for r in results]):.3f}")
    return results


if __name__ == "__main__":
    main()
