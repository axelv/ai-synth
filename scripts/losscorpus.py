"""Render once, score many: the cached audio the loss bake-off runs against.

Rendering costs 1.19 s; scoring a candidate loss costs milliseconds. So the corpus is
rendered a single time and every candidate is screened against the cached audio, which
turns "try five losses" into "try thirty" at no extra render cost.

What goes in, per target, and why each is there:

- `truth`, the answer. Every screen is measured relative to it.
- `radius` samples, random directions at known parameter distance. These map how the
  loss reads distance.
- `pedestal` samples, ONE parameter moved by 1e-8. The render differs by about -85 dB,
  which nothing can hear; the incumbent charges 0.194 for it. A candidate that also
  charges for this cannot be optimised, whatever else it does well.
- `fitted`, the vector the self-recovery bench actually produced. A real adversary at
  rms 0.31 that the incumbent scores at 1.31.
- `attractor`, where a local polish converges from near truth. The sharpest adversary
  there is: same distance from truth as a random near point, far lower loss under the
  incumbent, which is exactly why the search settles there instead of at the answer.
- `shuffled` and `delayed`, the deliberately-wrong controls CLAUDE.md already calibrates
  against. Not renders. They guard the screens against a degenerate winner: a loss that
  returns nearly zero for everything passes every smoothness test ever devised.

Mono, because the incumbent objective is mono and stereo width is a separate question
this bake-off deliberately defers. float32, because the effects being measured live at
-85 dB and float16 would erase them.

Run:  PYTHONPATH=scripts uv run python scripts/losscorpus.py --only t00
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import librosa
import numpy as np

import synth
from basin import perturb, rms
from bend2 import bend_curve
from selfrecover import TIER_A, load_target
from stage2 import CORE, DUR, SR, Objective, load_notes, run_cma

CORPUS = os.path.join(TIER_A, "corpus")
RADII = (0.0002, 0.002, 0.02, 0.05, 0.10, 0.33)
PEDESTAL = ("cutoff", "dlyTime")


def attractor(obj: Objective, truth: np.ndarray, args) -> np.ndarray:
    """Where a local polish from near truth actually ends up."""
    x0 = perturb(truth, 0.01, np.random.default_rng(args.seed + 991))
    free_all = [p.name for p in synth.PARAMS]
    xc, _ = run_cma(obj, x0, CORE, 2, 0.02, 301, "attr-core")
    xf, _ = run_cma(obj, xc, free_all, args.attractor_gens, 0.014, 302, "attr-full")
    return np.clip(np.asarray(xf), 0.0, 1.0)


def build(meta: dict, args) -> None:
    obj = Objective(load_notes(), target_path=meta["wav"])
    truth = np.array(meta["normalized"], dtype=float)
    rng = np.random.default_rng(args.seed)

    samples: list[tuple[str, float, np.ndarray | None]] = [("truth", 0.0, truth)]
    for r in RADII:
        for k in range(args.dirs):
            x = perturb(truth, r, rng)
            samples.append((f"radius:{r}:{k}", rms(x, truth), x))
    for name in PEDESTAL:
        x = truth.copy()
        i = synth.PARAM_INDEX[name]
        x[i] = min(1.0, truth[i] + 1e-8)
        samples.append((f"pedestal:{name}", rms(x, truth), x))

    fit = os.path.join(TIER_A, f"{meta['id']}_recover.json")
    if os.path.exists(fit):
        x = np.array(json.load(open(fit))["normalized"], dtype=float)
        samples.append(("fitted", rms(x, truth), x))

    xa = attractor(obj, truth, args)
    samples.append(("attractor", rms(xa, truth), xa))

    audio: dict[str, np.ndarray] = {}
    labels, dists = [], []
    for label, d, x in samples:
        audio[label] = obj.render(x).mean(axis=0).astype(np.float32)
        labels.append(label)
        dists.append(d)
        print(f"[{meta['id']}] {label:22s} rms {d:.4f}")

    # The wrong-answer controls, built from the target itself rather than rendered.
    tgt = librosa.load(meta["wav"], sr=SR, mono=True)[0].astype(np.float32)
    frames = tgt[: len(tgt) // 512 * 512].reshape(-1, 512)
    audio["shuffled"] = frames[rng.permutation(len(frames))].reshape(-1).copy()
    audio["delayed"] = np.concatenate([np.zeros(SR, np.float32), tgt])[: len(tgt)]
    for label in ("shuffled", "delayed"):
        labels.append(label)
        dists.append(float("nan"))     # no parameter vector, so no distance

    os.makedirs(CORPUS, exist_ok=True)
    np.savez(os.path.join(CORPUS, f"{meta['id']}.npz"),
             target=tgt, labels=np.array(labels), dists=np.array(dists), **audio)
    print(f"[{meta['id']}] wrote {len(labels)} samples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=TIER_A)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--dirs", type=int, default=2)
    ap.add_argument("--attractor-gens", type=int, default=25)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    metas = [load_target(p)
             for p in sorted(glob.glob(os.path.join(args.targets, "t[0-9][0-9].json")))]
    if args.only:
        metas = [m for m in metas if m["id"] in args.only]
    for m in metas:
        build(m, args)


if __name__ == "__main__":
    main()
