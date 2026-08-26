"""Does a more diverse crowd actually raise the retrieval margin, or only look like it?

The day-1 calibration used a crowd of random perturbations at rms 0.33 around one target,
all from one synth playing one note list. That is a fair model of a library of near
relatives and a poor model of a real preset collection, which spans architectures and
sound designers. The obvious objection is that a real crowd is far more spread out, and
a more spread out crowd should be easier to stand out from.

Arithmetic says that objection is viable. Scaling every crowd distance by k gives
margin(k) = (k*mu - near) / (k*sd), which climbs to a ceiling of 1/CoV: measured at 3.8
to 4.9, against the 3.42 a library of 2000 demands. On three of four targets a k of only
1.3 to 1.6 would clear it.

That is a prediction, and it is cheap to test without any preset plumbing: draw patches
from the whole parameter box instead of a shell around one target, which is the widest
crowd this synth can produce, and see whether the margin climbs the way the model says.
If it does not climb here it will not climb for real presets either, and the mapping work
is not worth starting. If it does, the extrapolation has survived its first real check.

Note what this cannot settle. Random draws from a parameter box are not designed presets.
Designed sounds are plausible by construction and may cluster far tighter than random
ones, which is the direction that would hurt. This probe tests the mechanism, not the
population.

Run:  PYTHONPATH=scripts uv run python scripts/widecrowd.py --n 128
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import synth
from bend2 import bend_curve
from earpanel import (COMBO_ALL, blocks, delevel, describe, dist, needed_margin,
                      radius_labels, weights)
from losscorpus import CORPUS
from selfgen import PEAK_MAX, RMS_MIN, sample_vectors
from stage2 import DUR, SR, load_notes

WIDE = os.path.join(CORPUS, "wide.npz")


def build(path: str, n: int, seed: int, scale_clip: bool = False) -> dict[str, np.ndarray]:
    notes = load_notes()
    r = synth.PadRenderer(n_voices=24)
    r.set_bend(bend_curve(int(DUR * SR) + SR))

    # iid EQ, not selfgen's smooth 4-mode curve: the smooth manifold exists to make
    # targets plausible, and plausibility is exactly the constraint being relaxed here.
    # Rejecting clippers throws away three quarters of the draws, which is affordable at
    # 60 patches and not at 2000. Scaling them down instead keeps the timbre and loses
    # only the level, and the fingerprint is level-invariant anyway.
    cand = sample_vectors(n if scale_clip else n * 2, "iid", seed)
    out, dropped = {}, {"clip": 0, "quiet": 0, "nonfinite": 0}
    for x in cand:
        if len(out) >= n:
            break
        a = synth.render_with(x, notes, DUR, renderer=r)
        if not np.isfinite(a).all():
            dropped["nonfinite"] += 1
            continue
        peak = float(np.abs(a).max())
        if peak > PEAK_MAX:
            if not scale_clip:
                dropped["clip"] += 1
                continue
            a = a * (PEAK_MAX / peak)
            dropped["clip"] += 1
        if float(np.sqrt((a ** 2).mean())) < RMS_MIN:
            dropped["quiet"] += 1
            continue
        out[f"w{len(out):03d}"] = (a.mean(axis=0) if a.ndim > 1 else a).astype(np.float32)
    print(f"kept {len(out)} of {len(cand)} draws; dropped {dropped}")
    os.makedirs(CORPUS, exist_ok=True)
    np.savez(path, **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--block", default="all4")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--scale-clip", action="store_true",
                    help="scale loud draws down instead of discarding them")
    ap.add_argument("--wide", default=WIDE)
    args = ap.parse_args()

    if args.rebuild or not os.path.exists(args.wide):
        build(args.wide, args.n, args.seed, args.scale_clip)
    with np.load(args.wide, allow_pickle=False) as z:
        wide_audio = {k: z[k] for k in z.files}
    print(f"wide crowd: {len(wide_audio)} patches\n")

    desc = {os.path.basename(p)[:-4]: describe(p, True)
            for p in sorted(glob.glob(os.path.join(CORPUS, "t[0-9][0-9].npz")))}
    tids = sorted(desc)
    w = weights(desc, args.block)

    def fp(a: np.ndarray) -> np.ndarray:
        b = {k: delevel(v) for k, v in blocks(a.astype(np.float64)).items()}
        return np.concatenate([b[q] for q in COMBO_ALL])

    wide = [fp(a) for a in wide_audio.values()]

    print(f"{'tgt':4s} | {'--- tight crowd (rms 0.33) ---':^30s} | "
          f"{'--- wide crowd (whole box) ---':^30s} | {'rank of the':>12s}")
    print(f"{'':4s} | {'mu':>7s} {'CoV':>6s} {'margin':>7s} {'ceil':>6s} | "
          f"{'mu':>7s} {'CoV':>6s} {'margin':>7s} {'ceil':>6s} | {'relative':>12s}")
    rows = []
    for tid in tids:
        d = desc[tid]
        t = d["truth"][args.block]
        near = float(np.mean([dist(t, d[l][args.block], w)
                              for l in radius_labels(d, 0.05)]))
        tight = np.array([dist(t, desc[o][l][args.block], w)
                          for o in tids for l in radius_labels(desc[o], 0.33)]
                         + [dist(t, desc[o]["truth"][args.block], w)
                            for o in tids if o != tid])
        wd = np.array([dist(t, f, w) for f in wide])
        mk = lambda c: ((c.mean() - near) / c.std(ddof=1), c.mean() / c.std(ddof=1),
                        c.std(ddof=1) / c.mean(), c.mean())
        mt, ct, covt, mut = mk(tight)
        mw, cw, covw, muw = mk(wd)
        rank = 1 + int((wd < near).sum())          # empirical, no gaussian assumption
        rows.append((mt, mw, covt, covw, rank, muw / mut))
        print(f"{tid:4s} | {mut:7.1f} {covt:6.3f} {mt:7.2f} {ct:6.2f} | "
              f"{muw:7.1f} {covw:6.3f} {mw:7.2f} {cw:6.2f} | {rank:5d} of {len(wd)+1:5d}")

    mt, mw = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])
    print(f"\nmean margin  tight {mt:.2f}  ->  wide {mw:.2f}   "
          f"({'up' if mw > mt else 'DOWN'} {abs(mw-mt):.2f})")
    print(f"mean crowd distance grew {np.mean([r[5] for r in rows]):.2f}x "
          f"(this is the measured k)")
    print(f"mean CoV     tight {np.mean([r[2] for r in rows]):.3f}  ->  "
          f"wide {np.mean([r[3] for r in rows]):.3f}")
    print(f"relative ranked 1st on {sum(1 for r in rows if r[4] == 1)} of {len(rows)} targets")
    print(f"\nfor reference, margin a library needs: "
          f"N=165 {needed_margin(165):.2f}, N=2000 {needed_margin(2000):.2f}, "
          f"N=10000 {needed_margin(10000):.2f}")


if __name__ == "__main__":
    main()
