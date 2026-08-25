"""Screen candidate losses against the cached corpus, cheap tests first.

The bake-off exists because the incumbent objective was measured to rank patches wrongly:
see `basin.py` and CLAUDE.md. A candidate has to earn an expensive CMA-ES recovery run,
and these screens are what it earns it with. All four run against `losscorpus` renders,
so a candidate costs seconds here and hours only if it survives.

Every screen is reported in units of the loss's OWN dynamic range,

    unit = L(rms 0.10) - L(truth)

so candidates on wildly different scales are comparable and no candidate wins by being
small. A loss with a non-positive unit is broken and is reported as such rather than
scored.

- `discrim`, is it awake. min(L(shuffled), L(delayed)) in units. A loss returning nearly
  zero for everything passes every smoothness test ever devised, so this runs first and
  a low value disqualifies. CLAUDE.md's own calibration says a deliberately-wrong
  control must sit clearly above a plausible-but-wrong candidate.
- `pedestal`, is it continuous. L(one parameter moved 1e-8) in units. The render differs
  by about -85 dB. The incumbent spends 10% of its entire range on this; it is the
  measured reason the search cannot descend. Lower is better, target <= 0.01.
- `near_wins`, is the minimum in the right place. (L(attractor) - L(near truth)) in
  units, where `near` is the closest sampled radius. The attractor is where a local
  polish actually converges, rms ~0.017 from truth. Under the incumbent it scores BELOW
  points ten times closer to the answer, which is the whole failure in one number.
  Positive is necessary; a candidate that is negative here cannot be optimised to truth
  no matter how smooth it is.
- `spearman`, does loss order distance. Rank correlation between the loss and true
  normalised parameter distance across the corpus, adversaries included.

Run:  PYTHONPATH=scripts uv run python scripts/bakeoff.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
from scipy.stats import spearmanr

import losses as L
from losscorpus import CORPUS

# The radius at which the escape was actually measured: a polish started 0.001 from
# truth walked away to 0.015+. So 0.002 is the neighbourhood a real initialiser could
# land in, and the one the attractor has to lose against. 0.0002 is reported alongside
# it because at that radius even the incumbent wins, and quoting only the flattering
# radius is how a screen gets rigged.
NEAR = 0.002
NEAR_TIGHT = 0.0002
UNIT = 0.10        # the radius that defines one unit of dynamic range


def pick(labels: np.ndarray, prefix: str) -> list[int]:
    return [i for i, s in enumerate(labels) if str(s).startswith(prefix)]


def screen(name: str, factory, npz) -> dict:
    labels = list(npz["labels"])
    dists = np.asarray(npz["dists"], dtype=float)
    target = np.asarray(npz["target"], dtype=np.float32)

    t0 = time.time()
    score = factory(target, L.SR)
    vals = np.array([score(np.asarray(npz[str(s)], dtype=np.float32)) for s in labels])
    ms = 1000.0 * (time.time() - t0) / max(len(labels), 1)

    idx = {str(s): i for i, s in enumerate(labels)}
    truth = vals[idx["truth"]]
    far = float(np.mean([vals[i] for i in pick(labels, f"radius:{UNIT}:")]))
    unit = far - truth
    if not np.isfinite(unit) or unit <= 0:
        return {"loss": name, "broken": True, "unit": float(unit), "ms": ms}

    ped = float(np.mean([vals[i] for i in pick(labels, "pedestal:")]))
    near = float(np.min([vals[i] for i in pick(labels, f"radius:{NEAR}:")]))
    tight = float(np.min([vals[i] for i in pick(labels, f"radius:{NEAR_TIGHT}:")]))
    attr = vals[idx["attractor"]]
    ctrl = float(min(vals[idx["shuffled"]], vals[idx["delayed"]]))
    nuis = vals[idx["nuisance"]] if "nuisance" in idx else float("nan")

    fin = np.isfinite(dists)
    rho = float(spearmanr(dists[fin], vals[fin]).statistic)

    return {
        "loss": name, "broken": False, "ms": ms, "unit": float(unit),
        "discrim": (ctrl - truth) / unit,
        "pedestal": (ped - truth) / unit,
        "near_wins": (attr - near) / unit,
        "near_tight": (attr - tight) / unit,
        # What the loss charges for a difference no parameter can produce. Every bit of
        # this is an irreducible floor the fit can never remove, and if it is comparable
        # to the loss's whole dynamic range then the fit is mostly chasing phase.
        "nuisance": (nuis - truth) / unit,
        "spearman": rho,
        "raw": {"truth": float(truth), "near": near, "tight": tight,
                "nuisance": float(nuis), "attractor": float(attr),
                "far": far, "control": ctrl, "pedestal": ped},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--only", action="append", default=[], help="loss name, repeatable")
    ap.add_argument("--out", default="out/tierA/bakeoff.json")
    args = ap.parse_args()

    found = L.load_candidates()
    if found:
        print(f"loaded candidate modules: {', '.join(found)}")
    names = [n for n in L.LOSSES if not args.only or n in args.only]
    paths = sorted(glob.glob(os.path.join(args.corpus, "t[0-9][0-9].npz")))
    if not paths:
        raise SystemExit(f"no corpus in {args.corpus}; run losscorpus.py first")
    print(f"{len(names)} losses against {len(paths)} targets\n")

    rows: dict[str, list[dict]] = {}
    for name in names:
        rows[name] = []
        for p in paths:
            with np.load(p, allow_pickle=False) as npz:
                rows[name].append(screen(name, L.LOSSES[name], npz))

    def mean(name: str, key: str) -> float:
        v = [r[key] for r in rows[name] if not r["broken"]]
        return float(np.mean(v)) if v else float("nan")

    # The incumbent sets the discrimination bar rather than a number chosen in advance.
    # A replacement is allowed to be no blinder than the thing it replaces, and the
    # incumbent's own value (a frame-shuffled target reads as LESS wrong than a patch
    # 0.10 away) is a measured property of this near-stationary pad, not a failure of
    # the screen.
    base = mean("incumbent", "discrim") if "incumbent" in rows else 0.5

    print(f"{'loss':20s} {'discrim':>8s} {'pedestal':>9s} {'nuisance':>9s} "
          f"{'near@2e-3':>10s} {'near@2e-4':>10s} {'spearman':>9s} {'sig/nui':>8s}   verdict")
    ranked = sorted(names, key=lambda n: (-mean(n, "near_wins"), mean(n, "pedestal")))
    summary = []
    for n in ranked:
        if all(r["broken"] for r in rows[n]):
            print(f"{n:20s} {'BROKEN':>8s}")
            continue
        d, p, w, t, s = (mean(n, "discrim"), mean(n, "pedestal"), mean(n, "near_wins"),
                         mean(n, "near_tight"), mean(n, "spearman"))
        nz = mean(n, "nuisance")
        why = []
        if d < base:
            why.append(f"blinder than incumbent ({d:.2f} < {base:.2f})")
        if p > 0.01:
            why.append("pedestal")
        if w <= 0.0:
            why.append("minimum misplaced")
        if s < 0.5:
            why.append("ranks badly")
        # A loss that charges as much for unfittable phase as for a real parameter move
        # of rms 0.10 is spending its whole range on something the fit cannot change.
        if nz > 1.0:
            why.append(f"nuisance {nz:.2f}")
        verdict = "SCREEN C" if not why else "rejected: " + ", ".join(why)
        # The ratio that ranks candidates sensibly: parameter signal per unit of
        # unfittable phase. The incumbent's is negative, which is the whole problem in
        # one number.
        ratio = w / nz if nz > 0 else float("nan")
        print(f"{n:20s} {d:8.2f} {p:9.4f} {nz:9.3f} {w:10.4f} {t:10.4f} {s:9.3f} "
              f"{ratio:8.2f}   {verdict}")
        summary.append({"loss": n, "discrim": d, "pedestal": p, "nuisance": nz,
                        "near_wins": w, "near_tight": t, "spearman": s,
                        "sig_per_nuisance": ratio, "ms": mean(n, "ms"),
                        "verdict": verdict})

    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "per_target": rows}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
