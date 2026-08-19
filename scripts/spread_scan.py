"""How much stereo width does the loss actually want, and how much does the clip have?

The stage-2 loss is mono, so it cannot arbitrate width at all: it only sees `spread`
through the second-order effect the pan gains have on the mono sum. A one-dimensional
scan at the delivered patch therefore separates two different questions that adding the
parameter conflates - which spread the optimiser picked, and which spread actually
matches the original's L/R decorrelation. Both go to out/spread_scan.json.
"""

from __future__ import annotations

import argparse
import json

import librosa

from metrics import stereo_decorrelation
from stage2 import Objective, load_notes
from synth import PARAM_INDEX, PARAMS, denorm, normalize_one, pad_normalized

SR = 44100
VALUES = (0.0, 0.2, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="out/patch.json")
    ap.add_argument("--out", default="out/spread_scan.json")
    args = ap.parse_args()

    orig, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    obj = Objective(load_notes())
    base = pad_normalized(json.load(open(args.patch))["normalized"])
    prm = PARAMS[PARAM_INDEX["spread"]]
    rows = []
    for v in VALUES:
        x = base.copy()
        x[PARAM_INDEX["spread"]] = normalize_one(prm, v)
        aud = obj.render(x)
        rows.append({"spread": v, "loss": obj.loss_of(aud),
                     "decorr": stereo_decorrelation(aud)})
        print(f"spread {v:.2f}  loss {rows[-1]['loss']:.4f}  decorr {rows[-1]['decorr']:.4f}")

    target = stereo_decorrelation(orig)
    best_loss = min(rows, key=lambda r: r["loss"])
    closest = min(rows, key=lambda r: abs(r["decorr"] - target))
    out = {"patch": args.patch, "delivered_spread": float(denorm(base)["spread"]),
           "original_decorr": target, "rows": rows,
           "lowest_loss": best_loss, "closest_decorr": closest}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"original decorr {target:.4f}; loss prefers spread {best_loss['spread']:.2f} "
          f"(decorr {best_loss['decorr']:.4f}); closest width is spread "
          f"{closest['spread']:.2f} (decorr {closest['decorr']:.4f}, loss {closest['loss']:.4f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
