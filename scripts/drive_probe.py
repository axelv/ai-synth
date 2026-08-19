"""Does `drive` earn its place if the brightness controls are allowed to retune with it?

The drive x spread grid in verify_new_stages rejects every nonzero drive, but that grid
holds the other 27 parameters at a patch fitted without a waveshaper, which is not a
fair test: a shaper adds harmonics everywhere, so it can only pay off if the filter and
the tilt EQ give the extra brightness back. This runs the same CMA-ES search twice from
the same base, once with drive free and once with drive absent, so the difference
between the two arms is the parameter and nothing else.

Free set is the brightness axes a waveshaper trades against: cutoff, reso, envAmt, fS,
sqrMix, tilt, outGain. Result goes to out/drive_probe.json.
"""

from __future__ import annotations

import argparse
import json

import librosa

from metrics import band_db_error, lta_band_error
from stage2 import Objective, load_notes, run_cma
from synth import PARAM_INDEX, denorm, pad_normalized

SR = 44100
BRIGHT = ["cutoff", "reso", "envAmt", "fS", "sqrMix", "tilt", "outGain"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="out/patch_grid.json")
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--drive0", type=float, default=0.16, help="where the drive arm starts")
    args = ap.parse_args()

    orig, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    obj = Objective(load_notes())
    base = pad_normalized(json.load(open(args.base))["normalized"])
    base[PARAM_INDEX["drive"]] = 0.0

    out: dict[str, object] = {"base": args.base, "gens": args.gens, "sigma": args.sigma}
    for arm, free, d0 in (("no_drive", BRIGHT, 0.0),
                          ("with_drive", BRIGHT + ["drive"], args.drive0)):
        x0 = base.copy()
        x0[PARAM_INDEX["drive"]] = d0
        before = obj.calls
        x, loss = run_cma(obj, x0, free, args.gens, args.sigma, 700, arm)
        aud = obj.render(x)
        out[arm] = {
            "loss": loss,
            "renders": obj.calls - before,
            "drive": float(denorm(x)["drive"]),
            "mid_signed_db": band_db_error(orig, aud)["mean_signed_db"],
            "bands_db": lta_band_error(orig, aud),
            "normalized": x.tolist(),
        }
        d = out[arm]
        print(f"{arm:10s} loss {d['loss']:.4f}  drive {d['drive']:.3f}  "
              f"250-900 {d['mid_signed_db']:+.2f} dB  ({d['renders']} renders)")

    with open("out/drive_probe.json", "w") as fh:
        json.dump(out, fh, indent=2)
    delta = out["with_drive"]["loss"] - out["no_drive"]["loss"]
    print(f"with_drive - no_drive = {delta:+.4f} (negative means the shaper helped)")
    print("wrote out/drive_probe.json")


if __name__ == "__main__":
    main()
