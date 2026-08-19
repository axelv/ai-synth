"""Measure the two documented defects on rendered audio, before and after a refit.

Neither defect is in the stage-2 loss: the loss is mono, so L/R decorrelation is
invisible to it, and its spectral-convergence term is only loosely tied to absolute
level, so a band deficit can survive a lower loss. That is exactly why adding drive
and spread cannot be judged by the loss alone. Both numbers therefore come from here,
from a re-render through PadRenderer, and land in out/defect_check.json so every
figure quoted in out/report.md is traceable to a file.

    uv run python scripts/defect_check.py --wav delivered=out/render.wav \
        --patch refit27=out/patch_27.json --patch extended=out/patch_ext.json
"""

from __future__ import annotations

import argparse
import json

import librosa
import numpy as np

from metrics import band_db_error, lta_band_error, report, stereo_decorrelation
from stage2 import Objective, load_notes
from synth import denorm, pad_normalized, write_render

SR = 44100


def measure(orig: np.ndarray, audio: np.ndarray, obj: Objective) -> dict[str, object]:
    return {
        "loss": obj.loss_of(audio),
        "decorr": stereo_decorrelation(audio),
        "rms": float(np.sqrt((audio.astype(np.float64) ** 2).mean())),
        "mid": band_db_error(orig, audio),
        "bands_db": lta_band_error(orig, audio),
        "metrics": report(orig, audio),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", action="append", default=[], help="label=path, measured as is")
    ap.add_argument("--patch", action="append", default=[], help="label=patch.json, re-rendered")
    ap.add_argument("--out", default="out/defect_check.json")
    args = ap.parse_args()

    orig, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    obj = Objective(load_notes())
    out: dict[str, object] = {
        "original": {"decorr": stereo_decorrelation(orig),
                     "rms": float(np.sqrt((orig.astype(np.float64) ** 2).mean()))},
    }

    for spec in args.wav:
        label, _, path = spec.partition("=")
        a, _ = librosa.load(path, sr=SR, mono=False)
        out[label] = measure(orig, a, obj) | {"source": path}

    for spec in args.patch:
        label, _, path = spec.partition("=")
        z = pad_normalized(json.load(open(path))["normalized"])
        a = obj.render(z)
        wav = f"out/{label}_render.wav"
        write_render(wav, a)
        out[label] = measure(orig, a, obj) | {"source": path, "render": wav,
                                              "params": denorm(z)}

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"{'label':12s} {'loss':>8s} {'decorr':>8s} {'250-900':>8s} {'2-6k':>8s} {'rms':>7s}")
    for label, d in out.items():
        if label == "original":
            print(f"{label:12s} {'-':>8s} {d['decorr']:8.4f} {'-':>8s} {'-':>8s} {d['rms']:7.4f}")
            continue
        print(f"{label:12s} {d['loss']:8.4f} {d['decorr']:8.4f} "
              f"{d['mid']['mean_signed_db']:8.2f} {d['bands_db']['2000_6000']:8.2f} "
              f"{d['rms']:7.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
