"""Tier A: targets this synth rendered itself, with the parameters recorded.

Why this exists. `data/original.wav` is one clip, so every stage-2 result is n=1 and
nothing can be validated: a change that helps here may be helping the pad or helping
the optimiser and there is no way to tell them apart. A self-rendered target has a
known answer, so the question "did the optimiser find it" has a yes or no.

The targets are reachable BY CONSTRUCTION. That is the whole point, and it is why the
renderer here has to be the one `stage2.Objective` uses: the same 24-voice PadRenderer,
the same notes from the frozen transcription, and the same sample-accurate bend curve.
A target rendered without the bend would sit outside the set the fit can reach and the
bench would report an optimiser failure that is really a setup mistake.

Two things a naive sampler gets wrong here, both measured:

- Clipping. `outGain` reaches 1.8 and 26 EQ bands reach +18 dB each, so a uniform draw
  overshoots full scale often. soundfile clips on write, and a clipped target is not
  reachable by any parameter vector, so those draws are rejected rather than saved.
- Independent per-band EQ gains. The 26-band cascade has a near-singular alternating
  direction; `fit_eq_full` penalises curvature precisely to keep out of it. A target
  drawn with independent band gains has that direction baked in and is unrecoverable
  for a reason that is about identifiability, not about the optimiser. The default
  therefore draws the curve from four low-order DCT modes. `--eq-mode iid` restores the
  independent draw for anyone who wants to measure the pathological corner, and
  `--eq-mode flat` pins the whole bank at 0 dB to test the 29 macros alone.

Run:  PYTHONPATH=scripts uv run python scripts/selfgen.py --n 8
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import qmc

import synth
from bend2 import bend_curve
from stage2 import DUR, SR, load_notes

TIER_A = "out/tierA"

EQ_DOF = 4          # DCT modes behind a smooth EQ curve
EQ_SCALE = 6.0      # dB per mode; four modes clip to the +-18 dB rail only rarely

# A target has to survive the round trip to PCM_24 unchanged, and has to be loud enough
# that the loss is looking at the patch rather than at the dither floor. The incumbent
# render sits at rms 0.16, so 0.02 is about 18 dB below anything we would call a patch.
PEAK_MAX = 0.99
RMS_MIN = 0.02


def eq_curve(coeffs: np.ndarray, n: int = synth.N_EQ) -> np.ndarray:
    """Low-order DCT modes -> 26 band gains in dB, clipped to the bank's rail."""
    i = np.arange(n)
    modes = np.stack([np.cos(np.pi * k * (i + 0.5) / n) for k in range(len(coeffs))])
    return np.clip(EQ_SCALE * (coeffs @ modes), -synth.EQ_LIMIT, synth.EQ_LIMIT)


def sample_vectors(n: int, eq_mode: str, seed: int) -> np.ndarray:
    """Sobol draws in the normalised box, one row per candidate patch.

    Sobol rather than uniform random: with 29 macros and 8 targets, an unstratified draw
    leaves whole corners of the box empty, and the bench's job is coverage.
    """
    names = [p.name for p in synth.PARAMS]
    n_eq = sum(1 for s in names if s.startswith("eq"))
    n_macro = len(names) - n_eq
    d = n_macro if eq_mode == "flat" else n_macro + (n_eq if eq_mode == "iid" else EQ_DOF)
    m = int(np.ceil(np.log2(max(n, 2))))
    u = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m)
    out = np.zeros((len(u), len(names)))
    out[:, :n_macro] = u[:, :n_macro]
    if eq_mode == "iid":
        out[:, n_macro:] = u[:, n_macro:]
    elif eq_mode == "smooth":
        for j, row in enumerate(u):
            g = eq_curve(2.0 * row[n_macro:] - 1.0, n_eq)
            out[j, n_macro:] = (g + synth.EQ_LIMIT) / (2.0 * synth.EQ_LIMIT)
    else:
        out[:, n_macro:] = 0.5   # 0 dB is the cascade's identity
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="targets to keep")
    ap.add_argument("--eq-mode", choices=("smooth", "iid", "flat"), default="smooth")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=TIER_A)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    notes = load_notes()
    renderer = synth.PadRenderer(n_voices=24)
    renderer.set_bend(bend_curve(int(DUR * SR) + SR))

    # Oversample: draws that clip or come out near-silent are thrown away, and the
    # Sobol balance matters less than getting `n` usable targets.
    cand = sample_vectors(args.n * 4, args.eq_mode, args.seed)
    kept, rejected = [], []
    for x in cand:
        if len(kept) >= args.n:
            break
        audio = synth.render_with(x, notes, DUR, renderer=renderer)
        peak = float(np.abs(audio).max())
        rms = float(np.sqrt((audio ** 2).mean()))
        if not np.isfinite(audio).all():
            rejected.append("nonfinite")
            continue
        if peak > PEAK_MAX:
            rejected.append(f"clip peak={peak:.2f}")
            continue
        if rms < RMS_MIN:
            rejected.append(f"quiet rms={rms:.4f}")
            continue

        tid = f"t{len(kept):02d}"
        wav = os.path.join(args.out, f"{tid}.wav")
        synth.write_render(wav, audio)
        meta = {
            "id": tid,
            "wav": wav,
            "eq_mode": args.eq_mode,
            "seed": args.seed,
            # param_names pins the contract harder than a length does: PARAMS is
            # append-only, so a length check catches an append but not a reorder, and
            # a reorder would silently compare truth against the wrong slider.
            "n_params": len(synth.PARAMS),
            "param_names": [p.name for p in synth.PARAMS],
            "normalized": [float(v) for v in x],
            "params": synth.denorm(x),
            "peak": peak,
            "rms": rms,
        }
        with open(os.path.join(args.out, f"{tid}.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        kept.append(tid)
        print(f"{tid}: peak {peak:.3f}  rms {rms:.4f}  -> {wav}")

    print(f"\nkept {len(kept)}, rejected {len(rejected)}")
    reasons = [r.split(" ")[0] for r in rejected]
    for r in sorted(set(reasons)):
        print(f"  {reasons.count(r)}x {r}")
    if len(kept) < args.n:
        raise SystemExit(f"only {len(kept)} of {args.n} targets survived; widen the draw")


if __name__ == "__main__":
    main()
