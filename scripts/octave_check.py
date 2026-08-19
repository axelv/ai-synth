"""Decide the bass octave by low-band energy, which chroma cannot see.

The voicing search maximised chroma agreement and pushed every bass note up an
octave. Chroma is octave-invariant, so it cannot arbitrate this; sub-80 Hz
energy can, and the fundamentals were measured directly at 43.65 / 34.65 /
38.89 / 46.25 Hz.
"""

from __future__ import annotations

import json

import librosa
import numpy as np

from build_midi import INTRO_END, eval_patch
from metrics import report
from synth import PadRenderer

SR = 44100
DUR = 17.904

VOICINGS = [
    (3.45, 4.30, [24, 28, 33]),
    (4.30, 7.50, [7, 19, 24, 28]),
    (7.45, 10.45, [19, 24, 28, 31]),
    (10.40, 13.40, [7, 19, 24, 28]),
    (13.35, 16.10, [16, 19, 24, 28]),
    (16.05, 17.90, [19, 24, 28, 31]),
]
BASS_MEASURED = [27, 29, 25, 27, 30, 29]   # from pitch_probe (fundamentals)
BASS_SEARCH = [27, 41, 37, 39, 42, 41]     # what the chroma search chose
INTRO = [53, 55, 57, 59, 60, 62, 64, 65]


def build(basses):
    notes = [(p, 60, 0.02, INTRO_END - 0.02) for p in INTRO]
    for (t0, t1, vo), b in zip(VOICINGS, basses):
        d = t1 - t0
        notes.append((b, 100, t0, d))
        for s in vo:
            notes.append((b + s, 66, t0, d))
    return notes


def band_profile(y: np.ndarray) -> dict[str, float]:
    S = np.abs(librosa.stft(y, n_fft=8192, hop_length=2048))
    fr = librosa.fft_frequencies(sr=SR, n_fft=8192)
    # only the sustained section, where the bass is actually present
    t = librosa.times_like(S[0], sr=SR, hop_length=2048)
    sel = t >= 3.6
    out = {}
    tot = 0.0
    for name, (lo, hi) in {"<60": (0, 60), "60-120": (60, 120), "120-300": (120, 300), "300-1k": (300, 1000), ">1k": (1000, 22050)}.items():
        m = (fr >= lo) & (fr < hi)
        e = float(np.sqrt((S[np.ix_(m, np.where(sel)[0])] ** 2).sum()))
        out[name] = e
        tot += e
    return {k: v / tot * 100 for k, v in out.items()}


def main() -> None:
    orig, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    orig_st, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    r = PadRenderer(n_voices=64)
    r.set_params(eval_patch())

    print(f"{'':22s}" + "".join(f"{k:>9s}" for k in ("<60", "60-120", "120-300", "300-1k", ">1k")))
    po = band_profile(orig)
    print(f"{'ORIGINAL':22s}" + "".join(f"{po[k]:8.1f}%" for k in po))

    results = {}
    for label, basses in (("measured (F1/C#1/..)", BASS_MEASURED), ("search (F2/C#2/..)", BASS_SEARCH)):
        r.set_notes(build(basses))
        aud = r.render(DUR)
        pr = band_profile(aud.mean(axis=0))
        m = report(orig_st, aud)
        err = sum(abs(pr[k] - po[k]) for k in po)
        results[label] = (pr, m, err)
        print(f"{label:22s}" + "".join(f"{pr[k]:8.1f}%" for k in pr) + f"   band-err={err:5.1f}  chroma={m['chroma_agree']:.4f}  mel={m['mel_dist']:.2f}")

    best = min(results.items(), key=lambda kv: kv[1][2])
    print(f"\nlow-band evidence favours: {best[0]}")


if __name__ == "__main__":
    main()
