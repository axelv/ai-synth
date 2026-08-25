"""Add a nuisance sample to the corpus: same parameters, unfittable difference.

Faust oscillators are free running from the start of the render, so the phase they are at
when a note begins depends on WHEN the note begins, not on any parameter. Push every note
later by a fraction of a second, render, and trim the lead-in back off: the notes, their
pitches, their timing relative to each other and every one of the 55 parameters are
unchanged, and the audio still differs. Nothing in PARAMS reaches that difference, so
whatever a loss charges for it is charged for nothing.

Voice count was the obvious knob and it does not work: at 24 or 32 voices, with at most a
handful of notes sounding at once, allocation is identical and the render is bit for bit
the same. Recorded because it looks like it should work.

This is the sharpest screen in the set and it was nearly missed. A candidate can be
perfectly insensitive to inaudible change and still be useless if it reads a nuisance as
a parameter error, because then its minimum sits wherever the nuisance happens to point.

Appends `nuisance` to each existing corpus npz rather than rebuilding, since the attractor
polish in losscorpus costs half an hour and none of it changes here.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import synth
from bend2 import bend_curve
from losscorpus import CORPUS
from selfrecover import TIER_A, load_target
from stage2 import DUR, SR, load_notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, default=0.37,
                    help="seconds to push every note later; not a multiple of any "
                         "partial period here, so no pitch lands back in phase")
    args = ap.parse_args()

    notes = load_notes()
    off = int(round(args.shift * SR))
    shifted = [(p, v, s + args.shift, d) for p, v, s, d in notes]
    r = synth.PadRenderer(n_voices=24)
    r.set_notes(shifted)
    # The bend is sample indexed, so it has to slide with the notes or the glide would
    # land on the wrong ones and this would stop being a pure nuisance.
    r.set_bend(np.concatenate([np.ones(off), bend_curve(int(DUR * SR) + SR)]))

    for path in sorted(glob.glob(os.path.join(CORPUS, "t[0-9][0-9].npz"))):
        tid = os.path.basename(path)[:-4]
        meta = load_target(os.path.join(TIER_A, f"{tid}.json"))
        truth = np.array(meta["normalized"], dtype=float)
        r.set_params(synth.denorm(truth))
        audio = r.render(DUR + args.shift).mean(axis=0)[off:]

        with np.load(path, allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
        ref = data["truth"]
        n = min(len(ref), len(audio))
        rel = float(np.abs(audio[:n] - ref[:n]).max() / (np.abs(ref).max() + 1e-12))
        S = lambda x: np.abs(np.fft.rfft(x[:n]))
        spec = float(np.linalg.norm(S(audio) - S(ref)) / (np.linalg.norm(S(ref)) + 1e-12))
        data["nuisance"] = audio.astype(np.float32)
        data["labels"] = np.append(data["labels"], "nuisance")
        data["dists"] = np.append(data["dists"], np.nan)   # zero parameter distance
        np.savez(path, **data)
        print(f"{tid}: nuisance +{args.shift}s, waveform delta {rel:.3f}, "
              f"relative spectral delta {spec:.4f}")


if __name__ == "__main__":
    main()
