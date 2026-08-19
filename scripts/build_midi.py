"""Stage 1 (cont.): build MIDI variants and pick the voicing by measured score.

Established by analysis (see report):
  bass line  F1 -> C#1 -> D#1 -> F#1 -> F1   (exact A440 equal temperament)
  key        Bb minor / Db major
  intro      same pad, highpassed (<2% energy under 200 Hz) until the bass
             drops in at 3.45 s

The upper voicing is genuinely ambiguous from the spectrum alone, because a
sawtooth's own harmonics plus a ~70-cent unison spread reproduce much of the
"chord". So we render each hypothesis and let the metrics decide.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from metrics import report
from synth import PadRenderer, denorm, norm_defaults

SR = 44100
DUR = 17.904

# (start, end, bass_midi, upper_triad_semitones_above_bass)
# triads read off the NNLS residuals + Bb-minor context
CHORDS = [
    (3.45, 7.50, 29, [16, 19, 24]),   # F1  -> A3 C4 F4   (F major)
    (7.45, 10.45, 25, [21, 25, 28]),  # C#1 -> Bb3 Db4 F4 (Db6 / Bbm7)
    (10.40, 13.40, 27, [21, 24, 28]), # D#1 -> C4 Eb4 G4  (Eb6 / Cm7)
    (13.35, 16.10, 30, [16, 20, 23]), # F#1 -> Bb2 Db3 F3 (Gb major)
    (16.05, 17.90, 29, [16, 19, 24]), # F1  -> A3 C4 F4   (F major)
]

INTRO_END = 3.55


def variant(name: str) -> list[tuple[int, int, float, float]]:
    """Return [(pitch, vel, start, dur)] for a named hypothesis."""
    notes: list[tuple[int, int, float, float]] = []

    # ---- intro: pad is highpassed in the original, so it is voiced in the
    # register that actually sounds (F3/F4) rather than as a sub-heavy F1.
    if name != "bass_only":
        for p, v in ((53, 62), (57, 55), (60, 58), (65, 52)):
            notes.append((p, v, 0.02, INTRO_END - 0.02))
    else:
        notes.append((41, 70, 0.02, INTRO_END - 0.02))

    for t0, t1, bass, triad in CHORDS:
        dur = t1 - t0
        notes.append((bass, 100, t0, dur))
        if name == "bass_only":
            continue
        if name in ("bass_oct", "full"):
            notes.append((bass + 12, 72, t0, dur))
        if name in ("triad", "full"):
            for s in triad:
                notes.append((bass + s, 64, t0, dur))
    return notes


def to_pm(notes, path: str) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=82.0)
    inst = pretty_midi.Instrument(program=90, name="pad")
    for p, v, s, d in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p), start=float(s), end=float(s + d)))
    pm.instruments.append(inst)
    pm.write(path)


def eval_patch() -> dict[str, float]:
    """Reference patch used only to compare transcription variants.

    Prefers a calibrated patch if one has been fitted, so that voicing choices
    are judged through a synth whose spectral balance resembles the target
    rather than through an arbitrarily dark one.
    """
    import os

    if os.path.exists("out/patch_ref.json"):
        with open("out/patch_ref.json") as fh:
            return json.load(fh)["params"]

    p = denorm(norm_defaults())
    p.update(
        detune=65.0, uniMix=0.85, subLvl=0.35, sqrMix=0.0,
        cutoff=900.0, reso=1.0, envAmt=400.0, kbdTrk=0.25,
        aA=0.6, aD=1.2, aS=0.9, aR=1.8, fA=0.8, fD=1.5, fS=0.8,
        lfoAmt=0.0, chDepth=0.25, chRate=0.4, dlyWet=0.0,
        revSize=0.9, revDamp=0.5, revWet=0.45, tilt=-0.2, outGain=0.5,
    )
    return p


def main() -> None:
    orig, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    r = PadRenderer(n_voices=32)
    r.set_params(eval_patch())

    rows = []
    for name in ("bass_only", "bass_oct", "triad", "full"):
        notes = variant(name)
        r.set_notes(notes)
        aud = r.render(DUR)
        m = report(orig, aud)
        m["variant"] = name
        m["n_notes"] = len(notes)
        rows.append(m)
        print(
            f"{name:10s} notes={len(notes):3d}  chroma={m['chroma_agree']:.4f}  "
            f"mel={m['mel_dist']:.3f}  env_l1={m['env_l1']:.3f}  onsetF={m['onset_f']:.3f}"
        )
        sf.write(f"out/stage1_{name}.wav", aud.T, SR)

    best = max(rows, key=lambda m: m["chroma_agree"] - 0.02 * m["mel_dist"])
    print(f"\nbest variant: {best['variant']}")
    notes = variant(best["variant"])
    to_pm(notes, "out/transcription.mid")
    with open("out/stage1_metrics.json", "w") as fh:
        json.dump({"variants": rows, "chosen": best["variant"]}, fh, indent=2)
    print("wrote out/transcription.mid")


if __name__ == "__main__":
    main()
