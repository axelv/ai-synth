"""Adopt the author's chord structure: F > Dbm > Ebsus2 (last bent up +3 at half length).

The author wrote the part in GarageBand and cannot export MIDI, so the chord
qualities come from them. Everything checkable was checked against the audio
first: the +3.00 semitone ratio and the 13.25 s midpoint are exact, and the
sus2-vs-major distinction was shown to be unresolvable by measurement because
Bb3 and Ab4 coincide with harmonics 5 and 9 of the Gb bass.

Compares the author's structure against my earlier 6-region all-major reading on
the stage-2 loss and on chroma.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import soundfile as sf

import bend2
from build_midi import to_pm
from metrics import chroma, mel_db, report
from stage2 import Objective
from synth import PadRenderer, denorm

SR = 44100
DUR = 17.904
INTRO = [49, 53]  # C#3 + F3, measured; a third apart as in the DAW screenshot

# author's structure. (label, t0, t1, bass_midi, offsets above bass)
AUTHOR = [
    ("F major",   3.45,  7.50, 29, [16, 19, 24, 28]),
    ("Db minor",  7.45, 10.45, 25, [12, 19, 24, 27, 31]),
    ("Ebsus2",   10.40, 16.10, 27, [12, 19, 24, 26, 31]),   # bent +3 at 13.35
    ("F major",  16.05, 17.90, 29, [12, 19, 24, 28, 31]),
]
# same, but keeping my short 3.45-4.30 region separate instead of folding into F
AUTHOR_SPLIT = [
    ("Eb(top)",   3.45,  4.30, 39, [24, 28, 33]),
    ("F major",   4.30,  7.50, 29, [16, 19, 24, 28]),
    ("Db minor",  7.45, 10.45, 25, [12, 19, 24, 27, 31]),
    ("Ebsus2",   10.40, 16.10, 27, [12, 19, 24, 26, 31]),
    ("F major",  16.05, 17.90, 29, [12, 19, 24, 28, 31]),
]


def build(regions, intro, vel):
    notes = [(p, vel["intro"], 0.02, 3.53) for p in intro]
    for _, t0, t1, bass, offs in regions:
        d = t1 - t0
        notes.append((bass, 100, t0, d))
        for s in offs:
            notes.append((bass + s, vel["upper"], t0, d))
    return notes


def main() -> None:
    choice = json.load(open("out/stage1_choice.json"))
    patch = json.load(open("out/patch.json"))
    px = np.array(patch["normalized"], dtype=float)
    vel = choice["velocities"]

    orig, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    orig_st, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    co, Mo = chroma(orig), mel_db(orig)

    n = int(DUR * SR) + SR
    curve_two = bend2.bend_curve(n)
    from bend import bend_curve as curve_intro_only

    def evaluate(notes, curve, label):
        obj.renderer.set_notes(notes)
        obj.renderer.set_bend(curve)
        loss = obj(px)
        a = obj.renderer.render(DUR)
        m = report(orig_st, a)
        print(f"  {label:34s} loss={loss:.4f}  chroma={m['chroma_agree']:.4f}  "
              f"mel={m['mel_dist']:.2f}  env={m['env_l1']:.4f}")
        return loss, m, a

    obj = Objective([(60, 100, 0.0, 1.0)])  # notes replaced per evaluation

    print("comparison (same fitted patch throughout):")
    mine = [(r["t0"], r["t1"], r["bass"], r["voicing"]) for r in choice["regions"]]
    mine_notes = [(p, vel["intro"], 0.02, 3.53) for p in choice["intro"]]
    for t0, t1, bass, offs in mine:
        d = t1 - t0
        mine_notes.append((bass, 100, t0, d))
        for s in offs:
            mine_notes.append((bass + s, vel["upper"], t0, d))
    l0, m0, _ = evaluate(mine_notes, curve_intro_only(n), "mine: 6 regions, all major")

    l1, m1, a1 = evaluate(build(AUTHOR, INTRO, vel), curve_two, "author: F/Dbm/Ebsus2+bend")
    l2, m2, a2 = evaluate(build(AUTHOR_SPLIT, INTRO, vel), curve_two, "author + separate 3.45-4.30")

    best = min(((l1, AUTHOR, a1, m1, "author"), (l2, AUTHOR_SPLIT, a2, m2, "author_split")))
    loss, regions, aud, m, tag = best
    print(f"\nadopting: {tag}  (loss {loss:.4f})")
    for label, t0, t1, bass, offs in regions:
        print(f"  [{t0:5.2f}-{t1:5.2f}] {label:9s} {librosa.midi_to_note(bass):>4s} + "
              + " ".join(librosa.midi_to_note(bass + x) for x in offs))

    notes = build(regions, INTRO, vel)
    to_pm(notes, "out/transcription.mid")
    sf.write("out/render.wav", aud.T, SR)
    patch["metrics"] = m
    patch["loss"] = loss
    json.dump(patch, open("out/patch.json", "w"), indent=2)
    choice["structure"] = "author (GarageBand): F > Dbm > Ebsus2, last chord bent +3 at half length"
    choice["regions"] = [
        {"t0": t0, "t1": t1, "bass": b, "voicing": o, "label": lab}
        for lab, t0, t1, b, o in regions
    ]
    choice["intro"] = INTRO
    choice["bend"] = bend2.describe()
    choice["metrics_vs_mine"] = {"author_loss": loss, "mine_loss": l0,
                                 "author_chroma": m["chroma_agree"],
                                 "mine_chroma": m0["chroma_agree"]}
    json.dump(choice, open("out/stage1_choice.json", "w"), indent=2)
    print(f"\nwrote out/transcription.mid ({len(notes)} notes) + out/render.wav")


if __name__ == "__main__":
    main()
