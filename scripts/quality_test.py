"""Test the user's chord qualities against the audio.

User (GarageBand, cannot export MIDI) reports the progression as
F > Dbm > Ebsus2, with the last chord pitched up at half its length.

My transcription had all-major triads (F, Db, Eb, Gb, F). The qualities differ on
one note each, which is directly testable:
  Db major (Db F  Ab)  vs  Db minor (Db Fb Ab)   -> F3 vs E3
  Eb major (Eb G  Bb)  vs  Ebsus2  (Eb F  Bb)    -> G3 vs F3
and region 5 should be region 4's voicing transposed +3 semitones if it really is
the same chord bent up.
"""

from __future__ import annotations

import json

import librosa
import numpy as np

from bend import bend_curve
from metrics import chroma, mel_db
from synth import PadRenderer, denorm

SR = 44100
DUR = 17.904

REGIONS = [
    (3.45, 4.30, 39),
    (4.30, 7.50, 29),
    (7.45, 10.45, 25),
    (10.40, 13.40, 27),
    (13.35, 16.10, 30),
    (16.05, 17.90, 29),
]

# candidate voicings per region, labelled by chord quality
CANDIDATES: dict[int, dict[str, list[int]]] = {
    1: {  # F1
        "F major (mine)": [16, 19, 24, 28],
        "F major alt": [19, 24, 28, 31],
        "F minor": [15, 19, 24, 27],
        "Fsus2": [14, 19, 24, 26],
    },
    2: {  # Db1
        "Db major (mine)": [12, 19, 24, 28, 31],
        "Db minor": [12, 19, 24, 27, 31],
        "Db minor alt": [19, 24, 27, 31],
        "Dbm add9": [12, 19, 24, 27, 26],
        "Dbsus2": [12, 19, 24, 26, 31],
    },
    3: {  # Eb1
        "Eb major (mine)": [12, 19, 24, 28, 31],
        "Ebsus2": [12, 19, 24, 26, 31],
        "Ebsus2 alt": [19, 24, 26, 31],
        "Ebsus4": [12, 19, 24, 29, 31],
        "Eb minor": [12, 19, 24, 27, 31],
    },
    4: {  # Gb1 = Eb bent +3
        "Gb major (mine)": [12, 16, 19, 24, 28],
        "Gbsus2 (=Ebsus2+3)": [12, 19, 24, 26, 31],
        "Gbsus2 alt": [19, 24, 26, 31],
        "Gbsus4": [12, 19, 24, 29, 31],
    },
    5: {  # F1
        "F major (mine)": [12, 19, 24, 28, 31],
        "F major alt": [19, 24, 28, 31],
        "F minor": [12, 19, 24, 27, 31],
        "Fsus2": [12, 19, 24, 26, 31],
    },
}


def main() -> None:
    choice = json.load(open("out/stage1_choice.json"))
    patch = json.load(open("out/patch.json"))
    px = np.array(patch["normalized"], dtype=float)
    intro = choice["intro"]
    vel = choice["velocities"]

    orig, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    r = PadRenderer(n_voices=48)
    r.set_params(denorm(px))
    r.set_bend(bend_curve(int(DUR * SR) + SR))
    co, Mo = chroma(orig), mel_db(orig)

    base = [list(reg["voicing"]) for reg in choice["regions"]]

    def build(voic):
        notes = [(p, vel["intro"], 0.02, 3.53) for p in intro]
        for (t0, t1, bass), vo in zip(REGIONS, voic):
            d = t1 - t0
            notes.append((bass, 100, t0, d))
            for s in vo:
                notes.append((bass + s, vel["upper"], t0, d))
        return notes

    def region_score(voic, t0, t1):
        r.set_notes(build(voic))
        a = r.render(DUR).mean(axis=0)
        cr = chroma(a)
        n = min(co.shape[1], cr.shape[1])
        f0, f1 = int(t0 * SR / 512), min(int(t1 * SR / 512), n)
        ca = float((co[:, f0:f1] * cr[:, f0:f1]).sum(axis=0).mean())
        Mr = mel_db(a)
        k = min(Mo.shape[1], Mr.shape[1])
        md = float(np.abs((Mo[:, f0:f1] - Mo.mean()) - (Mr[:, f0:min(f1, k)] - Mr.mean())).mean())
        return ca, md

    chosen = list(base)
    for idx, opts in CANDIDATES.items():
        t0, t1, bass = REGIONS[idx]
        print(f"\nregion {idx}  [{t0:5.2f}-{t1:5.2f}]  bass {librosa.midi_to_note(bass)}")
        rows = []
        for name, vo in opts.items():
            trial = list(chosen)
            trial[idx] = vo
            ca, md = region_score(trial, t0 + 0.15, t1)
            rows.append((ca - 0.02 * md, ca, md, name, vo))
            tag = " ".join(librosa.midi_to_note(bass + x) for x in vo)
            print(f"  {name:22s} chroma={ca:.4f} mel={md:.2f}  {tag}")
        rows.sort(reverse=True)
        best = rows[0]
        chosen[idx] = best[4]
        print(f"  -> {best[3]}  (chroma {best[1]:.4f})")

    print("\nfinal per-region choice:")
    for (t0, t1, bass), vo in zip(REGIONS, chosen):
        print(f"  [{t0:5.2f}-{t1:5.2f}] {librosa.midi_to_note(bass):>4s} + "
              + " ".join(librosa.midi_to_note(bass + x) for x in vo))
    json.dump({"voicings": chosen}, open("out/quality_test.json", "w"), indent=2)


if __name__ == "__main__":
    main()
