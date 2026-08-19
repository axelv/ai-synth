"""Stage 1 final pass: split the long opening region and search bass + voicing jointly.

pitch_probe suggested a chord change inside 3.45-7.50 (D#2-ish at 3.5-4.2, F1
from 4.2), which the merged region could not represent.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import soundfile as sf

from build_midi import INTRO_END, eval_patch, to_pm
from metrics import HOP, chroma, mel_db, report
from refine_midi import VOICINGS, region_score
from synth import PadRenderer

SR = 44100
DUR = 17.904

# (t0, t1, [bass candidates])
# Bass octaves are FIXED to the directly measured fundamentals (pitch_probe:
# 78.10 / 43.65 / 34.65 / 38.89 / 46.25 / 43.65 Hz). They are not searched:
# the search objective is chroma-dominated and chroma is octave-invariant, so
# it pulled every bass up an octave despite the measured fundamental being the
# strongest peak in the spectrum.
REGIONS = [
    (3.45, 4.30, [39]),
    (4.30, 7.50, [29]),
    (7.45, 10.45, [25]),
    (10.40, 13.40, [27]),
    (13.35, 16.10, [30]),
    (16.05, 17.90, [29]),
]

INTRO = [53, 55, 57, 59, 60, 62, 64, 65]


def build(intro, regions_choice) -> list[tuple[int, int, float, float]]:
    notes = [(p, 60, 0.02, INTRO_END - 0.02) for p in intro]
    for (t0, t1, _), (bass, vo) in zip(REGIONS, regions_choice):
        d = t1 - t0
        notes.append((bass, 100, t0, d))
        for s in vo:
            notes.append((bass + s, 66, t0, d))
    return notes


def main() -> None:
    orig, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    orig_st, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    r = PadRenderer(n_voices=64)
    r.set_params(eval_patch())

    def render(notes):
        r.set_notes(notes)
        return r.render(DUR).mean(axis=0)

    choice = [(cands[0], [19, 24, 28]) for _, _, cands in REGIONS]

    for pass_no in range(2):
        print(f"===== pass {pass_no + 1} =====")
        for i, (t0, t1, cands) in enumerate(REGIONS):
            best, best_s, best_ca = choice[i], -9.0, 0.0
            for bass in cands:
                for vo in VOICINGS:
                    trial = list(choice)
                    trial[i] = (bass, vo)
                    ca, md = region_score(orig, render(build(INTRO, trial)), t0 + 0.1, t1)
                    s = ca - 0.02 * md
                    if s > best_s:
                        best, best_s, best_ca = (bass, vo), s, ca
            choice[i] = best
            bass, vo = best
            tag = " ".join(librosa.midi_to_note(bass + x) for x in vo) or "(bass only)"
            print(
                f"  r{i} [{t0:5.2f}-{t1:5.2f}] bass={librosa.midi_to_note(bass):>4s} "
                f"{str(vo):20s} chroma={best_ca:.4f}  {tag}"
            )

    notes = build(INTRO, choice)
    r.set_notes(notes)
    st = r.render(DUR)
    m = report(orig_st, st)
    print("\nfinal stage-1 metrics:", {k: round(v, 4) for k, v in m.items()})
    to_pm(notes, "out/transcription.mid")
    sf.write("out/stage1_final.wav", st.T, SR)
    with open("out/stage1_choice.json", "w") as fh:
        json.dump(
            {
                "intro": INTRO,
                "regions": [
                    {"t0": t0, "t1": t1, "bass": int(b), "voicing": v}
                    for (t0, t1, _), (b, v) in zip(REGIONS, choice)
                ],
                "metrics": m,
            },
            fh,
            indent=2,
        )
    print("wrote out/transcription.mid + out/stage1_choice.json")


if __name__ == "__main__":
    main()
