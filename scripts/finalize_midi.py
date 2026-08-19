"""Fix chord-boundary gaps in the transcription.

The comparison plot showed a level dip at every chord change: notes ended exactly
where the next chord began, so the release gap produced a re-attack the original
does not have. Real pads overlap. Overlap is a note *duration* property, so this
belongs to stage 1; the amount is chosen by measurement.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import soundfile as sf

from build_midi import INTRO_END, eval_patch, to_pm
from metrics import env_l1, mel_dist, report
from polish_midi import REGIONS
from synth import PadRenderer

SR = 44100
DUR = 17.904


def build(intro, voicings, overlap: float):
    notes = [(p, 60, 0.02, INTRO_END + overlap) for p in intro]
    for i, ((t0, t1, bass), vo) in enumerate(zip(REGIONS, voicings)):
        # extend past the boundary so the next chord overlaps this one's tail
        d = (t1 - t0) + (overlap if i < len(REGIONS) - 1 else 0.0)
        notes.append((bass, 100, t0, d))
        for s in vo:
            notes.append((bass + s, 66, t0, d))
    return notes


def main() -> None:
    choice = json.load(open("out/stage1_choice.json"))
    intro = choice["intro"]
    voicings = [r["voicing"] for r in choice["regions"]]

    orig, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    orig_st, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    r = PadRenderer(n_voices=64)
    r.set_params(eval_patch())

    best, best_s = 0.0, 1e9
    for ov in (0.0, 0.3, 0.6, 1.0, 1.5, 2.0):
        r.set_notes(build(intro, voicings, ov))
        a = r.render(DUR).mean(axis=0)
        e = env_l1(orig, a)
        md = mel_dist(orig, a)
        s = e + 0.02 * md
        print(f"  overlap {ov:4.1f}s  env_l1={e:.4f}  mel={md:.3f}  score={s:.4f}")
        if s < best_s:
            best, best_s = ov, s
    print(f"-> overlap {best:.1f}s")

    notes = build(intro, voicings, best)
    r.set_notes(notes)
    st = r.render(DUR)
    m = report(orig_st, st)
    print("stage-1 metrics after overlap fix:", {k: round(v, 4) for k, v in m.items()})
    to_pm(notes, "out/transcription.mid")
    sf.write("out/stage1_final.wav", st.T, SR)
    choice["overlap"] = best
    choice["metrics"] = m
    with open("out/stage1_choice.json", "w") as fh:
        json.dump(choice, fh, indent=2)
    print(f"wrote out/transcription.mid ({len(notes)} notes)")


if __name__ == "__main__":
    main()
