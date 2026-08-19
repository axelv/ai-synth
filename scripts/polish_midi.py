"""Final stage-1 pass with globally monotone acceptance.

Per-region greedy updates interact through the reverb tail, so a change that
helps one region can hurt the whole. Here a candidate is only accepted if the
full-clip score improves, which makes the pass monotone by construction.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import soundfile as sf

from build_midi import INTRO_END, eval_patch, to_pm
from metrics import chroma, mel_db, report
from refine_midi import VOICINGS
from synth import PadRenderer

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
INTRO_OPTS = {
    "chromatic": [53, 55, 57, 59, 60, 62, 64, 65],
    "cluster": [53, 57, 60, 62, 65, 67],
    "cluster_hi": [57, 60, 62, 65, 67, 69, 72],
    "F_triad_mid": [53, 57, 60, 65],
    "chromatic_hi": [57, 59, 60, 62, 64, 65, 67, 69],
    "F_wide": [41, 53, 57, 60, 65],
}
START = [[21, 25, 28], [16, 19, 24, 28], [12, 19, 24, 28, 31],
         [12, 19, 24, 28, 31], [12, 16, 19, 24, 28], [7, 19, 24, 28]]


def build(intro, voicings):
    notes = [(p, 60, 0.02, INTRO_END - 0.02) for p in intro]
    for (t0, t1, bass), vo in zip(REGIONS, voicings):
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
    co = chroma(orig)
    Mo = mel_db(orig)

    def global_score(notes) -> tuple[float, float, float]:
        r.set_notes(notes)
        a = r.render(DUR).mean(axis=0)
        cr = chroma(a)
        n = min(co.shape[1], cr.shape[1])
        ca = float((co[:, :n] * cr[:, :n]).sum(axis=0).mean())
        Mr = mel_db(a)
        k = min(Mo.shape[1], Mr.shape[1])
        md = float(np.abs((Mo[:, :k] - Mo.mean()) - (Mr[:, :k] - Mr.mean())).mean())
        return ca - 0.02 * md, ca, md

    intro = INTRO_OPTS["chromatic"]
    voic = [list(v) for v in START]
    best_s, best_ca, best_md = global_score(build(intro, voic))
    print(f"start: score={best_s:.4f} chroma={best_ca:.4f} mel={best_md:.2f}")

    for pass_no in range(3):
        changed = False
        for name, opt in INTRO_OPTS.items():
            s, ca, md = global_score(build(opt, voic))
            if s > best_s + 1e-5:
                intro, best_s, best_ca, best_md, changed = opt, s, ca, md, True
                print(f"  intro -> {name:14s} score={s:.4f} chroma={ca:.4f}")
        for i, (t0, t1, bass) in enumerate(REGIONS):
            for vo in VOICINGS:
                trial = [list(v) for v in voic]
                trial[i] = list(vo)
                s, ca, md = global_score(build(intro, trial))
                if s > best_s + 1e-5:
                    voic, best_s, best_ca, best_md, changed = trial, s, ca, md, True
                    tag = " ".join(librosa.midi_to_note(bass + x) for x in vo) or "(bass only)"
                    print(f"  r{i} -> {str(vo):22s} score={s:.4f} chroma={ca:.4f} mel={md:.2f}  {tag}")
        print(f"pass {pass_no + 1}: score={best_s:.4f} chroma={best_ca:.4f} mel={best_md:.2f}")
        if not changed:
            print("converged")
            break

    notes = build(intro, voic)
    r.set_notes(notes)
    st = r.render(DUR)
    m = report(orig_st, st)
    print("\nFINAL stage-1:", {k: round(v, 4) for k, v in m.items()})
    for (t0, t1, bass), vo in zip(REGIONS, voic):
        print(f"  [{t0:5.2f}-{t1:5.2f}] {librosa.midi_to_note(bass):>4s} + "
              + " ".join(librosa.midi_to_note(bass + x) for x in vo))
    to_pm(notes, "out/transcription.mid")
    sf.write("out/stage1_final.wav", st.T, SR)
    with open("out/stage1_choice.json", "w") as fh:
        json.dump({"intro": intro,
                   "regions": [{"t0": t0, "t1": t1, "bass": b, "voicing": v}
                               for (t0, t1, b), v in zip(REGIONS, voic)],
                   "metrics": m}, fh, indent=2)
    print("wrote out/transcription.mid")


if __name__ == "__main__":
    main()
