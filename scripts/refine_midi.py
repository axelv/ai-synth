"""Stage 1 refinement: choose each chord's voicing by region-local measured score.

Renders the whole arrangement each time (so reverb bleed between chords is
represented) but scores only the frames belonging to the region being tuned.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import soundfile as sf

from build_midi import CHORDS, INTRO_END, eval_patch, to_pm
from metrics import HOP, chroma, mel_db
from synth import PadRenderer

SR = 44100
DUR = 17.904

# semitone offsets above the bass note
VOICINGS: list[list[int]] = [
    [],
    [12],
    [12, 19],
    [12, 24],
    [7, 12, 19],
    [16, 19, 24],
    [15, 19, 24],
    [12, 16, 19],
    [12, 15, 19],
    [19, 24, 28],
    [21, 25, 28],
    [16, 19, 24, 28],
    [15, 19, 22, 24],
    [12, 19, 24, 28],
    [24, 28, 31],
    # second-pass additions: wider spreads, added 6ths/9ths, doubled roots
    [12, 24, 28, 31],
    [19, 24, 28, 31],
    [24, 28, 31, 36],
    [12, 19, 24, 28, 31],
    [24, 28, 33],
    [24, 27, 31],
    [19, 24, 28, 33],
    [7, 19, 24, 28],
    [12, 16, 19, 24, 28],
    [24, 28, 31, 40],
    [28, 31, 36],
    [24, 31, 36],
]

INTRO_OPTIONS: dict[str, list[int]] = {
    "F_triad_mid": [53, 57, 60, 65],
    "F_triad_hi": [57, 60, 65, 69],
    "F_oct": [53, 65],
    "F_wide": [41, 53, 57, 60, 65],
    "cluster": [53, 57, 60, 62, 65, 67],
    "F_low": [41, 45, 48, 53],
    "cluster_wide": [53, 56, 57, 60, 62, 65, 67, 69],
    "cluster_hi": [57, 60, 62, 65, 67, 69, 72],
    "F_dbmaj": [53, 56, 60, 65, 68],
    "chromatic": [53, 55, 57, 59, 60, 62, 64, 65],
    "F_triad_x2": [53, 57, 60, 65, 69, 72],
}


def region_score(orig_m: np.ndarray, rend_m: np.ndarray, t0: float, t1: float) -> tuple[float, float]:
    """(chroma agreement, mel distance) restricted to a time window."""
    co, cr = chroma(orig_m), chroma(rend_m)
    n = min(co.shape[1], cr.shape[1])
    f0 = int(t0 * SR / HOP)
    f1 = min(int(t1 * SR / HOP), n)
    if f1 <= f0:
        return 0.0, 99.0
    ca = float((co[:, f0:f1] * cr[:, f0:f1]).sum(axis=0).mean())
    Mo, Mr = mel_db(orig_m), mel_db(rend_m)
    Mo, Mr = Mo[:, f0:f1], Mr[:, f0:f1]
    md = float(np.abs((Mo - Mo.mean()) - (Mr - Mr.mean())).mean())
    return ca, md


def build(intro: list[int], voicings: list[list[int]]) -> list[tuple[int, int, float, float]]:
    notes: list[tuple[int, int, float, float]] = []
    for p in intro:
        notes.append((p, 60, 0.02, INTRO_END - 0.02))
    for (t0, t1, bass, _), vo in zip(CHORDS, voicings):
        d = t1 - t0
        notes.append((bass, 100, t0, d))
        for s in vo:
            notes.append((bass + s, 66, t0, d))
    return notes


def main() -> None:
    orig, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    r = PadRenderer(n_voices=48)
    r.set_params(eval_patch())

    def render(notes) -> np.ndarray:
        r.set_notes(notes)
        return r.render(DUR).mean(axis=0)

    # --- intro ---
    best_intro, best_intro_s = None, -9.0
    print("intro options:")
    for name, pitches in INTRO_OPTIONS.items():
        notes = build(pitches, [c[3] for c in CHORDS])
        ca, md = region_score(orig, render(notes), 0.1, INTRO_END)
        s = ca - 0.02 * md
        print(f"  {name:14s} chroma={ca:.4f} mel={md:.2f} score={s:.4f}")
        if s > best_intro_s:
            best_intro, best_intro_s = pitches, s
    print(f"  -> intro {best_intro}")

    # --- chords, one region at a time, two passes (choices interact via reverb) ---
    chosen = [c[3] for c in CHORDS]
    for pass_no in range(2):
        print(f"\n===== pass {pass_no + 1} =====")
        for i, (t0, t1, bass, _) in enumerate(CHORDS):
            best_v, best_s = chosen[i], -9.0
            for vo in VOICINGS:
                trial = list(chosen)
                trial[i] = vo
                ca, md = region_score(orig, render(build(best_intro, trial)), t0 + 0.1, t1)
                s = ca - 0.02 * md
                if s > best_s:
                    best_v, best_s, best_ca = vo, s, ca
            chosen[i] = best_v
            tag = " ".join(librosa.midi_to_note(bass + x) for x in best_v) or "(bass only)"
            print(
                f"  region {i} [{t0:5.2f}-{t1:5.2f}] {librosa.midi_to_note(bass):>4s} -> "
                f"{str(best_v):24s} chroma={best_ca:.4f}  {tag}"
            )

    notes = build(best_intro, chosen)
    aud_full = render(notes)
    from metrics import report

    orig_st, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    r.set_notes(notes)
    st = r.render(DUR)
    m = report(orig_st, st)
    print("\nfinal stage-1 metrics:", {k: round(v, 4) for k, v in m.items()})
    to_pm(notes, "out/transcription.mid")
    sf.write("out/stage1_final.wav", st.T, SR)
    with open("out/stage1_choice.json", "w") as fh:
        json.dump({"intro": best_intro, "voicings": chosen, "metrics": m}, fh, indent=2)
    print("wrote out/transcription.mid")


if __name__ == "__main__":
    main()
