"""Two-segment pitch-bend lane, matching how the clip was actually made.

Segment 1 — intro riser: measured glide F2 -> C#3 (+7.85 st) over 0-3.45 s.
Segment 2 — the Eb chord is pitched up +3 semitones at half its length
            (10.40-16.10 s, bent from 13.35 s), then released at 16.05 s.

Both the amount (+3.00 st) and the timing (midpoint 13.25 s vs measured boundary
13.35 s) were confirmed independently from the audio before adopting this.
"""

from __future__ import annotations

import librosa
import numpy as np

SR = 44100
TARGET_MIDI = 49       # C#3 — the pitch the intro glide lands on
INTRO_END = 3.45
BEND_UP_AT = 13.35     # half way through the Eb chord
BEND_RELEASE_AT = 16.05
BEND_SEMITONES = 3.0
RAMP = 0.20            # ~0.2 s, read off the automation lane in the DAW


def bend_curve(n_samples: int, sr: int = SR, path: str = "out/intro_f0.npy") -> np.ndarray:
    t_meas, f0 = np.load(path)
    target = float(librosa.midi_to_hz(TARGET_MIDI))
    ratio = f0 / target

    t = np.arange(n_samples) / sr
    curve = np.interp(t, t_meas, ratio, left=ratio[0], right=1.0)
    curve[t >= INTRO_END] = 1.0

    # smooth the seam into unity at the drop
    seam = (t >= INTRO_END - 0.15) & (t < INTRO_END)
    if seam.any():
        w = np.linspace(0.0, 1.0, int(seam.sum()))
        curve[seam] = curve[seam] * (1 - w) + w

    up = float(2 ** (BEND_SEMITONES / 12))
    rise = (t >= BEND_UP_AT) & (t < BEND_UP_AT + RAMP)
    curve[rise] = 1.0 + (up - 1.0) * np.linspace(0.0, 1.0, int(rise.sum()))
    curve[(t >= BEND_UP_AT + RAMP) & (t < BEND_RELEASE_AT)] = up
    fall = (t >= BEND_RELEASE_AT) & (t < BEND_RELEASE_AT + RAMP)
    curve[fall] = up + (1.0 - up) * np.linspace(0.0, 1.0, int(fall.sum()))
    curve[t >= BEND_RELEASE_AT + RAMP] = 1.0
    return curve.astype(np.float32)


def describe() -> str:
    t, f0 = np.load("out/intro_f0.npy")
    return (
        f"intro glide {f0[2]:.1f} Hz ({librosa.hz_to_note(f0[2])}) -> {f0[-3]:.1f} Hz "
        f"({librosa.hz_to_note(f0[-3])}), {12 * np.log2(f0[-3] / f0[2]):+.2f} st over "
        f"{t[-3] - t[2]:.2f}s; then +{BEND_SEMITONES:.0f} st at {BEND_UP_AT}s, "
        f"released at {BEND_RELEASE_AT}s"
    )


if __name__ == "__main__":
    c = bend_curve(int(17.904 * SR))
    print(describe())
    for x in (0.0, 2.0, 3.4, 5.0, 13.3, 13.6, 15.0, 16.0, 16.4, 17.5):
        i = min(int(x * SR), len(c) - 1)
        st = 12 * np.log2(c[i])
        print(f"  t={x:5.2f}s  bend={c[i]:.4f}  ({st:+.2f} semitones)")
