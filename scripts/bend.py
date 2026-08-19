"""Pitch-bend curve for the intro glide, taken from the measured f0 trajectory.

The intro is one voice gliding F2 -> C#3 (+7.85 semitones over ~3.5 s). It is
rendered as a note held at the *target* pitch (C#3) with the bend ramping up to
1.0, so that when the bend reaches unity at the drop the sustained chords are
unaffected.
"""

from __future__ import annotations

import librosa
import numpy as np

SR = 44100
TARGET_MIDI = 49  # C#3, the pitch the glide lands on
INTRO_END = 3.50


def bend_curve(n_samples: int, sr: int = SR, path: str = "out/intro_f0.npy") -> np.ndarray:
    """Audio-rate frequency multiplier: the glide during the intro, 1.0 after."""
    t, f0 = np.load(path)
    target = float(librosa.midi_to_hz(TARGET_MIDI))
    ratio = f0 / target

    times = np.arange(n_samples) / sr
    curve = np.interp(times, t, ratio, left=ratio[0], right=1.0)
    # hold exactly 1.0 once the drop arrives so the chords are never bent
    curve[times >= INTRO_END] = 1.0
    # short crossfade into unity to avoid a discontinuity at the seam
    seam = (times >= INTRO_END - 0.15) & (times < INTRO_END)
    if seam.any():
        w = np.linspace(0.0, 1.0, int(seam.sum()))
        curve[seam] = curve[seam] * (1 - w) + 1.0 * w
    return curve.astype(np.float32)


def describe() -> str:
    t, f0 = np.load("out/intro_f0.npy")
    return (
        f"glide {f0[2]:.1f} Hz ({librosa.hz_to_note(f0[2])}) -> {f0[-3]:.1f} Hz "
        f"({librosa.hz_to_note(f0[-3])}), {12 * np.log2(f0[-3] / f0[2]):+.2f} semitones "
        f"over {t[-3] - t[2]:.2f}s"
    )


if __name__ == "__main__":
    c = bend_curve(int(17.904 * SR))
    print(describe())
    print(f"curve: n={len(c)} start={c[0]:.4f} at3.0s={c[int(3.0 * SR)]:.4f} "
          f"at4.0s={c[int(4.0 * SR)]:.4f} min={c.min():.4f} max={c.max():.4f}")
