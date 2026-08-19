"""Look for a pitch bend in the sustained section, not just the intro.

Measures the bass fundamental's deviation from equal temperament over time. A
bend shows up as a sustained drift away from 0 cents within a held chord,
distinct from a step at a chord change (which is just a new note).
"""

from __future__ import annotations

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = 44100
N_FFT = 16384
HOP = 512

# held chords from stage 1, with the measured bass fundamental
CHORDS = [
    (3.45, 4.30, 39),
    (4.30, 7.50, 29),
    (7.45, 10.45, 25),
    (10.40, 13.40, 27),
    (13.35, 16.10, 30),
    (16.05, 17.90, 29),
]


def refine(S, freqs, t_idx, f_target, harmonic):
    """Interpolated frequency of a given harmonic near its expected place."""
    f = f_target * harmonic
    tol = max(4.0, f * 0.03)
    m = np.where((freqs >= f - tol) & (freqs <= f + tol))[0]
    if len(m) < 3:
        return np.nan
    col = S[m, t_idx]
    i = int(np.argmax(col))
    gi = m[i]
    if gi <= 0 or gi >= len(freqs) - 1:
        return np.nan
    a, b, c = S[gi - 1, t_idx], S[gi, t_idx], S[gi + 1, t_idx]
    d = 0.5 * (a - c) / (a - 2 * b + c + 1e-20)
    return (freqs[gi] + d * (freqs[1] - freqs[0])) / harmonic


def main() -> None:
    y, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    t = librosa.times_like(S[0], sr=SR, hop_length=HOP)

    all_t, all_c = [], []
    print("chord      bass   cents deviation over the held chord (start -> end)")
    for t0, t1, midi in CHORDS:
        f_nom = float(librosa.midi_to_hz(midi))
        sel = np.where((t >= t0 + 0.15) & (t <= t1 - 0.10))[0]
        cents = []
        for ti in sel:
            # harmonic 4 is strong and well resolved for these low fundamentals
            f = refine(S, freqs, ti, f_nom, 4)
            cents.append(np.nan if not np.isfinite(f) else 1200 * np.log2(f / f_nom))
        cents = np.array(cents, float)
        good = np.isfinite(cents)
        if good.sum() < 5:
            continue
        all_t.append(t[sel][good])
        all_c.append(cents[good])
        c = cents[good]
        n = max(3, len(c) // 6)
        print(f"{t0:5.2f}-{t1:5.2f} {librosa.midi_to_note(midi):>4s}  "
              f"{c[:n].mean():+7.1f} -> {c[-n:].mean():+7.1f} cents   "
              f"(min {c.min():+.0f}, max {c.max():+.0f}, drift {c[-n:].mean() - c[:n].mean():+.1f})")

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    for tt, cc in zip(all_t, all_c):
        ax.plot(tt, cc, lw=1.5)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    for t0, t1, _ in CHORDS:
        ax.axvline(t0, color="r", alpha=0.25, lw=1.0)
    ax.set(xlabel="time (s)", ylabel="cents from equal temperament",
           title="bass fundamental tuning drift within each held chord")
    ax.grid(alpha=0.3)
    fig.savefig("out/late_bend.png", dpi=115)
    print("\nwrote out/late_bend.png")


if __name__ == "__main__":
    main()
