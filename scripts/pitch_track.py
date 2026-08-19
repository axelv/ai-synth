"""Track the fundamental over time to test for pitch bend / glide.

Static-window analysis cannot tell a glide from a chord: a partial sweeping
through a band looks like several different notes when successive windows are
analysed separately. This uses a harmonic-sum salience over a fine f0 grid with
a continuity (Viterbi) constraint, so it follows one voice instead of hopping
between partials.

Harmonics 1-3 are excluded from the salience because the first 3.45 s of the
clip is highpassed and has no energy there.
"""

from __future__ import annotations

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = 44100
N_FFT = 8192
HOP = 256
F_LO, F_HI = 30.0, 130.0
HARMONICS = range(4, 25)


def salience(S: np.ndarray, freqs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """(n_grid, n_frames) harmonic-sum salience."""
    L = np.log1p(S / (S.max() + 1e-12) * 1000.0)
    out = np.zeros((len(grid), S.shape[1]))
    df = freqs[1] - freqs[0]
    for gi, f0 in enumerate(grid):
        acc = np.zeros(S.shape[1])
        for k in HARMONICS:
            f = f0 * k
            if f >= freqs[-1]:
                break
            b = int(round(f / df))
            if 1 <= b < len(freqs) - 1:
                acc += L[b - 1 : b + 2].max(axis=0) / (k**0.3)
        out[gi] = acc
    return out


def viterbi(sal: np.ndarray, grid: np.ndarray, jump_penalty: float = 9.0) -> np.ndarray:
    """Best continuous path through the salience surface."""
    n_g, n_t = sal.shape
    cents = 1200 * np.log2(grid / grid[0])
    cost = -sal / (sal.max() + 1e-9)
    dp = cost[:, 0].copy()
    ptr = np.zeros((n_g, n_t), dtype=int)
    for t in range(1, n_t):
        trans = np.abs(cents[:, None] - cents[None, :]) / 100.0  # semitone distance
        tot = dp[None, :] + jump_penalty * trans
        ptr[:, t] = np.argmin(tot, axis=1)
        dp = cost[:, t] + tot[np.arange(n_g), ptr[:, t]]
    path = np.zeros(n_t, dtype=int)
    path[-1] = int(np.argmin(dp))
    for t in range(n_t - 1, 0, -1):
        path[t - 1] = ptr[path[t], t]
    return grid[path]


def main() -> None:
    y, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    t = librosa.times_like(S[0], sr=SR, hop_length=HOP)

    grid = F_LO * 2 ** (np.arange(0, 1201) * (np.log2(F_HI / F_LO) / 1200))
    sal = salience(S, freqs, grid)
    f0 = viterbi(sal, grid)
    midi = librosa.hz_to_midi(f0)

    print(" time      f0      midi   note   dev(cents)")
    for i in range(0, len(t), 12):
        n = round(midi[i])
        print(f"{t[i]:5.2f} {f0[i]:8.2f} {midi[i]:8.2f} {librosa.midi_to_note(int(n)):>5s} "
              f"{(midi[i] - n) * 100:+8.0f}")

    for a, b, lbl in ((0.30, 3.30, "intro"), (3.60, 7.40, "chord 1"), (7.60, 10.30, "chord 2")):
        m = (t >= a) & (t <= b)
        v = f0[m]
        st = 12 * np.log2(v[-1] / v[0])
        print(f"\n{lbl:8s} {a:.2f}-{b:.2f}s : {v[0]:6.2f} -> {v[-1]:6.2f} Hz  "
              f"({st:+.2f} semitones, min {v.min():.2f} max {v.max():.2f})")

    fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
    L = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(L, sr=SR, hop_length=HOP, x_axis="time", y_axis="log", ax=ax[0])
    for k in (1, 2, 4, 6, 8):
        ax[0].plot(t, f0 * k, lw=1.3, alpha=0.9, label=f"tracked f0 x{k}")
    ax[0].set(ylim=(28, 2200), title="tracked fundamental and its harmonics over the spectrogram")
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(t, midi, color="C3", lw=1.6)
    for s in range(24, 42):
        ax[1].axhline(s, color="k", alpha=0.12, lw=0.6)
    ax[1].set(xlabel="time (s)", ylabel="MIDI pitch", title="pitch trajectory (gridlines = semitones)")
    ax[1].grid(alpha=0.2, axis="x")
    fig.savefig("out/pitch_track.png", dpi=110)
    print("\nwrote out/pitch_track.png")
    np.save("out/f0_track.npy", np.vstack([t, f0]))


if __name__ == "__main__":
    main()
