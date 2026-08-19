"""Measure the intro pitch bend by following individual partials.

The intro's partials sweep upward. Fixed-window spectra sample a moving partial
at different points and mistake it for a chord, so partials are tracked here by
nearest-neighbour continuation in log frequency, then checked for a common f0.
"""

from __future__ import annotations

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = 44100
N_FFT = 4096
HOP = 256
T_END = 3.60


def frame_peaks(col: np.ndarray, freqs: np.ndarray, lo: float, hi: float, rel: float = 0.06):
    m = (freqs >= lo) & (freqs <= hi)
    idx = np.where(m)[0]
    peak = col[idx].max()
    out = []
    df = freqs[1] - freqs[0]
    for i in idx[1:-1]:
        if col[i] > col[i - 1] and col[i] >= col[i + 1] and col[i] > rel * peak:
            a, b, c = col[i - 1], col[i], col[i + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c + 1e-20)
            out.append((float(freqs[i] + d * df), float(col[i])))
    return out


def follow(S, freqs, t, start_frame: int, start_hz: float, max_jump_cents: float = 120.0):
    """Follow one partial forward and backward in time."""
    traj = np.full(len(t), np.nan)
    cur = start_hz
    for f in range(start_frame, len(t)):
        pk = frame_peaks(S[:, f], freqs, cur * 0.75, cur * 1.45)
        if not pk:
            break
        best = min(pk, key=lambda z: abs(1200 * np.log2(z[0] / cur)))
        if abs(1200 * np.log2(best[0] / cur)) > max_jump_cents:
            break
        cur = best[0]
        traj[f] = cur
    cur = start_hz
    for f in range(start_frame - 1, -1, -1):
        pk = frame_peaks(S[:, f], freqs, cur * 0.6, cur * 1.35)
        if not pk:
            break
        best = min(pk, key=lambda z: abs(1200 * np.log2(z[0] / cur)))
        if abs(1200 * np.log2(best[0] / cur)) > max_jump_cents:
            break
        cur = best[0]
        traj[f] = cur
    return traj


def main() -> None:
    y, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    y = y[: int(T_END * SR)]
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    t = librosa.times_like(S[0], sr=SR, hop_length=HOP)

    # seed from strong peaks mid-way through the intro, where lines are clear
    seed_f = int(1.8 * SR / HOP)
    seeds = sorted(frame_peaks(S[:, seed_f], freqs, 200.0, 1400.0), key=lambda z: -z[1])[:8]
    print(f"seeds at t={t[seed_f]:.2f}s:", " ".join(f"{f:.1f}" for f, _ in seeds))

    trajs = []
    for f0, _ in seeds:
        tr = follow(S, freqs, t, seed_f, f0)
        if np.isfinite(tr).sum() > 0.5 * len(t):
            trajs.append(tr)
    print(f"{len(trajs)} partials tracked across most of the intro\n")

    print(" time  " + "".join(f"{i:>9d}" for i in range(len(trajs))))
    for i in range(0, len(t), 20):
        print(f"{t[i]:5.2f} " + "".join(f"{tr[i]:9.1f}" if np.isfinite(tr[i]) else "        -" for tr in trajs))

    # Are they harmonics of one gliding voice? Divide by best integer k.
    print("\nratios to the lowest tracked partial (should be near-integer if one voice):")
    base = trajs[int(np.nanargmin([np.nanmean(tr) for tr in trajs]))]
    for i, tr in enumerate(trajs):
        r = np.nanmean(tr / base)
        print(f"  partial {i}: mean ratio {r:6.3f}  -> harmonic {round(r)}")

    print("\nglide extent per partial:")
    for i, tr in enumerate(trajs):
        v = tr[np.isfinite(tr)]
        idx = np.where(np.isfinite(tr))[0]
        st = 12 * np.log2(v[-1] / v[0])
        print(f"  partial {i}: {v[0]:7.1f} -> {v[-1]:7.1f} Hz over "
              f"{t[idx[0]]:.2f}-{t[idx[-1]]:.2f}s  = {st:+.2f} semitones")

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    L = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(L, sr=SR, hop_length=HOP, x_axis="time", y_axis="log", ax=ax)
    for tr in trajs:
        ax.plot(t, tr, lw=1.8, alpha=0.95)
    ax.set(ylim=(120, 2000), title="intro: tracked partials (all sweeping upward together)")
    fig.savefig("out/intro_glide.png", dpi=115)
    print("\nwrote out/intro_glide.png")
    np.save("out/intro_trajs.npy", np.vstack([t] + trajs))


if __name__ == "__main__":
    main()
