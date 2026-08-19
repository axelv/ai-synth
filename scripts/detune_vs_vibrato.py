"""Decide the movement mechanism: static unison detune vs LFO vibrato.

Static detune is a fixed *ratio*, so a partial cluster's width in Hz grows
linearly with harmonic number. Vibrato produces sidebands at the LFO rate, so
the spacing in Hz is the same at every harmonic. Measuring cluster width across
harmonics separates the two.
"""

from __future__ import annotations

import librosa
import numpy as np

SR = 44100
T0, T1 = 5.20, 7.40  # stable F1 chord
F0 = 43.6535         # measured fundamental


def hi_res(seg: np.ndarray, sr: int, pad: int = 4):
    w = np.hanning(len(seg))
    n_fft = 1 << (int(np.ceil(np.log2(len(seg)))) + pad)
    S = np.abs(np.fft.rfft(seg * w, n=n_fft))
    fr = np.fft.rfftfreq(n_fft, 1 / sr)
    return fr, S


def cluster(fr, S, centre, halfwidth):
    m = (fr >= centre - halfwidth) & (fr <= centre + halfwidth)
    f, s = fr[m], S[m]
    if not len(s):
        return None
    s = s / s.max()
    # peaks above 25% of the cluster max
    idx = [i for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > 0.25]
    if len(idx) < 2:
        return {"n": len(idx), "width": 0.0, "peaks": [float(f[i]) for i in idx]}
    fs = [float(f[i]) for i in idx]
    return {"n": len(idx), "width": max(fs) - min(fs), "peaks": fs}


def main() -> None:
    y, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    seg = y[int(T0 * SR) : int(T1 * SR)]
    fr, S = hi_res(seg, SR)
    print(f"segment {T0}-{T1}s  resolution {fr[1] - fr[0]:.3f} Hz\n")

    print(f"{'harm':>4s} {'centre Hz':>10s} {'peaks':>6s} {'width Hz':>9s} {'width/harm':>11s} {'width cents':>12s}")
    rows = []
    for k in (3, 5, 6, 8, 10, 12, 16, 20):
        c = F0 * k
        if c > 4000:
            break
        hw = max(6.0, c * 0.035)
        r = cluster(fr, S, c, hw)
        if r is None or r["n"] < 2:
            print(f"{k:4d} {c:10.1f} {r['n'] if r else 0:6d}        --          --           --")
            continue
        cents = 1200 * np.log2((min(r["peaks"]) + r["width"]) / min(r["peaks"]))
        rows.append((k, r["width"], cents))
        print(f"{k:4d} {c:10.1f} {r['n']:6d} {r['width']:9.2f} {r['width'] / k:11.2f} {cents:12.1f}")

    if len(rows) >= 3:
        ks = np.array([r[0] for r in rows], float)
        w = np.array([r[1] for r in rows], float)
        cents = np.array([r[2] for r in rows], float)
        # detune  -> width proportional to k (width/k constant, cents constant)
        # vibrato -> width constant in Hz
        cv_ratio = np.std(w / ks) / (np.mean(w / ks) + 1e-9)
        cv_flat = np.std(w) / (np.mean(w) + 1e-9)
        print(f"\nwidth/harmonic  spread (CV) = {cv_ratio:.3f}   <- low favours STATIC DETUNE")
        print(f"width (Hz)      spread (CV) = {cv_flat:.3f}   <- low favours VIBRATO/CHORUS")
        print(f"cluster width in cents: mean {cents.mean():.0f}, std {cents.std():.0f}")
        print("\nverdict:", "STATIC UNISON DETUNE" if cv_ratio < cv_flat else "FIXED-RATE MODULATION")


if __name__ == "__main__":
    main()
