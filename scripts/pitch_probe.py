"""High-resolution partial analysis with explicit octave disambiguation.

For each stable chord region: interpolate peak frequencies, then score harmonic
combs while explicitly checking whether the odd harmonics that distinguish f0
from 2*f0 are actually present.
"""

from __future__ import annotations

import librosa
import numpy as np

SR = 44100
REGIONS = [
    (3.60, 4.90),
    (5.10, 6.45),
    (6.60, 7.40),
    (7.60, 8.60),
    (8.80, 10.30),
    (10.60, 11.60),
    (11.80, 12.70),
    (13.50, 14.50),
    (14.70, 15.90),
    (16.10, 17.00),
    (17.15, 17.85),
]


def peaks(seg: np.ndarray, sr: int, fmax: float = 1200.0, thresh: float = 0.03):
    """Interpolated spectral peaks of a stationary segment."""
    w = np.hanning(len(seg))
    n_fft = 1 << int(np.ceil(np.log2(len(seg)))) + 2  # zero-pad x4
    S = np.abs(np.fft.rfft(seg * w, n=n_fft))
    fr = np.fft.rfftfreq(n_fft, 1 / sr)
    S /= S.max() + 1e-12
    out = []
    for i in range(2, len(S) - 2):
        if fr[i] > fmax:
            break
        if S[i] > S[i - 1] and S[i] >= S[i + 1] and S[i] > thresh:
            a, b, c = S[i - 1], S[i], S[i + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c + 1e-20)
            out.append((float(fr[i] + d * (fr[1] - fr[0])), float(b)))
    return out


def comb_score(pk, f0: float, n_harm: int = 16):
    """(total score, per-harmonic amps) for a comb at f0."""
    amps = np.zeros(n_harm + 1)
    for k in range(1, n_harm + 1):
        f = f0 * k
        tol = max(1.5, f * 0.012)
        hit = [a for (fq, a) in pk if abs(fq - f) <= tol]
        if hit:
            amps[k] = max(hit)
    score = sum(amps[k] / (k**0.5) for k in range(1, n_harm + 1))
    return score, amps


def main() -> None:
    y, sr = librosa.load("data/original.wav", sr=SR, mono=True)
    for t0, t1 in REGIONS:
        seg = y[int(t0 * sr) : int(t1 * sr)]
        pk = peaks(seg, sr)
        if not pk:
            continue
        # candidate f0s: every peak below 200 Hz, plus its octave up
        cands = sorted({round(f, 2) for f, a in pk if f < 200})
        scored = []
        for f0 in cands:
            s, amps = comb_score(pk, f0)
            # does the octave-up hypothesis explain it as well?
            s2, amps2 = comb_score(pk, f0 * 2)
            # odd harmonics of f0 (3,5,7) are the discriminator
            odd = amps[3] + amps[5] + amps[7]
            scored.append((s, f0, amps, odd, s2))
        scored.sort(reverse=True)
        s, f0, amps, odd, s2 = scored[0]
        midi = librosa.hz_to_midi(f0)
        print(f"\n[{t0:5.2f}-{t1:5.2f}] f0={f0:7.2f}Hz  midi={midi:6.2f} ({librosa.hz_to_note(f0)})  score={s:.2f}")
        print(f"    odd-harm(3,5,7)={odd:.3f}  octave-up score={s2:.2f}  -> f0 {'CONFIRMED' if odd > 0.15 else 'WEAK'}")
        print("    harmonics:", " ".join(f"{k}:{amps[k]:.2f}" for k in range(1, 13)))
        # residual peaks not explained by the comb
        unexp = []
        for fq, a in pk:
            if a < 0.08:
                continue
            k = round(fq / f0)
            if k < 1 or abs(fq - k * f0) > max(1.5, k * f0 * 0.012):
                unexp.append((fq, a))
        if unexp:
            print("    unexplained:", " ".join(f"{fq:.1f}({a:.2f},{librosa.hz_to_note(fq)})" for fq, a in unexp[:10]))


if __name__ == "__main__":
    main()
