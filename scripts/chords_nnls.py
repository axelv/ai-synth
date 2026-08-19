"""Chord estimation by non-negative decomposition over a saw-harmonic dictionary.

Unlike greedy correlation, NNLS is penalised for templates that predict partials
which are not actually present, so it resolves the octave ambiguity that a plain
harmonic-comb maximiser gets wrong on bass-heavy material.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
from scipy.optimize import nnls

SR = 44100
N_FFT = 16384
FLO, FHI = 28.0, 3500.0
PLO, PHI = 21, 88
N_HARM = 20

REGIONS: list[tuple[float, float]] = [
    (3.45, 4.95),
    (4.95, 7.45),
    (7.45, 10.40),
    (10.40, 13.35),
    (13.35, 16.05),
    (16.05, 17.90),
]


def smooth_env(L: np.ndarray, w: int = 201) -> np.ndarray:
    k = np.ones(w) / w
    return np.convolve(L, k, mode="same")


def build_dict(freqs: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Column per MIDI pitch: gaussian bumps at harmonics, unit-normalised."""
    pitches = list(range(PLO, PHI + 1))
    T = np.zeros((len(freqs), len(pitches)))
    for j, p in enumerate(pitches):
        f0 = float(librosa.midi_to_hz(p))
        col = np.zeros_like(freqs)
        for k in range(1, N_HARM + 1):
            f = f0 * k
            if f > FHI:
                break
            # width absorbs unison detune (~25 cents) and reverb smear
            sigma = max(2.5, f * 0.014)
            col += np.exp(-0.5 * ((freqs - f) / sigma) ** 2) / (k**0.5)
        n = np.linalg.norm(col)
        if n > 0:
            T[:, j] = col / n
    return T, pitches


def observed(y: np.ndarray, t0: float, t1: float, freqs_mask: np.ndarray) -> np.ndarray:
    seg = y[int(t0 * SR) : int(t1 * SR)]
    if len(seg) < N_FFT:
        seg = np.pad(seg, (0, N_FFT - len(seg)))
    S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=N_FFT // 4))
    mag = np.median(S, axis=1)[freqs_mask]
    # flatten the global spectral tilt so the fit is about comb structure
    L = np.log(mag + 1e-8)
    flat = np.exp(L - smooth_env(L))
    flat -= flat.min()
    return flat / (np.linalg.norm(flat) + 1e-9)


def main() -> None:
    y, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    all_f = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mask = (all_f >= FLO) & (all_f <= FHI)
    freqs = all_f[mask]
    T, pitches = build_dict(freqs)

    out = []
    for t0, t1 in REGIONS:
        # late-stable window: least contaminated by the previous chord's tail
        a = t0 + 0.5 * (t1 - t0)
        b = min(t1 - 0.05, a + 1.4)
        obs = observed(y, a, b, mask)
        act, res = nnls(T, obs)
        order = np.argsort(-act)
        top = act[order[0]]
        keep = [(pitches[i], act[i] / top) for i in order if act[i] > top * 0.15]
        keep.sort()
        names = " ".join(f"{librosa.midi_to_note(p)}({w:.2f})" for p, w in keep)
        print(f"[{t0:5.2f}-{t1:5.2f}] resid={res:.4f}  {names}")
        out.append({"t0": t0, "t1": t1, "notes": [[int(p), float(w)] for p, w in keep]})

    with open("out/chords_nnls.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote out/chords_nnls.json")


if __name__ == "__main__":
    main()
