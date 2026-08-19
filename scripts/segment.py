"""Segment the pad into stable chord regions and identify the notes in each.

basic-pitch shreds sustained reverb-heavy pads into ghost notes, so instead we
find chord-change boundaries with spectral novelty, then fit a harmonic comb per
segment with iterative subtraction (kills octave/fifth ghosts).
"""

from __future__ import annotations

import json

import librosa
import numpy as np

SR = 44100
CLIP = "data/original.wav"
HOP = 512


def boundaries(y: np.ndarray, sr: int) -> np.ndarray:
    """Chord-change times from CQT flux, restricted to sustained region."""
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz("C1"), n_bins=84))
    L = librosa.amplitude_to_db(C, ref=np.max)
    # positive flux only: chord changes light up new partials
    flux = np.maximum(0.0, np.diff(L, axis=1)).sum(axis=0)
    flux = flux / (flux.max() + 1e-9)
    times = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=HOP)

    peaks = librosa.util.peak_pick(
        flux, pre_max=20, post_max=20, pre_avg=40, post_avg=40, delta=0.06, wait=40
    )
    return times[peaks]


def harmonic_fit(
    mag: np.ndarray, freqs: np.ndarray, sr: int, lo: int = 21, hi: int = 96, n_notes: int = 8
) -> list[tuple[int, float]]:
    """Greedy harmonic-comb fit with subtraction. Returns [(midi, strength)]."""
    resid = mag.copy()
    n_harm = 12
    picked: list[tuple[int, float]] = []

    def comb_bins(midi: int) -> list[np.ndarray]:
        f0 = librosa.midi_to_hz(midi)
        out = []
        for k in range(1, n_harm + 1):
            f = f0 * k
            if f >= sr / 2 - 50:
                break
            # tolerance ~ half a semitone at that harmonic, min 3 bins
            tol = max(3.0, f * 0.02)
            out.append(np.where(np.abs(freqs - f) <= tol)[0])
        return out

    combs = {m: comb_bins(m) for m in range(lo, hi + 1)}

    for _ in range(n_notes):
        best, best_score = None, 0.0
        for m, bins in combs.items():
            score = 0.0
            for k, b in enumerate(bins, start=1):
                if len(b):
                    score += resid[b].max() / (k**0.7)
            if score > best_score:
                best, best_score = m, score
        if best is None or best_score <= 0:
            break
        # subtract the explained comb from the residual
        for k, b in enumerate(combs[best], start=1):
            if len(b):
                resid[b] *= 0.05
        picked.append((best, float(best_score)))
    return picked


def segment_spectrum(y: np.ndarray, sr: int, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Average magnitude spectrum over the stable middle of a segment."""
    pad = min(0.25, (t1 - t0) * 0.2)
    a, b = int((t0 + pad) * sr), int((t1 - pad) * sr)
    seg = y[a:b]
    n_fft = 32768
    if len(seg) < n_fft:
        n_fft = 1 << int(np.floor(np.log2(max(len(seg), 1024))))
    S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=n_fft // 4))
    mag = np.median(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return mag, freqs


def main() -> None:
    y, sr = librosa.load(CLIP, sr=SR, mono=True)
    dur = len(y) / sr

    bounds = boundaries(y, sr)
    print("raw boundaries:", np.round(bounds, 3).tolist())

    # Build segment edges over the sustained part; treat 0..first as the riser.
    edges = [0.0] + [float(b) for b in bounds if 0.3 < b < dur - 0.3] + [dur]
    edges = sorted(set(np.round(edges, 3)))
    # merge segments shorter than 0.6s
    merged = [edges[0]]
    for e in edges[1:]:
        if e - merged[-1] < 0.6:
            continue
        merged.append(e)
    if merged[-1] < dur:
        merged[-1] = dur
    print("segments:", [(round(merged[i], 2), round(merged[i + 1], 2)) for i in range(len(merged) - 1)])

    results = []
    for i in range(len(merged) - 1):
        t0, t1 = merged[i], merged[i + 1]
        mag, freqs = segment_spectrum(y, sr, t0, t1)
        picked = harmonic_fit(mag, freqs, sr)
        if not picked:
            continue
        top = picked[0][1]
        keep = [(m, s) for m, s in picked if s > top * 0.22]
        keep.sort()
        names = [f"{librosa.midi_to_note(m)}({s / top:.2f})" for m, s in keep]
        print(f"\n[{t0:6.2f}-{t1:6.2f}]  {' '.join(names)}")
        results.append({"t0": t0, "t1": t1, "notes": [[int(m), float(s / top)] for m, s in keep]})

    with open("out/segments.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote out/segments.json")


if __name__ == "__main__":
    main()
