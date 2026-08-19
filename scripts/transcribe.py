"""Stage 1: recover the MIDI.

The clip is a slow pad: a moving bass note under a sustained voicing, with heavy
reverb bleeding each chord into the next. Notes are chosen by greedy search that
renders candidate voicings through a neutral supersaw and compares *whitened*
spectra (tilt-invariant, so the filter setting cannot bias note choice).
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from synth import PadRenderer, denorm, norm_defaults

SR = 44100
CLIP = "data/original.wav"

# Chord regions with a confidently-identified bass fundamental (from pitch_probe).
# (t_start, t_end) of the *stable* part of each chord.
REGIONS: list[tuple[float, float]] = [
    (3.45, 4.95),
    (4.95, 7.45),
    (7.45, 10.40),
    (10.40, 13.35),
    (13.35, 16.05),
    (16.05, 17.90),
]

CAND_LO, CAND_HI = 24, 84  # C1..C6
MAX_NOTES = 6


def whiten(mag: np.ndarray, smooth: int = 141) -> np.ndarray:
    """Log-magnitude minus its smoothed envelope: keeps comb structure, drops tilt."""
    L = np.log(mag + 1e-8)
    k = np.ones(smooth) / smooth
    env = np.convolve(L, k, mode="same")
    w = L - env
    return w / (np.linalg.norm(w) + 1e-9)


def seg_spectrum(y: np.ndarray, sr: int, t0: float, t1: float, n_fft: int = 16384) -> np.ndarray:
    a, b = int(t0 * sr), int(t1 * sr)
    seg = y[a:b]
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=n_fft // 4))
    return np.median(S, axis=1)


def band(mag: np.ndarray, freqs: np.ndarray, lo: float = 30.0, hi: float = 5000.0) -> np.ndarray:
    m = (freqs >= lo) & (freqs <= hi)
    return mag[m]


def neutral_params() -> dict[str, float]:
    """Bright, dry, sustained patch: maximises harmonic visibility for note ID."""
    p = denorm(norm_defaults())
    p.update(
        detune=45.0, uniMix=0.8, subLvl=0.0, sqrMix=0.0,
        cutoff=9000.0, reso=0.7, envAmt=0.0, kbdTrk=0.0,
        aA=0.01, aD=0.05, aS=1.0, aR=0.2, fA=0.01, fD=0.05, fS=1.0,
        lfoAmt=0.0, chDepth=0.0, dlyWet=0.0, revWet=0.0, tilt=0.0, outGain=0.8,
    )
    return p


def render_chord(r: PadRenderer, notes: list[int], dur: float = 2.0) -> np.ndarray:
    r.set_notes([(p, 100, 0.0, dur * 0.9) for p in notes])
    a = r.render(dur)
    return a.mean(axis=0)


def score(target_w: np.ndarray, cand: np.ndarray, freqs: np.ndarray) -> float:
    cw = whiten(band(cand, freqs))
    return float(np.dot(target_w, cw))


def fit_region(r: PadRenderer, y: np.ndarray, t0: float, t1: float, freqs: np.ndarray) -> list[int]:
    """Greedy add-one-note search maximising whitened-spectrum similarity."""
    # analyse the late-stable part of the region (least contaminated by the
    # previous chord's reverb tail)
    a = t0 + 0.45 * (t1 - t0)
    b = min(t1 - 0.05, a + 1.2)
    tgt = whiten(band(seg_spectrum(y, SR, a, b), freqs))

    chosen: list[int] = []
    best_s = -1.0
    for _ in range(MAX_NOTES):
        best_add, best_add_s = None, best_s
        for p in range(CAND_LO, CAND_HI + 1):
            if p in chosen:
                continue
            cand_notes = sorted(chosen + [p])
            spec = band_spec(r, cand_notes, freqs)
            s = float(np.dot(tgt, whiten(spec)))
            if s > best_add_s:
                best_add, best_add_s = p, s
        if best_add is None or best_add_s < best_s + 0.006:
            break
        chosen = sorted(chosen + [best_add])
        best_s = best_add_s
        print(f"    + {librosa.midi_to_note(best_add):>5s}  score {best_s:.4f}  -> {[librosa.midi_to_note(c) for c in chosen]}")
    return chosen


_spec_cache: dict[tuple[int, ...], np.ndarray] = {}


def band_spec(r: PadRenderer, notes: list[int], freqs: np.ndarray) -> np.ndarray:
    key = tuple(notes)
    if key not in _spec_cache:
        aud = render_chord(r, list(notes))
        n_fft = 16384
        S = np.abs(librosa.stft(aud[int(0.4 * SR) : int(1.6 * SR)], n_fft=n_fft, hop_length=n_fft // 4))
        _spec_cache[key] = band(np.median(S, axis=1), freqs)
    return _spec_cache[key]


def main() -> None:
    y, _ = librosa.load(CLIP, sr=SR, mono=True)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=16384)
    r = PadRenderer(n_voices=24)
    r.set_params(neutral_params())

    out = []
    for t0, t1 in REGIONS:
        print(f"\n[{t0:5.2f}-{t1:5.2f}]")
        notes = fit_region(r, y, t0, t1, freqs)
        print(f"  => {[librosa.midi_to_note(n) for n in notes]}")
        out.append({"t0": t0, "t1": t1, "notes": notes})

    with open("out/chords.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote out/chords.json")


if __name__ == "__main__":
    main()
