"""Single-chord fitting bench: one held F major voicing, no pitch bend.

Why a window instead of the whole clip. The oracle ladder said the recoverable
error in the 18 s render is a static spectral envelope (a best-possible fixed EQ
buys 5x what a best-possible volume envelope buys), and the fitted curve is
non-monotonic, so a two-pole lowpass cannot reach it at any setting. That is a
timbre problem, and timbre is stationary, so it does not need 29 notes and a
measured bend curve to solve. It needs one steady chord.

The window is 4.95-7.45 s: highest frame-to-frame spectral stationarity of any
bend-free segment (0.9985), five notes held across the whole of it with no note
change and no attack inside it, and it sits 1.5 s clear of the intro glide and
6 s clear of the +3 semitone bend.

Everything here is a slice of the full render rather than a new rendering setup:
notes and bend curve are exactly stage 2's, the engine just stops at WIN_T1 and
the loss looks only at [WIN_T0, WIN_T1]. So no new modelling assumption is
introduced by narrowing the problem, and a patch fitted here can be dropped
straight back into the 18 s render.

Why the harmonic amplitudes are not CMA-ES parameters. Given the note pitches,
a candidate's partial amplitudes enter the spectrum linearly, so they are a
least-squares problem, not a search problem. Splitting them out is the whole
point: the linear part gets solved, and CMA-ES keeps only the handful of
genuinely nonlinear macro controls (envelopes, filter, effects).
"""

from __future__ import annotations

import auraloss
import librosa
import numpy as np
import pretty_midi
import torch

SR = 44100
WIN_T0 = 4.95
WIN_T1 = 7.45
PITCHES = (29, 45, 48, 53, 57)          # F1 A2 C3 F3 A3, all held across the window
H_TABLE = 64                            # wavetable harmonics; h=64 of F1 is 2.8 kHz, and
                                        # 99% of the window's energy is below 2.46 kHz
MIDI_PATH = "data/transcription.mid"
TARGET_PATH = "data/original.wav"


def win_slice(sr: int = SR) -> slice:
    return slice(int(WIN_T0 * sr), int(WIN_T1 * sr))


def n_window(sr: int = SR) -> int:
    s = win_slice(sr)
    return s.stop - s.start


def target(sr: int = SR) -> np.ndarray:
    """The window of the reference, mono. NOTE: original.wav is 48 kHz, so this
    must resample; reading it with soundfile and no resample silently compares
    audio that is 8.8% off in time and pitch."""
    y, _ = librosa.load(TARGET_PATH, sr=sr, mono=True)
    return y[win_slice(sr)].astype(np.float64)


def notes_upto(t1: float = WIN_T1, path: str = MIDI_PATH):
    """stage2.load_notes, truncated to what can sound before t1."""
    pm = pretty_midi.PrettyMIDI(path)
    out = [(n.pitch, n.velocity, float(n.start), float(n.end - n.start))
           for inst in pm.instruments for n in inst.notes if n.start < t1]
    return sorted(out, key=lambda z: z[2])


def f0s(pitches=PITCHES) -> np.ndarray:
    return librosa.midi_to_hz(np.asarray(pitches, dtype=float))


def partial_table(pitches=PITCHES, fmax: float = 16000.0):
    """(note_index, harmonic_number, frequency) for every audible partial."""
    rows = []
    for k, f0 in enumerate(f0s(pitches)):
        for h in range(1, int(fmax / f0) + 1):
            rows.append((k, h, h * f0))
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------- scoring

class WindowScore:
    """The stage-2 objective restricted to the window.

    Same FFT sizes and same w_env as stage2.Objective, so candidates rank the same
    way they would in the full fit. The absolute number is NOT comparable to the
    18 s loss (1.3823): different signal, different length. Compare within the
    window only.
    """

    def __init__(self, sr: int = SR, w_env: float = 0.35) -> None:
        self.sr = sr
        self.w_env = w_env
        t = torch.from_numpy(target(sr)).float().view(1, 1, -1)
        self.tgt = t
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048, 4096],
            hop_sizes=[128, 256, 512, 1024],
            win_lengths=[512, 1024, 2048, 4096],
            w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0,
        )
        self._tgt_env = self._env(t)

    @staticmethod
    def _env(x: torch.Tensor, hop: int = 512) -> torch.Tensor:
        f = x.reshape(1, 1, -1).unfold(-1, hop * 2, hop)
        e = f.pow(2).mean(-1).sqrt()
        return e / (e.mean() + 1e-9)

    def __call__(self, audio: np.ndarray) -> float:
        """audio is the WINDOW only: (n,) mono or (2, n) stereo."""
        m = np.asarray(audio, dtype=np.float64)
        m = m.mean(0) if m.ndim > 1 else m
        if not np.isfinite(m).all():
            return 1e6
        n = min(len(m), self.tgt.shape[-1])
        p = torch.from_numpy(m[:n].copy()).float().view(1, 1, -1)
        with torch.no_grad():
            v = float(self.mrstft(p, self.tgt[..., :n])) \
                + self.w_env * float((self._env(p) - self._tgt_env[..., :self._env(p).shape[-1]]).abs().mean())
        return v if np.isfinite(v) else 1e6

    def cos_theta(self, audio: np.ndarray, nfft: int = 2048) -> float:
        """Scale-invariant spectral shape agreement. Cannot be gamed by output level,
        unlike the loss, whose optimum sits at (level ratio) x cos(theta)."""
        m = np.asarray(audio, dtype=np.float64)
        m = m.mean(0) if m.ndim > 1 else m
        n = min(len(m), self.tgt.shape[-1])
        O = np.abs(librosa.stft(self.tgt.numpy().ravel()[:n], n_fft=nfft, hop_length=nfft // 4))
        R = np.abs(librosa.stft(m[:n], n_fft=nfft, hop_length=nfft // 4))
        O, R = O.astype(np.float64), R.astype(np.float64)
        return float((O * R).sum() / (np.linalg.norm(O) * np.linalg.norm(R) + 1e-30))


# ---------------------------------------------------------------- additive helpers

def harmonic_readout(x: np.ndarray, pitches=PITCHES, fmax: float = 16000.0,
                     sr: int = SR) -> np.ndarray:
    """Amplitude of x at each partial frequency, by windowed DFT at that exact bin.

    Reads amplitudes straight off the spectrum rather than solving a design matrix:
    over a 2.5 s window the partials of a 43.65 Hz fundamental are 20+ Hanning
    bin-widths apart, so they do not interfere.
    """
    tab = partial_table(pitches, fmax)
    w = np.hanning(len(x))
    xw = x * w
    t = np.arange(len(x)) / sr
    amp = np.empty(len(tab))
    norm = w.sum() / 2.0
    for i, (_, _, f) in enumerate(tab):
        c = (xw * np.cos(2 * np.pi * f * t)).sum()
        s = (xw * np.sin(2 * np.pi * f * t)).sum()
        amp[i] = np.hypot(c, s) / norm
    return amp


def additive_render(tab: np.ndarray, amp: np.ndarray, n: int, sr: int = SR,
                    seed: int = 0) -> np.ndarray:
    """Sum of sinusoids at the table's frequencies with the given amplitudes.

    Phases are random but seeded. Phase is not fitted: the loss is a magnitude
    STFT so it cannot see absolute phase, and a real oscillator bank has no way
    to reproduce a chosen one anyway.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for (_, _, f), a in zip(tab, amp):
        if a > 0:
            y += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return y


def single_cycle(a: np.ndarray, n: int = 2048) -> np.ndarray:
    """Harmonic amplitudes -> one cycle of the waveform, for export as a wavetable.

    Cosine phase for every harmonic. A wavetable's phase spectrum is arbitrary as
    far as this objective is concerned, and all-cosine keeps the table symmetric
    and its peak predictable.
    """
    spec = np.zeros(n // 2 + 1, dtype=complex)
    spec[1:len(a) + 1] = np.asarray(a, dtype=float)[:n // 2]
    y = np.fft.irfft(spec, n=n)
    return y / (np.abs(y).max() + 1e-12)


def describe() -> str:
    return (f"window {WIN_T0}-{WIN_T1}s ({n_window()/SR:.2f}s, {n_window()} samples), "
            f"pitches {list(PITCHES)} = "
            f"{[librosa.midi_to_note(p) for p in PITCHES]}, "
            f"{len(partial_table())} partials below 16 kHz")


if __name__ == "__main__":
    print(describe())
    tgt = target()
    sc = WindowScore()
    print(f"target rms {20*np.log10(np.sqrt((tgt**2).mean())):.2f} dB")
    print(f"score(target) = {sc(tgt):.6f}  cos_theta {sc.cos_theta(tgt):.4f}  (sanity: 0 and 1)")
    tab = partial_table()
    amp = harmonic_readout(tgt)
    rec = additive_render(tab, amp, len(tgt))
    print(f"score(additive readout) = {sc(rec):.6f}  cos_theta {sc.cos_theta(rec):.4f}")
