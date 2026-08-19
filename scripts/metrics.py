"""Objective metrics for both stages. Everything is computed on rendered audio."""

from __future__ import annotations

import librosa
import mir_eval
import numpy as np

SR = 44100
HOP = 512


def to_mono(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=0) if a.ndim > 1 else a


def match_len(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def onset_times(y: np.ndarray, sr: int = SR) -> np.ndarray:
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    return librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, units="time", backtrack=False)


def onset_f(ref: np.ndarray, est: np.ndarray, window: float = 0.15) -> tuple[float, float, float]:
    if len(ref) == 0 and len(est) == 0:
        return 1.0, 1.0, 1.0
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0, 0.0
    f, p, r = mir_eval.onset.f_measure(np.asarray(ref), np.asarray(est), window=window)
    return float(f), float(p), float(r)


def chroma(y: np.ndarray, sr: int = SR) -> np.ndarray:
    c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)


def chroma_agreement(a: np.ndarray, b: np.ndarray, sr: int = SR) -> float:
    """Mean per-frame cosine similarity of chromagrams."""
    ca, cb = chroma(a, sr), chroma(b, sr)
    n = min(ca.shape[1], cb.shape[1])
    return float((ca[:, :n] * cb[:, :n]).sum(axis=0).mean())


def pitch_class_hist(y: np.ndarray, sr: int = SR) -> np.ndarray:
    c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP).mean(axis=1)
    return c / (c.sum() + 1e-9)


def loudness_env(y: np.ndarray, sr: int = SR) -> np.ndarray:
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    return rms


def env_l1(a: np.ndarray, b: np.ndarray, sr: int = SR) -> float:
    ea, eb = match_len(loudness_env(a, sr), loudness_env(b, sr))
    # scale-invariant: normalize each to unit mean
    ea = ea / (ea.mean() + 1e-9)
    eb = eb / (eb.mean() + 1e-9)
    return float(np.abs(ea - eb).mean())


def mel_db(y: np.ndarray, sr: int = SR, n_mels: int = 128) -> np.ndarray:
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=HOP, fmax=14000)
    return librosa.power_to_db(M, ref=1e-6)


def mel_dist(a: np.ndarray, b: np.ndarray, sr: int = SR) -> float:
    Ma, Mb = mel_db(a, sr), mel_db(b, sr)
    n = min(Ma.shape[1], Mb.shape[1])
    Ma, Mb = Ma[:, :n], Mb[:, :n]
    # remove overall level so this measures spectral shape, not gain
    Ma = Ma - Ma.mean()
    Mb = Mb - Mb.mean()
    return float(np.abs(Ma - Mb).mean())


def stereo_decorrelation(a: np.ndarray) -> float:
    """1 - corr(L, R) on the raw waveforms. 0 is mono, 1 is fully decorrelated.

    The stage-2 loss is mono, so this is never optimised; it is only ever measured.
    """
    if a.ndim < 2 or a.shape[0] < 2:
        return 0.0
    lo, hi = a[0].astype(np.float64), a[1].astype(np.float64)
    lo = lo - lo.mean()
    hi = hi - hi.mean()
    d = np.sqrt((lo * lo).sum() * (hi * hi).sum())
    return 1.0 if d < 1e-20 else float(1.0 - (lo * hi).sum() / d)


def lta_spectrum(y: np.ndarray, sr: int = SR, n_fft: int = 8192,
                 hop: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Long-term average spectrum in dB, plus its bin frequencies."""
    s = np.abs(librosa.stft(to_mono(y), n_fft=n_fft, hop_length=hop)) ** 2
    return librosa.fft_frequencies(sr=sr, n_fft=n_fft), 10.0 * np.log10(s.mean(axis=1) + 1e-20)


BAND_EDGES = (20.0, 60.0, 250.0, 900.0, 2000.0, 6000.0, 16000.0)


def lta_band_error(orig: np.ndarray, rend: np.ndarray, edges: tuple[float, ...] = BAND_EDGES,
                   sr: int = SR) -> dict[str, float]:
    """Gain-aligned signed dB error of the long-term average spectrum, per octave band.

    Same alignment as band_db_error, so the numbers are comparable: whatever the render
    is missing in one band it must be carrying in another. That is what separates a
    waveshaper that added midrange harmonics from one that added broadband aliasing.
    """
    f, do = lta_spectrum(orig, sr)
    _, dr = lta_spectrum(rend, sr)
    wide = (f >= edges[0]) & (f <= edges[-1])
    offset = float((dr[wide] - do[wide]).mean())
    out: dict[str, float] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        out[f"{lo:.0f}_{hi:.0f}"] = float((dr[m] - do[m]).mean() - offset)
    return out


def band_db_error(orig: np.ndarray, rend: np.ndarray, lo: float = 250.0, hi: float = 900.0,
                  sr: int = SR, ref_lo: float = 20.0, ref_hi: float = 16000.0) -> dict[str, float]:
    """Signed dB error of the render's long-term average spectrum inside a band.

    Gain-aligned over the whole audible range first, because a render that is simply
    quieter is a different defect from one whose midrange is missing. Negative means
    the render is light in the band.
    """
    f, do = lta_spectrum(orig, sr)
    _, dr = lta_spectrum(rend, sr)
    wide = (f >= ref_lo) & (f <= ref_hi)
    offset = float((dr[wide] - do[wide]).mean())
    band = (f >= lo) & (f <= hi)
    err = dr[band] - do[band] - offset
    worst = int(np.argmin(err))
    return {
        "band_lo_hz": lo,
        "band_hi_hz": hi,
        "gain_offset_db": offset,
        "mean_signed_db": float(err.mean()),
        "mean_abs_db": float(np.abs(err).mean()),
        "worst_signed_db": float(err[worst]),
        "worst_hz": float(f[band][worst]),
    }


def report(orig: np.ndarray, rend: np.ndarray, sr: int = SR) -> dict[str, float]:
    o, r = to_mono(orig), to_mono(rend)
    o, r = match_len(o, r)
    f, p, rc = onset_f(onset_times(o, sr), onset_times(r, sr))
    return {
        "onset_f": f,
        "onset_p": p,
        "onset_r": rc,
        "chroma_agree": chroma_agreement(o, r, sr),
        "env_l1": env_l1(o, r, sr),
        "mel_dist": mel_dist(o, r, sr),
    }
