"""Candidate objectives for the loss bake-off, behind one interface.

Why there is a bake-off. The incumbent loss was measured to be unusable as a ranker of
patches: a 1e-8 change in one parameter moves the render by -85 dB and the loss by
+0.194, entirely from the log-magnitude term, and the objective's broad minimum sits
about rms 0.017 away from the true patch rather than at it. See `basin.py` and the
"settled by measurement" section of CLAUDE.md. So the question here is not which loss
scores lowest, it is which loss puts its minimum where the answer is.

The interface is a FACTORY, not a plain function:

    def my_loss(target: np.ndarray, sr: int) -> Callable[[np.ndarray], float]

The factory precomputes whatever depends only on the target, and returns a scorer that
takes one mono render and gives a float. That is the shape `stage2.Objective` already
has, and it matters here because the bake-off scores every candidate against a few
hundred cached renders per target: recomputing the target side each time would dominate.

Rules for a candidate:

- Mono in, float out. The corpus is mono because the incumbent objective is, and stereo
  width is a separate question deliberately deferred.
- Lower means closer. The screens assume that.
- Absolute scale is free. Every screen is a ratio, a rank or an argmin, so a candidate
  may return whatever units are natural.
- Deterministic, finite, and no new dependencies. numpy, scipy, librosa, torch and
  auraloss are all already here.
- Cheap enough to run a few hundred times. Aim under 50 ms per call at 17.9 s of audio.
"""

from __future__ import annotations

from typing import Callable, Protocol

import auraloss
import librosa
import numpy as np
import torch

SR = 44100
FFTS = (512, 1024, 2048, 4096)
HOPS = (128, 256, 512, 1024)


class Factory(Protocol):
    def __call__(self, target: np.ndarray, sr: int) -> Callable[[np.ndarray], float]: ...


LOSSES: dict[str, Factory] = {}


def register(name: str):
    def deco(fn: Factory) -> Factory:
        if name in LOSSES:
            raise ValueError(f"duplicate loss name {name!r}")
        LOSSES[name] = fn
        return fn
    return deco


# ---------------- shared helpers ----------------

def match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def mags(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    return np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft))


def env(x: np.ndarray, hop: int = 512) -> np.ndarray:
    """Unit-mean rms envelope. The incumbent's `w_env` term uses this shape."""
    f = np.lib.stride_tricks.sliding_window_view(x, hop * 2)[::hop]
    e = np.sqrt((f ** 2).mean(axis=-1))
    return e / (e.mean() + 1e-9)


# ---------------- the incumbent, as a control ----------------

@register("incumbent")
def incumbent(target: np.ndarray, sr: int = SR) -> Callable[[np.ndarray], float]:
    """Exactly `stage2.Objective.loss_of`: MRSTFT (sc + log-mag) plus 0.35 * env L1.

    Present so every screen reports the number the project has been optimising, next to
    whatever replaces it. Reproduces out/patch.json's recorded loss.
    """
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=list(FFTS), hop_sizes=list(HOPS), win_lengths=list(FFTS),
        w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0)
    t = torch.from_numpy(target.copy()).float().view(1, 1, -1)

    def _env_t(x: torch.Tensor, hop: int = 512) -> torch.Tensor:
        f = x.view(1, 1, -1).unfold(-1, hop * 2, hop)
        e = f.pow(2).mean(-1).sqrt()
        return e / (e.mean() + 1e-9)

    te = _env_t(t)

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        pt = torch.from_numpy(p.copy()).float().view(1, 1, -1)
        with torch.no_grad():
            spec = float(mrstft(pt, t[..., :len(p)]))
            e = float((_env_t(pt) - te[..., :_env_t(pt).shape[-1]]).abs().mean())
        v = spec + 0.35 * e
        return float(v) if np.isfinite(v) else 1e6
    return score


@register("sc_only")
def sc_only(target: np.ndarray, sr: int = SR) -> Callable[[np.ndarray], float]:
    """The incumbent with the log-magnitude term removed.

    The minimal edit implied by the measurement: on the 1e-8 perturbation that costs the
    incumbent 0.194, spectral convergence alone moves 0.000047. Included to find out what
    the log term was actually buying, which nobody here has measured.
    """
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=list(FFTS), hop_sizes=list(HOPS), win_lengths=list(FFTS),
        w_sc=1.0, w_log_mag=0.0, w_lin_mag=0.0)
    t = torch.from_numpy(target.copy()).float().view(1, 1, -1)

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        with torch.no_grad():
            v = float(mrstft(torch.from_numpy(p.copy()).float().view(1, 1, -1), t[..., :len(p)]))
        return v if np.isfinite(v) else 1e6
    return score


@register("logmag_floor")
def logmag_floor(target: np.ndarray, sr: int = SR, rel_db: float = -80.0
                 ) -> Callable[[np.ndarray], float]:
    """Log-magnitude, but with the magnitudes floored relative to each resolution's peak.

    The direct fix for the measured failure. auraloss adds a fixed small epsilon before
    the log, so bins that sit at -110 dB still contribute their full ratio; those are the
    near-empty bins between partials, and they are where an inaudible change becomes 0.19
    of loss. Flooring at `rel_db` below the target's own peak makes the log blind to
    anything already inaudible, without touching the bins that carry the timbre.
    """
    ref = [(f, h, mags(target, f, h)) for f, h in zip(FFTS, HOPS)]
    prepared = []
    for f, h, M in ref:
        floor = M.max() * (10.0 ** (rel_db / 20.0))
        prepared.append((f, h, floor, np.log(np.maximum(M, floor))))

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        tot = 0.0
        for f, h, floor, L in prepared:
            P = mags(p, f, h)
            n = min(P.shape[1], L.shape[1])
            tot += float(np.abs(np.log(np.maximum(P[:, :n], floor)) - L[:, :n]).mean())
        return tot / len(prepared)
    return score


@register("pow03")
def pow03(target: np.ndarray, sr: int = SR, p_exp: float = 0.3
          ) -> Callable[[np.ndarray], float]:
    """Multi-resolution distance on compressed magnitudes |X|^0.3.

    A power law compresses dynamic range like a log but stays finite and flat-gradiented
    at zero, so an empty bin cannot dominate. Standard in the DDSP literature for exactly
    the reason measured here.
    """
    ref = [(f, h, mags(target, f, h) ** p_exp) for f, h in zip(FFTS, HOPS)]

    def score(pred: np.ndarray) -> float:
        q, _ = match(pred, target)
        tot = 0.0
        for f, h, T in ref:
            P = mags(q, f, h) ** p_exp
            n = min(P.shape[1], T.shape[1])
            tot += float(np.abs(P[:, :n] - T[:, :n]).mean() / (T[:, :n].mean() + 1e-12))
        return tot / len(ref)
    return score


@register("mel_l1")
def mel_l1(target: np.ndarray, sr: int = SR, n_mels: int = 128,
           floor_db: float = -80.0) -> Callable[[np.ndarray], float]:
    """Mel-band log distance with a floor. Bands average the empty bins away.

    `mel_dist` is already one of the metrics the fit never sees and that has twice caught
    overfitting, so it is a natural candidate for something the fit SHOULD see.
    """
    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n_mels)

    def _mel_db(x: np.ndarray) -> np.ndarray:
        M = fb @ (mags(x, 2048, 512) ** 2)
        return np.maximum(10.0 * np.log10(M + 1e-12), floor_db)

    T = _mel_db(target)

    def score(pred: np.ndarray) -> float:
        q, _ = match(pred, target)
        P = _mel_db(q)
        n = min(P.shape[1], T.shape[1])
        return float(np.abs(P[:, :n] - T[:, :n]).mean())
    return score


@register("band26_env")
def band26_env(target: np.ndarray, sr: int = SR, floor_db: float = -80.0
               ) -> Callable[[np.ndarray], float]:
    """Per-band amplitude envelopes on the synth's own 26 third-octave centres.

    Deliberately matched to the EQ bank's geometry: if the patch's controls live on that
    grid, an objective on the same grid measures what the controls can actually change,
    and nothing finer. Time resolution is kept so envelopes still separate the ADSR
    stages, which a static spectrum cannot.
    """
    import synth  # local: keeps the loss registry importable without a Faust compile

    edges = synth.eq_band_freqs()
    fft, hop = 4096, 512
    freqs = librosa.fft_frequencies(sr=sr, n_fft=fft)
    # third-octave-ish support around each centre, matching EQ_Q's measured 0.508 octave
    fb = np.zeros((len(edges), len(freqs)))
    for i, fc in enumerate(edges):
        lo, hi = fc * 2 ** -0.254, fc * 2 ** 0.254
        fb[i] = ((freqs >= lo) & (freqs <= hi)).astype(float)
        if fb[i].sum() == 0:                       # bands below the bin spacing
            fb[i, np.argmin(np.abs(freqs - fc))] = 1.0
        fb[i] /= fb[i].sum()

    def _bands(x: np.ndarray) -> np.ndarray:
        return np.maximum(20.0 * np.log10(fb @ mags(x, fft, hop) + 1e-12), floor_db)

    T = _bands(target)

    def score(pred: np.ndarray) -> float:
        q, _ = match(pred, target)
        P = _bands(q)
        n = min(P.shape[1], T.shape[1])
        return float(np.abs(P[:, :n] - T[:, :n]).mean())
    return score


def load_candidates() -> list[str]:
    """Import every module under loss_candidates/, registering what it defines.

    Candidates live in their own package rather than in this file so a contributed loss
    cannot break the six baselines, and so `LOSSES` stays readable as the set of things
    the project itself stands behind.
    """
    import importlib
    import pkgutil

    import loss_candidates
    found = []
    for m in pkgutil.iter_modules(loss_candidates.__path__):
        importlib.import_module(f"loss_candidates.{m.name}")
        found.append(m.name)
    return found
