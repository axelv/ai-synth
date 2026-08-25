"""Four objectives taken from published practice, chosen against the measured failure.

The diagnosis is specific: auraloss builds its magnitudes as
`sqrt(clamp(re^2 + im^2, min=1e-8))`, so the log term's floor sits at magnitude 1e-4.
On an unnormalised STFT of a full-scale pad the peak magnitude is O(100), which puts
that floor near -120 dB relative to peak. The perturbation that costs the incumbent
0.194 lives at -85 dB, comfortably above the floor, so every near-empty bin between
partials contributes its full log ratio. An L1 over log magnitudes weights a bin by the
RELATIVE change in it, and relative change is unbounded exactly where there is no
energy. That is the whole bug, and it is a convention choice, not a law of spectral
losses.

The literature has four separate answers to it, and they are separate mechanisms rather
than four spellings of "compress the magnitudes", which is why all four are here:

- `nmr_mask`     measure error against a psychoacoustic masking threshold, so inaudible
                 is defined rather than approximated (MPEG psychoacoustic model 1, and
                 NMR as used in PEAQ / ITU-R BS.1387).
- `jtfs_lite`    joint time-frequency scattering, the representation reported to beat
                 multi-scale spectral loss specifically for synth parameter estimation
                 (Anden and Mallat 2014; Vahidi et al. 2023; Han et al. PNP 2023).
- `mfcc_match`   truncated cepstral distance, the standard fitness in the synthesiser
                 sound-matching line of work (Yee-King et al. 2018 and the GA work it
                 grew out of).
- `beta_kl`      the incumbent's own STFTs with only the divergence changed, to
                 generalised Kullback-Leibler, beta = 1 in the beta-divergence family
                 (Fevotte and Idier 2011). An ablation of the convention alone.

Rejected after thinking it through, recorded so nobody spends the afternoon:

- A faithful port of DDSP's `SpectralLoss` (Engel et al. 2020). Its `safe_log` floors at
  an ABSOLUTE 1e-5, which on these unnormalised magnitudes is about -140 dB relative to
  peak, i.e. lower than the incumbent's floor, not higher. Its actual protection is the
  co-equal linear-magnitude term, and `pow03` already covers compressed-magnitude L1
  better. Porting it would have measured the epsilon, not the idea.
- PCEN (Wang, Lostanlen et al.), the obvious "log blows up on quiet bins" replacement.
  Its adaptive gain divides each band by a smoothed copy of itself, so with the usual
  gain = 0.98 it retains band energy only as E^0.02. Twenty-six of these fifty-five
  parameters ARE per-band gains. A loss engineered to normalise away per-band gain
  cannot rank them. Discarded on that argument alone.
- CQT via `librosa.cqt`. Right answer to the pitch-resolution complaint (Turian and
  Henry 2020), wrong cost: hundreds of milliseconds on 17.9 s, against a 50 ms budget.
  `jtfs_lite` gets log-frequency geometry from a mel filterbank instead.
"""

from __future__ import annotations

from typing import Callable

import librosa
import numpy as np
import scipy.fft
import scipy.ndimage

from losses import SR, mags, match, register


def _power(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """|STFT|^2 without the intermediate magnitude. abs() then square costs a full extra
    pass over a 1025 x 1542 array, which is 20 ms of a 50 ms budget for nothing."""
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft)
    return (S.real ** 2 + S.imag ** 2).astype(np.float32)


def _finite(v: float) -> float:
    """Same contract as the incumbent: a non-finite render is worst, never NaN-poisons
    the ranking. Only reachable if the renderer hands back a blown-up signal."""
    return float(v) if np.isfinite(v) else 1e6

# ---------------------------------------------------------------- 1. masking / NMR


def _bark(f_hz: np.ndarray) -> np.ndarray:
    """Traunmuller's analytic Bark scale. Cheaper than a table and monotone to Nyquist."""
    z = (26.81 * f_hz) / (1960.0 + f_hz) - 0.53
    return np.maximum(z, 0.0)


def _ath_db(f_hz: np.ndarray) -> np.ndarray:
    """Terhardt's absolute threshold of hearing, dB SPL, f in kHz internally."""
    f = np.maximum(f_hz, 20.0) / 1000.0
    return (3.64 * f ** -0.8
            - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2)
            + 1e-3 * f ** 4)


@register("nmr_mask")
def nmr_mask(target: np.ndarray, sr: int = SR, n_bands: int = 26,
             spl_full_scale: float = 96.0) -> Callable[[np.ndarray], float]:
    """Noise-to-mask ratio: error energy divided by the target's own masking threshold.

    Every other candidate here approximates "inaudible" with a floor, a compression
    exponent or a filterbank. This one takes the definition from perceptual coding
    instead. The target's band energies are spread across critical bands with the
    standard asymmetric spreading function, offset by the signal-to-mask ratio, and
    floored by the absolute threshold in quiet; that gives a per-band, per-frame
    threshold below which added noise cannot be heard. The error is then charged only in
    units of that threshold. This is the NMR of ITU-R BS.1387, and the reason to reach
    for it here is that it is the only construction on the list whose insensitivity to a
    -85 dB perturbation is a stated design property rather than a side effect.

    The one deviation from the codec version: PEAQ takes the noise as the spectrum of
    the time-domain difference. Faust's oscillators are free-running, so two renders of
    the same patch differ in phase and a time-domain difference would measure phase.
    The noise here is the magnitude difference per bin, which is phase-blind, as
    CLAUDE.md requires.

    log1p rather than a hard max(0, dB) keeps the loss smooth and weakly informative
    below the mask instead of exactly flat, so a search still has something to follow
    once every band is already inaudible-close.
    """
    n_fft, hop = 2048, 512
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    z = _bark(freqs)

    # Equal-Bark band edges rather than the tabulated 24, so the top octave, which the
    # table stops short of, still gets bands. n_bands = 26 also lands close to the EQ's
    # own third-octave grid, which is the resolution the patch can actually control.
    edges = np.linspace(0.0, z[-1], n_bands + 1)
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, n_bands - 1)
    B = np.zeros((n_bands, len(freqs)))
    B[idx, np.arange(len(freqs))] = 1.0
    zc = 0.5 * (edges[:-1] + edges[1:])

    # Asymmetric spreading: steep below the masker, shallow above it. Slopes are the
    # usual simplified MPEG-1 figures; the level-dependent upper slope is dropped
    # because it would make the threshold depend on the render as well as the target.
    dz = zc[None, :] - zc[:, None]              # maskee minus masker, in Bark
    spread_db = np.where(dz >= 0, -12.0 * dz, 27.0 * dz)
    spread = 10.0 ** (spread_db / 10.0)

    # Signal-to-mask offset. Tonal maskers hide less than noise does, and a pad is
    # tonal, so this is the tonal branch of the model, 14.5 + z dB.
    smr = 10.0 ** (-(14.5 + zc) / 10.0)

    tgt_pow_f32 = _power(target, n_fft, hop)
    tgt_pow = tgt_pow_f32.astype(np.float64)
    band_t = B @ tgt_pow
    thresh = (spread.T @ band_t) * smr[:, None]

    # Absolute threshold, anchored by declaring a full-scale sine to be `spl_full_scale`.
    # Only the anchor is arbitrary; it sets where "quiet" stops mattering at all.
    band_f = np.array([freqs[idx == i].mean() if (idx == i).any() else freqs[-1]
                       for i in range(n_bands)])
    n_bins = np.maximum(B.sum(axis=1), 1.0)
    ref = (np.abs(tgt_pow).max() + 1e-30)
    ath = ref * 10.0 ** ((_ath_db(band_f) - spl_full_scale) / 10.0) * n_bins
    floor = thresh + ath[:, None]

    # Magnitudes for both sides come off the same code path, so scoring the target
    # against itself is exactly 0 rather than 1e-14 of float mismatch.
    tgt_mag = np.sqrt(tgt_pow_f32)

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        pm = np.sqrt(_power(p, n_fft, hop))
        n = min(pm.shape[1], tgt_mag.shape[1])
        noise = B @ ((pm[:, :n] - tgt_mag[:, :n]) ** 2).astype(np.float64)
        return _finite(np.log1p(noise / floor[:, :n]).mean())
    return score


# ---------------------------------------------------- 2. joint time-frequency scattering


def _morlet_time(n_rfft: int, xi: float, sigma: float) -> np.ndarray:
    """One-sided log-normal bandpass on the rfft grid: analytic, so slope sign survives."""
    w = np.arange(n_rfft, dtype=np.float64)
    out = np.zeros(n_rfft)
    m = w > 0
    out[m] = np.exp(-(np.log2(w[m] / xi) ** 2) / (2.0 * sigma ** 2))
    return out


def _gauss_freq(n: int, xi: float, sigma: float) -> np.ndarray:
    """Bandpass along the log-frequency axis. Signed centre, so +xi and -xi separate
    rising from falling spectral slopes, which is the 'joint' in joint scattering."""
    k = np.fft.fftfreq(n)
    return np.exp(-((k - xi) ** 2) / (2.0 * sigma ** 2))


@register("jtfs_lite")
def jtfs_lite(target: np.ndarray, sr: int = SR, n_bands: int = 48,
              seg: int = 16) -> Callable[[np.ndarray], float]:
    """Joint time-frequency scattering, second order, at a cost that fits the budget.

    The published claim this is here to test: for synthesiser parameter estimation
    specifically, not for waveform generation, a scattering representation ranks
    parameter distance better than a multi-scale spectral loss, because its second-order
    paths measure how the spectrum MOVES rather than where it sits. Half the audible
    identity of this patch is exactly that: supersaw beating, chorus rate, filter ADSR.
    A per-frame magnitude comparison sees those only indirectly, through frames that
    happen to differ.

    Construction, staying close to Anden and Mallat with cheap substitutions:

    - First order, |x * psi_lambda1|, approximated by a mel filterbank on a short STFT.
      That approximation is standard, and it buys the log-frequency geometry that a
      linear-frequency MSS lacks.
    - Second order, wavelets applied jointly along time and along log-frequency, then
      modulus. The temporal filters are analytic, so applying the frequential filter
      before the modulus separates rising from falling spectral slopes.
    - Local averaging at `seg` frames, about 190 ms, with the time axis kept rather than
      collapsed. A fully time-averaged scattering would score the 1 s delayed control as
      near-identical, which is precisely the degeneracy the corpus's controls exist to
      catch.

    Distances are taken on log(S + mu) with mu set per path from that path's own mean.
    This is the adaptive-epsilon convention: it makes the floor track the scale of the
    coefficient it protects, instead of being a constant that happens to sit at -120 dB
    on this material and somewhere else on the next.
    """
    n_fft, hop = 1024, 512
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_bands, fmin=30.0)

    # Modulation rates in cycles per frame; at 86 frames/s this spans about 1.7 to 19 Hz,
    # which is where tremolo, chorus and note-rate structure live.
    rates = (0.02, 0.045, 0.10, 0.22)
    fslopes = (0.0, 0.09, -0.09)

    def paths(x: np.ndarray) -> list[np.ndarray]:
        U1 = (fb @ mags(x, n_fft, hop)).astype(np.float32)
        nt = U1.shape[1]
        npad = int(2 ** np.ceil(np.log2(nt)))
        F = U1.shape[0]

        def avg(S: np.ndarray) -> np.ndarray:
            S = scipy.ndimage.uniform_filter1d(S[:, :nt], seg, axis=1, mode="nearest")
            return S[:, seg // 2::seg // 2]

        out = [avg(U1)]
        Xt = scipy.fft.rfft(U1.astype(np.float64), n=npad, axis=1)
        nr = Xt.shape[1]
        for xi in rates:
            band = Xt * _morlet_time(nr, xi * npad, 0.8)[None, :]
            # Analytic reconstruction: an rfft spectrum inverted as a full complex
            # spectrum keeps only positive frequencies, which is what we want.
            A = scipy.fft.ifft(band, n=npad, axis=1)[:, :nt].astype(np.complex64)
            Af = scipy.fft.fft(A, axis=0)
            for xf in fslopes:
                sig = 0.05 if xf == 0.0 else 0.045
                Y = Af * _gauss_freq(F, xf, sig).astype(np.complex64)[:, None]
                out.append(avg(np.abs(scipy.fft.ifft(Y, axis=0))))
        return out

    ref = paths(target)
    mus = [1e-3 * float(S.mean()) + 1e-20 for S in ref]
    logs = [np.log(S + mu) for S, mu in zip(ref, mus)]

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        tot = 0.0
        for S, mu, L in zip(paths(p), mus, logs):
            n = min(S.shape[1], L.shape[1])
            tot += float(np.abs(np.log(S[:, :n] + mu) - L[:, :n]).mean())
        return _finite(tot / len(logs))
    return score


# ---------------------------------------------------------------- 3. cepstral distance


@register("mfcc_match")
def mfcc_match(target: np.ndarray, sr: int = SR, n_mels: int = 128, n_mfcc: int = 40,
               floor_db: float = -70.0) -> Callable[[np.ndarray], float]:
    """Truncated MFCC distance, the standard fitness in synth sound matching.

    Yee-King et al. and the genetic-programming VST work before it score a candidate
    patch by frame-wise MFCC error, and that convention survives because the DCT
    truncation throws away exactly the part of the log-mel spectrum that a synthesiser
    cannot control: the fine structure between partials. Here that is not a nicety, it
    is the failing term. Because the DCT is orthonormal, keeping all 128 coefficients
    would reproduce a log-mel L2 exactly; the truncation IS the loss function.

    One deliberate deviation from the published recipe. Sound matching usually keeps 13
    coefficients. Twenty-six of the parameters in this patch are third-octave EQ gains,
    so a cepstral cutoff coarser than that grid makes half the parameter vector
    invisible. 40 coefficients over 128 mel bands resolves structure about 3 bands wide,
    finer than the EQ's 0.5-octave bandwidth, which is the condition for the loss to see
    what the patch can change. This is the trade the incumbent gets wrong from the other
    side: it resolves far finer than anything the controls can move.

    The floor is relative to the target's own peak, and at -70 dB rather than the usual
    -80: the perturbation that breaks the incumbent lives at -85 dB, so anything at or
    below that has to be clamped flat, not merely compressed.
    """
    n_fft, hop = 2048, 512
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    ref = float((fb @ _power(target, n_fft, hop)).max()) + 1e-30
    lo = ref * 10.0 ** (floor_db / 10.0)

    def _cep(x: np.ndarray) -> np.ndarray:
        M = fb @ _power(x, n_fft, hop)
        L = 10.0 * np.log10(np.maximum(M, lo))
        return scipy.fft.dct(L, type=2, axis=0, norm="ortho")[:n_mfcc]

    T = _cep(target)

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        P = _cep(p)
        n = min(P.shape[1], T.shape[1])
        d = P[:, :n] - T[:, :n]
        # L2 per frame then mean over frames, as the sound-matching papers report it:
        # a frame that is wrong in one coefficient is cheaper than one wrong in forty.
        return _finite(np.sqrt((d ** 2).sum(axis=0)).mean())
    return score


# ------------------------------------------------------------- 4. divergence ablation


@register("beta_kl")
def beta_kl(target: np.ndarray, sr: int = SR) -> Callable[[np.ndarray], float]:
    """The incumbent's own STFTs with only the divergence swapped, to generalised KL.

    The beta-divergence family indexes exactly the choice the incumbent got wrong.
    beta = 2 is Euclidean on power and weights a bin by its absolute error, so only the
    loudest partials count. beta = 0 is Itakura-Saito, scale-invariant per bin, which
    weights every bin by its RELATIVE error; an L1 on log magnitudes behaves the same
    way, and that is why a bin at -110 dB can outvote a partial. beta = 1, generalised
    Kullback-Leibler, sits between them: the cost of a bin is proportional to the
    target's energy there, so an empty bin cannot buy influence no matter how wrong it
    is, and a bin at -85 dB contributes at -85 dB.

    Nothing else changes. Same four resolutions, same hops, same normalisation by the
    target's total energy. If this alone recovers patches, the finding is that the
    representation was never the problem and the project needs a one-line change; if it
    does not, that is the cleanest possible evidence that the fix has to come from the
    representation, and the other three candidates are where to look. Either way it is
    the ablation the bake-off would otherwise be missing.

    Symmetrised, because the one-sided NMF form is blind in the direction that matters
    for search: with only d(target || pred), a render that invents energy where the
    target has none is charged almost nothing.
    """
    ffts = (512, 1024, 2048, 4096)
    hops = (128, 256, 512, 1024)
    eps = np.float32(1e-20)
    ref = []
    for f, h in zip(ffts, hops):
        t = _power(target, f, h) + eps
        ref.append((f, h, t, float(t.sum())))     # the normaliser never depends on pred

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        tot = 0.0
        for f, h, t, tsum in ref:
            q = _power(p, f, h) + eps
            n = min(q.shape[1], t.shape[1])
            tt, qq = t[:, :n], q[:, :n]
            # KL(t||q) + KL(q||t) collapses to (t - q) log(t/q). Worth the algebra: the
            # naive form is six passes over a 257 x 6169 array at the finest resolution.
            d = (tt - qq) * np.log(tt / qq)
            tot += float(d.sum()) / (tsum if n == t.shape[1] else float(tt.sum()))
        return _finite(tot / len(ref))
    return score
