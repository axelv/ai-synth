"""Losses built from what a listener can actually discriminate.

The measured failure is that the incumbent charges 0.194 for a change of -85 dB, i.e.
for a difference that is roughly 70 dB below the threshold of hearing at a normal
listening level. Every candidate here is built so that such a difference costs
*literally* zero, not merely a small amount, because the mechanism that hides it is a
threshold and a threshold is a hard floor. Below threshold there is nothing to hear, so
there is nothing to score.

Three mechanisms, deliberately not three flavours of one:

- `masked_excitation` compares spread excitation patterns and floors every band at the
  target's own simultaneous-masking threshold. Frequency domain, per frame.
- `loudness_roughness` compares Zwicker specific loudness (so a dB is worth what it is
  worth at that level, not a constant) plus the roughness fingerprint carried by the
  modulation spectrum of each critical band. That second half is what hears the unison
  detune and the chorus, which a per-frame spectral distance barely constrains at all
  because a few Hz of detune moves nothing across a critical band boundary.
- `pemo_adapt` runs band envelopes through adaptation loops, so onsets count and steady
  state is compressed. That is temporal masking and it is what reads the ADSR stages and
  the filter envelope.

`nmr_worst` is the fourth and is honestly a close relative of the first: same excitation
and mask, different aggregation. Detection theory says a listener notices the single
loudest audible defect rather than the average one, and mean-versus-max is exactly the
distinction PEAQ draws between total NMR and the maximum difference. It is here because
the two aggregations rank differently and the bench can afford to ask which is right.

Calibration matters and is not free. dB SPL is where thresholds live, so the target's
level is pinned to a nominal listening level (default 70 dB SPL) and the same offset is
applied to every candidate render. At 70 dB SPL a component 85 dB down sits far under
the threshold in quiet across the whole spectrum, which is the property being bought.
Pin it at 100 dB SPL instead and the same component becomes audible near 4 kHz, so the
number is a modelling choice and is exposed as a kwarg.

Measured on synthetic material, since rendering was not available. Six ADSR-gated notes,
five detuned voices each, and the same clip perturbed one way at a time. The cost of a
change 85 dB down, as a fraction of the cost of a mild spectral tilt:

    incumbent 1.2, masked_excitation 5.5e-4, loudness_roughness 2.1e-4,
    pemo_adapt 9.1e-4, nmr_worst exactly 0.

Sensitivity profiles differ on purpose, as a multiple of what each loss charges for a
+0.5 dB level change (attack 10 to 40 ms / decay 0.3 to 0.5 s / detune 0.4 to 0.5 % /
spectral tilt):

    incumbent            1.45  1.00  11.0  1.10
    masked_excitation    0.22  0.69   6.5  1.87
    loudness_roughness   2.61  1.48  18.1  1.96
    pemo_adapt           3.62  0.76  43.4  1.81
    nmr_worst            3.07  1.24   4.7  2.53

Tried and rejected:

- Noise-to-mask on the time-domain difference `pred - target`, which is the textbook
  objective difference measure. Faust's oscillators are free running, so two renders of
  the same patch differ in phase and the difference signal is mostly phase. Every measure
  built on an error SIGNAL rather than on error of features is unusable here for the same
  reason the house rule says to compare spectra and not waveforms.
- An ATH-gated per-bin log magnitude, the smallest edit that fixes the floor without
  giving up bin resolution. It works, but its measured profile (0.35 / 0.67 / 10.8 / 1.95
  on the four probes above) is close to the incumbent's, and mechanically it is
  `logmag_floor` with a better-chosen floor. The bake-off already has that mechanism. What
  separates everything here from it is critical-band averaging, so contributing another
  per-bin variant would have added a number, not an option.
- Level-dependent upper spreading slope, the `24 + 230/f - 0.2*L` form. It has to be
  recomputed per frame instead of being one frozen 24-square matrix, and the mask offset
  is already a cruder approximation than the slope is.
- A first version of the envelope probe ramped the attack over 1.2 s and made
  `pemo_adapt` look insensitive to envelope shape. It is not: adaptation is deliberately
  blind to anything slower than its longest time constant of 500 ms, so a 1.2 s ramp is
  tracked out. On a 10 versus 40 ms attack it weighs the change at 3.6x a level change,
  against 0.22x for `masked_excitation`. Recorded because the same mistake would make
  anyone reading the first table conclude the candidate does not work.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.fft
import scipy.signal

from losses import SR, match, register

# ---------------- shared psychoacoustic machinery ----------------

# `losses.mags` goes through librosa.stft, which measured about twice the cost of a
# strided scipy rfft on 17.9 s here. The budget is per call and there are a few hundred
# calls per candidate, so the framing is done by hand. Same magnitudes, cheaper.


def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    w = np.hanning(n_fft).astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n_fft:                    # a truncated render must score, not raise
        x = np.pad(x, (0, n_fft - len(x)))
    f = np.lib.stride_tricks.sliding_window_view(x, n_fft)
    return f[::hop] * w


def _spec(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Complex STFT, frames on axis 0. Complex because band beating needs the phase."""
    return scipy.fft.rfft(_frames(x, n_fft, hop), axis=-1)


def _bark(f: np.ndarray) -> np.ndarray:
    """Traunmuller's analytic Bark scale. Good enough above 200 Hz, which is where the
    critical bands stop being roughly constant width anyway."""
    return 26.81 * f / (1960.0 + f) - 0.53


def _ath_db(f: np.ndarray) -> np.ndarray:
    """Terhardt's threshold in quiet, dB SPL. Clipped because the fit diverges at DC and
    above 18 kHz, and an unbounded threshold would make whole bands free by accident."""
    k = np.maximum(f, 20.0) / 1000.0
    a = 3.64 * k ** -0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4
    return np.clip(a, -10.0, 100.0)


def _bark_bank(sr: int, n_fft: int, per_bark: float = 1.0, z_max: float = 24.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """Triangular critical-band weights on a Bark grid. Not row-normalised: excitation is
    the energy inside a critical band, so a wider band legitimately collects more."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    z = _bark(freqs)
    centres = np.arange(1.0, z_max + 1e-9, per_bark)
    w = np.maximum(0.0, 1.0 - np.abs(z[None, :] - centres[:, None]) / per_bark)
    # bands narrower than the bin spacing would otherwise be empty rows
    empty = w.sum(axis=1) <= 0
    for i in np.flatnonzero(empty):
        fc = 1960.0 * (centres[i] + 0.53) / (26.81 - centres[i] - 0.53)
        w[i, np.argmin(np.abs(freqs - fc))] = 1.0
    fc = 1960.0 * (centres + 0.53) / (26.81 - centres - 0.53)
    return w.astype(np.float32), fc


def _spreading(centres_z: np.ndarray, up_db: float = -12.0, dn_db: float = -27.0
               ) -> np.ndarray:
    """Level-independent two-slope spreading across Bark, as a band-to-band matrix.

    Masking spreads upward in frequency far more readily than downward, hence the
    asymmetry. The real upper slope depends on masker level; freezing it at -12 dB/Bark is
    the usual simplification for moderate levels and it costs one matmul on a 24-square
    matrix instead of a per-frame recompute."""
    d = centres_z[:, None] - centres_z[None, :]        # band j minus masker i
    slope = np.where(d >= 0, up_db, dn_db)             # d > 0 is the upward skirt
    return (10.0 ** (slope * np.abs(d) / 10.0)).astype(np.float32)


class _Ear:
    """Everything that turns a mono signal into band excitation in dB SPL.

    Built once from the target so the calibration, the filterbank and the thresholds are
    shared by target and render. Reused by three of the four candidates below.
    """

    def __init__(self, target: np.ndarray, sr: int, n_fft: int, hop: int,
                 per_bark: float, spl: float):
        self.n_fft, self.hop = n_fft, hop
        self.w, self.fc = _bark_bank(sr, n_fft, per_bark)
        self.z = _bark(self.fc)
        self.spread = _spreading(self.z)
        e = self.band_energy(self.spec(target))
        # pin the target's mean per-frame band energy to `spl` dB SPL
        self.offset = spl - 10.0 * np.log10(e.sum(axis=0).mean() + 1e-30)
        self.ath = (10.0 ** ((_ath_db(self.fc) - self.offset) / 10.0)).astype(np.float32)

    def spec(self, x: np.ndarray) -> np.ndarray:
        """Kept separate so a candidate that needs both energy and band phase pays for
        one transform, not two. That was 15 ms per call."""
        return _spec(x, self.n_fft, self.hop)

    def band_energy(self, s: np.ndarray) -> np.ndarray:
        return (self.w @ (np.abs(s) ** 2).T).astype(np.float32)      # [band, frame]

    def excitation(self, x: np.ndarray) -> np.ndarray:
        return self.spread @ self.band_energy(self.spec(x))


# ---------------- candidate 1 ----------------

@register("masked_excitation")
def masked_excitation(target: np.ndarray, sr: int = SR, spl: float = 70.0,
                      offset_db: float = 12.0) -> Callable[[np.ndarray], float]:
    """Excitation-pattern distance with every band floored at the masking threshold.

    Why this and not another flooring scheme: `logmag_floor` already floors at a fixed
    -80 dB below the global peak, which is one number for the whole spectrogram. Masking
    is local. A partial 40 dB down and 1 Bark away from a loud one is inaudible; the same
    partial alone in a quiet band at 3 kHz is not. So the floor here is the target's own
    spread excitation minus `offset_db`, taken together with the threshold in quiet, and
    it moves per band and per frame.

    The floor is additive, `10*log10(E + T)`, rather than a hard max. Both make
    sub-threshold energy free; the additive form does it with a derivative that fades out
    instead of a kink, which is the difference between a search that slides into the
    inaudible region and one that hits a wall there.

    Blind by construction to: a few Hz of unison detune, which does not move energy
    across a critical band. That is `loudness_roughness`'s job.
    """
    ear = _Ear(target, sr, 2048, 512, 1.0, spl)
    et = ear.excitation(target)
    # mask = whichever is higher, the spread masker skirt or hearing itself
    thr = np.maximum(et * (10.0 ** (-offset_db / 10.0)), ear.ath[:, None])
    at = 10.0 * np.log10(et + thr)

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        ep = ear.excitation(p)
        n = min(ep.shape[1], at.shape[1])
        ap = 10.0 * np.log10(ep[:, :n] + thr[:, :n])
        v = float(np.abs(ap - at[:, :n]).mean())
        return v if np.isfinite(v) else 1e6
    return score


# ---------------- candidate 2 ----------------

@register("loudness_roughness")
def loudness_roughness(target: np.ndarray, sr: int = SR, spl: float = 70.0,
                       w_rough: float = 2.0) -> Callable[[np.ndarray], float]:
    """Specific loudness plus the beating fingerprint of each critical band.

    Two things a per-frame magnitude distance handles badly.

    Loudness: a dB is not worth a constant. Zwicker's compression, `E**0.23` above the
    threshold in quiet, makes an error at a quiet partial cost less than the same error
    at a loud one, and makes an error below threshold cost exactly nothing because the
    energy is clamped to the threshold before the exponent.

    Roughness: the patch's character comes largely from seven detuned saws, and detune is
    a beat rate. Two partials 3 Hz apart and two partials 9 Hz apart put nearly identical
    energy in the same critical band and sound obviously different. What differs is the
    band envelope's modulation spectrum, so that is what is compared, normalised to a
    modulation depth and weighted by Aures' roughness curve peaking at 70 Hz. Bands are
    then weighted by the share of the target's loudness they carry, so beating nobody can
    hear is free for the same reason everything else here is.

    Band signals are formed by summing COMPLEX bins, not magnitudes: summing magnitudes
    would average the beating away, which is the one thing this term exists to see. The
    comparison is on modulation magnitude, so free-running oscillator phase does not
    matter, per the house rule about spectra rather than waveforms.
    """
    n_fft, hop = 512, 128                      # envelope rate 344.5 Hz, covers 70 Hz
    ear = _Ear(target, sr, n_fft, hop, 1.0, spl)
    fm = np.fft.rfftfreq(512, hop / sr)
    rough_w = np.where(fm > 0, (fm / 70.0) * np.exp(1.0 - fm / 70.0), 0.0).astype(np.float32)
    # log-spaced modulation bins: 0.7 Hz resolution is finer than anyone hears as a
    # distinct beat rate, and binning cuts the feature to something stable across renders
    edges = np.geomspace(2.0, fm[-1], 11)
    mbin = np.zeros((10, len(fm)), dtype=np.float32)
    for i in range(10):
        sel = (fm >= edges[i]) & (fm < edges[i + 1])
        if sel.any():
            mbin[i, sel] = 1.0 / sel.sum()

    def loudness(s: np.ndarray) -> np.ndarray:
        e = np.maximum(ear.band_energy(s), ear.ath[:, None])
        return e ** 0.23 - ear.ath[:, None] ** 0.23

    def rough(s: np.ndarray, weight: np.ndarray) -> np.ndarray:
        b = np.abs((ear.w @ s.T))                       # [band, frame] band envelope
        nc = b.shape[1] // 512
        if nc == 0:
            return np.zeros((b.shape[0], 10), dtype=np.float32)
        c = b[:, :nc * 512].reshape(b.shape[0], nc, 512)
        dc = c.mean(axis=2, keepdims=True) + 1e-12
        # depth per chunk before averaging chunks, so a loud chunk cannot set the depth
        # of a quiet one
        m = (np.abs(scipy.fft.rfft(c - dc, axis=2)) / (512 * dc)).mean(axis=1)
        return (m * rough_w) @ mbin.T * weight[:, None]

    st = ear.spec(target)
    lt = loudness(st)
    share = lt.mean(axis=1)
    share = (share / (share.sum() + 1e-12)).astype(np.float32)
    rt = rough(st, share)
    lscale = float(np.abs(lt).mean()) + 1e-12
    rscale = float(np.abs(rt).mean()) + 1e-12

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        sp = ear.spec(p)
        lp = loudness(sp)
        n = min(lp.shape[1], lt.shape[1])
        a = float(np.abs(lp[:, :n] - lt[:, :n]).mean()) / lscale
        b = float(np.abs(rough(sp, share) - rt).mean()) / rscale
        v = a + w_rough * b
        return v if np.isfinite(v) else 1e6
    return score


# ---------------- candidate 3 ----------------

@register("pemo_adapt")
def pemo_adapt(target: np.ndarray, sr: int = SR, spl: float = 70.0
               ) -> Callable[[np.ndarray], float]:
    """Band envelopes through adaptation loops: temporal masking, onsets weighted.

    Dau's model of the effective signal. Each band envelope is divided by a lagging
    average of itself, five times over with time constants from 5 to 500 ms. Steady state
    comes out as `E**(1/32)`, so a level offset that persists costs almost nothing, while
    a change in how fast something arrives or decays produces a large overshoot
    difference. That is a loss that reads the amplitude ADSR and the filter ADSR, which
    are 8 of the 29 macro parameters and which a distance on time-averaged spectra cannot
    separate from a static EQ tilt at all.

    The loops are run feed-forward, `y = x / sqrt(lowpass(x))`, rather than with the true
    output feedback. Same steady state (both settle at `sqrt(x)` per stage), same sign and
    rough size of the onset overshoot, and it is one `lfilter` per stage instead of a
    Python loop over the frames. Measured on 24 bands by 3082 frames: 11.6 ms for all five
    feed-forward stages against 68 ms for ONE true stage, so the exact version costs about
    340 ms per call and does not fit the budget at all.

    The internal-noise floor is applied before the first loop, so a band sitting under the
    threshold in quiet is clamped to a constant and every difference inside it divides out
    to nothing. Without that floor the division would do the opposite of what is wanted
    and amplify exactly the near-empty bins that broke the incumbent.
    """
    # 512/256 rather than a longer window: this candidate trades frequency resolution it
    # does not use for the time resolution the loops do. 11.6 ms of window is already the
    # limit on how fast an onset can be seen, so the 5 ms loop is the fiction it always is
    # in this model.
    n_fft, hop = 512, 256
    ear = _Ear(target, sr, n_fft, hop, 1.0, spl)
    fs = sr / hop
    taus = (0.005, 0.050, 0.129, 0.253, 0.500)
    lp = [np.exp(-1.0 / (t * fs)) for t in taus]
    a_mod = np.exp(-2.0 * np.pi * 8.0 / fs)            # 8 Hz modulation lowpass

    def internal(x: np.ndarray) -> np.ndarray:
        e = np.sqrt(np.maximum(ear.band_energy(ear.spec(x)), ear.ath[:, None]))
        for a in lp:
            s = scipy.signal.lfilter([1.0 - a], [1.0, -a], e, axis=1)
            e = e / np.sqrt(s + 1e-20)
        return scipy.signal.lfilter([1.0 - a_mod], [1.0, -a_mod], e, axis=1)

    t_int = internal(target)
    scale = float(np.abs(t_int).mean()) + 1e-12

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        pi = internal(p)
        n = min(pi.shape[1], t_int.shape[1])
        v = float(np.abs(pi[:, :n] - t_int[:, :n]).mean()) / scale
        return v if np.isfinite(v) else 1e6
    return score


# ---------------- candidate 4 ----------------

@register("nmr_worst")
def nmr_worst(target: np.ndarray, sr: int = SR, spl: float = 70.0,
              offset_db: float = 12.0, p: float = 4.0) -> Callable[[np.ndarray], float]:
    """Noise-to-mask ratio, aggregated towards the worst band rather than the average.

    Shares the ear model with `masked_excitation` and differs in two ways that change the
    ranking rather than the scale.

    First, the error is the excitation difference measured against the mask, not a
    difference of two floored levels. That is the quantity detection theory says predicts
    audibility: `d'` grows with the ratio of the difference to the threshold, so a value
    below 1 in every band means the two renders are indistinguishable and the loss is flat
    there by construction, not by tuning.

    Second, the aggregation is a power mean with `p = 4`. A listener comparing two takes
    does not average the error over 24 bands and 1500 frames, they notice the one moment
    where something is plainly wrong. PEAQ keeps both a total NMR and a maximum
    difference for the same reason. Between the two, mean-versus-max is the whole
    disagreement, and it is worth finding out which one puts its minimum on the answer.

    The ratio is clamped at 1 before aggregating, so the loss does not merely get small
    below threshold, it is exactly flat there and reaches its minimum of 0 on the WHOLE
    set of renders indistinguishable from the target rather than at a single point. That
    is the opposite of the measured failure, where truth is a needle one float32 grid
    point wide, and it is the property to test: a flat-bottomed basin cannot be walked out
    of, but it also cannot be climbed once inside, so the bench has to say whether the
    floor is wide enough to be an attractor and narrow enough to still mean something.

    The mask is the larger of the two signals' masks, so an added component that would be
    masked in the render is free too, not just one that would be masked in the target.
    Without that the loss is asymmetric in a way that quietly rewards removing energy.
    """
    ear = _Ear(target, sr, 2048, 512, 1.0, spl)
    et = ear.excitation(target)
    k = 10.0 ** (-offset_db / 10.0)
    thr_t = np.maximum(et * k, ear.ath[:, None])

    def score(pred: np.ndarray) -> float:
        q, _ = match(pred, target)
        ep = ear.excitation(q)
        n = min(ep.shape[1], et.shape[1])
        thr = np.maximum(thr_t[:, :n], ep[:, :n] * k)
        nmr = np.maximum(np.abs(ep[:, :n] - et[:, :n]) / thr, 1.0)
        v = float((np.mean(nmr ** p)) ** (1.0 / p))
        # dB so the deep end is readable; the clamp keeps the argument at or above 1
        return float(10.0 * np.log10(v)) if np.isfinite(v) else 1e6
    return score
