"""Candidate losses that compare STATISTICS, never time-frequency bins pointwise.

The measured failure is specific: the incumbent's log-magnitude term reads near-empty
bins between partials, where a 1e-8 parameter step changes the content by O(1) in
relative terms while changing the audio by -85 dB. Every fix in `losses.py` attacks that
by making the per-bin comparison gentler (floor it, compress it, average it into bands).
This file attacks it by not comparing bins at all.

A pad is a texture, not a waveform. What a listener has access to after a second of a
sustained pad is a set of running averages: how much energy sits in each critical band,
how deeply and how fast each band fluctuates, which bands fluctuate together, how the
spectrum's centre of mass moves. Those are the quantities the 55 controls set, and they
are all averages over many frames and many bins. An average over N near-empty bins of a
quantity that is individually O(1) wrong and collectively -85 dB is -85 dB wrong. The
fine structure the incumbent chokes on is exactly what a statistic integrates away, and
it is integrated away for free rather than by a threshold that has to be tuned.

Four mechanisms, deliberately not variations on one idea:

  texture_stats   McDermott-Simoncelli sound texture: subband envelope moments,
                  cross-band envelope correlations, modulation-band power.
  cepstral_gauss  Model each block's MFCC+delta frames as a multivariate Gaussian and
                  compare the Gaussians in closed form. Distribution against
                  distribution; no frame ever meets its counterpart.
  band_quantiles  Histograms. Match the quantile function of each band's level, and of
                  its frame-to-frame change. Asks how often and how fast, not when.
  spectral_ot     Optimal transport. Treat the spectrogram as mass on the log-frequency
                  and time axes and measure how far the mass has to move. A -85 dB bin
                  carries -85 dB of mass, so it can only ever move -85 dB of it.

Measured on a 17.9 s synthetic pad (sines only, so the bins between partials are
genuinely empty, which is where the incumbent detonates). The screen is the ratio of a
barely-audible real change, a 0.1 dB/octave spectral tilt, to the -85 dB pedestal that
one float32 parameter step produces. Above 1 the loss can see a real change through the
noise; below 1 it cannot, and the search is optimising the noise.

    incumbent        0.2        band26_env      0.1        pow03           0.03
    logmag_floor    12.9        mel_l1         31.3
    texture_stats   27.9        band_quantiles 30.8
    cepstral_gauss  4.8e3       spectral_ot    4.2e3

Every candidate here also puts its argmin exactly at the truth and is monotone on both
sides of it, checked by sweeping a spectral parameter (tilt) and a temporal one (decay
rate) with the -85 dB pedestal added to every point so the comparison is like for like.
On the decay sweep, moving 0.03 dB changes the incumbent's reading by 1% of what the
pedestal already contributes and changes spectral_ot's by 2e5%.

Tried and rejected:

- Per-band relative floors in `band_quantiles`. Flooring each band against its OWN peak
  gives a band that never rises above -100 dB a full 70 dB of dynamic range to fill with
  numerical noise, which reintroduces the incumbent's exact failure one band at a time.
  The floor has to be global.
- Purely order-free scoring (band_quantiles and cepstral_gauss over the whole clip).
  Both ranked the 1 s-delayed control as milder than a plainly audible tilt, because a
  bag of frames has no timeline; `losscorpus` carries `delayed` and `shuffled` precisely
  to catch that. Both now take their statistics over eight 2.2 s blocks, which restores
  the coarse timeline without restoring frame alignment.
- Scale-free statistics in inaudible bands. `texture_stats` first weighted every band's
  envelope shape equally, and since those statistics are scale-free by construction they
  measured pure numerical noise in the empty top bands: the pedestal scored 0.57 against
  0.18 for a real change, i.e. the incumbent's bug rebuilt in a new domain. Fixed by
  weighting shape statistics out below -70 dB relative to the loudest band.
- Wasserstein between the two clips' frame distributions (a real N x N transport
  problem, Sinkhorn or sliced). Right idea, ~100x over budget at 769 frames a side, and
  `cepstral_gauss` gets most of it in closed form for microseconds.
- librosa.stft for the front end. Measured 40 ms at 1024/512 on 17.9 s against 7 ms for
  a strided rfft doing the same thing, and the whole budget is 50 ms. `mags()` is left
  alone for the baselines that can afford it.
- Scale-invariant statistics only (correlations, skew, normalised modulation). Overall
  gain is a fitted parameter, so every candidate keeps at least one term that reads
  absolute level, verified by scoring a +0.5 dB copy well above the pedestal.
"""

from __future__ import annotations

from typing import Callable

import librosa
import numpy as np
import scipy.fft

from losses import SR, match, register

_WINDOWS: dict[int, np.ndarray] = {}
_BANKS: dict[tuple, np.ndarray] = {}


def _power(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Power spectrogram, frames x bins, float32 in and out.

    Hand-rolled instead of `mags()` because it measured 5x faster on 17.9 s (7 ms against
    40 ms at 1024/512) and four candidates each pay for it inside a 50 ms budget. No
    centring: frame k is samples [k*hop, k*hop+n_fft), applied identically to both
    signals, so the two spectrograms stay frame-aligned.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    w = _WINDOWS.get(n_fft)
    if w is None:
        w = np.hanning(n_fft).astype(np.float32)
        _WINDOWS[n_fft] = w
    f = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * w
    S = scipy.fft.rfft(f, axis=-1)
    return S.real ** 2 + S.imag ** 2


def _bank(sr: int, n_fft: int, n_bands: int, fmin: float, fmax: float) -> np.ndarray:
    """Row-sum-normalised mel triangles: each band is a weighted MEAN of bin powers.

    Mean rather than sum so wide high bands and narrow low bands are directly comparable
    as levels, which is what a band-gain curve is.
    """
    key = (sr, n_fft, n_bands, fmin, fmax)
    fb = _BANKS.get(key)
    if fb is None:
        fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_bands,
                                 fmin=fmin, fmax=fmax, norm=None)
        fb = fb / (fb.sum(axis=1, keepdims=True) + 1e-12)
        fb = fb.astype(np.float32)
        _BANKS[key] = fb
    return fb


def _guard(v: float) -> float:
    return float(v) if np.isfinite(v) else 1e6


# ---------------------------------------------------------------------------


@register("texture_stats")
def texture_stats(target: np.ndarray, sr: int = SR) -> Callable[[np.ndarray], float]:
    """McDermott-Simoncelli sound texture statistics on 32 subband envelopes.

    The claim behind this representation is that a sustained texture is perceptually
    determined by a modest set of time-averaged statistics of cochlear subband envelopes,
    and that two signals sharing them sound the same however different their waveforms.
    That is the exact property wanted here: the loss must be blind to everything a
    listener is blind to, and only these statistics survive the averaging.

    Six classes, each reading different controls:
      level         mean amplitude per band. This is the EQ curve plus the filter's
                    steady state, so it reads most of the 55 parameters directly.
      cv/skew/kurt  shape of each band's envelope distribution. Attack sharpness and
                    sustain depth live here; a fast percussive envelope is high kurtosis,
                    a flat pad is low.
      correlation   which bands rise and fall together. A resonant lowpass sweeping
                    couples neighbouring bands and decouples distant ones, so this reads
                    cutoff, resonance and the filter ADSR in a way no static spectrum can.
      modulation    where each band's fluctuation energy sits in rate. Chorus rate,
                    vibrato, delay repeats and LFOs all land here as octave-band power.

    Each class is divided by the target's own scale for that class so the six can be
    summed without a tuned weight. Level gets 2x because 26 of the 55 parameters are
    band gains and it is the only class that sees them cleanly.

    The five shape classes are weighted by band audibility and the level class is not,
    and that asymmetry is the whole fix for the measured bug. Every shape statistic is
    scale-free by construction, so in a band holding nothing it is a perfect measurement
    of numerical noise: the first version of this loss scored the -85 dB pedestal at 0.57
    against 0.18 for a real 0.5 dB/octave tilt, reproducing the incumbent's failure
    exactly. Weighting the shape classes out below -70 dB relative to the loudest band
    took the pedestal to 0.00036 against 0.010 for a 0.1 dB/octave tilt. The level class
    stays unweighted, so a patch still cannot hide energy in a band the target leaves
    empty; it is protected by its floor instead, which is a different mechanism because
    it is a one-sided clamp rather than a discount.
    """
    n_fft, hop, n_bands = 1024, 256, 32
    fb = _bank(sr, n_fft, n_bands, 40.0, 16000.0)
    fps = sr / hop

    def _envelopes(x: np.ndarray) -> np.ndarray:
        return np.sqrt(fb @ _power(x, n_fft, hop).T)          # bands x frames

    At = _envelopes(target)
    n_t = At.shape[1]
    # -70 dB below the LOUDEST band, not below the mean band. Against the mean the floor
    # landed at about -100 dB relative to the loudest band, which is under the render's
    # own noise, so the empty top bands were unclamped and the -85 dB pedestal moved them
    # 4 dB each. That single choice was 96% of the failing score.
    floor = float(At.mean(axis=1).max()) * 10.0 ** -3.5 + 1e-30

    # Octave modulation bands from 0.5 Hz up. Below 0.5 Hz on an 18 s clip there are
    # fewer than 10 cycles, so the estimate is the clip's own envelope, not a rate.
    mfreq = np.fft.rfftfreq(n_t, 1.0 / fps)
    mod_edges = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, fps / 2.0])
    mod_mat = np.zeros((len(mfreq), len(mod_edges) - 1), dtype=np.float32)
    for i in range(len(mod_edges) - 1):
        mod_mat[(mfreq >= mod_edges[i]) & (mfreq < mod_edges[i + 1]), i] = 1.0
    n_mod = mod_mat.shape[1]

    def _stats(A: np.ndarray) -> tuple[np.ndarray, ...]:
        mu = A.mean(axis=1, keepdims=True)
        lvl = np.log(np.maximum(mu[:, 0], floor))
        a = A / (mu + floor)                                  # unit mean, scale free
        cv = a.std(axis=1)
        d = a - a.mean(axis=1, keepdims=True)
        # Clipped to the range real envelopes occupy. A band that is exactly silent in
        # the render has cv = 0 and an unclipped skew of -1e27, which turned the score
        # for a silent candidate into 7.5e8 and would have swamped any screen that
        # averages or ratios scores. Real pad envelopes sit inside these bounds, so the
        # clip is inactive anywhere the loss is being asked a real question.
        sk = np.clip((d ** 3).mean(axis=1) / (cv ** 3 + 1e-6), -20.0, 20.0)
        ku = np.clip((d ** 4).mean(axis=1) / (cv ** 4 + 1e-6), 0.0, 100.0)
        # 0.3 compression before the correlation and modulation stats, as in the original:
        # it is roughly cochlear and it stops one loud attack from setting every
        # correlation in the clip.
        c = a ** 0.3
        c = c - c.mean(axis=1, keepdims=True)
        c = c / (np.sqrt((c ** 2).mean(axis=1, keepdims=True)) + 1e-9)
        corr = (c @ c.T) / c.shape[1]
        M = np.abs(scipy.fft.rfft(c, axis=-1)) ** 2
        n = min(M.shape[1], mod_mat.shape[0])
        mod = M[:, :n] @ mod_mat[:n]
        mod = mod / (mod.sum(axis=1, keepdims=True) + 1e-12)  # fraction of fluctuation
        return lvl, cv, sk, ku, corr, mod

    iu = np.triu_indices(n_bands, 1)
    Lt, Vt, Kt, Ut, Ct, Mt = _stats(At)
    Ct = Ct[iu]

    # Audibility of each band in the target: full weight down to 55 dB below the loudest
    # band, ramping to zero at 70 dB. Below that the band's envelope shape is a
    # measurement of the render's numerical noise and nothing else. Measured from the
    # UNFLOORED level, and the ramp reaches zero exactly at the floor, so a band the
    # level term has clamped contributes no shape statistic at all rather than the 0.33
    # weight the floored level would have implied.
    lvl_db = 20.0 * np.log10(At.mean(axis=1) + 1e-30)
    w = np.clip((lvl_db - (lvl_db.max() - 70.0)) / 15.0, 0.0, 1.0)
    w = w / (w.sum() + 1e-12)
    wc = (w[:, None] * w[None, :])[iu]
    wc = wc / (wc.sum() + 1e-12)

    # Scale of each class in the target, floored so a degenerate target cannot make a
    # denominator vanish and turn a small difference into a huge score.
    s_lvl = max(float(Lt.std()), 0.5)
    s_cv = float(w @ np.abs(Vt)) + 0.05
    s_sk = float(w @ np.abs(Kt)) + 1.0
    s_ku = float(w @ np.abs(Ut)) + 3.0

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        L, V, K, U, C, M = _stats(_envelopes(p))
        v = (2.0 * np.abs(L - Lt).mean() / s_lvl
             + float(w @ np.abs(V - Vt)) / s_cv
             + float(w @ np.abs(K - Kt)) / s_sk
             + float(w @ np.abs(U - Ut)) / s_ku
             + 4.0 * float(wc @ np.abs(C[iu] - Ct))
             + n_mod * float(w @ np.abs(M - Mt).mean(axis=1)))
        return _guard(v)

    return score


@register("cepstral_gauss")
def cepstral_gauss(target: np.ndarray, sr: int = SR, n_mels: int = 48, n_cep: int = 16,
                   n_blocks: int = 8) -> Callable[[np.ndarray], float]:
    """Model each 2.2 s block's MFCC+delta frames as a Gaussian, compare in closed form.

    The purest reading of "compare distributions, not frames": a block becomes a cloud of
    32-dimensional points and the score is the Jeffreys (symmetric KL) divergence between
    the two clouds' Gaussian fits, summed over blocks. No frame is ever compared to its
    counterpart, so frame-level fine structure has nowhere to enter; what survives is the
    mean timbre (the means), how the timbre varies and which bands covary (the
    covariance), and how fast it moves (the delta block).

    Two reasons this shape is right here rather than just mean and variance. First, the
    covariance is where a resonant filter shows up: cutoff and resonance do not move the
    average spectrum much, they move the direction along which it swings. Second, a
    Gaussian divergence is scale-free per dimension, so cepstral coefficients with wildly
    different variances contribute comparably without hand-set weights.

    Blocks, rather than one Gaussian for the clip, because a single Gaussian is a bag of
    frames with no timeline: it scored the 1 s-delayed control at 4.08 against 4.77 for a
    2 dB/octave tilt, ranking a whole second of drift as milder than a tilt. Blocking
    lifts the delayed control to 5.50 and the shuffled one to 11.3, both clear of every
    perturbation smaller than a 2 dB/octave tilt, which at 5.68 still edges past the
    delay. That last inversion is left standing rather than tuned away: a pad delayed one
    second is 94% the same audio, a plus or minus 9 dB tilt across the spectrum is a
    different instrument, and forcing the order would mean weakening the timbre term that
    is the point of the loss. Eight blocks is also about the most the covariance can
    carry: at 192 frames per block against 32 dimensions the estimate is still well
    conditioned, and finer blocks would turn this into frame matching by another route.

    The mel log is floored 75 dB below the loudest band. Below that a band contains
    nothing a listener can reach, and letting it into a log is precisely the incumbent's
    bug, one band wider.
    """
    n_fft, hop = 2048, 512
    fb = _bank(sr, n_fft, n_mels, 30.0, 18000.0)
    Mt = fb @ _power(target, n_fft, hop).T
    floor = float(Mt.max()) * 10.0 ** -7.5 + 1e-30
    block = max(Mt.shape[1] // n_blocks, 1)
    dim = 2 * n_cep

    def _feat(x: np.ndarray) -> np.ndarray:
        L = np.log(np.maximum(fb @ _power(x, n_fft, hop).T, floor))
        C = scipy.fft.dct(L, type=2, norm="ortho", axis=0)[:n_cep]
        D = np.diff(C, axis=1, prepend=C[:, :1])
        return np.concatenate([C, D], axis=0).T               # frames x 2*n_cep

    def _gauss(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = F.mean(axis=0)
        X = F - mu
        S = (X.T @ X) / max(len(F), 1)
        # Ridge at 5e-3 of the mean eigenvalue. Delta features are near-collinear with
        # their own cepstra, so the raw covariance is close to singular and its inverse
        # would amplify exactly the tiny differences this loss exists to ignore.
        return mu, S + (np.trace(S) / dim * 5e-3 + 1e-12) * np.eye(dim)

    def _blocks(x: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        F = _feat(x)
        nb = min(len(F) // block, n_blocks)
        return [_gauss(F[i * block:(i + 1) * block]) for i in range(nb)]

    ref = [(mu, S, np.linalg.inv(S)) for mu, S in _blocks(target)]

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        got = _blocks(p)
        if not got:
            return 1e6
        tot = 0.0
        for (mu_t, S_t, Si_t), (mu_p, S_p) in zip(ref, got):
            try:
                Si_p = np.linalg.inv(S_p)
            except np.linalg.LinAlgError:
                return 1e6
            d = mu_p - mu_t
            tot += 0.5 * (float(np.einsum("ij,ji->", Si_t, S_p))
                          + float(np.einsum("ij,ji->", Si_p, S_t))
                          + float(d @ (Si_t + Si_p) @ d)) - dim
        # log1p, not for the ranking (it is monotone, so ranks and argmins are identical)
        # but for the range. A raw Jeffreys divergence spans 2e-4 near truth to 9e5 for
        # white noise; a screen that reads a ratio of two of those is reading an artefact
        # of the exponential, and log1p is the identity where it matters.
        return _guard(np.log1p(max(tot, 0.0) / max(len(got), 1)))

    return score


@register("band_quantiles")
def band_quantiles(target: np.ndarray, sr: int = SR, n_bands: int = 26,
                   n_q: int = 24, n_blocks: int = 8) -> Callable[[np.ndarray], float]:
    """Match the histogram of each band's level, and of its frame-to-frame change.

    Everything else in the bake-off asks "was band b at the right level at frame t". This
    asks "how much of the clip did band b spend near level L", which is the same question
    a listener answers and throws away the alignment that makes the incumbent brittle.
    Comparing two histograms through their quantile functions is the 1-D Wasserstein
    distance, and it is a sorted subtraction, so the whole thing costs a sort.

    An ADSR is a statement about how long a band spends at each level, so the level
    quantiles read the amplitude and filter envelopes without any onset detection. The
    26 bands are on the EQ bank's own grid: if the controls live there, an objective that
    resolves nothing finer cannot be pushed around by structure the controls cannot move.

    The second term is the distribution of first differences, in dB per frame. Without it
    the loss is order-free and scores frame-shuffled audio at nearly zero, which is the
    degenerate winner `losscorpus.shuffled` exists to catch. With it, shuffling shows up
    as a distribution of jumps tens of dB wide against a pad's fraction of a dB, while
    the statistic stays a histogram and never regains alignment sensitivity.

    Histograms are taken per 2.2 s block rather than over the whole clip. Fully global
    histograms scored the 1 s-delayed control at 1.08 against 1.55 for a plainly audible
    2 dB/octave tilt, which is the wrong order: a time shift is invisible to a global
    histogram by construction. Eight blocks pin the coarse timeline and lift the delayed
    control to 2.20, clear of the tilt at 1.48, while costing only a little pedestal
    insensitivity (the tilt-to-pedestal ratio goes 38.8 to 30.8). A pad is stationary
    over a couple of seconds, so order-freedom within a block is all the robustness that
    was ever wanted.
    """
    n_fft, hop = 1024, 512
    fb = _bank(sr, n_fft, n_bands, 40.0, 16000.0)
    Bt = fb @ _power(target, n_fft, hop).T
    # One global floor, 70 dB below the loudest band-frame. Per-band floors were tried
    # and are wrong: they hand a band that is 100 dB down its own 70 dB of range to fill
    # with numerical noise, which is the incumbent's failure rebuilt band by band.
    floor = float(Bt.max()) * 10.0 ** -7.0 + 1e-30
    # Block length fixed by the target so pred blocks cover the same instants. A short
    # pred simply contributes fewer blocks rather than silently resampling the timeline.
    block = max(Bt.shape[1] // n_blocks, 1)
    qidx = np.linspace(0, block - 1, n_q).round().astype(int)

    def _quantiles(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        L = 10.0 * np.log10(np.maximum(fb @ _power(x, n_fft, hop).T, floor))
        nb = min(L.shape[1] // block, n_blocks)
        L = L[:, :nb * block]
        D = np.diff(L, axis=1, append=L[:, -1:])
        out = []
        for A in (L, D):
            S = np.sort(A.reshape(n_bands, nb, block), axis=2)
            out.append(S[:, :, qidx])
        return out[0], out[1], nb

    Qt, Dt, _ = _quantiles(target)

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        Q, D, nb = _quantiles(p)
        if nb < 1:                     # shorter than one block: nothing to compare
            return 1e6
        return _guard(np.abs(Q - Qt[:, :nb]).mean() + np.abs(D - Dt[:, :nb]).mean())

    return score


@register("spectral_ot")
def spectral_ot(target: np.ndarray, sr: int = SR,
                n_bins: int = 120) -> Callable[[np.ndarray], float]:
    """Optimal transport of spectral mass along the log-frequency line, plus level.

    Every other candidate compares how much energy is in a band. This one compares where
    the energy is, and charges by how far it has to move to agree, in octaves. That
    changes the failure mode completely. A bin sitting at -85 dB holds -85 dB of mass, so
    however wrong it is in relative terms it can only ever be charged -85 dB of transport:
    the insensitivity is structural, not a floor someone chose. It also makes the loss
    smooth in the parameters that shift a spectrum rather than scale it, which is most of
    them here: cutoff, detune, the tilt of the EQ curve. Under a per-bin distance, moving
    a partial off a bin and onto the next one is two full errors and a flat region in
    between; under transport it is a straight ramp of the width of the move.

    Mass is conserved exactly. FFT bins are monotone in frequency, so the log grid is
    built by `reduceat` over bin index ranges, every bin lands in exactly one band, and
    the W1 integral uses the real (non-uniform) spacing of the resulting band centres.

    Two transports, one per axis, which is the sliced Wasserstein distance of the
    time-frequency mass taken along its two marginals. Across frequency, per frame, in
    octaves. Along time, over the whole clip, in seconds. The time term is what stops the
    frequency term being fooled: with frequency alone the shuffled control scored 0.46
    against 0.49 for a 2 dB/octave tilt, because a shuffled pad is still made of frames
    that each look like a pad. Adding it puts shuffled at 0.89 and delayed at 1.26
    against 0.63 for the tilt. Transporting mass along the time axis charges a 1 s delay
    almost exactly 1 s, which is the most honest number this loss produces.

    Frames are weighted by the geometric mean of the two clips' relative energy there, so
    the shape of a silent frame, which is noise with a shape, cannot outvote a sustained
    chord. The level term restores what per-frame normalisation removes: it is the only
    part that sees overall gain, and at 0.05 per dB one dB of broadband level error costs
    the same as moving the whole spectrum a twentieth of an octave.
    """
    n_fft, hop = 2048, 1024
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.logspace(np.log10(30.0), np.log10(18000.0), n_bins + 1)
    idx = np.unique(np.clip(np.searchsorted(freqs, edges), 1, len(freqs)))
    starts, stop = idx[:-1], int(idx[-1])
    # Position of each band on the log-frequency line, and the gap to the next one. W1 is
    # the integral of |CDF difference| along that line, so the gaps are the measure.
    pos = np.log2(np.sqrt(freqs[starts] * freqs[np.minimum(idx[1:], len(freqs) - 1)]))
    step = np.diff(pos, append=pos[-1]).astype(np.float32)

    def _profile(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = _power(x, n_fft, hop)[:, :stop]
        m = np.add.reduceat(P, starts, axis=1)                # frames x bands
        tot = m.sum(axis=1)
        C = np.cumsum(m / (tot[:, None] + 1e-30), axis=1)
        return C, tot

    Ct, tot_t = _profile(target)
    lfloor = float(tot_t.max()) * 1e-6 + 1e-30                # -60 dB below the loudest frame
    lvl_t = 10.0 * np.log10(np.maximum(tot_t, lfloor))
    wt_t = tot_t / (tot_t.sum() + 1e-30)
    dt = hop / sr

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        C, tot = _profile(p)
        n = min(len(C), len(Ct))
        # Both terms renormalise over the frames compared, so a truncated render would
        # otherwise be scored on its prefix alone and a half-length one came out at 2e-8,
        # near-perfect. The other candidates reject a short render through their block
        # arithmetic; this one has to say so.
        if n < 0.9 * len(Ct):
            return 1e6
        wt_p = tot[:n] / (tot[:n].sum() + 1e-30)
        w = np.sqrt(wt_t[:n] * wt_p)
        shape = float((w * (np.abs(C[:n] - Ct[:n]) * step).sum(axis=1)).sum()
                      / (w.sum() + 1e-30))
        # Transport along time. Renormalised over the frames actually compared so a
        # truncated pred is not charged for the mass it never had a chance to place.
        time = float(np.abs(np.cumsum(wt_p) - np.cumsum(wt_t[:n] / (wt_t[:n].sum() + 1e-30))).sum() * dt)
        lvl = 10.0 * np.log10(np.maximum(tot[:n], lfloor))
        return _guard(shape + time + 0.05 * float(np.abs(lvl - lvl_t[:n]).mean()))

    return score
