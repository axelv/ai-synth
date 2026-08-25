"""Candidates written from the optimisation-landscape lens.

The question here is not what a human hears, it is whether a derivative-free optimiser
can descend the surface. Four properties decide that, and the incumbent fails all four.

1. Bounded influence. The incumbent's log-magnitude term is a mean over ~1.6M STFT bins,
   most of which sit between partials and carry no energy. Each of those can contribute
   an unbounded log ratio, so a broadband change 85 dB down moves the mean by more than
   the largest genuine improvement the project ever found. No single bin may move the
   total by more than a bounded amount, and that bound has to scale with how much of the
   signal the bin actually carries.

2. Effective dimension. A mean of N independent near-zero terms has a value set by the
   noise in those terms, and that noise does not shrink as the patch approaches truth.
   1.6M votes to resolve 55 unknowns is 1.6M chances for the surplus to outvote the
   signal. Either weight the terms so the effective N is small, or pool before comparing.

3. Conditioning. CMA-ES adapts to a covariance, so it survives ill-conditioning but pays
   for it in evaluations, and a direction that is flat to within the term-noise of point 2
   is not merely slow, it is unrecoverable. Terms that live in different units have to be
   put on one scale explicitly, and any direction known to be near-singular in the
   parameterisation is worth giving its own term.

4. Proportionality. The value must move smoothly with a parameter, not in steps and not
   in spikes. Differentiable transforms only; no hard floors, since a max() is a kink an
   optimiser can lodge against and its crossing set moves from candidate to candidate.

One finding here is bigger than any of the four candidates and belongs at the top. All
of these were screened on a synthetic pad, against a broadband change 85 dB down, against
deliberately wrong controls, and against a nuisance the bake-off does not test: re-rolling
the initial phase of the oscillators. Phase is not a fittable parameter, but detuned
partials beat, so phase changes the magnitude surface. It costs the incumbent 0.68 of its
whole rms-0.10 dynamic range, and it costs these candidates 0.037 (graduated_blur), 0.34
(slope_conditioned), 0.48 (softlog_huber) and 0.71 (energy_weighted_log). Every one of
those is far above the pedestal any of them charges, and above what a real parameter move
of rms 0.001 costs. If that carries over to the renderer, then beating phase, not the
pedestal, sets the floor on how finely any of these can resolve a patch. Smoothing buys
it back, and it has to be smoothing in frequency as well as in time: blurring only along
time left the cost at 0.31, blurring in both directions took it to 0.037. Worth measuring
on real renders before tuning anything else.

Costs, measured as the minimum of repeated calls on 17.9 s at 44100 Hz while the machine's
CPUs were saturated by a render job, so the true idle figures are lower: 12 ms
(softlog_huber), 18 ms (graduated_blur), 10 ms (energy_weighted_log), 8 ms
(slope_conditioned), against 213 ms for the incumbent. Each returns exactly 0.0 against
its own target, is finite on silence, on a 40x overdriven signal and on a truncated one,
and is unchanged by float64 input.

Measured on synthetic signals here and rejected, recorded so nobody re-opens them:

- A trimmed mean of the incumbent's per-bin log residual, dropping the worst 20% of bins.
  The cleanest test of "a minority of empty bins carries the pedestal", and it fails: on
  a pad plus a -85 dB broadband floor, the pedestal was still 813x a plausible parameter
  move, because the empty bins are the majority, not a minority, so the trim never
  reaches them. `np.partition` also costs 25-80 ms per resolution. The pedestal is a
  property of most of the domain, so it has to be fixed by transform or by weight, never
  by rank.
- Fully redescending penalties (Geman-McClure, Tukey). They bound each bin's contribution
  to a constant, which kills the pedestal but flattens the far field, and `bakeoff`'s
  `discrim` requires deliberately-wrong audio to sit at least 3x the rms-0.10 dynamic
  range above truth. Everything here stays linear or grows in the far field.
- A hard relative floor as the whole fix. That is the `logmag_floor` baseline. It works
  on the pedestal but it is a kink: bins cross the floor as the patch moves, so the loss
  is piecewise-smooth with a boundary that depends on the candidate. The soft reference
  used below is that idea made analytic.
- Per-call normalisation by the prediction's own energy. It makes the loss invariant to
  the one parameter that is trivially recoverable and hands the optimiser a free flat
  direction.
- Cosine distance on spectra. Bounded above by 2, and grossly wrong audio gets close
  enough to that bound to compress `discrim`.
- A short descriptor vector compared by squared distance. Built it, then found
  `blind_descriptor` in this same package already occupies that mechanism; the marginal
  value was a scale normalisation, which is a variation and not a candidate.
"""

from __future__ import annotations

from typing import Callable

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter

from losses import SR, mags, match, register

# One mid resolution is enough for anything that pools or blurs afterwards, and it is the
# cheapest STFT of the useful ones: 17 ms against 40 ms for 2048/512 on this machine.
_FFT, _HOP = 2048, 1024


def _f32(x: np.ndarray) -> np.ndarray:
    """Float32 throughout. Halves the elementwise cost and matches librosa's own dtype."""
    return np.asarray(x, dtype=np.float32)


def _bandmat(sr: int, n_fft: int, n_bands: int, lo: float, hi: float) -> np.ndarray:
    """Overlapping triangular pooling onto log-spaced bands.

    Triangles rather than rectangles so a partial drifting between two bands moves the
    pooled level continuously. A rectangular bank makes every band a step function of
    pitch, which is exactly the kind of staircase an optimiser cannot descend.
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    edges = np.geomspace(lo, hi, n_bands + 2)
    fb = np.zeros((n_bands, len(freqs)), dtype=np.float32)
    for i in range(n_bands):
        a, c, b = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs > a) & (freqs <= c)
        dn = (freqs > c) & (freqs < b)
        fb[i, up] = (freqs[up] - a) / (c - a)
        fb[i, dn] = (b - freqs[dn]) / (b - c)
        s = fb[i].sum()
        if s > 0:
            fb[i] /= s
        else:                               # bands narrower than the bin spacing
            fb[i, np.argmin(np.abs(freqs - c))] = 1.0
    return fb


def _softdb(B: np.ndarray, ref: float) -> np.ndarray:
    """dB with a soft floor: 20 log10(B + ref).

    Above `ref` this is dB, so a gain error costs the same wherever it happens, which is
    what fitting 26 EQ gains needs. Below `ref` it flattens smoothly to a constant, so a
    band that is silent in the target cannot report an unbounded ratio when the render
    puts something inaudible there. Unlike max(B, ref) it has no kink and no crossing set.
    """
    # float64 out: the pooled surface is ~50k numbers, so carrying it exactly costs
    # nothing, and a float32 subtraction of two dB surfaces has an absolute error near
    # 1e-5 dB, which is inside the range these candidates are being asked to resolve.
    return 20.0 * np.log10(B.astype(np.float64) + ref)


# --------------------------------------------------------------------------------------
# 1. soft reference plus bounded influence, per bin
# --------------------------------------------------------------------------------------

@register("softlog_huber")
def softlog_huber(target: np.ndarray, sr: int = SR, ref_db: float = -55.0,
                  delta: float = 0.10) -> Callable[[np.ndarray], float]:
    """Per-bin residual read through log1p compression and a pseudo-Huber penalty.

    Two independent mechanisms, one for each half of the measured failure.

    The compression is c(M) = log1p(M / ref) with ref a fixed level 55 dB under the
    target's loudest bin at that resolution. Above ref it is a logarithm, keeping the
    dynamic-range sensitivity that makes log-magnitude worth having on the partials that
    carry the timbre. Below ref it is linear in amplitude, so an empty bin's residual is
    proportional to the absolute energy injected there rather than to its ratio against
    nothing. That is the whole pedestal: a broadband floor 85 dB down is a 25 dB jump for
    a bin sitting at -110 dB, which log reads as 2.9 nats and log1p against a -55 dB
    reference reads as 0.031, and which the quadratic knee below then charges as 0.0005.

    The penalty is pseudo-Huber, d^2 (sqrt(1 + (r/d)^2) - 1): quadratic within `delta`,
    asymptotically linear outside it, analytic everywhere. Its derivative is bounded by
    `delta`, which caps how far one bin can pull the argmin, and the quadratic knee
    suppresses small residuals by a further factor r/(2 delta) while leaving large ones at
    full weight. The two mechanisms compose: the compression turns the pedestal into a
    small residual, and the knee then charges the square of it.

    `ref` is anchored to the target, never to the prediction, so the scorer is a fixed
    function of its argument and two candidates are always measured on one ruler.

    One resolution, 2048/1024, not the incumbent's four. This was measured rather than
    assumed: adding a fine 512/256 resolution alongside it left the pedestal unchanged
    (0.0029 against 0.0031) and made discrimination worse, 4.05 against 4.93 in units of
    the r=0.1 move and 6.12 against 12.77 in units of r=0.03, for 51 ms per call instead
    of 23 ms. The fine resolution resolves detail that no parameter controls, so what it
    contributes to a mean is mostly the term-noise of point 2 in the module docstring.

    `ref_db` is the knob that trades the two screens against each other and it was swept,
    not guessed. On the synthetic pad below, pedestal and discrimination in units of the
    r=0.1 move: -35 dB gives 0.00007 and 2.15, -45 gives 0.00044 and 2.51, -55 gives
    0.00289 and 4.05, -65 gives 0.01575 and 8.22. Lowering the reference hands back the
    log's reach on quiet detail and takes back the pedestal, monotonically. -55 dB is the
    most permissive setting still inside the 0.01 pedestal target with 3x to spare, and it
    is the only one that also clears the discrimination gate of 3.0. `delta`
    matters far less: over 0.01 to 1.0 the pedestal moves 0.0034 to 0.0001 and
    discrimination stays within 20%.
    """
    tgt = _f32(target)
    M = mags(tgt, _FFT, _HOP)
    ref = np.float32(float(M.max()) * (10.0 ** (ref_db / 20.0)) + 1e-20)
    C = np.log1p(M / ref)
    d2 = np.float32(delta * delta)

    def score(pred: np.ndarray) -> float:
        p, _ = match(_f32(pred), tgt)
        u = mags(p, _FFT, _HOP)
        u /= ref
        np.log1p(u, out=u)
        n = min(u.shape[1], C.shape[1])
        u = u[:, :n] - C[:, :n]
        u *= u
        u /= d2                                    # u = (residual / delta)^2
        # sqrt(1+u) - 1 written as u / (sqrt(1+u) + 1). The direct form cancels in
        # float32 for exactly the residuals this loss exists to resolve, and the whole
        # question is whether a 1e-6 relative move is visible at all.
        v = np.sqrt(u + np.float32(1.0))
        v += np.float32(1.0)
        u /= v
        val = float(d2) * float(u.mean(dtype=np.float64))
        return val if np.isfinite(val) else 1e6
    return score


# --------------------------------------------------------------------------------------
# 2. graduated non-convexity, folded into one static number
# --------------------------------------------------------------------------------------

@register("graduated_blur")
def graduated_blur(target: np.ndarray, sr: int = SR, n_bands: int = 64,
                   ref_db: float = -60.0) -> Callable[[np.ndarray], float]:
    """One time-frequency surface compared at three blur scales, each variance-whitened.

    Continuation methods exist for exactly the geometry measured here: a true optimum that
    is a needle beside a broad wrong basin. Blur the objective and the needle merges with
    its neighbourhood into one wide bowl whose minimum is near truth; sharpen and the bowl
    resolves onto it. A derivative-free optimiser cannot be handed a schedule, so the
    scales are summed into a single value instead. Far from truth the coarse terms carry
    the residual and they vary smoothly over parameter steps large enough for CMA-ES to
    sample; near truth the coarse terms have gone small and quadratic and the finest term
    decides.

    Each scale is divided by the variance of the target's own structure at that scale.
    This is the conditioning move, not a cosmetic one: blurring shrinks a residual, so
    without the whitening the finest term would dominate at every distance and the
    continuation would do nothing at all. With it the three terms are commensurate and no
    single scale owns the Hessian.

    There is deliberately no unblurred term. That was measured, and it is the most useful
    thing this candidate turned up. Re-rolling the initial phase of the oscillators, which
    is not a fittable parameter, changes the detuning beat pattern and so changes the
    magnitude surface: it costs the incumbent 0.68 of its whole rms-0.10 dynamic range.
    Including an unblurred scale here costs 0.15 of it; dropping it costs 0.037. Every
    other screen improved at the same time, discrimination from 10.5 to 11.6 and the
    radius at which a real parameter move first outweighs the pedestal from 1e-3 to 1e-4,
    at identical runtime. The finest scale worth keeping is set by what a parameter can
    control, not by what the STFT can resolve, and below that the surface is a record of
    beating phase.

    Squared rather than absolute residuals: quadratic at the optimum is the best-
    conditioned bowl a derivative-free optimiser can be given, and superlinear growth in
    the far field buys headroom on `discrim`.

    Blur is applied to the residual, not to each surface separately, which is the same
    thing for a linear operator and costs one filter pass instead of two.
    """
    tgt = _f32(target)
    fb = _bandmat(sr, _FFT, n_bands, 30.0, 18000.0)
    ref = np.float32(float(np.max(fb @ mags(tgt, _FFT, _HOP)))
                     * (10.0 ** (ref_db / 20.0)) + 1e-20)
    # (bands, frames) sigmas. At hop 1024 a frame is 23 ms, so these run from 46 ms and a
    # sixth of an octave up to 1.1 s and an octave, which is the scale a note event has.
    scales = ((1.0, 2.0), (3.0, 8.0), (8.0, 48.0))

    def surface(x: np.ndarray) -> np.ndarray:
        return _softdb(fb @ mags(x, _FFT, _HOP), ref)

    T = surface(tgt)
    wts = [1.0 / (float(gaussian_filter(T, s, mode="nearest").var()) + 1e-9)
           for s in scales]

    def score(pred: np.ndarray) -> float:
        p, _ = match(_f32(pred), tgt)
        P = surface(p)
        n = min(P.shape[1], T.shape[1])
        R0 = P[:, :n] - T[:, :n]
        tot = 0.0
        for s, w in zip(scales, wts):
            R = gaussian_filter(R0, s, mode="nearest")
            tot += w * float((R * R).mean(dtype=np.float64))
        v = tot / len(scales)
        return float(v) if np.isfinite(v) else 1e6
    return score


# --------------------------------------------------------------------------------------
# 3. shrink the effective number of terms by weighting, not by pooling
# --------------------------------------------------------------------------------------

@register("energy_weighted_log")
def energy_weighted_log(target: np.ndarray, sr: int = SR, gate_db: float = -45.0,
                        eps_db: float = -110.0, exponent: float = 2.0
                        ) -> Callable[[np.ndarray], float]:
    """Log ratio per bin, weighted by the target's own energy share within its own row.

    The minimal edit that follows from the diagnosis, if the diagnosis is read as a
    counting problem rather than a transform problem. The log ratio is a good measure of
    how wrong a bin is; the mistake is averaging it uniformly, which lets a million bins
    that carry no signal outvote the ten thousand that do. So keep the log exactly as it
    is and change what the average is over.

    Weights are w[f,t] proportional to T[f,t]^`exponent`, normalised to sum to one along
    time within each frequency row. Normalising per row and not globally is deliberate: a
    global energy weight would make the loss see only the loudest few partials and go
    blind to the 26 EQ gains, most of which control bands 20 to 40 dB down. Per row, every
    frequency keeps an equal vote on its own gain while the frames in which that band is
    silent lose theirs, and those frames are where most of the pedestal lives.

    The row gate then removes rows the target leaves genuinely empty, softly, on a
    Michaelis-Menten curve in power rather than a threshold, so no row ever crosses in or
    out. Only rows more than `gate_db` below the loudest are materially affected: a row
    exactly at the gate keeps half its vote, one 10 dB below it keeps a tenth.

    Squared log ratio, not absolute: quadratic at the optimum, and it grows quadratically
    in the far field, which is where `discrim` is won.

    Both knobs were swept, and the result corrected the reasoning above rather than
    confirming it. Once rows are normalised, the pedestal that survives is carried by
    near-empty ROWS, not by the silent frames within a row, so `gate_db` is the knob that
    matters and `exponent` is nearly inert: at a -60 dB gate the pedestal is 0.022, 0.016,
    0.013 and 0.010 of the r=0.1 move for exponents 2, 3, 4 and 5, all of them at or over
    the 0.01 target, while at a -45 dB gate it is 0.0023, 0.0020, 0.0017 and 0.0014, all
    of them comfortably inside it. So the gate is set at -45 dB and the exponent is left
    at 2, plain energy share, which is the one value with a meaning.

    An earlier version of this computed the gate's row level from the weight normaliser,
    which is only an rms when the exponent is 2. It made the exponent look like the
    important knob and the gate like a formality. Recorded because the wrong conclusion
    was reachable from a clean-looking sweep.
    """
    tgt = _f32(target)
    T = mags(tgt, _FFT, _HOP)
    eps = np.float32(float(T.max()) * 10.0 ** (eps_db / 20.0) + 1e-30)
    Tl = np.log(T + eps)

    w = T.astype(np.float64) ** exponent
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)   # each row now sums to one
    # the gate reads a true per-row rms, which is not the same quantity as the weight
    # normaliser once `exponent` is anything but 2
    p2 = (T.astype(np.float64) ** 2).mean(axis=1)
    g = p2 / (p2 + p2.max() * 10.0 ** (gate_db / 10.0) + 1e-300)
    w *= (g / (g.sum() + 1e-300))[:, None]                  # total weight one
    W = w.astype(np.float32)

    def score(pred: np.ndarray) -> float:
        p, _ = match(_f32(pred), tgt)
        P = mags(p, _FFT, _HOP)
        n = min(P.shape[1], T.shape[1])
        r = np.log(P[:, :n] + eps) - Tl[:, :n]
        r *= r
        r *= W[:, :n]
        # renormalised, so a render that ends early is not rewarded for the frames it
        # never produced. The full-length case is the common one and costs nothing.
        v = (float(r.sum(dtype=np.float64))
             / (1.0 if n == T.shape[1] else float(W[:, :n].sum(dtype=np.float64)) + 1e-30))
        return v if np.isfinite(v) else 1e6
    return score


# --------------------------------------------------------------------------------------
# 4. put curvature into the direction that is known to be flat
# --------------------------------------------------------------------------------------

@register("slope_conditioned")
def slope_conditioned(target: np.ndarray, sr: int = SR, n_bands: int = 48,
                      ref_db: float = -50.0, blur: tuple[float, float] = (1.0, 4.0)
                      ) -> Callable[[np.ndarray], float]:
    """Band-level residual plus its differences across frequency and across time.

    The complement of `graduated_blur`, and worth measuring beside it: that one sums
    low-pass copies of the residual, this one applies difference operators to a single
    mildly smoothed one. Composed, smooth-then-difference is a bandpass, so if the
    bake-off separates the two it says where in scale the wrong basin lives, which no
    single number from either alone can say.

    The motivation is specific rather than general. `fit_eq_full` records that the 26-band
    bank has a near-singular alternating direction, and that fitting its gains without a
    curvature penalty walks into it; the same history records a fit that came back as a
    comb tuned to one chord's partials. A direction that is near-singular in the forward
    map is a direction in which the loss is flat, and flat here means flat to within the
    term-noise of a million-term mean, which is to say unrecoverable. Applying the first
    and second difference across bands to the residual and scoring those puts curvature
    back exactly there: a comb, alternating band to band, is nearly invisible to the level
    term and maximal under second differences. The time difference is included on the same
    argument for envelopes, since two patches whose band levels average the same but whose
    attacks differ have a small level residual and a large difference residual.

    Every term is divided by the variance of the target in that same domain, so the four
    are commensurate and no direction is small merely because of its units. That whitening
    is what makes this a preconditioner rather than a fourth opinion, and it is where the
    value is: with the smoothing in place the target's own band-to-band alternation is
    small, so its reciprocal is large, and a comb in the residual is charged accordingly.
    Measured, `blur` raised discrimination from 1.7 to 33.5 in units of the r=0.1 move and
    from 8.4 to 185 in units of r=0.03, left the pedestal at 1e-5, and cut the cost of the
    unfittable oscillator-phase nuisance described in `graduated_blur` from 0.42 to 0.34.
    """
    tgt = _f32(target)
    fb = _bandmat(sr, _FFT, n_bands, 30.0, 18000.0)
    ref = np.float32(float(np.max(fb @ mags(tgt, _FFT, _HOP)))
                     * (10.0 ** (ref_db / 20.0)) + 1e-20)

    def surface(x: np.ndarray) -> np.ndarray:
        return gaussian_filter(_softdb(fb @ mags(x, _FFT, _HOP), ref), blur,
                               mode="nearest")

    def ops(S: np.ndarray) -> tuple[np.ndarray, ...]:
        return (S, np.diff(S, axis=0), np.diff(S, n=2, axis=0), np.diff(S, axis=1))

    T = surface(tgt)
    wts = [1.0 / (float(o.var()) + 1e-9) for o in ops(T)]

    def score(pred: np.ndarray) -> float:
        p, _ = match(_f32(pred), tgt)
        P = surface(p)
        n = min(P.shape[1], T.shape[1])
        R = P[:, :n] - T[:, :n]
        v = sum(w * float((o * o).mean(dtype=np.float64))
                for w, o in zip(wts, ops(R))) / len(wts)
        return v if np.isfinite(v) else 1e6
    return score
