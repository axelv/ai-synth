"""Two losses for the family the other six lenses left out: CALIBRATED ones.

Read the twenty-four candidates already in this package and one thing is true of every
single one of them. Each computes a feature vector from the render, computes the same
feature vector from the target, and charges some fixed norm of the difference. The
features vary enormously (Bark excitation, adaptation loops, texture moments, scattering
paths, quantiles, transport plans) and the norms vary a little (L1, L2, power mean,
Jeffreys). What none of them has is any notion of HOW MUCH A GIVEN FEATURE IS WORTH. The
weights are all set by hand from physical reasoning: 6 dB is a clearly audible band
error, 0.5 octaves a clearly audible brightness error, 1/variance of the target's own
structure, a share of the target's loudness. Reasonable numbers, every one, and not one
of them was measured against the thing that actually decides whether a feature carries
information about a parameter.

That matters because the bake-off has two screens, not one, and the pool is aimed at
only the first. `pedestal` asks whether an inaudible change is cheap; every module here
reports its pedestal ratio proudly and most of them fix it. `near_wins` asks whether the
minimum is in the right place, and that is the failure that actually stops the search:
a local polish started 0.001 from truth walks to an attractor 0.017 away and stays there
regardless of where it started. Making a loss blind to the inaudible does not move that
attractor. Making it FLATTER can even entrench it, because a flat-bottomed basin is a
wide one. Nothing in the pool addresses it directly, and `landscape.py` comes closest by
accident: it measured that re-rolling the oscillators' initial phase, which is not a
fittable parameter at all, costs the incumbent 0.68 of its whole dynamic range and costs
its own best candidate 0.037, both far above what a real parameter move of rms 0.001
costs. Then it hand-tuned a Gaussian blur until that number came down.

That measurement is the diagnosis and it deserves to be the mechanism. If a feature moves
a lot under a change no parameter can produce, then a disagreement in that feature is not
evidence about the patch, and an objective that charges for it is being steered by noise.
Its minimum then sits wherever the noise happens to balance, which is a broad basin in the
wrong place. So:

  `nuisance_whitened`  measures the nuisance instead of guessing at it. Build an ensemble
                       of variants of the TARGET that differ only in things no control can
                       set, take the per-feature standard deviation over that ensemble,
                       and charge every feature in units of its own nuisance sigma. This
                       is a whitened matched filter, or Fisher's within-class scatter, and
                       it is the standard answer to exactly this problem in every other
                       field that has it. Absent here.

  `control_projection` measures distance in the units the search moves in. Split the
                       static spectral disagreement into the part the 26-band EQ can
                       actually produce, using the bank's own first-order response, and
                       the part it cannot; charge the first in dB of slider travel and the
                       second at a premium. "Rank patches by how close their parameters
                       are" is the stated requirement, and this is the only candidate that
                       computes a number in parameter units. Also absent: no other
                       candidate solves anything, they all only transform and subtract.

Both are deliberately built on the plainest front end in the package, a log-band dB
surface floored 70 dB under the target's loudest band, the one shared by `mel_l1`,
`band26_env`, `blind_*`, `graduated_blur` and `slope_conditioned`. That is not a lack of
imagination, it is the control: if either of these beats that group on `near_wins`, the
credit belongs to the calibration and not to a cleverer representation, and if it does
not then the calibration idea is dead and the bake-off learned something for the price of
two candidates instead of six.

Measured here, on synthetic material only, since rendering was not available (17.9 s at
44100 Hz, machine loaded by a render job, best of five calls):

    nuisance_whitened   14 ms per call, 1.0 s one-time factory
    control_projection  17 ms per call, 0.16 s factory
    incumbent, same run 200 ms per call

The factory figures are per TARGET, not per call, and `nuisance_whitened` spends nearly
all of its on building the eight-member ensemble. Worth stating because `bakeoff.screen`
divides factory plus all scores by the number of labels and so folds a one-time second
into the per-call figure.

On the standard probes, as a fraction of what a plainly audible 1 dB/octave tilt costs, a
broadband perturbation 85 dB down costs nuisance_whitened 6.0e-4 and control_projection
9.1e-4, against 0.21 for the incumbent on the identical pair. On a seven-point tilt sweep
with that same perturbation added at every point, all three put the argmin at truth, so
the sweep separates nothing; the sweep is one-dimensional and the measured failure is a
basin offset in 55, which is why the real test is `bakeoff`'s `near_wins` and not
anything that can be checked here. Both return exactly 0.0 against their own target, in
float32 or float64, and both are finite on silence, on white noise, on a 40x overdriven
signal, on a half-length render and on a 0.5 s target, with no numpy warnings raised.

Tried and rejected:

- Whitening by a FULL covariance over the feature ensemble rather than per-dimension.
  Correct in principle and unusable in practice: 1280 features from 8 ensemble members
  gives a rank-7 estimate, so the pseudo-inverse amplifies seven arbitrary directions and
  ignores the rest. Diagonal plus local smoothing of the sigma map is what a sample of
  that size can actually support, and the smoothing is doing the job a low-rank shrinkage
  would: nuisance sensitivity varies smoothly over the time-frequency plane, so
  neighbouring features are legitimately pooled to estimate it.
- Including a level jitter among the nuisances. outGain is a fitted parameter, so
  declaring level a nuisance would make the loss blind to it. Every nuisance in the
  ensemble has to be something the 55 controls provably cannot produce, which is a much
  shorter list than it first looks: initial oscillator phase, and where the analysis
  frames happen to land. Both are in; nothing else qualified.
- The all-pass phase nuisance without restoring the target's broadband envelope. The
  group delay needed to decorrelate two partials a few Hz apart is of order 1/detune,
  i.e. 0.1 s, and that much dispersion also smears the note attacks, which are set by aA.
  Left alone it taught the whitener that onsets are noise and cost the metric its only
  read on the amplitude envelope. Restoring the envelope keeps the beat re-roll, which is
  the real nuisance, and drops the smearing, which is an artefact of modelling it as a
  filter.
- Charging the EQ correction as an L2 norm rather than L1. The bank's gains are 26
  independent sliders and the honest measure of "how far did you have to move the
  controls" is total travel, not Euclidean distance in a space where no rotation means
  anything.
- Fitting a time-varying correction as well as a static one, so the dynamic term would
  also be reachability-split. The filter ADSR and the envelopes are what produce
  time-varying spectral change and their response is not linear in dB, so the projection
  would have been onto a basis nobody measured. The dynamic term is left as the plainest
  possible L1 on the band surface precisely so that any difference between this candidate
  and `mel_l1` is attributable to the static split alone.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.fft
import scipy.ndimage

from losses import SR, mags, match, register

# 2048/1024 is the cheapest resolution that still resolves the EQ bank's 0.5-octave
# bandwidth at its lowest centre, and both candidates pool it immediately, so anything
# finer would only add structure no control can move. Shared so the two are comparable.
_FFT, _HOP = 2048, 1024
_FLOOR_DB = -70.0        # below this, relative to the target's loudest band, nothing counts


def _logband(n_fft: int, n_bands: int, sr: int, lo: float = 30.0,
             hi: float = 18000.0) -> tuple[np.ndarray, np.ndarray]:
    """Area-normalised rectangular log-spaced bands, plus their geometric centres.

    Written out here rather than imported from a sibling candidate: importing
    `loss_candidates.blind` for its private helper would register four unrelated losses as
    a side effect of importing these two, and the registry is order-sensitive.
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(lo, hi, n_bands + 1)
    fb = np.zeros((n_bands, freqs.size), dtype=np.float32)
    cen = np.sqrt(edges[:-1] * edges[1:])
    for i in range(n_bands):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if sel.any():
            fb[i, sel] = 1.0 / sel.sum()
        else:                                   # narrower than the bin spacing
            fb[i, np.argmin(np.abs(freqs - cen[i]))] = 1.0
    return fb, cen


def _band_db(x: np.ndarray, fb: np.ndarray, floor: float) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(fb @ mags(x, _FFT, _HOP), floor))


def _floor_of(bands: np.ndarray) -> float:
    """One absolute silence level, taken from the target and applied to both signals, so a
    quiet render is never graded on its own curve."""
    return float(bands.max()) * 10.0 ** (_FLOOR_DB / 20.0) + 1e-30


def _guard(v: float) -> float:
    return float(v) if np.isfinite(v) else 1e6


# --------------------------------------------------------------------------------------
# 1. charge every feature in units of its own nuisance sigma
# --------------------------------------------------------------------------------------


def _phase_reroll(x: np.ndarray, rng: np.random.Generator, sr: int,
                  corr_hz: float = 5.0) -> np.ndarray:
    """The target with every partial's phase re-rolled and nothing else changed.

    Faust's oscillators are free running, so the phase each voice starts at is not a
    function of any parameter. It is not inaudible either: seven detuned saws beat against
    each other, and which partials are momentarily in phase decides the shape of every
    band's envelope over the beat period. That is the single largest thing in a render
    that no patch can control.

    Modelled as an all-pass with a random phase that decorrelates over `corr_hz`. The
    scale is forced: to change the relative phase of two partials that far apart, the
    phase has to swing by order one over that interval, which is a group delay of order
    1 / corr_hz, i.e. 0.03 s here. That dispersion also smears the note attacks, which a
    phase re-roll does not, so the target's own broadband envelope is divided back in
    afterwards. What survives is a signal with the same spectrum, the same envelope and a
    different beat pattern, which is what a re-seeded render is.
    """
    n = len(x)
    nf = scipy.fft.next_fast_len(n)
    X = scipy.fft.rfft(np.asarray(x, dtype=np.float32), nf)
    k = max(int(round(corr_hz * nf / sr)), 1)
    th = scipy.ndimage.uniform_filter1d(
        rng.standard_normal(X.size).astype(np.float32), k, mode="nearest")
    th *= np.float32(np.pi / (th.std() + 1e-12))
    # built by parts rather than as exp(1j*th): a Python 1j promotes the whole array to
    # complex128 and doubles the cost of the inverse transform for no extra precision
    rot = np.empty(th.size, dtype=np.complex64)
    rot.real, rot.imag = np.cos(th), np.sin(th)
    y = scipy.fft.irfft(X * rot, nf)[:n].astype(np.float32)

    # restore the target's broadband envelope, 20 ms rectangular, so the dispersion cannot
    # masquerade as a slower attack
    w = max(int(0.020 * sr), 1)
    ex = np.sqrt(scipy.ndimage.uniform_filter1d(np.asarray(x, np.float32) ** 2, w))
    ey = np.sqrt(scipy.ndimage.uniform_filter1d(y ** 2, w))
    return y * (ex / (ey + np.float32(1e-9) + np.float32(0.01) * float(ex.max())))


@register("nuisance_whitened")
def nuisance_whitened(target: np.ndarray, sr: int = SR, n_bands: int = 40,
                      n_blocks: int = 32, n_ens: int = 8, seed: int = 0
                      ) -> Callable[[np.ndarray], float]:
    """Log-band dB surface, each feature divided by how much a non-parameter moves it.

    The construction. Pool the render into a 40 band by 32 block dB surface, 1280
    features. Build eight variants of the TARGET that differ from it only in quantities no
    control can set: the phase each oscillator starts at, and where the analysis frames
    land. Take the standard deviation of each feature across that ensemble. That sigma is
    the irreducible disagreement between two renders of the SAME patch, so it is the
    resolution limit on that feature and nothing finer than it is evidence. Score the mean
    square of the residual measured in those units, which is the whitened matched filter
    and reads as a d-prime: 1.0 means the render differs from the target by about as much
    as two renders of one patch differ from each other.

    Why this should move `near_wins` and not merely `pedestal`. Every other candidate sets
    its weights from what a listener notices. That is the right calibration for audibility
    and the wrong one for identifiability, and the two come apart precisely here: beating
    phase is highly audible AND carries no information about the patch, so a perceptual
    weighting gives it a large vote and a nuisance weighting gives it none. A loss steered
    by a feature that carries no information has its minimum wherever that feature happens
    to balance, and the measured attractor at rms 0.017 that does not depend on the start
    is what a minimum determined by something other than the parameters looks like.

    The pedestal falls out rather than being aimed at. A change 85 dB down moves a band
    level by order 1e-5 dB and the smallest sigma in the ensemble is 0.02 dB, so it scores
    around 1e-3 sigma, squared. Nothing was floored to make that happen.

    Measured on a synthetic pad, the sigma map is not flat, which is the whole question:
    2.9 dB in the mid bands where five detuned voices beat against each other, 0.25 dB in
    the top bands where they do not, around a median of 1.78. So the metric weights a
    top-band error about twelve times a mid-band one of the same size in dB. Every other
    candidate in the package weights those two the same, or weights the mid band HIGHER
    because it is louder and therefore more audible. If the bake-off's `near_wins` moves,
    that factor of twelve is where to look for the reason.

    Two honest costs. The sigma estimate comes from eight samples, so it is good to about
    30% per feature before smoothing and better after; the smoothing over a 3 by 3
    neighbourhood is legitimate because nuisance sensitivity varies smoothly over the
    time-frequency plane, not because eight is enough. And the ensemble models two
    nuisances, not all of them; anything else that is not a function of the parameters, and
    the renderer surely has more, keeps its full vote. Both push the same way: the metric
    is conservative, downweighting less than it should, never more.
    """
    tgt = np.asarray(target, dtype=np.float32)
    fb, _ = _logband(_FFT, n_bands, sr)
    floor = _floor_of(fb @ mags(tgt, _FFT, _HOP))

    n_ref = _band_db(tgt, fb, floor).shape[1]
    # Block layout is fixed by the target so every signal is pooled on the same instants.
    # Deduplicated because a clip shorter than two frames per block would otherwise give
    # empty blocks and divide by zero; on 17.9 s this is a no-op.
    bnd = np.unique(np.linspace(0, n_ref, min(n_blocks, max(n_ref // 2, 1)) + 1
                                ).round().astype(int))
    n_blocks = len(bnd) - 1

    def pool(D: np.ndarray) -> tuple[np.ndarray, int]:
        n = D.shape[1]
        nb = int(np.searchsorted(bnd, n, side="right")) - 1   # complete blocks only
        if nb < 1:
            return np.zeros((D.shape[0], 0), dtype=np.float64), 0
        return (np.add.reduceat(D[:, :bnd[nb]], bnd[:nb], axis=1)
                / np.diff(bnd[:nb + 1])), nb

    def feature(x: np.ndarray, off: int = 0) -> np.ndarray:
        # the offset shifts where the frames land without changing how many there are, so
        # every ensemble member pools onto the same block layout as the target
        if off:
            x = np.concatenate([x[off:], np.zeros(off, dtype=np.float32)])
        return pool(_band_db(x, fb, floor))[0]

    ft = feature(tgt)

    rng = np.random.default_rng(seed)
    ens = np.stack([feature(_phase_reroll(tgt, rng, sr), int(rng.integers(0, _HOP)))
                    for _ in range(n_ens)])
    sig = ens.std(axis=0, ddof=1)
    sig = scipy.ndimage.uniform_filter(sig, size=3, mode="nearest")
    med = float(np.median(sig))
    # Bounded both ways. The absolute 0.02 dB is where float32 render noise lives, so no
    # feature is ever declared infinitely informative; the 20x cap stops a single
    # ill-estimated sigma from silencing a feature outright. The relative lower bound is
    # deliberately loose at 0.05: measured on a synthetic pad the sigma map runs from 0.25
    # dB in the top bands to 2.9 dB in the beating mid-bands around a median of 1.78, so a
    # tighter one would clip the quiet end, which is where the informative features are.
    w = np.clip(np.maximum(sig, 0.02), 0.05 * med, 20.0 * med)

    def score(pred: np.ndarray) -> float:
        p, _ = match(np.asarray(pred, dtype=np.float32), tgt)
        fp, nb = pool(_band_db(p, fb, floor))
        if nb < 0.9 * n_blocks:            # a render too short to pool on this layout
            return 1e6
        r = (fp - ft[:, :nb]) / w[:, :nb]
        return _guard(np.sqrt(float((r * r).mean())))
    return score


# --------------------------------------------------------------------------------------
# 2. measure the disagreement in the units the controls are dialled in
# --------------------------------------------------------------------------------------


@register("control_projection")
def control_projection(target: np.ndarray, sr: int = SR, n_bands: int = 60,
                       lam: float = 0.05, w_unreach: float = 2.0, w_dyn: float = 1.0,
                       scale_db: float = 3.0) -> Callable[[np.ndarray], float]:
    """How much EQ would fix this render, plus what no EQ can fix.

    Every other candidate answers "how different do these two sound". This one answers
    "how far apart are the patches", which is what the bake-off is actually ranking, by
    solving a small inverse problem instead of taking a norm.

    The mechanism. A cascade of peaking sections adds in dB exactly, and to first order in
    gain a bell of Q at fc contributes g / (1 + (Q(fc/f - f/fc))^2) dB. So the 26-band bank
    is a linear map A from a gain vector in dB to a dB curve, and it is a map with a known
    range: some static spectral shapes are one slider move away and others are outside it
    entirely. Given the time-averaged dB disagreement between render and target, solve for
    the gain vector that best explains it, then report three things:

      the travel      mean |g| in dB, how far the sliders would have to move
      the unreachable what the best possible gain setting still leaves, charged double
      the dynamic     the part of the disagreement that varies over time, which no static
                      EQ addresses at all

    Why the split should help where a plain band distance does not. Under any norm on the
    band surface, two renders equally far from the target score the same whether their
    error is one slider away or outside the bank entirely; the optimiser is told the same
    thing about both and has to discover the difference by trial. Here the two are
    different numbers with different prices. And the solve is regularised by curvature,
    which `fit_eq_full` records as the thing that keeps the fit out of the bank's
    near-singular alternating direction. That regularisation now does double duty: a
    residual lying in the alternating direction is not cheaply reachable, so it is charged
    to the unreachable term at the premium rather than being absorbed as free travel. A
    comb error is exactly what the project has measured a mismatched EQ fit walking into,
    and this is the only candidate in the pool that prices it as one.

    Insensitivity to the inaudible is inherited, not engineered: every term is a difference
    of band dB floored 70 dB below the target's loudest band, so a perturbation 85 dB down
    moves an audible band by under 0.001 dB and a floored one by nothing.

    The solve was checked against known moves rather than assumed to work. Applying the
    exact analog bell cascade to the target and asking the projection to name it: a single
    +6 dB band at 902 Hz comes back on the right band, right sign, at 0.79 of its size and
    correlated +0.96 with the truth; a smooth 6 dB tilt across the bank comes back at 0.84
    (the shortfall is the top and bottom bands, which the audibility weight excludes
    because the target has nothing there); an alternating plus or minus 3 dB comb comes
    back at 0.30. That last number is the regulariser working as intended, not a failure:
    the comb is the bank's near-singular direction, it is not cheaply reachable, and two
    thirds of it is therefore charged to `unreach` at the premium rather than counted as
    free travel. Under any plain norm on the band surface those three would be ranked by
    size alone.

    Weights. `w_unreach` at 2.0 is the only genuinely tuned number and it says a static
    error the bank cannot make is worth twice one it can, because the first means the core
    patch is wrong and the second means a slider is. `scale_db` at 3 dB is a unit, not a
    tuning: it makes the whole score read as multiples of a clearly audible band error.
    """
    import synth  # local, as in losses.band26_env: keeps the registry import cheap

    tgt = np.asarray(target, dtype=np.float32)
    fb, cen = _logband(_FFT, n_bands, sr)
    floor = _floor_of(fb @ mags(tgt, _FFT, _HOP))
    T = _band_db(tgt, fb, floor).astype(np.float64)

    fc = synth.eq_band_freqs()
    q = synth.EQ_Q
    u = q * (fc[:, None] / cen[None, :] - cen[None, :] / fc[:, None])
    A = 1.0 / (1.0 + u * u)                        # 26 bands x n_bands, dB per dB

    # Audibility of each analysis band in the target, so a band the target leaves empty
    # neither drives the fit nor scores. Full weight to 45 dB below the loudest band,
    # nothing below 65.
    lvl = T.mean(axis=1)
    w = np.clip((lvl - (lvl.max() - 65.0)) / 20.0, 0.0, 1.0)
    w = w / (w.sum() + 1e-12)

    # Second difference across the 26 gains. Penalising curvature rather than size is the
    # settled result from fit_eq_full: without it the solve walks into the bank's
    # near-singular alternating direction and reports free travel that is not free.
    n_eq = len(fc)
    D2 = (np.eye(n_eq)[:-2] - 2.0 * np.eye(n_eq)[1:-1] + np.eye(n_eq)[2:])
    G = A @ (w[:, None] * A.T)
    R = D2.T @ D2
    # lam is dimensionless: the two traces are matched first, so the same value means the
    # same trade-off whatever the band count or the weighting.
    lam_eff = lam * float(np.trace(G)) / (float(np.trace(R)) + 1e-12)
    solve = np.linalg.solve(G + lam_eff * R + 1e-9 * np.eye(n_eq), A * w[None, :])

    def score(pred: np.ndarray) -> float:
        p, _ = match(np.asarray(pred, dtype=np.float32), tgt)
        P = _band_db(p, fb, floor).astype(np.float64)
        n = min(P.shape[1], T.shape[1])
        if n < 0.9 * T.shape[1]:
            return 1e6
        res = T[:, :n] - P[:, :n]
        d = res.mean(axis=1)                       # the static disagreement, dB per band
        g = solve @ d                              # the EQ move that best explains it
        left = d - A.T @ g                         # what the bank cannot produce
        travel = float(np.abs(g).mean())
        unreach = float(np.sqrt(w @ (left * left)))
        dyn = float(w @ np.abs(res - d[:, None]).mean(axis=1))
        v = (travel + w_unreach * unreach + w_dyn * dyn) / scale_db
        return _guard(v)
    return score
