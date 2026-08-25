"""Four objectives designed from the failure, not from the incumbent.

The measurement says the incumbent is a sum over time-frequency bins of a quantity that
is unbounded below, so a bin holding nothing can outvote every bin holding the sound.
The usual reflex is to repair that sum: floor it, compress it, average it into bands.
This file does not do that. It asks instead what a distance between two renders of the
SAME synth should be when the goal is to recover the parameters, and it deliberately
gives up things the incumbent takes for granted:

  - that the score is a sum over bins            (`blind_descriptor` reads 78 numbers)
  - that the signals share a time axis           (`blind_bag` throws time order away)
  - that both sides are treated alike            (`blind_explain` judges pred by a model
                                                  built only from the target)
  - that the interesting content is the spectrum (`blind_modspec` scores the envelopes'
                                                  spectra, not the signal's)

The unifying argument. Two renders of one synth differ only through 55 controls. The
information needed to tell those controls apart is tiny: a filter curve, a few rates, a
few times, a level per band. Anything an objective measures beyond that is not evidence
about the parameters, it is noise with a vote. Every candidate here therefore passes the
audio through a hard information bottleneck first and only then compares. That is what
makes an inaudible change invisible: it is not floored into submission, it is not
compressed, it is simply not represented.

The one piece of hygiene shared by all four is `_db`: every dB is taken against an
absolute floor 70 dB below the TARGET's loudest band, fixed once in the factory. Both
signals are measured against the same physical level, so a quiet render cannot earn a
more forgiving reference, and content already 70 dB down cannot be read at all. That is
the smallest possible statement of "inaudible means invisible", and it is a property of
the representation rather than a term in the loss.

Rejected along the way, recorded so nobody re-spends the time:

  - Rank correlation per frequency band (1 - Spearman rho on band levels). Scale-free and
    outlier-proof, but it discards magnitude entirely, so a patch with twice the EQ tilt
    is indistinguishable from the right one. Half the parameters here ARE the EQ.
  - Self-referential contrasts only (compare each signal to a time-shifted copy of
    itself, then compare the two contrasts). Invariant to any static shaping, which is
    exactly the 26 EQ gains plus the filter. Blind to half the search space by design.
  - Cross-correlation or any waveform alignment term. Faust oscillators are free-running,
    so phase is not a function of the parameters at all.
  - A per-window minimum or best-window score. It rewards a patch that gets one chord
    right, which is the failure mode already recorded in CLAUDE.md.
  - Long Welch segments in `blind_modspec` (6 s, no smearing along the modulation axis).
    Sharper rate resolution, but a beating peak that crosses a bin boundary is charged
    the full depth of the periodogram valley next to it, which made a 1 percent detune
    error cost four times a 2x cutoff error. Rate resolution finer than the ear's is the
    same mistake as amplitude resolution finer than the ear's.
  - A multirate front end (decimate 8x for the low bands, where a synth's sub and cutoff
    live, and analyse the rest at full rate). Right idea, wrong budget: scipy's
    `resample_poly` alone measured 135 ms, three times the whole allowance.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from losses import SR, env, mags, match, register

FLOOR_DB = -70.0        # below this, relative to the target's loudest band, nothing counts


# ---------------- shared bottleneck machinery ----------------


def _logband_matrix(n_fft: int, n_bands: int, sr: int = SR,
                    fmin: float = 30.0, fmax: float = 18000.0) -> np.ndarray:
    """Rectangular, area-normalised log-spaced band filters.

    Rectangular rather than triangular because the point is an average, not a smooth
    interpolation: each band answers "how much energy is here" and nothing finer. Log
    spacing because the EQ bank being fitted is third-octave, so this is the geometry the
    controls themselves have. Bands narrower than the bin spacing take their single
    nearest bin, so no row is ever empty and no dB is ever -inf.
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    fb = np.zeros((n_bands, freqs.size))
    for i in range(n_bands):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if not sel.any():
            fb[i, np.argmin(np.abs(freqs - np.sqrt(edges[i] * edges[i + 1])))] = 1.0
        else:
            fb[i, sel] = 1.0 / sel.sum()
    return fb


def _floor_of(bands: np.ndarray) -> float:
    """The absolute amplitude that counts as silence, from the target's own loudest band.

    Taken once, from the target, and then applied to both signals. Deriving it per signal
    would let a quiet render be graded on its own curve.
    """
    return float(bands.max()) * 10.0 ** (FLOOR_DB / 20.0) + 1e-30


def _db(a: np.ndarray, floor: float) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(a, floor))


def _smooth_rows(M: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """Row-wise convolution. A loop over rows, because np.apply_along_axis costs 10x here."""
    return np.stack([np.convolve(r, kern, mode="same") for r in M])


# ---------------- 1. a descriptor, not a sum ----------------


@register("blind_descriptor")
def blind_descriptor(target: np.ndarray, sr: int = SR, deadband: float = 0.2
                     ) -> Callable[[np.ndarray], float]:
    """Read 78 numbers off each render and compare those, with a deadband.

    Why not a sum over bins. A sum over 1025 x 1542 bins casts 1.6 million votes to
    resolve 55 unknowns; the surplus is where the measured failure lives. So this reads a
    fixed, short list of quantities, each on the list because a synth control moves it:
    band levels (the EQ curve and where the filter sits), per-band dynamic range (the
    amp and filter envelopes, and the env amount), per-band fluctuation left after a
    200 ms trend is removed (detune beating and chorus depth, which a still spectrum
    averages away), centroid percentiles (the cutoff and how far it sweeps), envelope
    rise and fall rates (attack and release), and the slope of the tail (delay and reverb
    decay). Nothing else is represented, so nothing else can vote.

    Why a deadband. The requirement is not "small changes cost little", it is "changes
    nothing can hear cost nothing". Every feature is expressed in dB or in semitones,
    units in which a fixed threshold means the same thing everywhere, so subtracting
    `deadband` from each absolute difference and clamping at zero makes the score exactly
    flat in a neighbourhood of truth. Truth is then interior to the minimum by
    construction rather than by luck, which is precisely the property the incumbent was
    measured not to have. A perturbation 85 dB down moves these features by order 1e-5,
    four orders inside the threshold.

    Known cost, and intended: two patches that agree to within the deadband on all 78
    features tie at zero. They are indistinguishable by ear and the objective says so.
    A tie is a better answer than a confident wrong ordering.
    """
    n_fft, hop, n_b = 2048, 1024, 24
    fb = _logband_matrix(n_fft, n_b, sr)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    logf = np.log2(np.maximum(freqs, 20.0))
    smooth = int(round(0.2 * sr / hop)) | 1      # odd, ~200 ms, for the detrend
    kern = np.ones(smooth) / smooth
    n_tail = int(1.5 * sr / hop)

    def _features(x: np.ndarray, floor: float) -> np.ndarray:
        S = mags(x, n_fft, hop)
        bd = _db(fb @ S, floor)                                   # 24 bands x frames
        med = np.median(bd, axis=1)
        p10, p90 = np.percentile(bd, [10, 90], axis=1)
        if bd.shape[1] > 2 * smooth:
            trend = _smooth_rows(bd, kern)
            # edges dropped: "same" convolution tapers there, which would read as flutter
            fluct = (bd - trend)[:, smooth:-smooth].std(axis=1)
        else:
            fluct = np.zeros(n_b)
        w = S.sum(axis=0) + 1e-12
        cent = (logf @ S) / w                                     # log2 Hz, per frame
        cq = np.percentile(cent, [10, 50, 90]) * 12.0             # semitones
        e = env(x, hop)
        edb = _db(e, e.max() * 10.0 ** (FLOOR_DB / 20.0) + 1e-30)
        rate = np.percentile(np.diff(edb), [5, 95])               # dB per hop
        tail = edb[-n_tail:]
        t_ax = np.arange(tail.size) * hop / sr
        slope = float(np.polyfit(t_ax, tail, 1)[0]) * 0.1 if tail.size > 8 else 0.0
        return np.concatenate([med, p90 - p10, fluct, cq, rate, [slope]])

    floor = _floor_of(fb @ mags(target, n_fft, hop))
    ft = _features(target, floor)

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        d = np.abs(_features(p, floor) - ft)
        v = np.maximum(d - deadband, 0.0).mean()
        return float(v) if np.isfinite(v) else 1e6
    return score


# ---------------- 2. the envelopes' spectrum, not the signal's ----------------


@register("blind_modspec")
def blind_modspec(target: np.ndarray, sr: int = SR, rel_floor_db: float = -40.0
                  ) -> Callable[[np.ndarray], float]:
    """Compare modulation spectra: what each band's LEVEL does over time.

    A different domain, not a repaired version of the same one. Most of this patch's
    identity is rates and times rather than a still spectrum: detune shows up as beating
    at a few Hz, chorus as its LFO rate, the filter ADSR as the shape of every band's
    rise and fall, the delay as a comb along the modulation axis, the reverb as a lowpass
    on it. All of that is the Fourier transform of each band's envelope, which the
    incumbent never forms and can only infer from frame-by-frame disagreement.

    The bottleneck is the whole point: 16 bands x 80 modulation bins, Welch-averaged
    over overlapping 3 s segments with each segment's mean removed, then smeared along
    the modulation axis. Averaging across
    segments is the second reason an 85 dB-down perturbation cannot register: its
    contribution is incoherent between segments and averages towards zero, whereas a
    changed LFO rate is coherent and survives.

    Mean removal makes the modulation term blind to static level, which would leave the
    26 EQ gains unmeasured, so the per-band means ride along as the zero-frequency term.
    """
    n_fft, hop, n_b = 1024, 512, 16
    fb = _logband_matrix(n_fft, n_b, sr)
    seg, step = 256, 128                        # 3 s at the 86 Hz envelope rate
    win = np.hanning(seg)
    keep = 80                                   # modulation bins out to about 27 Hz
    # smear the modulation axis: a rate that moves by one bin should cost about what a
    # rate that moves by one bin sounds like, and a bare periodogram bin charges far more
    # than that for a sharp beating peak crossing a boundary. Measured on a 1 percent
    # detune error, which moves the 24th harmonic's beat by 1.5 bins: 6 s segments with
    # no smearing scored 1.91, these 3 s segments with it score 0.79, against 0.49 for a
    # 2x cutoff change. Still the most detune-sensitive of the four, deliberately.
    sm = np.hanning(5)[1:-1]
    sm /= sm.sum()

    def _parts(x: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
        bd = _db(fb @ mags(x, n_fft, hop), floor)                 # bands x frames, dB
        means = bd.mean(axis=1)
        acc = np.zeros((n_b, keep))
        cnt = 0
        for s in range(0, max(bd.shape[1] - seg, 0) + 1, step):
            f = bd[:, s:s + seg]
            f = (f - f.mean(axis=1, keepdims=True)) * win
            acc += np.abs(np.fft.rfft(f, axis=1)[:, 1:keep + 1]) ** 2
            cnt += 1
        return means, _smooth_rows(acc / max(cnt, 1), sm)

    floor = _floor_of(fb @ mags(target, n_fft, hop))
    mt, Pt = _parts(target, floor)
    ref = Pt.max(axis=1, keepdims=True) * 10.0 ** (rel_floor_db / 10.0) + 1e-30
    Lt = 10.0 * np.log10(np.maximum(Pt, ref))

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        mp, Pp = _parts(p, floor)
        mod = float(np.abs(10.0 * np.log10(np.maximum(Pp, ref)) - Lt).mean())
        static = float(np.abs(mp - mt).mean())
        v = mod + static
        return float(v) if np.isfinite(v) else 1e6
    return score


# ---------------- 3. one-sided: judge pred by the target's own vocabulary ----------------


@register("blind_explain")
def blind_explain(target: np.ndarray, sr: int = SR, k: int = 8
                  ) -> Callable[[np.ndarray], float]:
    """Not a metric and not symmetric: how well the target's own model explains a render.

    The target is the only thing there is evidence about, so it should set the terms of
    the comparison. Its log-band spectrogram is factored once into `k` orthonormal
    spectral shapes, the smallest set that reconstructs it. Those shapes are the only
    questions a candidate render is ever asked: project it onto them and compare the
    eight time-courses. Anything lying outside the span costs nothing, and a change 85 dB
    down lies almost entirely outside it, because the span was fitted to the loud
    structure and nothing else.

    The second term is a complexity check rather than a distance: how much of each signal
    its OWN rank-k picture fails to capture. A render smoother than the target (cutoff
    too low, detune too small) reconstructs better than the target does, a rougher one
    worse, and either way the discrepancy is evidence about parameters that the projected
    time-courses alone would miss. Subtracting the target's own residual is what makes a
    signal score exactly 0 against itself in spite of the truncation.

    Deliberately not a metric: score(a, b) != score(b, a), and no triangle inequality is
    claimed or wanted. Every screen downstream is a rank or an argmin, and asymmetry buys
    the thing a metric cannot have, which is a model of what matters.
    """
    n_fft, hop, n_b = 2048, 1024, 48
    fb = _logband_matrix(n_fft, n_b, sr)

    # a 5-tap Hann along time, about 120 ms. Measured: it improves the ratio between a
    # 2x cutoff change and an 85 dB-down perturbation by 3.5x, because the perturbation's
    # contribution is incoherent frame to frame and averages out while the cutoff's does
    # not. It does NOT reduce this term's sensitivity to beat PHASE, which remains its
    # weak point and the reason the other three candidates exist: on a 1 percent detune
    # error it still scores twice what a 2x cutoff error scores.
    tsm = np.hanning(7)[1:-1]
    tsm /= tsm.sum()

    def _X(x: np.ndarray, floor: float) -> np.ndarray:
        bd = _db(fb @ mags(x, n_fft, hop), floor)
        return _smooth_rows(bd, tsm)

    floor = _floor_of(fb @ mags(target, n_fft, hop))
    Xt = _X(target, floor)
    # eigen-decompose the n_b x n_b gram rather than SVD the full matrix: same subspace,
    # and the factory cost stops depending on clip length
    _, V = np.linalg.eigh(Xt @ Xt.T)
    U = np.ascontiguousarray(V[:, ::-1][:, :k])                  # n_b x k, orthonormal
    At = U.T @ Xt                                                # k x frames
    scale = np.abs(At).mean(axis=1)
    scale = np.maximum(scale, 0.02 * scale.max()) + 1e-9         # a near-dead component
    res_t = float(np.linalg.norm(Xt - U @ At) / (np.linalg.norm(Xt) + 1e-12))

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        Xp = _X(p, floor)
        n = min(Xp.shape[1], At.shape[1])
        Ap = U.T @ Xp[:, :n]
        traj = float((np.abs(Ap - At[:, :n]).mean(axis=1) / scale).mean())
        res_p = float(np.linalg.norm(Xp - U @ (U.T @ Xp)) / (np.linalg.norm(Xp) + 1e-12))
        v = traj + 10.0 * abs(res_p - res_t)
        return float(v) if np.isfinite(v) else 1e6
    return score


# ---------------- 4. a two-sample test on frames, with time thrown away ----------------


@register("blind_bag")
def blind_bag(target: np.ndarray, sr: int = SR, n_dir: int = 64, seed: int = 0
              ) -> Callable[[np.ndarray], float]:
    """Could one patch have produced both renders? A two-sample test, order ignored.

    The question worth answering here is not "do these line up frame by frame" but "do
    they come from the same generator". That is a two-sample test, and a test needs no
    time axis. The note list is frozen upstream, so time order is not a free parameter of
    the search; insisting on it only imports alignment fragility, and with free-running
    oscillators the fine structure is not a function of the parameters anyway.

    So each render becomes a bag of 40-dimensional frames, 20 log-band levels in dB plus
    their first differences, and the two bags are compared by sliced Wasserstein-1:
    project onto a fixed set of random directions, sort each projection, take the mean
    absolute gap. The first differences are what keep attack and release visible once
    order is gone, since a fast attack means large positive jumps exist in the bag
    regardless of when they happened.

    Sorting is what buys the robustness. An 85 dB-down perturbation permutes almost
    nothing and displaces each projected value by ~1e-5 dB, and the score is LINEAR in
    that displacement rather than logarithmic in a ratio, which is the whole difference
    between 1e-5 and the incumbent's +0.194.

    Blind spot, stated up front rather than discovered later: a render made of the right
    frames in the wrong order scores 0. Correct for this corpus, wrong for one where the
    notes are being fitted too.
    """
    n_fft, hop, n_b = 2048, 1024, 20
    fb = _logband_matrix(n_fft, n_b, sr)
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((n_dir, 2 * n_b))
    D /= np.linalg.norm(D, axis=1, keepdims=True)

    def _proj(x: np.ndarray, floor: float) -> np.ndarray:
        bd = _db(fb @ mags(x, n_fft, hop), floor)
        d = np.diff(bd, axis=1, prepend=bd[:, :1])
        return np.sort(D @ np.vstack([bd, d]), axis=1)

    floor = _floor_of(fb @ mags(target, n_fft, hop))
    Pt = _proj(target, floor)
    scale = float(Pt.std()) + 1e-9               # only so the number reads O(1)

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        Pp = _proj(p, floor)
        # match() equalises length, so the two bags have the same count and the sorted
        # rows pair off directly; min() only guards a ragged final frame
        n = min(Pp.shape[1], Pt.shape[1])
        v = float(np.abs(Pp[:, :n] - Pt[:, :n]).mean()) / scale
        return float(v) if np.isfinite(v) else 1e6
    return score
