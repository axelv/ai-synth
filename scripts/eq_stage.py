"""Absolute-frequency peaking-EQ stage: the shaping the current filter cannot reach.

Why this exists. The oracle ladder said the recoverable error in the render is a
STATIC spectral envelope, and the curve it wants is non-monotonic: +7.1 dB at
250-900 Hz, +3.3 at 900-2000, -4.6 at 2000-6000, +2.1 at 6000-16000. fi.resonlp +
fi.lowpass(2) + a one-pole tilt is monotonic above its cutoff at every setting, so
that shape is outside the reachable set. 7500 CMA-ES renders moving the loss by 0.012
is what that looks like from the inside.

Why peaking bands at FIXED centres. Cascaded second-order sections multiply, so in dB
the cascade response is the exact SUM of the individual band responses with no
interaction term: three bands at +6 dB measure 1.8e-06 dB from the sum of the three
taken alone. Fixed centres therefore make the gains a linear problem, which is why
fit_gains is a least-squares solve and not a search. And a fixed log-spaced grid of
peaking gains is just an EQ curve, so the result loads into any DAW instead of being a
private parameterisation.

Why fi.svf.bell and not the fi.peak_eq the brief suggested. peak_eq is unusable here,
and not for a subtle reason: ONE flat peak_eq band at 40 Hz costs 0.69 of loss (2.2522
against 1.5605) while leaving cos theta untouched at 0.7515, i.e. it adds junk without
changing the spectrum's shape. At Lfx=0 its numerator and denominator polynomials are
equal in exact arithmetic, but they are computed through different expressions in
float32, and at 40 Hz the pole radius is 0.998, so the surviving mismatch is amplified
by 1/(1-r)^2 and leaves a real -52 dB resonance. That lands in the near-empty bins
between the partials, which is exactly where the log-magnitude term does its work, so
the loss sees it at three times the size of the whole prize. It also passes an impulse
test at -90 dB and only fails on sustained input, which is why check_identity uses
2.5 s of the real render and not just an impulse. peak_eq_rm is worse: read its
routing and the Lfx=0 case is the bare allpass rather than the identity, which at
40 Hz smears phase over hundreds of ms (2.39 flat). fi.svf.bell is a TPT
state-variable bell whose output mix is `v0 + k*(A*A-1)*v1`, so at 0 dB the second
term is multiplied by exactly zero and the identity is structural rather than
numerical: measured bit-exact, and -113 dB of float32 junk against a float64
convolution of its own measured impulse response even with all 26 bands driven to
+-18 dB. That is check_numerics, and it is the first thing to run if anyone swaps the
primitive again.

Why 26 bands, when the brief expected 9 to 12. Measured with the gains optimised
straight against WindowScore and the real cascade in the loop, so each number is that
band set's own ceiling (select_bands, out/eq_select.json), at Q 1.4:

    bands    8      10      12      15      20      26      32
    score  1.4167  1.3678  1.3571  1.3340  1.3026  1.2978  1.2970

against 1.5605 with no EQ. It plateaus at 26: the last six bands buy 0.0008. Twelve
bands cannot reach the 1.3424 that has to be beaten at any gain setting, and 26 can.
26 over 40-16000 Hz is 2.89 bands/octave, which is also the resolution the
partial-amplitude fit independently picked out: 3 bands/octave, 34 parameters, 4.20 dB
residual where a 177-parameter wavetable left 4.47 dB. Two unrelated measurements
landing on third-octave is the reason to stop there rather than at 32.

Why Q 2.8. Scanned at 26 bands (out/eq_select.json). The reachable score plateaus in
the same place the bank stops being redundant: Q 0.7 reaches 1.3356, then 1.0 -> 1.3095,
1.4 -> 1.2978, 2.0 -> 1.2868, 2.8 -> 1.2859. Q 2.8 gives each bell a measured 0.508
octave bandwidth against the 0.346 octave spacing, i.e. 1.5x overlap, and that is what
makes the design matrix well conditioned: condition number 6.6, against 15 at Q 2.0
and 51 at Q 1.4. A well-conditioned bank matters twice over, because the near-null
directions of a redundant bank are also flat directions for whatever search fits the
patch later. It costs almost nothing on the round trip (0.37 dB rms on the plateaus
against 0.30 dB at the best Q).

One correction to the brief. The four-step curve above is a summary of the fitted
third-octave EQ, not that EQ. Realised by this stage it takes the render from 1.5605
to 1.4938 and cos theta from 0.7516 to 0.7855, which is real but is a quarter of what
the same 26 bands reach when they are fitted to the loss instead (1.2859). Applied as
an ideal zero-phase gain curve it is worse still, 1.9379, because its own broadband
level is 3 dB hot and the loss prefers level = ratio x cos theta. So check_round_trip
fits it as an expressiveness test, which is what the brief asks for, but it is the
wrong thing to aim the synth at; out/eq_warm_start.json holds gains that are not.

Everything above was measured by rendering through real Faust. The two facts that
decided the design, peak_eq's arithmetic and where the band count stops paying, are
both invisible to an analytic filter model.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.signal import fftconvolve

import faust_probe as fp

SR = 44100
F_LO = 40.0            # below the F1 fundamental at 43.65 Hz
F_HI = 16000.0         # top of the band the oracle curve is defined over
N_BANDS = 26           # third-octave spacing; set by select_bands()
BAND_Q = 2.8           # set by the Q scan at that spacing
GAIN_LIMIT = 18.0

IMP_N = 32768          # 0.74 s, long enough for the 40 Hz band's tail to reach -300 dB
BASIS_GAIN = 6.0       # reference gain the basis is measured at
BASIS_PATH = "out/eq_basis.npz"
WARM_PATH = "out/eq_warm_start.json"

# The oracle ladder's curve. Piecewise constant in absolute frequency, and undefined
# below 250 Hz, because the ladder never constrained that region.
TARGET_STEPS = ((250.0, 900.0, 7.1), (900.0, 2000.0, 3.3),
                (2000.0, 6000.0, -4.6), (6000.0, 16000.0, 2.1))


# ---------------------------------------------------------------- the stage itself

def band_freqs(n: int = N_BANDS, lo: float = F_LO, hi: float = F_HI) -> np.ndarray:
    return np.geomspace(lo, hi, n)


def faust_source(n: int = N_BANDS, q: float = BAND_Q, name: str = "eqCurve",
                 lo: float = F_LO, hi: float = F_HI) -> str:
    """One mono peaking-EQ cascade plus its sliders, for `par(i, 2, eqCurve)`.

    Every gain defaults to 0 dB, which is the exact identity, bit for bit, so
    appending these parameters leaves an existing patch rendering unchanged: the same
    contract drive and spread were added under.
    """
    f = band_freqs(n, lo, hi)
    sliders = "\n".join(
        f'eq{i} = hslider("eq{i}", 0, {-GAIN_LIMIT:g}, {GAIN_LIMIT:g}, 0.001);'
        f'   // {f[i]:.0f} Hz' for i in range(n))
    chain = " : ".join(f"fi.svf.bell({f[i]:.4f}, {q:g}, eq{i})" for i in range(n))
    return f"{sliders}\n{name} = _ : {chain};\n"


EQ_FAUST = faust_source()


def param_specs(n: int = N_BANDS) -> tuple[tuple[str, float, float, float], ...]:
    """(name, lo, hi, default) per band, to be appended to synth.PAD_PARAMS."""
    return tuple((f"eq{i}", -GAIN_LIMIT, GAIN_LIMIT, 0.0) for i in range(n))


def gain_dict(gains) -> dict[str, float]:
    return {f"eq{i}": float(v) for i, v in enumerate(np.asarray(gains, float))}


# ---------------------------------------------------------------- measurement

def _probe_dsp(n: int, q: float, lo: float = F_LO, hi: float = F_HI) -> str:
    return ('import("stdfaust.lib");\n' + faust_source(n, q, "eqCurve", lo, hi)
            + "process = eqCurve;\n")


class _FxEngine:
    """faust_probe.render_fx with the engine kept, so one DSP can be re-rendered.

    render_fx builds a fresh engine per call at 0.28 s; the same render on a reused
    engine costs 5 ms. That is the difference between fitting the gains against real
    Faust and fitting them against a model of Faust, so it is worth the extra class.
    Same graph, same processor: check_engine measures that the two agree bit for bit
    rather than assuming it.
    """

    def __init__(self, dsp: str, x: np.ndarray, sr: int = SR) -> None:
        import dawdreamer as daw
        sig = np.asarray(x, dtype=np.float32)
        sig = sig[None, :] if sig.ndim == 1 else sig
        self.dur = sig.shape[1] / sr
        self.engine = daw.RenderEngine(sr, fp.BLOCK)
        play = self.engine.make_playback_processor("src", sig)
        self.proc = self.engine.make_faust_processor("eq")
        self.proc.faust_libraries_path = fp.FAUST_LIBS
        if not self.proc.set_dsp_string(dsp):
            raise RuntimeError("faust compile failed:\n" + dsp)
        self.pidx = {d["label"]: d["index"]
                     for d in self.proc.get_parameters_description()}
        self.engine.load_graph([(play, []), (self.proc, ["src"])])

    def render(self, params: dict[str, float]) -> np.ndarray:
        for name, val in params.items():
            self.proc.set_parameter(self.pidx[name], float(val))
        self.engine.render(self.dur)
        return self.engine.get_audio()


_ENGINES: dict[tuple, _FxEngine] = {}


def _engine(dsp: str, x: np.ndarray, key: tuple) -> _FxEngine:
    if key not in _ENGINES:
        _ENGINES[key] = _FxEngine(dsp, x)
    return _ENGINES[key]


def impulse_response(gains, q: float = BAND_Q, lo: float = F_LO, hi: float = F_HI,
                     n_imp: int = IMP_N) -> np.ndarray:
    g = np.asarray(gains, dtype=float)
    e = _engine(_probe_dsp(len(g), q, lo, hi), fp.impulse(n_imp),
                ("imp", len(g), q, lo, hi, n_imp))
    return e.render(gain_dict(g))[0].astype(np.float64)


def response(gains, q: float = BAND_Q, lo: float = F_LO, hi: float = F_HI,
             n_imp: int = IMP_N) -> tuple[np.ndarray, np.ndarray]:
    """Measured dB magnitude response of the cascade at these gains.

    Impulse rather than swept sine: for an LTI cascade both give the same transfer
    function, and the impulse needs no deconvolution, so nothing but the filter is in
    the answer. render_fx's graph adds no latency (the impulse returns at sample 0),
    and magnitude is blind to latency anyway.
    """
    y = impulse_response(gains, q, lo, hi, n_imp)
    return (np.fft.rfftfreq(n_imp, 1.0 / SR),
            20.0 * np.log10(np.abs(np.fft.rfft(y)) + 1e-300))


def sweep_response(gains, q: float = BAND_Q, n: int = 1 << 19
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Same thing from an exponential sweep, by dividing spectra.

    An independent check on `response`: it drives the filter with sustained broadband
    signal instead of one sample, so anything that only misbehaves under load shows up
    here. That is not hypothetical, it is how peak_eq's float32 defect was found, and
    it is invisible to the impulse. Only valid inside the sweep's 20 Hz to 20 kHz
    span; 12 s of sweep leaves ~50 usable bins per third octave at 250 Hz.
    """
    x = fp.sweep(n)
    g = np.asarray(gains, dtype=float)
    y = fp.render_fx(_probe_dsp(len(g), q), gain_dict(g), x,
                     tail=0.2)[0].astype(np.float64)
    X = np.fft.rfft(np.concatenate([x, np.zeros(len(y) - len(x))]))
    Y = np.fft.rfft(y)
    return (np.fft.rfftfreq(len(y), 1.0 / SR),
            20.0 * np.log10(np.abs(Y) / (np.abs(X) + 1e-30) + 1e-300))


def measure_basis(n: int = N_BANDS, q: float = BAND_Q, lo: float = F_LO,
                  hi: float = F_HI, gain: float = BASIS_GAIN,
                  n_imp: int = IMP_N) -> dict[str, np.ndarray]:
    """Each band's realised response at +gain dB with the others flat.

    Measured through the full cascade rather than a lone band, so that "the others
    flat" is tested rather than asserted.
    """
    f = np.fft.rfftfreq(n_imp, 1.0 / SR)
    rows = np.empty((n, len(f)))
    for i in range(n):
        g = np.zeros(n)
        g[i] = gain
        rows[i] = response(g, q, lo, hi, n_imp)[1]
    return {"f": f, "basis_db": rows, "freqs": band_freqs(n, lo, hi),
            "q": np.array(float(q)), "gain": np.array(float(gain))}


def save_basis(basis: dict[str, np.ndarray], path: str = BASIS_PATH) -> str:
    np.savez_compressed(path, **basis)
    return path


def load_basis(path: str = BASIS_PATH) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def default_basis(n: int = N_BANDS, q: float = BAND_Q,
                  path: str = BASIS_PATH) -> dict[str, np.ndarray]:
    """The cached basis if it is the one being asked for, otherwise measure and cache.

    A caller with a curve to realise should not have to know that the design matrix
    came out of 26 Faust renders, but it also must not silently use a basis measured
    for a different band set, so the cache is checked against n and q rather than
    trusted.
    """
    try:
        b = load_basis(path)
        if len(b["freqs"]) == n and float(b["q"]) == float(q):
            return b
    except FileNotFoundError:
        pass
    b = measure_basis(n, q)
    save_basis(b, path)
    return b


# ---------------------------------------------------------------- fitting

def _design(f_target: np.ndarray, basis: dict[str, np.ndarray]) -> np.ndarray:
    """Band responses per dB of gain, resampled onto the target frequencies.

    Interpolated in log frequency because the bands are log-spaced while the FFT grid
    is linear, so below a few hundred Hz there are few bins per octave.
    """
    fb, B = basis["f"], basis["basis_db"]
    ok = fb > 0
    lf, lt = np.log(fb[ok]), np.log(np.asarray(f_target, dtype=float))
    cols = np.stack([np.interp(lt, lf, b[ok]) for b in B], axis=1)
    return cols / float(basis["gain"])


def fit_gains(f_target: np.ndarray, g_target: np.ndarray,
              basis: dict[str, np.ndarray] | None = None,
              weights: np.ndarray | None = None,
              ridge: float = 1e-2, free_offset: bool = False,
              limit: float = GAIN_LIMIT,
              ridge_center: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Band gains in dB that best realise a desired log-gain curve.

    THE entry point for a caller with a curve. f_target/g_target may be sampled
    anywhere: partial frequencies, third-octave centres, an FFT grid. Bounded least
    squares, because a gain outside the slider range is not realisable and an
    unbounded solve will happily ask for one.

    ridge penalises mean squared gain on the same normalised scale as the curve
    residual. Its job is not accuracy but sanity: overlapping bands leave the solve
    nearly singular in the alternating direction, and a +18/-18 sawtooth that averages
    out to the right curve would sit where the bell shape is least gain-independent.

    free_offset adds a frequency-independent term, for callers who can absorb a
    broadband level in outGain. Peaking bands cannot produce one. It is not shrunk by
    the ridge: a level is not a suspicious thing for the fit to want.

    ridge_center is what the ridge pulls towards, and exists for refine_gains: when
    the solve returns a correction rather than a gain vector, the penalty still has to
    apply to the resulting total. Pulling the correction towards zero instead lets
    each iteration undo the last one's regularisation, and the gains then walk out to
    the bounds.
    """
    t = np.asarray(g_target, dtype=float)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    # a curve is allowed to be undefined somewhere, e.g. brief_curve below 250 Hz,
    # and those points must drop out rather than poison the solve
    keep = np.isfinite(t) & np.isfinite(w)
    basis = default_basis() if basis is None else basis
    A = _design(np.asarray(f_target, dtype=float)[keep], basis)
    t, w = t[keep], w[keep] / (w[keep].mean() + 1e-30)
    n = A.shape[1]
    if free_offset:
        A = np.hstack([A, np.ones((len(t), 1))])
    m, k = A.shape
    sw = np.sqrt(w / m)[:, None]
    A_, t_ = A * sw, t * sw[:, 0]
    if ridge > 0.0:
        c = np.zeros(n) if ridge_center is None else np.asarray(ridge_center, float)
        R = np.zeros((n, k))
        R[:, :n] = np.sqrt(ridge / n) * np.eye(n)
        A_ = np.vstack([A_, R])
        t_ = np.concatenate([t_, np.sqrt(ridge / n) * c])
    lo, hi = np.full(k, -limit), np.full(k, limit)
    if free_offset:
        lo[-1], hi[-1] = -60.0, 60.0
    sol = lsq_linear(A_, t_, bounds=(lo, hi)).x
    return sol[:n], (float(sol[-1]) if free_offset else 0.0)


def refine_gains(f_target: np.ndarray, g_target: np.ndarray,
                 basis: dict[str, np.ndarray] | None, gains: np.ndarray,
                 weights: np.ndarray | None = None, ridge: float = 1e-2,
                 iters: int = 3, limit: float = GAIN_LIMIT) -> np.ndarray:
    """Gauss-Newton on top of the linear solve, with Faust as the forward model.

    Needed because a bell's shape is not exactly proportional to its own gain: the
    svf bell's damping carries a 1/A term, so a basis measured at +6 dB does not
    describe a band sitting at +12. The basis is still a usable Jacobian (its error is
    second order in the gain), so one measured residual per step goes a long way.

    The step is backtracked against the MEASURED residual rather than trusted. On a
    target the cascade cannot reach, an undamped step chases structure the basis
    cannot represent and ends up worse than not refining at all. Each trial is one
    impulse render on the cached engine, so backtracking is nearly free.
    """
    basis = default_basis() if basis is None else basis
    q, lo, hi = float(basis["q"]), float(basis["freqs"][0]), float(basis["freqs"][-1])
    t = np.asarray(g_target, dtype=float)
    lt = np.log(np.asarray(f_target, dtype=float))
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)

    def realised(v: np.ndarray) -> np.ndarray:
        fr, db = response(v, q, lo, hi)
        return np.interp(lt, np.log(fr[1:]), db[1:])

    ok = np.isfinite(t)

    def cost(v: np.ndarray) -> float:
        r = (realised(v) - t)[ok]
        return float((w[ok] * r * r).sum() / w[ok].sum() + ridge * (v * v).mean())

    g = np.clip(np.asarray(gains, dtype=float), -limit, limit)
    best = cost(g)
    for _ in range(iters):
        d, _ = fit_gains(f_target, t - realised(g), basis, weights, ridge=ridge,
                         limit=2 * limit, ridge_center=-g)
        for step in (1.0, 0.5, 0.25, 0.125):
            cand = np.clip(g + step * d, -limit, limit)
            c = cost(cand)
            if c < best:
                g, best = cand, c
                break
        else:
            break
    return g


def brief_curve(f: np.ndarray) -> np.ndarray:
    """TARGET_STEPS sampled at f; NaN where the oracle ladder said nothing."""
    f = np.asarray(f, dtype=float)
    out = np.full(len(f), np.nan)
    for f0, f1, g in TARGET_STEPS:
        out[(f >= f0) & (f < f1)] = g
    out[f == TARGET_STEPS[-1][1]] = TARGET_STEPS[-1][2]
    return out


# ---------------------------------------------------------------- on real material

def _score():
    import chord
    global _SCORE
    if _SCORE is None:
        _SCORE = chord.WindowScore()
    return _SCORE


_SCORE = None


def render_window(patch_path: str = "out/patch.json") -> np.ndarray:
    """The current synth over the chord window, mono. Imported lazily because the
    fitting half of this module has no business pulling in torch."""
    import chord
    import synth
    from bend2 import bend_curve
    r = synth.PadRenderer(n_voices=24)
    r.set_notes(chord.notes_upto())
    r.set_params(json.load(open(patch_path))["params"])
    r.set_bend(bend_curve(int(chord.WIN_T1 * SR) + SR))
    return r.render(chord.WIN_T1)[:, chord.win_slice()].mean(0)


def eq_window(win: np.ndarray, gains, q: float = BAND_Q, lo: float = F_LO,
              hi: float = F_HI) -> np.ndarray:
    """`win` through the real cascade. Authoritative: the audio comes out of Faust.

    The cache key hashes the SIGNAL, not just its length. _FxEngine bakes the playback
    buffer in at construction, so a key of (len(gains), q, lo, hi, len(win)) silently
    returns the first signal of that length filtered, for every later signal of the
    same length. That is invisible when the window is fixed, which is how the gain fits
    use this, and it is wrong the moment a caller sweeps a synth parameter and re-renders:
    it made sixteen different filter settings report a loss identical to six decimals.
    Hashing 18 s of audio costs about as much as the render it protects, which is the
    right trade for a function whose whole purpose is to be authoritative.
    """
    x = np.ascontiguousarray(win, dtype=np.float32)
    tag = hashlib.blake2b(x.tobytes(), digest_size=16).hexdigest()
    e = _engine(_probe_dsp(len(gains), q, lo, hi), x,
                ("win", len(gains), q, lo, hi, len(win), tag))
    return e.render(gain_dict(gains))[0].astype(np.float64)


def score_gains(win: np.ndarray, gains, q: float = BAND_Q, lo: float = F_LO,
                hi: float = F_HI) -> tuple[float, float]:
    """WindowScore and cos theta of `win` put through the cascade at these gains.

    Cheap enough to sit inside an optimiser because the engine is cached, so there is
    no reason to fit against a surrogate. One was tried and abandoned: superposing the
    measured band curves and applying them zero-phase agreed with Faust to 0.6 dB mean
    over every bin carrying signal, and still mispredicted the loss by 0.92, because
    the two differed by 5 dB in the near-empty bins BETWEEN the partials. That was
    peak_eq's float32 noise showing up as a model error, and it is the same trap the
    pure-tone additive bank fell into.
    """
    y = eq_window(win, gains, q, lo, hi)
    sc = _score()
    return sc(y), sc.cos_theta(y)


# ---------------------------------------------------------------- verification

def _err(got: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    e = np.abs(np.asarray(got) - np.asarray(want))
    return float(e.max()), float(np.sqrt((e ** 2).mean()))


def check_engine(n: int = N_BANDS, q: float = BAND_Q) -> dict[str, float]:
    """The cached engine against faust_probe.render_fx, on identical input."""
    g = np.linspace(-9.0, 9.0, n)
    a = impulse_response(g, q)
    b = fp.render_fx(_probe_dsp(n, q), gain_dict(g),
                     fp.impulse(IMP_N))[0].astype(np.float64)
    return {"max_sample_diff": float(np.abs(a - b).max())}


def check_stereo(n: int = N_BANDS, q: float = BAND_Q) -> dict[str, float]:
    """The fragment used the way synth.py's effect chain will use it.

    tiltEQ sits in `par(i, 2, tiltEQ)`, so the fragment has to compile as one mono
    curve applied to both channels off the same sliders. Compiling it here means the
    integrator finds out now rather than after wiring it in, and it also checks that
    the two channels come out matched instead of one of them picking up the other's
    gain through some routing accident.
    """
    dsp = ('import("stdfaust.lib");\n' + faust_source(n, q)
           + "process = _,_ : par(i, 2, eqCurve);\n")
    g = np.linspace(-9.0, 9.0, n)
    x = fp.impulse(IMP_N, channels=2)
    y = fp.render_fx(dsp, gain_dict(g), x)
    mono = impulse_response(g, q)
    return {"channels": float(y.shape[0]),
            "l_minus_r_max": float(np.abs(y[0] - y[1]).max()),
            "vs_mono_max": float(np.abs(y[0] - mono).max())}


def check_identity(n: int = N_BANDS, q: float = BAND_Q,
                   win: np.ndarray | None = None) -> dict[str, float]:
    """Flat is the identity, on an impulse and on 2.5 s of the real render.

    The sustained test is not redundant. peak_eq passed the impulse test at -90 dB and
    failed this one at -52 dB, because its error needs a long input to build up.
    """
    d = impulse_response(np.zeros(n), q)
    d[0] -= 1.0
    f, db = response(np.zeros(n), q)
    band = (f >= F_LO) & (f <= F_HI)
    win = render_window() if win is None else win
    w32 = np.asarray(win, np.float32).astype(np.float64)
    flat = eq_window(win, np.zeros(n), q)
    r = flat - w32
    # linear ratios rather than dB, because these come out at exactly zero and
    # -6000 dB from a log of 1e-300 reads like a measurement when it is a floor
    return {"impulse_residual_max": float(np.abs(d).max()),
            "max_response_dev_db": float(np.abs(db[band]).max()),
            "sustained_residual_rel": float(np.sqrt((r ** 2).mean())
                                            / np.sqrt((w32 ** 2).mean())),
            "score_flat": _score()(flat), "score_input": _score()(w32)}


def check_numerics(n: int = N_BANDS, q: float = BAND_Q, amps=(6.0, 12.0, 18.0),
                   win: np.ndarray | None = None) -> list[dict]:
    """Is the cascade's output the LTI filtering it claims to be?

    Compares Faust against a float64 convolution of the cascade's OWN measured impulse
    response, so the only thing being tested is arithmetic. This is the check that
    rejected peak_eq, and the first one to run if the primitive is ever swapped.
    """
    win = render_window() if win is None else win
    w32 = np.asarray(win, np.float32).astype(np.float64)
    rng = np.random.default_rng(0)
    rows = []
    for amp in amps:
        g = rng.uniform(-amp, amp, n)
        ir = impulse_response(g, q)
        y = eq_window(win, g, q)
        ref = fftconvolve(w32, ir)[:len(w32)]
        r = y - ref
        rows.append({
            "gain_spread_db": amp,
            "junk_db_rel": 20.0 * np.log10(np.sqrt((r ** 2).mean())
                                           / np.sqrt((y ** 2).mean()) + 1e-300),
            "ir_tail_db": 20.0 * np.log10(np.abs(ir[-2000:]).max() + 1e-300),
            "faust_score": _score()(y), "float64_score": _score()(ref)})
    return rows


def check_linearity(n: int = N_BANDS, q: float = BAND_Q,
                    gains=(1.5, 3.0, 6.0, 9.0, 12.0, 18.0)) -> dict:
    """Does gain g give g/6 times the +6 dB response, and do bands superpose?

    The first question is whether the least-squares solve is valid on its own; the
    second is whether the cascade really adds in dB, which is what makes the design
    matrix a fixed basis rather than a function of the operating point.
    """
    f = np.fft.rfftfreq(IMP_N, 1.0 / SR)
    band = (f >= F_LO) & (f <= F_HI)
    mid = n // 2
    ref = np.zeros(n)
    ref[mid] = BASIS_GAIN
    r6 = response(ref, q)[1]
    rows = []
    for g in gains:
        v = np.zeros(n)
        v[mid] = g
        mx, rms = _err(response(v, q)[1][band], (r6 * g / BASIS_GAIN)[band])
        v[mid] = -g
        mxc, _ = _err(response(v, q)[1][band], (-r6 * g / BASIS_GAIN)[band])
        rows.append({"gain_db": g, "max_dev_db": mx, "rms_dev_db": rms,
                     "cut_max_dev_db": mxc})
    idx = (1, mid, n - 2)
    three = np.zeros(n)
    sm = np.zeros(len(f))
    for i in idx:
        three[i] = BASIS_GAIN
        v = np.zeros(n)
        v[i] = BASIS_GAIN
        sm += response(v, q)[1]
    mx, rms = _err(response(three, q)[1][band], sm[band])
    return {"per_gain": rows, "superposition_max_db": mx,
            "superposition_rms_db": rms}


def check_round_trip(n: int = N_BANDS, q: float = BAND_Q, npts: int = 400,
                     refine: int = 3, basis: dict | None = None) -> dict:
    """Fit the brief's curve, render the result, and score the realisation.

    Reported from the impulse response and independently from the swept sine, because
    a stage that only behaves under an impulse is not a stage.

    Two error figures. `max` is over the whole 250-16000 Hz grid and is dominated by
    the step discontinuities at 900, 2000 and 6000 Hz, which no finite-Q filter of any
    order reaches. `max_flat` excludes a sixth of an octave either side of each step,
    so it asks how well the plateaus themselves are hit.
    """
    basis = measure_basis(n, q) if basis is None else basis
    ft = np.geomspace(250.0, F_HI, npts)
    want = brief_curve(ft)
    g0, _ = fit_gains(ft, want, basis)
    g = refine_gains(ft, want, basis, g0, iters=refine) if refine else g0

    flat = np.ones(len(ft), bool)
    for e in [f1 for _, f1, _ in TARGET_STEPS[:-1]] + [250.0]:
        flat &= (ft < e / 2 ** (1 / 6)) | (ft > e * 2 ** (1 / 6))

    out: dict = {}
    for tag, (fr, db) in (("impulse", response(g, q)), ("sweep", sweep_response(g, q))):
        got = np.interp(np.log(ft), np.log(fr[1:]), db[1:])
        out[f"{tag}_max_db"], out[f"{tag}_rms_db"] = _err(got, want)
        out[f"{tag}_max_flat_db"], out[f"{tag}_rms_flat_db"] = _err(got[flat], want[flat])
    fr0, db0 = response(g0, q)
    out["linear_only_rms_db"] = _err(
        np.interp(np.log(ft), np.log(fr0[1:]), db0[1:]), want)[1]
    out["max_abs_gain_db"] = float(np.abs(g).max())
    out["gains"] = [float(v) for v in g]
    return out


def select_bands(counts=(8, 10, 12, 15, 20, 26, 32), qs=(1.4,),
                 maxfev: int = 6000, path: str | None = WARM_PATH) -> list[dict]:
    """Choose the band count by measuring what each one can actually reach.

    Per configuration, the gains are optimised straight against WindowScore with the
    real cascade in the loop, so the number reported is that band set's ceiling: the
    best it could do with the rest of the synth left exactly as fitted. That is a
    measurement of the reachable set, NOT a result. A fitted EQ bolted onto a finished
    render is what the brief rules out as an answer, and refitting the synth around
    the stage is the next agent's job. For scale, the render with no EQ scores 1.5605
    and the documented number to beat is 1.3424.

    Powell rather than CMA-ES: a static gain curve is nearly separable across bands,
    each gain owning its own piece of the spectrum, which is the one situation where
    coordinate-wise line search is the efficient choice.
    """
    win = render_window()
    sc = _score()
    rows = [{"n": 0, "q": 0.0, "score": sc(win), "cos": sc.cos_theta(win),
             "max_abs_gain_db": 0.0, "nfev": 0, "gains": []}]
    for q in qs:
        for n in counts:
            r = minimize(lambda v: score_gains(win, v, q)[0], np.zeros(n),
                         method="Powell", bounds=[(-GAIN_LIMIT, GAIN_LIMIT)] * n,
                         options={"maxfev": maxfev, "xtol": 1e-2, "ftol": 1e-5})
            rows.append({"n": n, "q": q, "score": float(r.fun),
                         "cos": score_gains(win, r.x, q)[1],
                         "max_abs_gain_db": float(np.abs(r.x).max()),
                         "nfev": int(r.nfev), "gains": [float(v) for v in r.x]})
            print(f"  n={n:3d} q={q:4.1f} score {r.fun:.4f} "
                  f"cos {rows[-1]['cos']:.4f} |g|max {rows[-1]['max_abs_gain_db']:5.2f} "
                  f"nfev {r.nfev}")
    if path is not None:
        with open(path, "w") as fh:
            json.dump(min(rows[1:], key=lambda d: d["score"]), fh, indent=2)
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true",
                    help="sweep band count and Q against the loss, and pick")
    args = ap.parse_args()

    if args.select:
        rows = select_bands()
        with open("out/eq_select.json", "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote out/eq_select.json and {WARM_PATH}")
        return

    print(f"stage: {N_BANDS} bells, Q {BAND_Q}, {F_LO:.0f}-{F_HI:.0f} Hz "
          f"({(N_BANDS - 1) / np.log2(F_HI / F_LO):.2f} bands/octave)")
    print("centres " + " ".join(f"{f:.0f}" for f in band_freqs()))

    eng = check_engine()
    print(f"\ncached engine vs faust_probe.render_fx: max sample diff "
          f"{eng['max_sample_diff']:.1e}")
    st = check_stereo()
    print(f"in par(i, 2, eqCurve): {int(st['channels'])} channels, L-R "
          f"{st['l_minus_r_max']:.1e}, vs the mono chain {st['vs_mono_max']:.1e}")

    win = render_window()
    ident = check_identity(win=win)
    print(f"flat = identity: impulse residual {ident['impulse_residual_max']:.1e}, "
          f"response deviation {ident['max_response_dev_db']:.1e} dB, "
          f"2.5 s of render {ident['sustained_residual_rel']:.1e} relative "
          f"(requirement -80 dB, i.e. 1e-4)")
    print(f"  score through the flat cascade {ident['score_flat']:.4f} "
          f"vs {ident['score_input']:.4f} for its own input")

    print("\nis it LTI in float32 (Faust vs float64 convolution of its own IR)")
    for r in check_numerics(win=win):
        print(f"  gains +-{r['gain_spread_db']:4.1f} dB: junk {r['junk_db_rel']:7.1f} dB, "
              f"IR tail {r['ir_tail_db']:6.0f} dB, score {r['faust_score']:.4f} "
              f"vs {r['float64_score']:.4f}")

    lin = check_linearity()
    print("\nlinearity in dB (deviation from g/6 x the +6 dB response)")
    for r in lin["per_gain"]:
        print(f"  {r['gain_db']:+6.1f} dB  max {r['max_dev_db']:6.3f}  "
              f"rms {r['rms_dev_db']:6.3f}  cut {r['cut_max_dev_db']:6.3f}")
    print(f"  3 bands at +6 dB vs the sum of the three alone: "
          f"max {lin['superposition_max_db']:.1e} dB")

    basis = measure_basis()
    rt = check_round_trip(basis=basis)
    print("\nround trip on the brief's curve (+7.1/+3.3/-4.6/+2.1)")
    for tag in ("impulse", "sweep"):
        print(f"  {tag:>7}: max {rt[f'{tag}_max_db']:5.2f} dB  "
              f"rms {rt[f'{tag}_rms_db']:5.2f} dB   plateaus only: "
              f"max {rt[f'{tag}_max_flat_db']:5.2f}  rms {rt[f'{tag}_rms_flat_db']:5.2f}")
    print(f"  linear solve alone, no refinement: rms {rt['linear_only_rms_db']:.2f} dB")
    print(f"  largest band gain {rt['max_abs_gain_db']:.2f} dB")
    s, c = score_gains(win, rt["gains"])
    print(f"  that curve on the render: score {s:.4f} cos {c:.4f} "
          f"(flat {ident['score_flat']:.4f}/{_score().cos_theta(win):.4f})")

    path = save_basis(basis | {"round_trip_gains": np.array(rt["gains"])})
    print(f"\nwrote {path}")
    with open("out/eq_verify.json", "w") as fh:
        json.dump({"n_bands": N_BANDS, "q": BAND_Q,
                   "centres_hz": band_freqs().tolist(), "engine": eng, "stereo": st,
                   "identity": ident, "numerics": check_numerics(win=win),
                   "linearity": lin, "round_trip": rt,
                   "brief_curve_on_render": {"score": s, "cos": c}}, fh, indent=2)
    print("wrote out/eq_verify.json")


if __name__ == "__main__":
    main()
