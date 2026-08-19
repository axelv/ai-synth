"""Fit the two timbre stages and the macro controls on the chord window.

The split that makes this cheap. Three of the coordinate blocks are not search
problems:

  * the 26 band gains are a cascade of peaking sections AFTER the voice sum, so in dB
    they add and the whole stage commutes with outGain and with the mono downmix the
    loss takes. That means a rendered clip put through the cascade IS the render the
    synth would have produced with those sliders set, so the gains can be fitted without
    re-rendering the synth at all: one cascade render per trial at 5 ms instead of a
    0.6 s poly render. It only holds with the pre-roll: fed the bare 2.5 s window the
    cascade starts from zero state where the synth's had been running since t=0, and at
    the fitted gains that startup is 6.2e-02 of the window in relative L2 and 0.0030 of
    loss. Fed everything from t=0 it is 1.9e-06 and 0.0000 (`check`).
  * outGain is a scalar at the end of the same chain, so the best level is a 1-D search
    on an array already in memory. It is fitted last and never traded against timbre,
    because the loss's preferred level is (true ratio) x cos(theta): a level deficit is
    a symptom, not a defect.
  * the harmonic table enters each partial's amplitude multiplicatively, so its 64
    log gains are a weighted linear least-squares problem in dB against the measured
    partial residual. It does NOT commute with the filter, so it costs one render per
    solve, plus a Faust recompile because the table is baked into the source.

Only the oscillator/filter/envelope/FX macros are genuinely nonlinear, and they are the
only block CMA-ES is pointed at.

Why the EQ is polished against the loss and not only against the log-amplitude ratio, as
the plan asked. Because the log-amplitude ratio is a surrogate and it is measurably the
wrong one. Solved in closed form from the 796 partial ratios and applied through the real
cascade, it scores 1.5896 against 1.5605 for NO EQ at all: it makes things worse, and it
does so while nudging cos theta up (0.7554 from 0.7516), so it is not a level artefact.
harmonic_readout samples the spectrum only AT the partial frequencies, while the
log-magnitude STFT term also weighs the bins BETWEEN them, which on this material is
where the differences live: it is the same trap that made a pure-tone bank at exactly the
fitted amplitudes score 1.949 against the 1.561 of the render it was meant to replace. So
the closed-form solve is kept for the one block that has no cheaper option, the harmonic
table, and the band gains are optimised against WindowScore itself with real Faust
renders in the loop, which is affordable only because of the commutation above.

The ladder this produced, every line a Faust render scored by chord.WindowScore on
4.95-7.45 s (target 0, lower better), against 1.3424 for the render post-processed
through a fitted third-octave EQ:

    out/patch.json                                       1.5605  cos 0.7516
    + closed-form band gains from the partial ratios     1.5896  cos 0.7554
    + band gains optimised on the loss                   1.2808  cos 0.8561
    + macros by CMA-ES, gains re-solved per candidate    1.2643  cos 0.8656
    + fitted harmonic table at morph 1                   1.2604  cos 0.8634   REJECTED
    + additive bank at the sawtooth's own amplitudes     (1.2604  cos 0.8679)
    + bank length fitted (h = 160) and a longer polish    1.2565  cos 0.8679

Everything reported comes from a Faust render scored by chord.WindowScore, and cos theta
is reported with every score because the loss alone can be lowered by turning the output
down. That is not hypothetical here: the fitted harmonic table lowered the score from
1.2643 to 1.2604 while LOWERING cos theta from 0.8656 to 0.8634, and an equally scored
patch with the table left flat kept cos theta at 0.8679, so the table's apparent win was
not a shape improvement and it was thrown away (phase_scan).

Two things the final patch confirms rather than assumes. Its window sits at -19.52 dB rms
against the target's -18.28, i.e. 1.24 dB down, where 20*log10(cos theta) is 1.23 dB: the
level is exactly the least-squares shrinkage the objective asks for, so there is nothing
to "fix" about it. And on the whole 18 s clip, with the same notes and the same measured
bend, the patch scores 1.5311 against 1.5446 for out/patch.json, so 2.5 s of chord did not
buy its improvement by wrecking the other 15.5.

Budget, and where the improvement did and did not come from. About 690 window renders in
total (0.3 s each on the bank-free build, 20 s with the bank) and roughly 60,000 cascade
renders, which are what the band gains cost. The two CMA-ES passes are 630 of those
renders and bought 0.0165; the band gains, at no synth renders at all beyond the one per
polish, bought 0.28. The second CMA pass over the 13 gap-filling macros (detune, unison,
chorus, delay, reverb, LFO, the coordinates the cascade cannot imitate) improved on its
start in none of its 16 generations, and a final 5000-evaluation polish moved the gains
nowhere, so both blocks are reported as converged rather than as out of budget.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.optimize import lsq_linear, minimize

import chord
import eq_stage
import synth
from bend2 import bend_curve

SR = chord.SR
H_FIT = chord.H_TABLE                   # 64 fitted harmonics; above it the saw tail
N_EQ = synth.N_EQ
EQ_LIMIT = synth.EQ_LIMIT
DB = 20.0 / np.log(10.0)

PRE_ROLL = 0.5          # seconds of clip fed to the cascade before the window starts
OUT_PATCH = "out/patch_chord.json"
OUT_WAV = "out/patch_chord_window.wav"

EQ_IDX = np.array([synth.PARAM_INDEX[f"eq{i}"] for i in range(N_EQ)])
I_GAIN = synth.PARAM_INDEX["outGain"]
I_MORPH = synth.PARAM_INDEX["wtMorph"]

# The macros CMA-ES is allowed to move. tilt is left out on purpose: it is a redundant
# coordinate now (any tilt is also a set of band gains) and giving the search two ways to
# spell the same shape only widens the plateau. drive stays out for the reason
# drive_probe.py established, that it is worse at every setting on this material.
MACROS = [
    "detune", "uniMix", "subLvl", "sqrMix", "cutoff", "reso", "envAmt", "kbdTrk",
    "fA", "fD", "fS", "aA", "aD", "aS", "aR", "lfoRate", "lfoAmt", "spread",
    "chRate", "chDepth", "dlyTime", "dlyFb", "dlyWet", "revSize", "revDamp", "revWet",
]


# ---------------------------------------------------------------- rendering bench

class Bench:
    """Window renders and their scores, plus the cheap post-render cascade.

    One PadRenderer per DSP string. A build carrying the 128-partial bank costs 16 s to
    compile and 5 s per window render against 0.6 s without it, so a rebuild is
    something to count, not something to do in a loop.
    """

    def __init__(self) -> None:
        self.score = chord.WindowScore()
        self.notes = chord.notes_upto()
        self.bend = bend_curve(int(chord.WIN_T1 * SR) + SR)
        self.tab = chord.partial_table()
        self.tgt = chord.target()
        self.tgt_amp = chord.harmonic_readout(self.tgt)
        self.basis = eq_stage.default_basis()
        self._r: dict[str, synth.PadRenderer] = {}
        self._eq: _Cascade | None = None
        self._eqw: _Cascade | None = None
        self.renders = 0
        self.builds = 0

    def renderer(self, dsp: str) -> synth.PadRenderer:
        r = self._r.get(dsp)
        if r is None:
            self._r.clear()     # a bank build is large; do not hoard compiled graphs
            r = synth.PadRenderer(n_voices=24, dsp=dsp)
            r.set_notes(self.notes)
            r.set_bend(self.bend)
            self._r[dsp] = r
            self.builds += 1
        return r

    def window(self, x: np.ndarray, dsp: str) -> np.ndarray:
        """Mono from t=0 to WIN_T1. The whole clip and not just the window, because the
        post-render cascade needs the same pre-roll the synth's own cascade had."""
        r = self.renderer(dsp)
        r.set_params(synth.denorm(x))
        self.renders += 1
        return r.render(chord.WIN_T1).mean(0).astype(np.float64)

    @staticmethod
    def win(mono: np.ndarray) -> np.ndarray:
        return np.asarray(mono)[chord.win_slice()]

    def rate(self, mono: np.ndarray) -> tuple[float, float]:
        w = self.win(mono)
        return self.score(w), self.score.cos_theta(w)

    def cascade(self, mono: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """`mono` from t=0 through the real 26-band cascade at these gains."""
        if self._eq is None:
            self._eq = _Cascade(len(mono))
        return self._eq(mono, gains)

    def cascade_win(self, mono: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """The window only, but with PRE_ROLL seconds of the clip run in first.

        The same signal as cascade(...) sliced to the window, for 40% of the cost, which
        matters because this is the innermost loop of both the polish and the per-candidate
        profile. 0.5 s is 20 ring-downs of the slowest band (40 Hz at Q 2.8), and `check`
        measures the two against each other rather than trusting that.
        """
        sl = chord.win_slice()
        seg = np.asarray(mono)[sl.start - int(PRE_ROLL * SR):sl.stop]
        if self._eqw is None:
            self._eqw = _Cascade(len(seg))
        return self._eqw(seg, gains)[-(sl.stop - sl.start):]


class _Cascade:
    """The band cascade as a re-feedable Faust graph.

    eq_stage.eq_window caches its engine on the window's LENGTH, so it silently reuses
    the first signal it was ever given; this fit changes the window every time a macro
    moves, so the playback buffer has to be settable.
    """

    def __init__(self, n: int) -> None:
        import dawdreamer as daw

        import faust_probe as fp
        self.dur = n / SR
        self.engine = daw.RenderEngine(SR, fp.BLOCK)
        self.play = self.engine.make_playback_processor("src", np.zeros((1, n), np.float32))
        self.proc = self.engine.make_faust_processor("eq")
        self.proc.faust_libraries_path = fp.FAUST_LIBS
        if not self.proc.set_dsp_string(eq_stage._probe_dsp(N_EQ, eq_stage.BAND_Q)):
            raise RuntimeError("faust compile failed for the eq cascade")
        self.pidx = {d["label"]: d["index"] for d in self.proc.get_parameters_description()}
        self.engine.load_graph([(self.play, []), (self.proc, ["src"])])
        self._sig: np.ndarray | None = None
        self.calls = 0

    def __call__(self, mono: np.ndarray, gains: np.ndarray) -> np.ndarray:
        sig = np.asarray(mono, np.float32)[None, :]
        if self._sig is None or sig.shape != self._sig.shape or not np.array_equal(sig, self._sig):
            self.play.set_data(sig)
            self._sig = sig
        for i, v in enumerate(np.asarray(gains, float)):
            self.proc.set_parameter(self.pidx[f"eq{i}"], float(v))
        self.engine.render(self.dur)
        self.calls += 1
        return self.engine.get_audio()[0].astype(np.float64)


# ---------------------------------------------------------------- the linear blocks

def best_level(bench: Bench, mono: np.ndarray, span_db: float = 9.0, iters: int = 28,
               windowed: bool = False) -> tuple[float, float]:
    """The scalar gain that minimises the score, by golden section in dB.

    Exact rather than a heuristic: outGain multiplies at the end of the effect chain, so
    scaling the rendered window is what moving that slider does.
    """
    score = bench.score
    w = np.asarray(mono) if windowed else bench.win(mono)
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = -span_db, span_db
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = score(w * 10 ** (c / 20)), score(w * 10 ** (d / 20))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = score(w * 10 ** (c / 20))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = score(w * 10 ** (d / 20))
        if b - a < 0.01:
            break
    g = (a + b) / 2.0
    return float(g), float(score(w * 10 ** (g / 20)))


def clean_mask(tab: np.ndarray, tol: float = 3.0) -> np.ndarray:
    """Partials no other partial in the chord sits on top of.

    F3 is two octaves above F1 and C3 lands 0.13 Hz from F1's third harmonic, so a
    windowed DFT at those frequencies measures two notes at once. That is harmless for
    the EQ, which multiplies everything at a frequency by the same amount whatever
    produced it, but it makes the harmonic NUMBER of the residual ambiguous, so those
    rows carry no table information and get a zero row in the table block.
    """
    f = tab[:, 2]
    order = np.argsort(f)
    ok = np.ones(len(f), bool)
    for i, j in zip(order[:-1], order[1:]):
        if f[j] - f[i] < tol:
            ok[i] = ok[j] = False
    return ok


def solve_linear(bench: Bench, cur_amp: np.ndarray, g_prev: np.ndarray,
                 d_prev: np.ndarray, fit_table: bool, ridge_eq: float = 3e-2,
                 ridge_tab: float = 0.3, smooth_tab: float = 1.0,
                 floor_db: float = -55.0, clip_db: float = 15.0
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """One weighted least-squares step on the measured partial residual, in dB.

    Both stages are solved together because they overlap by construction: the 34
    parameter absolute-frequency curve leaves 4.20 dB on this material, the 177
    parameter harmonic-number curve 4.47 dB, and both together 2.21 dB, so solving one
    and then the other hands each the other's job. The unknowns are DELTAS on top of
    what the render already has, since the render is where the residual was measured.

    Weighted by target amplitude so partials in the noise cannot steer the fit, and the
    residual is clipped: a partial the render misses by 40 dB is a structural miss, and
    letting it pull on a band gain that also carries loud partials costs more than it
    buys. The table block is smoothed by a second difference penalty because harmonic
    number is a real axis but a 64-tooth comb in it is the readout's noise, not timbre.
    """
    tgt, cur = bench.tgt_amp, np.asarray(cur_amp, float)
    keep = tgt > tgt.max() * 10 ** (floor_db / 20)
    r = np.clip(DB * np.log(np.maximum(tgt, 1e-12) / np.maximum(cur, 1e-12)), -clip_db, clip_db)
    w = tgt / tgt.max()

    A_eq = eq_stage._design(bench.tab[:, 2], bench.basis)
    h = bench.tab[:, 1].astype(int)
    tab_ok = clean_mask(bench.tab) & (h <= H_FIT) & keep
    n_tab = H_FIT if fit_table else 0
    A_tab = np.zeros((len(r), n_tab))
    if fit_table:
        A_tab[np.where(tab_ok)[0], h[tab_ok] - 1] = 1.0
    A = np.hstack([A_eq, A_tab, np.ones((len(r), 1))])[keep]
    r, w = r[keep], w[keep]

    m = A.shape[0]
    sw = np.sqrt(w / w.mean() / m)[:, None]
    rows = [A * sw]
    rhs = [r * sw[:, 0]]
    # ridge on the TOTAL, not on the delta: pulling the delta towards zero lets each
    # round undo the last one's regularisation and the gains then walk to the rails
    k = A.shape[1]
    P = np.zeros((N_EQ, k))
    P[:, :N_EQ] = np.sqrt(ridge_eq / N_EQ) * np.eye(N_EQ)
    rows.append(P)
    rhs.append(-np.sqrt(ridge_eq / N_EQ) * g_prev)
    if fit_table:
        Q = np.zeros((n_tab, k))
        Q[:, N_EQ:N_EQ + n_tab] = np.sqrt(ridge_tab / n_tab) * np.eye(n_tab)
        rows.append(Q)
        rhs.append(-np.sqrt(ridge_tab / n_tab) * d_prev)
        D = np.zeros((n_tab - 2, k))
        for i in range(n_tab - 2):
            D[i, N_EQ + i:N_EQ + i + 3] = np.sqrt(smooth_tab / n_tab) * np.array([1.0, -2.0, 1.0])
        rows.append(D)
        rhs.append(-D[:, N_EQ:N_EQ + n_tab] @ d_prev)

    lo = np.concatenate([-EQ_LIMIT - g_prev, -clip_db - d_prev[:n_tab], [-24.0]])
    hi = np.concatenate([EQ_LIMIT - g_prev, clip_db - d_prev[:n_tab], [24.0]])
    sol = lsq_linear(np.vstack(rows), np.concatenate(rhs), bounds=(lo, hi)).x
    g = np.clip(g_prev + sol[:N_EQ], -EQ_LIMIT, EQ_LIMIT)
    d = d_prev + sol[N_EQ:N_EQ + n_tab] if fit_table else d_prev
    return g, np.clip(d, -clip_db, clip_db), float(sol[-1])


def eq_polish(bench: Bench, mono: np.ndarray, g0: np.ndarray, maxfev: int = 3000,
              level: bool = True) -> tuple[np.ndarray, float]:
    """The 26 gains straight against WindowScore, with the real cascade in the loop.

    No synth render inside: the cascade is the last linear stage before outGain, so
    `cascade(window, g)` is the window the synth renders with those sliders set.
    Powell because the bands are nearly separable in the loss (each bell covers half an
    octave), which is a coordinate structure a gradient-free line search exploits and
    CMA-ES does not.
    """
    sc = bench.score

    def f(g: np.ndarray) -> float:
        y = bench.cascade_win(mono, np.clip(g, -EQ_LIMIT, EQ_LIMIT))
        return sc(y * 10 ** (lvl[0] / 20)) if level else sc(y)

    lvl = [0.0]
    g = np.asarray(g0, float).copy()
    best = f(g)
    for _ in range(3):
        res = minimize(f, g, method="Powell",
                       bounds=[(-EQ_LIMIT, EQ_LIMIT)] * N_EQ,
                       options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-4})
        g = np.clip(res.x, -EQ_LIMIT, EQ_LIMIT)
        if level:
            db, v = best_level(bench, bench.cascade_win(mono, g), windowed=True)
            lvl[0] += db
        else:
            v = float(res.fun)
        if v > best - 1e-4:
            best = min(best, v)
            break
        best = v
    return g, lvl[0]


def profile_eq(bench: Bench, mono: np.ndarray, g0: np.ndarray, step_db: float = 0.5,
               iters: int = 1) -> tuple[np.ndarray, float, float]:
    """One measured gradient step on the band gains, then the level. The cheap profile.

    Why a candidate cannot be judged with the gains left where the last fit put them.
    The 26 gains are co-adapted to one exact render: with them pinned, four random macro
    perturbations at sigma 0.12 scored 1.402, 1.434, 1.396 and 1.888 against 1.281, and
    cos theta fell in every one. CMA-ES read that as "every direction is uphill" and sat
    still for five generations. So the objective a macro search must see is the score
    AFTER the linear block is re-solved, and the only question is how cheaply.

    A full polish is 2000+ cascade renders, 100x the cost of the render being judged. One
    forward-difference gradient over the 26 gains plus a backtracked line search is 32,
    which is 3x the render, and it is enough to remove the first-order penalty for moving
    a macro. It is a lower bound on what the block could do, so the ranking it produces is
    conservative rather than optimistic, and the winner gets a full polish afterwards.
    """
    sc = bench.score
    g = np.clip(np.asarray(g0, float).copy(), -EQ_LIMIT, EQ_LIMIT)

    def f(v: np.ndarray) -> float:
        return sc(bench.cascade_win(mono, np.clip(v, -EQ_LIMIT, EQ_LIMIT)))

    base = f(g)
    for _ in range(iters):
        grad = np.array([(f(g + step_db * e) - base) / step_db
                         for e in np.eye(N_EQ)])
        nrm = np.abs(grad).max()
        if nrm < 1e-9:
            break
        dirn = -grad / nrm
        moved = False
        for step in (1.5, 3.0, 6.0, 0.75, 0.375):
            cand = np.clip(g + step * dirn, -EQ_LIMIT, EQ_LIMIT)
            v = f(cand)
            if v < base:
                g, base, moved = cand, v, True
                break
        if not moved:
            break
    db, v = best_level(bench, bench.cascade_win(mono, g), span_db=6.0, iters=16,
                       windowed=True)
    return g, db, float(v)


# ---------------------------------------------------------------- patch bookkeeping

def start_vector() -> np.ndarray:
    return synth.pad_normalized(np.array(json.load(open("out/patch.json"))["normalized"]))


def flat_eq(x: np.ndarray) -> np.ndarray:
    """The same patch with the cascade at 0 dB.

    Every gain fit works on a render made with this vector, because the cascade is
    applied post hoc afterwards: feeding it a render that already carries the gains
    would apply them twice.
    """
    return with_eq(x, np.zeros(N_EQ))


def with_eq(x: np.ndarray, gains: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[EQ_IDX] = [synth.normalize_one(synth.PARAMS[i], float(v))
                 for i, v in zip(EQ_IDX, np.clip(gains, -EQ_LIMIT, EQ_LIMIT))]
    return y


def with_level(x: np.ndarray, db: float) -> tuple[np.ndarray, float]:
    """Fold a broadband dB offset into outGain, and report the dB it could not take.

    outGain is bounded, so a level the slider cannot reach has to be reported rather
    than silently dropped: the leftover is what the caller must absorb elsewhere.
    """
    p = synth.PARAMS[I_GAIN]
    want = synth.denorm(x)["outGain"] * 10 ** (db / 20)
    got = float(np.clip(want, p.lo, p.hi))
    y = x.copy()
    y[I_GAIN] = synth.normalize_one(p, got)
    return y, DB * np.log(want / got)


def eq_gains(x: np.ndarray) -> np.ndarray:
    v = synth.denorm(x)
    return np.array([v[f"eq{i}"] for i in range(N_EQ)])


def table_from_delta(d: np.ndarray, h: int = synth.WT_H) -> np.ndarray:
    """The table: the sawtooth's own amplitudes times the fitted dB delta below H_FIT,
    and the untouched sawtooth tail above it, which is what keeps wtMorph a timbre
    control instead of a level control.

    h is the number of harmonics the bank runs to, i.e. where the series stops. It is a
    real timbre coordinate and not a numerical detail: at the delivered patch, stopping at
    128 COST 0.018 of loss, and at the patch fitted here stopping at 160 BUYS 0.007 over
    the untruncated sawtooth (1.2565 against 1.2633, both with the gains re-polished). The
    sawtooth's top octaves are hotter than the target's, and the cascade cannot fix it
    there: third-octave spacing is 2.9 gains per octave, so it has no way to put a cliff
    between two partials. Measured by phase_trunc: h = 96 gives 1.2649, 128 gives 1.2589,
    160 gives 1.2565, 192 gives 1.2564, 224 gives 1.2569, so the optimum is a plateau from
    160 to 224 and 160 is the cheapest point on it.
    """
    a = 2.0 / (np.pi * np.arange(1, h + 1, dtype=float))
    k = min(len(d), h)
    a[:k] *= 10 ** (np.asarray(d, float)[:k] / 20)
    return a


def save_patch(x: np.ndarray, d: np.ndarray | None, score: float, cos: float,
               note: str, extra: dict | None = None, h: int = synth.WT_H) -> None:
    doc = {
        "window_score": float(score),
        "window_cos_theta": float(cos),
        "normalized": np.asarray(x, float).tolist(),
        "params": synth.denorm(x),
        # the harmonic table is baked into the Faust source, so the vector alone does not
        # reproduce the render: DSP = synth.dsp_source(np.array(wt_amps)), or synth.DSP
        # when wt_amps is null
        "wt_amps": None if d is None else table_from_delta(d, h).tolist(),
        "wt_delta_db": None if d is None else np.asarray(d, float).tolist(),
        "wt_h": None if d is None else int(h),
        "note": note,
    }
    if extra:
        doc.update(extra)
    with open(OUT_PATCH, "w") as fh:
        json.dump(doc, fh, indent=2)


def load_patch() -> dict:
    return json.load(open(OUT_PATCH))


def dsp_for(d: np.ndarray | None, h: int = synth.WT_H) -> str:
    return synth.DSP_SAW if d is None else synth.dsp_source(table_from_delta(d, h))


def report(tag: str, score: float, cos: float, bench: Bench) -> None:
    print(f"  {tag:<28s} score {score:.4f}  cos {cos:.4f}   "
          f"[{bench.renders} renders, {bench.builds} builds, "
          f"{sum(c.calls for c in (bench._eq, bench._eqw) if c)} cascade]")


# ---------------------------------------------------------------- phases

def phase_check(bench: Bench) -> None:
    """Is the post-render cascade the same thing as the sliders? Measure, do not assume.

    The whole fit leans on it: if it holds, 26 of the 56 coordinates cost cascade renders
    instead of poly renders, and if it does not, every gain trial costs a synth render.
    """
    x = start_vector()
    mono = bench.window(x, synth.DSP_SAW)
    s0, c0 = bench.rate(mono)
    report("patch.json (saw build)", s0, c0, bench)
    g = np.array(json.load(open(eq_stage.WARM_PATH))["gains"])
    post = bench.cascade(mono, g)
    sp, cp = bench.rate(post)
    report("warm-start EQ, post hoc", sp, cp, bench)
    insitu = bench.window(with_eq(x, g), synth.DSP_SAW)
    si, ci = bench.rate(insitu)
    report("warm-start EQ, in synth", si, ci, bench)
    a, b = bench.win(post), bench.win(insitu)
    print(f"  post hoc (pre-roll) vs in synth: rel L2 "
          f"{np.linalg.norm(a - b) / np.linalg.norm(b):.3e}, loss gap {abs(sp - si):.4f}")
    short = bench.cascade_win(mono, g)
    print(f"  post hoc ({PRE_ROLL:.1f}s pre-roll) vs in synth: rel L2 "
          f"{np.linalg.norm(short - b) / np.linalg.norm(b):.3e}, "
          f"loss gap {abs(bench.score(short) - si):.4f}")
    # the same thing fed only the window, which is what eq_stage measured post hoc
    bare = _Cascade(len(b))(bench.win(mono), g)
    print(f"  post hoc (window only) vs in synth: rel L2 "
          f"{np.linalg.norm(bare - b) / np.linalg.norm(b):.3e}, "
          f"loss gap {abs(bench.score(bare) - si):.4f}")
    t0 = time.time()
    bench.window(x, synth.DSP_SAW)
    t_saw = time.time() - t0
    t0 = time.time()
    mono_b = bench.window(synth.pad_normalized(x), synth.DSP)
    t_bank = time.time() - t0
    sb, cb = bench.rate(mono_b)
    report("full bank build, morph 0", sb, cb, bench)
    print(f"  render cost: saw {t_saw:.2f}s, bank {t_bank:.2f}s (plus compile)")


def phase_eq(bench: Bench, maxfev: int) -> None:
    """The absolute-frequency stage alone: closed form, then polished on the loss."""
    x = start_vector()
    mono = bench.window(flat_eq(x), synth.DSP_SAW)
    s0, c0 = bench.rate(mono)
    report("start", s0, c0, bench)

    amp = chord.harmonic_readout(bench.win(mono))
    g_cf, _, off = solve_linear(bench, amp, np.zeros(N_EQ), np.zeros(H_FIT), False)
    y = bench.cascade(mono, g_cf)
    s, c = bench.rate(y)
    print(f"  closed form from partial ratios: score {s:.4f} cos {c:.4f} "
          f"(offset {off:+.2f} dB), post hoc")
    db, s_l = best_level(bench, y)
    print(f"  + best level {db:+.2f} dB: score {s_l:.4f}")

    starts = {"closed form": g_cf, "warm start": np.array(json.load(open(eq_stage.WARM_PATH))["gains"])}
    best = None
    for name, g0 in starts.items():
        t0 = time.time()
        g, db = eq_polish(bench, mono, g0, maxfev=maxfev)
        y = bench.cascade(mono, g)
        s, c = bench.rate(y * 10 ** (db / 20))
        print(f"  polished from {name}: score {s:.4f} cos {c:.4f} "
              f"(level {db:+.2f} dB, {time.time()-t0:.0f}s), post hoc")
        if best is None or s < best[0]:
            best = (s, g, db)
    _, g, db = best
    xb, left = with_level(with_eq(x, g), db)
    mono2 = bench.window(xb, synth.DSP_SAW)
    s, c = bench.rate(mono2)
    report("EQ in synth (authoritative)", s, c, bench)
    if abs(left) > 0.05:
        print(f"  outGain could not take {left:+.2f} dB of the level")
    save_patch(xb, None, s, c, "eq only, closed form then polished on the loss")
    print(f"  wrote {OUT_PATCH}")


def phase_als(bench: Bench, rounds: int, maxfev: int, damp: float) -> None:
    """Alternate the harmonic table against the EQ, both solved rather than searched.

    The two stages are solved by different means, and not by choice. The EQ commutes with
    everything after the voice sum, so its 26 gains can be optimised against the loss
    itself for the price of cascade renders. The table sits inside the oscillator, ahead
    of the filter, so it cannot: it is fitted to the measured partial residual, which is
    a surrogate, and the round then has to be accepted or rejected by a real render. The
    joint solve is what keeps the table off the frequency axis (its 34-parameter curve
    and the 177-parameter harmonic-number one overlap, 4.20 and 4.47 dB alone against
    2.21 together), so the EQ columns are in the design matrix even though the polish and
    not the solve is what ships the gains.

    wtMorph is pinned at 1 rather than fitted: at morph m the leg's harmonic amplitudes
    are (1-m)*saw[h] + m*a[h], which is another table, so a free table at morph 1 already
    spans every intermediate morph and the coordinate would only add a flat direction.
    """
    doc = load_patch()
    incoming = float(doc["window_score"])
    x = _morph_on(np.array(doc["normalized"]))
    d = np.array(doc["wt_delta_db"]) if doc.get("wt_delta_db") else np.zeros(H_FIT)
    mono = bench.window(x, dsp_for(d))
    s, c = bench.rate(mono)
    # morph 1 with a flat table is the sawtooth truncated at 128 harmonics, so round 0
    # starts BEHIND the incoming patch by whatever that truncation costs
    report("round 0 (morph 1, this table)", s, c, bench)
    print(f"  incoming patch scored {incoming:.4f}; morph 1 costs {s - incoming:+.4f} "
          f"before the table moves")
    best = (s, c, x.copy(), d.copy())
    for it in range(1, rounds + 1):
        amp = chord.harmonic_readout(bench.win(mono))
        # the solve's EQ half is thrown away: measured on the window it is worse than no
        # EQ at all (1.5896 against 1.5605), because matching amplitudes AT the partials
        # says nothing about the bins between them, where this loss does much of its work
        _, d_new, _ = solve_linear(bench, amp, eq_gains(x), d, True)
        d_new = d + damp * (d_new - d)
        mono_t = bench.window(x, dsp_for(d_new))
        st, ct = bench.rate(mono_t)
        print(f"  round {it}: table step alone  score {st:.4f} cos {ct:.4f}")
        g, db = eq_polish(bench, bench.window(flat_eq(x), dsp_for(d_new)),
                          eq_gains(x), maxfev=maxfev)
        xn, left = with_level(with_eq(x, g), db)
        if abs(left) > 0.05:
            print(f"  outGain could not take {left:+.2f} dB of the level")
        mono_n = bench.window(xn, dsp_for(d_new))
        s, c = bench.rate(mono_n)
        report(f"round {it} (table + EQ)", s, c, bench)
        if s < best[0] - 1e-4:
            best = (s, c, xn.copy(), d_new.copy())
            x, d, mono = xn, d_new, mono_n
        else:
            print("  round did not improve on the best, stopping the ALS loop")
            break
    s, c, x, d = best
    if s >= incoming - 1e-4:
        print(f"  the harmonic-number stage did not pay: best {s:.4f} against "
              f"{incoming:.4f} without it, {OUT_PATCH} left alone")
        return
    save_patch(x, d, s, c, "harmonic table by least squares on the partial residual, "
                           "EQ polished on the loss, alternated")
    report("best (saved)", s, c, bench)


def _morph_on(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[I_MORPH] = 1.0
    return y


def phase_cma(bench: Bench, gens: int, sigma: float, popsize: int, seed: int,
              maxfev: int, profile: bool, macros: list[str], saw: bool) -> None:
    """CMA-ES on the nonlinear macros, with the linear block re-solved per candidate.

    The macros are the only coordinates that are not a least-squares problem, so they are
    the only ones here. tilt and outGain are excluded because the band gains and the
    profile's level already span them; the harmonic table is held fixed because moving it
    costs a Faust recompile.

    saw=True runs the search on the bank-free build and only the winner on the bank, which
    is the difference between 1.2 s and 25 s per candidate. It is a search shortcut and
    not a claim: the two builds differ above harmonic 160, so the winner is re-polished
    and re-rendered WITH the bank before any number is reported or saved.
    """
    import cma
    doc = load_patch()
    x = np.array(doc["normalized"])
    d = np.array(doc["wt_delta_db"]) if doc.get("wt_delta_db") else None
    h = int(doc.get("wt_h") or synth.WT_H)
    dsp = dsp_for(d, h)
    ref = float(doc["window_score"])
    x_search = x.copy()
    if saw:
        x_search[I_MORPH] = 0.0
    dsp_search = synth.DSP_SAW if saw else dsp
    idx = [synth.PARAM_INDEX[n] for n in macros]
    mono = bench.window(flat_eq(x_search), dsp_search)
    g0, db0, s0 = profile_eq(bench, mono, eq_gains(x_search))
    print(f"  search build starts at profiled {s0:.4f} "
          f"({'saw' if saw else 'bank'}), reference on the bank is {ref:.4f}")
    best = [s0, x_search.copy(), g0, db0]

    def f(z: np.ndarray) -> float:
        y = x_search.copy()
        y[idx] = np.clip(z, 0.0, 1.0)
        if not profile:
            return bench.score(bench.win(bench.window(y, dsp_search)))
        m = bench.window(flat_eq(y), dsp_search)
        g, db, v = profile_eq(bench, m, best[2])
        if v < best[0]:
            best[0], best[1], best[2], best[3] = v, y.copy(), g, db
        return v

    es = cma.CMAEvolutionStrategy(x_search[idx].tolist(), sigma,
                                  {"bounds": [0, 1], "popsize": popsize, "seed": seed,
                                   "verbose": -9, "maxiter": gens})
    t0 = time.time()
    for g in range(1, gens + 1):
        sols = es.ask()
        vals = [f(np.asarray(z)) for z in sols]
        es.tell(sols, vals)
        if g % 5 == 0 or g == 1:
            print(f"  gen {g:3d}  profiled best {best[0]:.4f}  gen min {min(vals):.4f}  "
                  f"({bench.renders} renders, {time.time()-t0:.0f}s)")
    # the profile is one gradient step and the search build may not be the shipped one, so
    # the winner gets the full polish and a real render of the build that will be reported
    xb = x.copy()
    xb[idx] = best[1][idx]
    gains, db = eq_polish(bench, bench.window(flat_eq(xb), dsp), best[2], maxfev=maxfev)
    xb, left = with_level(with_eq(xb, gains), db)
    if abs(left) > 0.05:
        print(f"  outGain could not take {left:+.2f} dB of the level")
    s, c = bench.rate(bench.window(xb, dsp))
    report("polished winner (in synth)", s, c, bench)
    if s < ref - 1e-4:
        save_patch(xb, d, s, c,
                   "macros by CMA-ES with the band gains profiled per candidate", h=h)
        print(f"  wrote {OUT_PATCH}")
    else:
        print(f"  no improvement on {ref:.4f}; {OUT_PATCH} left alone")


def phase_scan(bench: Bench, base: str, alphas: list[float], maxfev: int) -> None:
    """Damped versions of one solved table, each with the EQ re-polished around it.

    Why the step needs scanning rather than taking. The table is the one stage fitted to a
    surrogate (the measured partial residual) instead of to the loss, and the surrogate
    disagrees: the full step moved the score from 1.2643 to 1.2604 but moved cos theta the
    wrong way, 0.8654 to 0.8634, so the loss improvement is partly a level effect and not
    a shape one. A scan over the step length is the cheapest honest way to find out
    whether any amount of it is a real improvement, since each point costs one recompile
    and three renders of the 128-partial bank.
    """
    doc = json.load(open(base))
    x = _morph_on(np.array(doc["normalized"]))
    d_full = np.array(json.load(open(OUT_PATCH))["wt_delta_db"])
    print(f"  solved table delta: {d_full.min():+.1f} to {d_full.max():+.1f} dB, "
          f"rms {np.sqrt((d_full**2).mean()):.1f} dB")
    rows = []
    for a in alphas:
        d = a * d_full
        g, db = eq_polish(bench, bench.window(flat_eq(x), dsp_for(d)), eq_gains(x),
                          maxfev=maxfev)
        xn, _ = with_level(with_eq(x, g), db)
        s, c = bench.rate(bench.window(xn, dsp_for(d)))
        report(f"alpha {a:.2f}", s, c, bench)
        rows.append((s, c, a, xn.copy(), d.copy()))
    rows.sort(key=lambda r: r[0])
    s, c, a, xn, d = rows[0]
    print(f"  best alpha {a:.2f}")
    if s < float(doc["window_score"]) - 1e-4:
        save_patch(xn, d, s, c, f"harmonic table at {a:.2f} of the solved step, "
                                "EQ polished on the loss")
        print(f"  wrote {OUT_PATCH}")
    else:
        print(f"  nothing beat the base {float(doc['window_score']):.4f}; "
              f"{OUT_PATCH} left alone")


def phase_polish(bench: Bench, drop_table: bool, maxfev: int) -> None:
    """Re-polish the band gains on the saved patch, optionally after dropping the bank.

    Dropping it is worth testing rather than assuming: the scan left the table at zero
    delta, and a bank whose amplitudes ARE the sawtooth's is the sawtooth truncated at
    128 harmonics, which is 30x the render cost for a waveform the plain build already
    has. If the saw build reaches the same place the deliverable is a bare 56-vector that
    renders through synth.DSP with wtMorph at 0.
    """
    doc = load_patch()
    x = np.array(doc["normalized"])
    d = np.array(doc["wt_delta_db"]) if doc.get("wt_delta_db") else None
    h = int(doc.get("wt_h") or synth.WT_H)
    if drop_table:
        x = x.copy()
        x[I_MORPH] = 0.0
        d = None
    dsp = dsp_for(d, h)
    g, db = eq_polish(bench, bench.window(flat_eq(x), dsp), eq_gains(x), maxfev=maxfev)
    xn, left = with_level(with_eq(x, g), db)
    if abs(left) > 0.05:
        print(f"  outGain could not take {left:+.2f} dB of the level")
    s, c = bench.rate(bench.window(xn, dsp))
    report("polished" + (" (bank dropped)" if drop_table else ""), s, c, bench)
    if s < float(doc["window_score"]) - 1e-4:
        save_patch(xn, d, s, c, doc.get("note", "") + "; band gains re-polished"
                   + (", bank dropped" if drop_table else ""), h=h)
        print(f"  wrote {OUT_PATCH}")
    else:
        print(f"  did not beat {float(doc['window_score']):.4f}; {OUT_PATCH} left alone")


def phase_trunc(bench: Bench, hs: list[int], maxfev: int) -> None:
    """Where the additive bank should stop, with the band gains re-solved at each length.

    A brickwall at h*f0 is a shape the cascade cannot make: its bands are third-octave, so
    above a few kHz it has 2.9 gains per octave and cannot put a cliff between two
    partials. That makes the series length a genuine timbre coordinate rather than an
    implementation constant, which is why it is measured here instead of assumed at 128.
    """
    doc = load_patch()
    x = _morph_on(np.array(doc["normalized"]))
    d0 = np.array(doc["wt_delta_db"]) if doc.get("wt_delta_db") else np.zeros(H_FIT)
    rows = []
    for h in hs:
        d = d0[:h]
        g, db = eq_polish(bench, bench.window(flat_eq(x), dsp_for(d, h)), eq_gains(x),
                          maxfev=maxfev)
        xn, _ = with_level(with_eq(x, g), db)
        s, c = bench.rate(bench.window(xn, dsp_for(d, h)))
        report(f"bank to h={h}", s, c, bench)
        rows.append((s, c, h, xn.copy(), d.copy()))
    rows.sort(key=lambda r: r[0])
    s, c, h, xn, d = rows[0]
    if s < float(doc["window_score"]) - 1e-4:
        save_patch(xn, d, s, c, f"additive bank to h={h}, band gains polished on the loss",
                   h=h)
        print(f"  best h={h}; wrote {OUT_PATCH}")
    else:
        print(f"  nothing beat the base {float(doc['window_score']):.4f}; "
              f"{OUT_PATCH} left alone")


def phase_final(bench: Bench, full: bool) -> None:
    """Re-render the saved patch from scratch, write the window audio, and optionally
    check that the window fit did not wreck the other 15.5 seconds."""
    doc = load_patch()
    x = np.array(doc["normalized"])
    d = np.array(doc["wt_delta_db"]) if doc.get("wt_delta_db") else None
    h = int(doc.get("wt_h") or synth.WT_H)
    dsp = dsp_for(d, h)
    mono = bench.window(x, dsp)
    s, c = bench.rate(mono)
    report("saved patch, re-rendered", s, c, bench)
    r = bench.renderer(dsp)
    r.set_params(synth.denorm(x))
    synth.write_render(OUT_WAV, r.render(chord.WIN_T1)[:, chord.win_slice()])
    print(f"  window rms {20*np.log10(np.sqrt((bench.win(mono)**2).mean())):.2f} dB "
          f"against target {20*np.log10(np.sqrt((bench.tgt**2).mean())):.2f} dB")
    extra = {"window_wav": OUT_WAV, "wt_h": h}
    if full:
        # the window is 2.5 s of an 18 s clip and the other 15.5 s contain 24 more notes
        # and the measured bend, so a stage fitted here can pay for itself locally and
        # still cost the full fit; one render says which
        import stage2
        obj = stage2.Objective(stage2.load_notes(), dsp=dsp)
        loss = obj(x)
        print(f"  full 18 s stage2 loss {loss:.4f} against {1.5446:.4f} for out/patch.json")
        extra["full_clip_loss"] = float(loss)
    save_patch(x, d, s, c, doc.get("note", ""), extra, h=h)
    print(f"  wrote {OUT_WAV} and {OUT_PATCH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "eq", "als", "cma", "scan", "polish", "trunc", "final"])
    ap.add_argument("--base", default=OUT_PATCH, help="patch the scan starts from")
    ap.add_argument("--alphas", default="0,0.5")
    ap.add_argument("--hs", default="96,160", help="bank lengths for the trunc phase")
    ap.add_argument("--full", action="store_true",
                    help="also score the patch on the whole 18 s clip")
    ap.add_argument("--drop-table", action="store_true",
                    help="polish with the additive bank removed and wtMorph at 0")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--saw", action="store_true",
                    help="search on the bank-free build; verify the winner on the bank")
    ap.add_argument("--only", default=None,
                    help="comma separated subset of the macros to move")
    ap.add_argument("--no-profile", action="store_true",
                    help="score candidates with the band gains pinned instead of re-solved")
    ap.add_argument("--maxfev", type=int, default=2600)
    ap.add_argument("--damp", type=float, default=1.0,
                    help="fraction of the solved table step to take")
    args = ap.parse_args()

    print(chord.describe())
    bench = Bench()
    if args.phase == "check":
        phase_check(bench)
    elif args.phase == "eq":
        phase_eq(bench, args.maxfev)
    elif args.phase == "als":
        phase_als(bench, args.rounds, args.maxfev, args.damp)
    elif args.phase == "trunc":
        phase_trunc(bench, [int(v) for v in args.hs.split(",")], args.maxfev)
    elif args.phase == "polish":
        phase_polish(bench, args.drop_table, args.maxfev)
    elif args.phase == "scan":
        phase_scan(bench, args.base, [float(v) for v in args.alphas.split(",")], args.maxfev)
    elif args.phase == "cma":
        phase_cma(bench, args.gens, args.sigma, args.popsize, args.seed,
                  args.maxfev, not args.no_profile,
                  args.only.split(",") if args.only else MACROS, args.saw)
    else:
        phase_final(bench, args.full)
    print(f"total {bench.renders} window renders, {bench.builds} faust builds")


if __name__ == "__main__":
    main()
