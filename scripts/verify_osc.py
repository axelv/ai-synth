"""Measure torch_osc against Faust, harmonic by harmonic. Writes out/fixtures/osc.npz.

The oscillator is the one module where a small structural mistake is invisible in
the loss but fatal to the gradient: a saw with the wrong aliasing, or a unison
stack whose cluster width does not grow with harmonic number, still sounds like a
pad. So the checks here are spectral and per harmonic, not "it looks similar".

One divergence is expected and is not a bug in the port: Faust accumulates its
phasors in float32, which biases the phase increment by a fixed ~2e-4 Hz and so
drifts the waveform linearly in time. Every test therefore reports both a
short-window time-domain error (structure) and the drift-immune per-harmonic
magnitude error, and test `drift` proves the long-window residual is nothing but
that drift by fitting a single scalar frequency correction.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from faust_probe import SR, render_gen, save_fixture
from synth import NVOICE, PadRenderer
from torch_common import HOP, NoteEvents
from torch_osc import SQUARE_FMIN, osc_bank

N = 1 << 16
CPU = torch.device("cpu")
FIXTURES: dict[str, np.ndarray] = {}
RESULTS: list[dict[str, object]] = []

OSC_EXPR = f"""
oscmix(g) = os.sawtooth(g)*(1.0-sqrMix) + os.square(g)*sqrMix;
vib = pow(2.0, os.osc(lfoRate) * lfoAmt / 1200.0) * bendc;
centre = oscmix(freq * vib);
spread = par(i, {NVOICE}, oscmix(freq * vib * pow(2.0, (i - ({NVOICE}-1)/2.0) * detune / 1200.0)))
       :> _ / {NVOICE};
sub = os.osc(freq * 0.5 * vib) * subLvl;
process = (centre * (1.0 - uniMix) + spread * uniMix) + sub;
"""

DEFAULTS = dict(freq=220.0, detune=0.0, uniMix=1.0, subLvl=0.0, sqrMix=0.0,
                lfoRate=4.0, lfoAmt=0.0)

# A full patch with every non-oscillator stage as close to a straight wire as its
# range allows: one saw, filter above Nyquist-relevant content, no effects. Used
# by the two tests that need the whole DSP rather than the osc expression alone.
DRY_PATCH = dict(detune=0.0, uniMix=1.0, subLvl=0.0, sqrMix=0.0, cutoff=12000.0, reso=0.5,
                 envAmt=0.0, kbdTrk=0.0, fA=0.1, fD=0.1, fS=1.0, aA=0.001, aD=4.0, aS=1.0,
                 aR=0.01, lfoRate=4.0, lfoAmt=0.0, chRate=0.6, chDepth=0.0, dlyTime=0.35,
                 dlyFb=0.0, dlyWet=0.0, revSize=0.5, revDamp=0.5, revWet=0.0, tilt=0.0,
                 outGain=1.0)


def faust_osc(bend_expr: str = "1.0", n: int = N, **over: float) -> np.ndarray:
    """Render the Faust osc expression alone (no filter, no envelope, no gate)."""
    params = dict(DEFAULTS)
    params.update(over)
    dsp = ('import("stdfaust.lib");\n'
           + "".join(f'{k} = hslider("{k}", {v}, -20000, 20000, 0.000001);\n'
                     for k, v in params.items())
           + f"bendc = {bend_expr};\n" + OSC_EXPR)
    return render_gen(dsp, params, n / SR)[0].astype(np.float64)


def torch_osc(bend: np.ndarray | None = None, n: int = N, onsets=(0,), freqs=None,
              bend_block: int = 512, **over: float) -> np.ndarray:
    """Same expression through torch_osc.osc_bank. Returns (V, n)."""
    params = dict(DEFAULTS)
    params.update(over)
    fr = [params["freq"]] if freqs is None else list(freqs)
    ev = NoteEvents(
        freq=torch.tensor(fr, dtype=torch.float32),
        gain=torch.ones(len(fr)),
        onset=torch.tensor(list(onsets), dtype=torch.long),
        offset=torch.full((len(fr),), n, dtype=torch.long),
        n_samples=n, device=CPU,
    )
    b = torch.ones(n) if bend is None else torch.as_tensor(bend, dtype=torch.float32)
    p = {k: torch.tensor(float(v)) for k, v in params.items()}
    with torch.no_grad():
        return osc_bank(ev, p, b, SR, bend_block=bend_block).numpy().astype(np.float64)


# ------------------------------------------------------------------ analysis


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _circular(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest difference between two wrapped phases, in cycles.

    Plain |a - b| would report 1.0 for two phases that straddle a wrap and are
    in fact identical, which is how a phase error gets overstated.
    """
    d = a - b
    return float((d - torch.round(d)).abs().max())


def spectrum(x: np.ndarray) -> np.ndarray:
    w = np.blackman(len(x))  # -58 dB sidelobes, far below the aliasing we measure
    return np.abs(np.fft.rfft(x * w))


def harmonic_amps(mag: np.ndarray, f0: float, k_max: int, half: float) -> np.ndarray:
    """RMS amplitude in a +/- `half` Hz window around each harmonic of f0."""
    bins = np.arange(len(mag)) * SR / (2.0 * (len(mag) - 1))
    out = np.zeros(k_max)
    for k in range(1, k_max + 1):
        m = np.abs(bins - k * f0) <= half
        out[k - 1] = np.sqrt(np.sum(mag[m] ** 2))
    return out


def aliasing_energy(mag: np.ndarray, f0: float, half: float) -> float:
    """Fraction of spectral energy sitting away from any harmonic of f0 (and DC)."""
    bins = np.arange(len(mag)) * SR / (2.0 * (len(mag) - 1))
    near = bins < half
    for k in range(1, int(SR / 2 / f0) + 1):
        near |= np.abs(bins - k * f0) <= half
    e = mag ** 2
    return float(e[~near].sum() / e.sum())


def cluster_width(mag: np.ndarray, f0: float, k_max: int, halves: np.ndarray) -> np.ndarray:
    """Energy-weighted std deviation in Hz of each harmonic cluster.

    The window has to grow with k, because that is the whole point: a detuned
    unison stack spreads harmonic k over k times the width of harmonic 1.
    """
    bins = np.arange(len(mag)) * SR / (2.0 * (len(mag) - 1))
    out = np.zeros(k_max)
    for k in range(1, k_max + 1):
        m = np.abs(bins - k * f0) <= halves[k - 1]
        e = mag[m] ** 2
        c = np.sum(e * bins[m]) / max(np.sum(e), 1e-30)
        out[k - 1] = np.sqrt(np.sum(e * (bins[m] - c) ** 2) / max(np.sum(e), 1e-30))
    return out


def comb_spacing(mag: np.ndarray, centre: float, span: float, floor: float = 0.02) -> float:
    """Median spacing in Hz of the peaks around `centre`, with sub-bin interpolation."""
    df = SR / (2.0 * (len(mag) - 1))
    lo, hi = int((centre - span) / df), int((centre + span) / df)
    seg = mag[lo:hi]
    pos = []
    for i in range(3, len(seg) - 3):
        if seg[i] == seg[i - 3:i + 4].max() and seg[i] > floor * seg.max():
            a, b, c = np.log(seg[i - 1:i + 2] + 1e-30)
            pos.append((lo + i + 0.5 * (a - c) / (a - 2 * b + c)) * df)
    return float(np.median(np.diff(pos))) if len(pos) > 1 else 0.0


def record(test: str, metric: str, value: float, ok: bool) -> None:
    RESULTS.append({"test": test, "metric": metric, "value": float(value), "pass": bool(ok)})
    print(f"  {'PASS' if ok else 'FAIL'}  {test:<28} {metric:<44} {value:.4e}")


# ------------------------------------------------------------------ tests


def test_waveforms() -> None:
    """os.sawtooth and os.square at four frequencies, harmonic by harmonic."""
    print("\n[1] single oscillator, per-harmonic magnitude and aliasing")
    for name, sqr in (("saw", 0.0), ("square", 1.0)):
        for f in (40.0, 87.0, 220.0, 523.0):
            fa = faust_osc(freq=f, sqrMix=sqr)
            to = torch_osc(freq=f, sqrMix=sqr)[0]
            FIXTURES[f"faust_{name}_{int(f)}"] = fa.astype(np.float32)
            FIXTURES[f"torch_{name}_{int(f)}"] = to.astype(np.float32)
            ma, mt = spectrum(fa), spectrum(to)
            k = min(30, int(SR / 2 / f) - 1)
            ha, ht = (harmonic_amps(m, f, k, 6.0) for m in (ma, mt))
            aa, at = (aliasing_energy(m, f, 6.0) for m in (ma, mt))
            FIXTURES[f"harm_faust_{name}_{int(f)}"] = ha.astype(np.float32)
            FIXTURES[f"harm_torch_{name}_{int(f)}"] = ht.astype(np.float32)
            # a square's even harmonics are ~2e-5 of the fundamental, so a relative
            # error on them is meaningless: report those against the fundamental
            live = ha > 1e-3 * ha[0]
            rel_live = (np.abs(ht - ha) / ha)[live]
            record(f"{name} {f:.0f}Hz", f"max rel err, {int(live.sum())} of {k} harmonics >-60 dB",
                   rel_live.max(), rel_live.max() < 3e-3)
            record(f"{name} {f:.0f}Hz", "mean rel err, same harmonics", rel_live.mean(),
                   rel_live.mean() < 1e-3)
            worst = float((np.abs(ht - ha) / ha[0]).max())
            record(f"{name} {f:.0f}Hz", f"max abs err / fundamental, all {k} harmonics", worst,
                   worst < 1e-3)
            # the 30 above are the task's window; the rest of the harmonic series up
            # to Nyquist is 550 partials at 40 Hz and is only otherwise covered in
            # aggregate by the aliasing number, so check it explicitly too
            kn = int(SR / 2 / f) - 1
            hna, hnt = (harmonic_amps(m, f, kn, 6.0) for m in (ma, mt))
            FIXTURES[f"harmall_faust_{name}_{int(f)}"] = hna.astype(np.float32)
            FIXTURES[f"harmall_torch_{name}_{int(f)}"] = hnt.astype(np.float32)
            wn = float((np.abs(hnt - hna) / hna[0]).max())
            record(f"{name} {f:.0f}Hz", f"max abs err / fundamental, all {kn} harmonics to Nyquist",
                   wn, wn < 1e-3)
            ln = float(np.linalg.norm(hnt - hna) / np.linalg.norm(hna))
            record(f"{name} {f:.0f}Hz", "rel L2 of the harmonic-amplitude vector to Nyquist", ln,
                   ln < 3e-3)
            record(f"{name} {f:.0f}Hz", "aliasing energy fraction (faust)", aa, True)
            record(f"{name} {f:.0f}Hz", "aliasing energy fraction (torch)", at, True)
            record(f"{name} {f:.0f}Hz", "aliasing energy ratio torch/faust", at / aa,
                   0.5 < at / aa < 2.0)
            record(f"{name} {f:.0f}Hz", "rel L2, first 4096 samples",
                   relerr(to[:4096], fa[:4096]), relerr(to[:4096], fa[:4096]) < 5e-3)
            record(f"{name} {f:.0f}Hz", f"rel L2, all {N} samples", relerr(to, fa), True)


def test_square_clip() -> None:
    """Below 23.4489 Hz os.square stops tracking freq. The real render goes there.

    os.pulsetrainN clips its frequency at SRmax/(2*delmax), so the square of the
    lowest note in the MIDI (34.65 Hz) sits at 22.07 Hz at the bottom of the intro
    glide and comes out a semitone and a half sharp of the saw next to it. Nothing
    else in the suite reaches that branch, and getting it wrong would silently
    detune the square content of the bass notes.
    """
    print("\n[1b] os.square frequency clip at 23.4489 Hz")
    for f in (18.0, 22.07, 24.0):
        fa = faust_osc(freq=f, sqrMix=1.0)
        to = torch_osc(freq=f, sqrMix=1.0)[0]
        FIXTURES[f"faust_sqclip_{f:.0f}"] = fa.astype(np.float32)
        FIXTURES[f"torch_sqclip_{f:.0f}"] = to.astype(np.float32)
        for tag, x in (("faust", fa), ("torch", to)):
            # zero crossings of a square are its period, and the spectrum of a
            # 22 Hz square in a 65536-sample window is too coarse to read
            s = np.sign(x[512:])
            k = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
            f0 = SR / float(np.median(np.diff(k))) if len(k) > 1 else 0.0
            expect = max(f, SQUARE_FMIN)
            record(f"sqclip {f:.1f}Hz", f"measured f0 / max(f, 23.4489) ({tag})", f0 / expect,
                   abs(f0 / expect - 1) < 0.02)
        record(f"sqclip {f:.1f}Hz", "rel L2, first 4096 samples", relerr(to[:4096], fa[:4096]),
               relerr(to[:4096], fa[:4096]) < 5e-3)


def numpy_ptr_saw(f: float, n: int = N) -> np.ndarray:
    """os.sawtooth in plain float64 numpy: the referee for the drift question.

    Independent of both sides, and with no accumulation error worth naming, so it
    says which of the two is the one whose frequency is slightly off.
    """
    ph = np.cumsum(np.full(n, f / SR, dtype=np.float64))
    ph -= np.floor(ph)
    wrap = np.zeros(n, dtype=bool)
    wrap[1:] = ph[1:] < ph[:-1]
    p = np.where(wrap, 1.0 + ph * (2.0 - SR / f), ph)
    return 2.0 * p - 1.0


FIT_SPAN, FIT_STEPS = 2e-5, 401
FIT_STEP = 2.0 * FIT_SPAN / (FIT_STEPS - 1)  # relative frequency, the fit's resolution


def fit_offset(ref: np.ndarray, f: float) -> tuple[float, float]:
    """Relative frequency offset that best aligns torch_osc with `ref`, and the residual."""
    best = (1e9, 0.0)
    for d in np.linspace(-FIT_SPAN, FIT_SPAN, FIT_STEPS):
        r = relerr(torch_osc(freq=f * (1.0 + d))[0], ref)
        if r < best[0]:
            best = (r, d)
    return best[1], best[0]


def test_drift() -> None:
    """The long-window residual is Faust's float32 phasor drift and nothing else.

    Faust's saw2ptr is a recursive float32 accumulator, so the rounding of each
    p + t0 biases its effective frequency; this module's phase is exact to 1.4e-6
    cycles against float64 (test_precision), and float32 rounding of the increment
    itself is only worth 1.3e-5 Hz at 220 Hz, so a fitted offset an order of
    magnitude larger than that cannot be coming from here. The numpy referee makes
    that an observation rather than an inference.

    Also re-runs the aliasing comparison at the matched frequency: PTR aliasing
    energy depends on how the wrap instants fall inside a sample, so at a
    frequency that divides SR exactly (40 Hz -> 1102.5 samples) it is not a
    continuous function of freq and the raw ratio is not a like-for-like number.
    """
    print("\n[2] the long-window residual is a pure phase drift")
    for f in (40.0, 220.0, 523.0):
        fa = faust_osc(freq=f)
        ref64 = numpy_ptr_saw(f)
        d64 = fit_offset(ref64, f)[0]
        e64 = relerr(torch_osc(freq=f)[0], ref64)
        record(f"drift {f:.0f}Hz",
               f"fitted offset vs the float64 referee, Hz (grid {f * FIT_STEP:.1e})", f * d64,
               abs(d64) <= 1.5 * FIT_STEP)
        record(f"drift {f:.0f}Hz", f"rel L2 vs the float64 referee, all {N} samples, no fit",
               e64, e64 < 1e-3)
        dfa, rfa = fit_offset(fa, f)
        record(f"drift {f:.0f}Hz", "fitted freq offset vs faust, Hz", f * dfa,
               abs(f * dfa) < 2e-3)
        record(f"drift {f:.0f}Hz", "rel L2 vs faust after removing the drift", rfa, rfa < 1e-3)
        at = aliasing_energy(spectrum(torch_osc(freq=f * (1.0 + dfa))[0]), f, 6.0)
        aa = aliasing_energy(spectrum(fa), f, 6.0)
        record(f"drift {f:.0f}Hz", "aliasing ratio torch/faust at matched freq", at / aa,
               0.9 < at / aa < 1.1)


def test_unison() -> None:
    """detune shows up as cluster width growing linearly with harmonic number."""
    print("\n[3] 7-voice unison stack, detune 22 cents")
    det, f = 22.0, 220.0
    fa = faust_osc(freq=f, detune=det, uniMix=1.0)
    to = torch_osc(freq=f, detune=det, uniMix=1.0)[0]
    FIXTURES["faust_unison22"] = fa.astype(np.float32)
    FIXTURES["torch_unison22"] = to.astype(np.float32)
    ma, mt = spectrum(fa), spectrum(to)
    ratios = 2.0 ** ((np.arange(NVOICE) - (NVOICE - 1) / 2.0) * det / 1200.0)
    # clusters merge once their width exceeds the harmonic spacing: at 22 cents
    # harmonic 11 already spans 0.84 of f, so that is as far as width is measurable
    k = 11
    ka = np.arange(1, k + 1)
    halves = 0.55 * ka * f * (ratios[-1] - ratios[0]) + 3.0
    wa, wt = (cluster_width(m, f, k, halves) for m in (ma, mt))
    FIXTURES["width_faust"] = wa.astype(np.float32)
    FIXTURES["width_torch"] = wt.astype(np.float32)
    # width / harmonic number must be constant: that is what "detune" means.
    # harmonic 1 is narrower than the analysis window, so measure from k=4 up.
    for tag, w in (("faust", wa), ("torch", wt)):
        r = (w / ka)[3:]
        rel_spread = float(np.std(r) / np.mean(r))
        record("unison22", f"width/k constancy k=4..11 ({tag}), std/mean", rel_spread,
               rel_spread < 0.05)
    slope_a = float(np.polyfit(ka[3:], wa[3:], 1)[0])
    slope_t = float(np.polyfit(ka[3:], wt[3:], 1)[0])
    theory = float(np.std(f * ratios))  # 7 equal-amplitude partials, flat weighting
    record("unison22", "cluster width slope, Hz per harmonic (faust)", slope_a, True)
    record("unison22", "cluster width slope, Hz per harmonic (torch)", slope_t, True)
    record("unison22", "slope torch/faust", slope_t / slope_a, abs(slope_t / slope_a - 1) < 0.02)
    record("unison22", "slope torch / flat-weight theory", slope_t / theory,
           0.9 < slope_t / theory < 1.1)
    ha, ht = (harmonic_amps(m, f, k, halves.max()) for m in (ma, mt))
    err = np.abs(ht - ha) / ha
    record("unison22", f"max rel cluster-amplitude err over {k} harmonics", err.max(), err.max() < 0.02)
    record("unison22", "rel L2, first 4096 samples", relerr(to[:4096], fa[:4096]),
           relerr(to[:4096], fa[:4096]) < 5e-3)


def test_vibrato() -> None:
    """lfoAmt > 0: sidebands spaced by exactly lfoRate around each harmonic."""
    print("\n[4] vibrato sidebands")
    f, rate, amt = 220.0, 8.285240442665962, 27.629644226556263
    fa = faust_osc(freq=f, lfoRate=rate, lfoAmt=amt)
    to = torch_osc(freq=f, lfoRate=rate, lfoAmt=amt)[0]
    FIXTURES["faust_vibrato"] = fa.astype(np.float32)
    FIXTURES["torch_vibrato"] = to.astype(np.float32)
    for tag, x in (("faust", fa), ("torch", to)):
        # a high harmonic: the deviation scales with k, so the comb is widest there
        med = comb_spacing(spectrum(x), 8 * f, 5 * rate)
        record("vibrato", f"median sideband spacing / lfoRate ({tag})", med / rate,
               abs(med / rate - 1) < 0.01)
    record("vibrato", "rel L2, first 4096 samples", relerr(to[:4096], fa[:4096]),
           relerr(to[:4096], fa[:4096]) < 5e-3)
    mag_a, mag_t = spectrum(fa), spectrum(to)
    ha, ht = (harmonic_amps(m, f, 20, 6 * rate) for m in (mag_a, mag_t))
    err = np.abs(ht - ha) / ha
    record("vibrato", "max rel err over 20 harmonic groups", err.max(), err.max() < 0.02)


def test_full_patch() -> None:
    """Every parameter of out/patch.json at once, plus a per-sample bend ramp."""
    print("\n[5] full expression at the patch.json operating point")
    pj = json.load(open("out/patch.json"))["params"]
    over = dict(freq=220.0, detune=pj["detune"], uniMix=pj["uniMix"], subLvl=pj["subLvl"],
                sqrMix=0.35, lfoRate=pj["lfoRate"], lfoAmt=pj["lfoAmt"])
    fa = faust_osc(**over)
    to = torch_osc(**over)[0]
    FIXTURES["faust_patch"] = fa.astype(np.float32)
    FIXTURES["torch_patch"] = to.astype(np.float32)
    record("patch", "rel L2, first 4096 samples", relerr(to[:4096], fa[:4096]),
           relerr(to[:4096], fa[:4096]) < 5e-3)
    ma, mt = spectrum(fa), spectrum(to)
    ha, ht = (harmonic_amps(m, 220.0, 20, 40.0) for m in (ma, mt))
    err = np.abs(ht - ha) / ha
    record("patch", "max rel err over 20 harmonic groups", err.max(), err.max() < 0.02)

    ramp = 1.0 + 0.3 * np.arange(N) / SR
    fb = faust_osc(bend_expr="1.0 + 0.3*ba.time/ma.SR", **over)
    tb = torch_osc(bend=ramp, bend_block=1, **over)[0]
    FIXTURES["faust_bendramp"] = fb.astype(np.float32)
    FIXTURES["torch_bendramp"] = tb.astype(np.float32)
    record("bend ramp", "rel L2, first 4096 samples", relerr(tb[:4096], fb[:4096]),
           relerr(tb[:4096], fb[:4096]) < 5e-3)
    record("bend ramp", f"rel L2, all {N} samples", relerr(tb, fb), relerr(tb, fb) < 5e-2)


def test_onsets() -> None:
    """A voice's phase starts at its own note-on: rows must equal shifted probes."""
    print("\n[6] per-voice onsets")
    freqs = [55.0, 220.0, 415.3]
    onsets = [0, 8832, 30000]
    over = dict(detune=22.0, uniMix=0.8, subLvl=0.4, sqrMix=0.3, lfoRate=8.0, lfoAmt=27.0)
    bank = torch_osc(freqs=freqs, onsets=onsets, **over)
    for f, o in zip(freqs, onsets):
        fa = faust_osc(freq=f, n=N, **over)
        row = bank[freqs.index(f)]
        pre = float(np.abs(row[:o]).max()) if o else 0.0
        record(f"onset {o}", "max |output| before note-on", pre, pre == 0.0)
        m = min(4096, N - o)
        record(f"onset {o}", "rel L2 vs shifted probe, first 4096",
               relerr(row[o:o + m], fa[:m]), relerr(row[o:o + m], fa[:m]) < 5e-3)

    # the same question asked of dawdreamer's polyphony layer rather than of a
    # probe DSP: a note that starts late must sound like the same note shifted
    hold, late = 1.0, 0.5
    ren = []
    for start in (0.0, late):
        r = PadRenderer(n_voices=8)
        r.set_notes([(45, 100, start, hold)])
        r.set_params(DRY_PATCH)
        r.set_bend(None)
        ren.append(r.render(start + hold).mean(axis=0).astype(np.float64))
    k = int(late * SR)
    e = relerr(ren[1][k:], ren[0][:len(ren[1]) - k])
    record("onset shift", f"dawdreamer: rel L2 of the note at {late}s vs at 0s, shifted", e,
           e < 1e-6)


def test_bend_block() -> None:
    """dawdreamer applies parameter automation once per 512-sample block."""
    print("\n[7] bend automation granularity in dawdreamer")
    n = SR
    ramp = (1.0 + 0.3 * np.arange(n + SR) / SR).astype(np.float32)
    held = ramp.copy()
    for b in range(0, len(held), 512):
        held[b:b + 512] = ramp[b]

    def go(curve):
        r = PadRenderer(n_voices=8)
        r.set_notes([(45, 100, 0.0, 1.0)])
        r.set_params(DRY_PATCH)
        r.set_bend(curve)
        return r.render(1.0).mean(axis=0)

    a, b = go(ramp), go(held)
    d = float(np.abs(a - b).max())
    record("bend block", "max |render(ramp) - render(block-held ramp)|", d, d == 0.0)
    c = go(None)
    d2 = float(np.abs(a - c).max())
    record("bend block", "max |render(ramp) - render(no bend)| (control)", d2, d2 > 0.1)

    # what ignoring the block hold would cost on the real bend curve, in cycles
    from torch_common import bend_tensor, default_n_samples
    m = default_n_samples()
    curve = bend_tensor(m, CPU).double()
    step = curve.clone()
    for b in range(0, m - m % 512, 512):
        step[b:b + 512] = curve[b]
    for f in (34.65, 523.25):  # lowest and highest note in out/transcription.mid
        err = float(torch.cumsum((f / SR) * (curve - step), 0).abs().max())
        record("bend block", f"phase cost of ignoring the hold at {f:.0f} Hz, cycles", err,
               err > 0.1)


def test_velocity() -> None:
    """Voice gain is velocity/127, not its square, and it does not reach the osc.

    The amplitude stage owns this, but the whole surrogate hangs off it and the
    two candidate laws are a factor of two apart at velocity 64, so it gets
    measured rather than assumed. Amplitude is read as the RMS over the held part
    of the note, which is immune to where the peak of a single cycle lands.
    """
    print("\n[8b] MIDI velocity to voice gain")
    vels = (25, 50, 64, 100, 127)
    amps = []
    for v in vels:
        r = PadRenderer(n_voices=8)
        r.set_notes([(45, v, 0.0, 1.0)])
        r.set_params(DRY_PATCH)
        r.set_bend(None)
        a = r.render(1.0).mean(axis=0)[SR // 10:SR]
        amps.append(float(np.sqrt((a.astype(np.float64) ** 2).mean())))
    amp = np.array(amps)
    FIXTURES["velocity_rms"] = amp.astype(np.float32)
    FIXTURES["velocity_values"] = np.array(vels, dtype=np.float32)
    lin = np.array(vels) / 127.0
    err = np.abs(amp / amp[-1] - lin / lin[-1]).max()
    record("velocity", f"max dev from gain = vel/127 over {vels}", err, err < 1e-3)
    sq = lin ** 2
    err_sq = np.abs(amp / amp[-1] - sq / sq[-1]).max()
    record("velocity", "same against the squared law (must be worse)", err_sq, err_sq > 0.1)


def test_voice_reuse() -> None:
    """Does any of the 29 notes land on a voice whose phasors are already running?

    dsp_voice::keyOn does not clear voice state, so a reused voice starts mid
    phase and this module's "phase starts at the note-on" model would be wrong for
    it. Giving dawdreamer more voices than notes makes reuse impossible, so the
    difference between 24 and 64 voices is exactly the size of that error.
    """
    print("\n[8] voice reuse in dawdreamer at 24 voices")
    from bend2 import bend_curve
    from stage2 import load_notes
    from torch_common import DUR

    notes = load_notes()
    params = json.load(open("out/patch.json"))["params"]
    out = []
    for nv in (24, 64):
        r = PadRenderer(n_voices=nv)
        r.set_notes(notes)
        r.set_params(params)
        r.set_bend(bend_curve(int(DUR * SR) + SR))
        out.append(r.render(DUR).mean(axis=0).astype(np.float64))
    a, b = out
    FIXTURES["render_24_voices"] = a.astype(np.float32)
    FIXTURES["render_64_voices"] = b.astype(np.float32)
    e = relerr(a, b)
    record("voice reuse", f"{len(notes)} notes: rel L2, 24 voices vs 64 voices", e, e < 1e-6)
    d = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))
    record("voice reuse", "max abs diff / peak", d, d < 1e-6)


def test_precision_and_grad() -> None:
    """Phase accumulator accuracy, and that the gradient exists and is correct."""
    print("\n[9] phase accumulator precision and gradients")
    from torch_common import bend_tensor, default_n_samples
    from torch_osc import _wrapped_phase

    n = default_n_samples()
    bend = bend_tensor(n, CPU)
    devs = [CPU] + ([torch.device("mps")] if torch.backends.mps.is_available() else [])
    for f in (34.65, 220.0, 523.0):  # lowest note in the MIDI, mid, highest
        inc = (f * bend / SR).double()
        ref = torch.cumsum(inc, dim=0)
        ref = ref - torch.floor(ref)
        for dev in devs:
            got = _wrapped_phase((f * bend.to(dev) / SR)[None, :])[0].cpu().double()
            e = _circular(got, ref)
            record("precision", f"{f:.0f} Hz phase error over {n} samples on {dev.type}, cycles",
                   e, e < 1e-5)
        naive = torch.cumsum((f * bend / SR), dim=0)
        naive = _circular((naive - torch.floor(naive)).double(), ref)
        record("precision", f"{f:.0f} Hz, plain float32 cumsum (for contrast)", naive, True)

    # the vibrato LFO does not get the blocked treatment: it is a closed form in
    # time, lfo_rate * since / sr, so its phase carries the float32 resolution of
    # 148 cycles. What matters is the bias, because that is the part that
    # integrates into the carrier phase through vib
    rate = 8.285240442665962  # out/patch.json lfoRate
    t = torch.arange(n)
    x32 = rate * t.float() / SR
    x64 = rate * t.double() / SR
    lo32, lo64 = x32 - torch.floor(x32), x64 - torch.floor(x64)
    record("precision", f"LFO phase error at {rate:.2f} Hz over {n} samples, cycles",
           _circular(lo32.double(), lo64), _circular(lo32.double(), lo64) < 1e-4)
    d = lo32.double() - lo64
    bias = float((d - torch.round(d)).mean())
    record("precision", "same, mean (the part that integrates), cycles", abs(bias), abs(bias) < 1e-5)

    # float64 and a loss that is linear in the output, because the phase
    # parameters move the output by ~1e-5 per step and a float32 central
    # difference is then noise-limited, not the gradient.
    m = 20000
    ev = NoteEvents(freq=torch.tensor([55.0, 220.0], dtype=torch.float64), gain=torch.ones(2),
                    onset=torch.tensor([0, 4096]), offset=torch.tensor([m, m]),
                    n_samples=m, device=CPU)
    torch.manual_seed(0)
    w = torch.randn(2, m, dtype=torch.float64)
    one = torch.ones(m, dtype=torch.float64)
    base = dict(detune=22.0, uniMix=0.8, subLvl=0.4, sqrMix=0.3, lfoRate=8.0, lfoAmt=27.0)

    def loss(q: dict[str, torch.Tensor]) -> torch.Tensor:
        return (osc_bank(ev, q, one, SR) * w).sum()

    p = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in base.items()}
    loss(p).backward()
    for k in base:
        g = float(p[k].grad)
        best = (1e9, 0.0)
        # the step has to be swept: the output oscillates on a scale of 1e-6 in
        # lfoRate, so a central difference is only valid well below that
        for h in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            fd = []
            for s in (+1, -1):
                q = {kk: torch.tensor(v + (h * s if kk == k else 0.0), dtype=torch.float64)
                     for kk, v in base.items()}
                with torch.no_grad():
                    fd.append(float(loss(q)))
            num = (fd[0] - fd[1]) / (2 * h)
            rel = abs(g - num) / max(abs(num), 1e-12)
            best = min(best, (rel, h))
        record("gradient", f"d loss/d {k}: vs central difference (h={best[1]:.0e}), rel err",
               best[0], best[0] < 1e-6 and np.isfinite(g))


def stft_mag(x: np.ndarray) -> np.ndarray:
    """Magnitude STFT of the summed bank: what the stage-2 loss actually consumes.

    A drifting phase is nearly invisible here, which is the point of measuring it:
    a time-domain deviation only matters to the fit through this projection.
    """
    m = torch.from_numpy(x.sum(axis=0)).float()
    s = torch.stft(m, 2048, HOP, window=torch.hann_window(2048), return_complex=True)
    return s.abs().numpy().astype(np.float64)


def _f64_bank(n: int, norm: np.ndarray) -> np.ndarray:
    """The same bank in float64 on the CPU: the referee for the float32 question."""
    from stage2 import load_notes
    from torch_common import Patch, bend_tensor, schedule

    ev = schedule(load_notes(), n, CPU)
    ev.freq = ev.freq.double()
    p = {k: v.double() for k, v in Patch(norm).values().items()}
    with torch.no_grad():
        y = osc_bank(ev, p, bend_tensor(n, CPU).double(), SR)
    return y.numpy()


def test_scale() -> None:
    """The real thing: 29 notes, 789566 samples, forward and backward.

    The float32 deviation is reported against a float64 run of this same module,
    not against whichever device happened to go first. It is pure accumulated
    phase: the per-sample increment (hz/sr)*vib is itself rounded to float32, so
    9200 cycles of accumulation carry ~1e-3 cycles of drift no matter how
    carefully the sum is done, and MPS and the CPU round it differently. What the
    fit can resolve is the magnitude STFT, so that is reported too.
    """
    print("\n[10] full-scale forward and backward")
    import resource
    import time

    from stage2 import load_notes
    from torch_common import Patch, bend_tensor, default_n_samples, schedule

    from synth import PARAM_INDEX

    n = default_n_samples()
    norm = np.array(json.load(open("out/patch.json"))["normalized"])
    devs = [CPU] + ([torch.device("mps")] if torch.backends.mps.is_available() else [])
    ref = _f64_bank(n, norm)
    ref_mag = stft_mag(ref)
    FIXTURES["scale_f64_rms"] = np.sqrt((ref ** 2).mean(axis=1)).astype(np.float32)
    banks: list[np.ndarray] = []
    for dev in devs:
        ev = schedule(load_notes(), n, dev)
        bend = bend_tensor(n, dev)
        patch = Patch(norm).to(dev)
        for it in range(2):  # the second pass is the honest one: warm kernels
            patch.logits.grad = None
            t0 = time.time()
            y = osc_bank(ev, patch.values(), bend, SR)
            if dev.type == "mps":
                torch.mps.synchronize()
            fwd = time.time() - t0
            t0 = time.time()
            y.pow(2).mean().backward()
            if dev.type == "mps":
                torch.mps.synchronize()
            bwd = time.time() - t0
        tag = f"scale {dev.type}"
        record(tag, f"forward seconds, ({y.shape[0]}, {y.shape[1]})", fwd, fwd < 60)
        record(tag, "backward seconds", bwd, bwd < 120)
        record(tag, "max |osc| over the whole bank", float(y.detach().abs().max()),
               bool(y.isfinite().all()))
        g = patch.logits.grad
        record(tag, "non-finite gradient entries", float((~g.isfinite()).sum()),
               bool(g.isfinite().all()))
        for k in ("detune", "uniMix", "subLvl", "sqrMix", "lfoRate", "lfoAmt"):
            v = float(g[PARAM_INDEX[k]])
            record(tag, f"d loss/d {k} logit", v, v != 0.0)
        cur = y.detach().cpu().numpy().astype(np.float64)
        banks.append(cur)
        e = relerr(cur, ref)
        # 2e-2 because this is drift near the saw discontinuities, where a phase
        # error of eps cycles puts a full-scale sample on the wrong side of the
        # wrap: the L2 grows like sqrt(eps), not eps. Faust's own float32 phasor
        # drifts further than this (test_drift, "rel L2, all 65536 samples").
        record(tag, "rel L2 vs the same bank in float64", e, e < 2e-2)
        first = relerr(cur[:, :SR], ref[:, :SR])
        last = relerr(cur[:, -SR:], ref[:, -SR:])
        record(tag, "same, first second only", first, first < 2e-3)
        record(tag, "same, last second only (drift accumulates)", last, last < 2e-2)
        em = relerr(stft_mag(cur), ref_mag)
        record(tag, "magnitude-STFT rel L2 vs float64 (what the loss sees)", em, em < 5e-3)
    if len(banks) > 1:
        record("scale", "rel L2 cpu vs mps (observation, both above are the test)",
               relerr(banks[1], banks[0]), True)
        record("scale", "magnitude-STFT rel L2 cpu vs mps",
               relerr(stft_mag(banks[1]), stft_mag(banks[0])), True)
    record("scale", "peak RSS, GB",
           resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, True)


def main() -> None:
    test_waveforms()
    test_square_clip()
    test_drift()
    test_unison()
    test_vibrato()
    test_full_patch()
    test_onsets()
    test_bend_block()
    test_velocity()
    test_voice_reuse()
    test_precision_and_grad()
    test_scale()
    n_fail = sum(1 for r in RESULTS if not r["pass"])
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    FIXTURES["results_json"] = np.array(json.dumps(RESULTS))
    save_fixture("out/fixtures/osc.npz", **FIXTURES)
    with open("out/fixtures/osc_results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1)


if __name__ == "__main__":
    main()
