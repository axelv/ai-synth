"""Ground-truth checks for the two timbre stages once they are inside synth.DSP.

Four questions, all answered from rendered audio rather than from reading the source:

  1. is the integration append-only, and is the source that went in the source that was
     verified (eq_stage.EQ_FAUST string for string, wt_osc's fitted table numerically)
  2. is the identity real: does the delivered patch, padded with the new defaults,
     render to the same audio as the DSP that predates both stages
  3. the identity is not bit-exact, so what exactly is the residue: one float32 ulp at
     the oscillator, times the roundoff gain of a filter fitted to a 258 Hz cutoff
  4. are the new parameters actually connected, which the identity check alone cannot
     tell, since a stage wired to nothing also passes it

(4) is not a fit. The 26 gains come from out/eq_warm_start.json, which eq_stage measured
post hoc by putting the finished render through the same cascade; reproducing its score
from inside the synth is what proves the cascade sits in the signal path and that its
place in the chain (after tiltEQ, before outGain) is equivalent to the place it was
measured in. wtMorph=1 is expected to score WORSE than the identity, and that is the
point of the next agent's job, not a defect here.

Everything numeric lands in out/timbre_stages.json; stdout is scalars only.
"""

from __future__ import annotations

import json
import time

import numpy as np
import soundfile as sf

import chord
import eq_stage
import wt_osc
from bend2 import bend_curve
from stage2 import DUR, Objective, load_notes
from synth import (
    DSP,
    DSP_SAW,
    PARAMS,
    PadRenderer,
    eq_faust,
    norm_defaults,
    pad_normalized,
    wt_table,
)

SR = 44100
PATCH = "out/patch.json"
# out/render.wav is promoted with out/patch.json by promote_patch.py, so it is the
# delivered render of exactly this patch through the DSP that predates both stages: an
# external reference that does not depend on this file's reconstruction being right.
REFERENCE = "out/render.wav"
RECORDED_LOSS = 1.5446351990103722   # out/patch.json, the 18 s clip
BASELINE_WINDOW = 1.5605             # chord.WindowScore of the same patch, from the brief
WARM_PATH = "out/eq_warm_start.json"
OUT_PATH = "out/timbre_stages.json"

# The cascade is second in the chain's tail (tiltEQ, eqCurve, outGain), so removing it
# means removing both its sliders and its one use.
_EQ_USE = "       : par(i, 2, eqCurve)\n"
_OSCMIX_NEW = "oscmix(f) = sawLeg(f) * (1.0 - sqrMix) + os.square(f) * sqrMix;"
_OSCMIX_OLD = "oscmix(f) = os.sawtooth(f) * (1.0 - sqrMix) + os.square(f) * sqrMix;"


def _replace_once(s: str, old: str, new: str) -> str:
    if s.count(old) != 1:
        raise AssertionError(f"expected exactly one {old[:40]!r} in the DSP, found {s.count(old)}")
    return s.replace(old, new)


def _wt_block() -> str:
    """The oscillator stage's Faust text, as synth.dsp_source pasted it in."""
    from synth import wt_faust
    return wt_faust(wt_table())


def legacy_dsp() -> str:
    """synth.DSP with both timbre stages excised: the oscillator and effect chain that
    out/patch.json was fitted against.

    Excised rather than neutralised. Faust folds `x*(1.0-0.0) + y*0.0` back to x and a
    0 dB fi.svf.bell back to a wire, so a neutralised build would compile to this anyway,
    but then the A/B would be testing the simplifier instead of the wiring. Checked
    against git HEAD's copy of synth.py below, which is what makes this a reconstruction
    of the previous source rather than a claim about one.
    """
    s = _replace_once(DSP, eq_faust(), "")
    s = _replace_once(s, _EQ_USE, "")
    return _replace_once(s, _wt_block() + "\n" + _OSCMIX_NEW, _OSCMIX_OLD)


# ---------------------------------------------------------------- checks

def test_sources(patch: dict) -> dict[str, object]:
    """The three ways this integration could silently not be the verified stages."""
    names = [p.name for p in PARAMS]
    fitted = list(patch["params"])
    if names[:len(fitted)] != fitted:
        raise AssertionError("PARAMS no longer starts with the order patch.json was fitted in")

    specs = eq_stage.param_specs()
    got = [(p.name, p.lo, p.hi, p.default) for p in PARAMS[len(fitted):len(fitted) + len(specs)]]
    if got != [tuple(s) for s in specs]:
        raise AssertionError("appended EQ parameters are not eq_stage.param_specs()")

    ref = wt_osc.with_saw_tail(wt_osc.fit_harmonic_table()["amps"], len(wt_table()))
    table_err = float(np.abs(wt_table() - ref).max())
    if table_err > 1e-8:
        raise AssertionError(f"shipped harmonic table is not the fitted one: {table_err:.2e}")

    legacy = legacy_dsp()
    return {
        "n_params": len(PARAMS),
        "n_params_before": len(fitted),
        "eq_source_matches_eq_stage": eq_faust() == eq_stage.EQ_FAUST,
        "table_max_abs_err": table_err,
        "table_h": len(wt_table()),
        # the excision has to land back on the old oscillator, with nothing of either
        # stage left behind for the A/B to be an A/B
        "legacy_has_old_oscmix": _OSCMIX_OLD in legacy,
        "legacy_stage_tokens": sum(legacy.count(t) for t in ("eq0", "eqCurve", "wtMorph", "wtosc")),
    }


def test_identity(x: np.ndarray, notes) -> dict[str, float]:
    """The 18 s clip: new DSP at the new defaults against the DSP that predates them."""
    # Compile and render timed apart, because they scale differently with the bank and a
    # fit pays the first once and the second thousands of times.
    t0 = time.time()
    new_obj = Objective(notes, dsp=DSP)
    t_compile = time.time() - t0
    t0 = time.time()
    new_audio = new_obj.render(x)
    t_new = time.time() - t0

    old_obj = Objective(notes, dsp=legacy_dsp())
    t0 = time.time()
    old_audio = old_obj.render(x)
    t_old = time.time() - t0

    saw_obj = Objective(notes, dsp=DSP_SAW)
    saw_audio = saw_obj.render(x)
    ref = sf.read(REFERENCE, always_2d=True)[0].T.astype(np.float64)

    def diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        n = min(a.shape[1], b.shape[1])
        d = a[:, :n] - b[:, :n]
        return {"max_abs": float(np.abs(d).max()),
                "rel_l2": float(np.linalg.norm(d) / (np.linalg.norm(b[:, :n]) + 1e-12))}

    return {
        "recorded_loss": RECORDED_LOSS,
        "legacy_loss": old_obj.loss_of(old_audio),
        "new_loss": new_obj.loss_of(new_audio),
        "saw_build_loss": saw_obj.loss_of(saw_audio),
        "reference_loss": new_obj.loss_of(ref),
        "loss_delta": new_obj.loss_of(new_audio) - old_obj.loss_of(old_audio),
        "vs_legacy": diff(new_audio, old_audio),
        "vs_reference": diff(new_audio, ref),
        "legacy_vs_reference": diff(old_audio, ref),
        "saw_build_vs_legacy": diff(saw_audio, old_audio),
        "sec_per_render_new": t_new,
        "sec_per_render_legacy": t_old,
        "sec_compile_new": t_compile,
    }


# Where the identity's last 7.6e-05 of loss comes from. A live slider in the voice makes
# Faust emit the oscillator as `saw*(1-m) + bank*m` instead of folding it to `saw`, which
# at m=0 is the same number to within one float32 ulp but not always the same ulp. The
# fitted filter is what turns that into something the loss can see: a one-pole recursion
# at radius r accumulates its own rounding with gain about 1/(1-r), and this patch's
# cutoff is 258 Hz, so r = 0.964 and 1/(1-r) = 28 per pole. These two probes are the real
# poly voice with everything after the filter removed, so they isolate that chain.
_PROBE = """import("stdfaust.lib");
freq = hslider("freq", 440, 20, 8000, 0.001);
gain = hslider("gain", 0.5, 0, 1, 0.001);
gate = button("gate");
%(m)s
sawLeg(f) = os.sawtooth(f) * (1.0 - m) + os.osc(f) * m;
effect = _,_;
process = %(body)s * gain <: _,_;
"""
_PROBE_NOTE = [(53, 100, 0.05, 1.2)]

_WIN_ENGINES: dict[str, PadRenderer] = {}


def render_window(params: dict[str, float], dsp: str = DSP) -> np.ndarray:
    """The chord window, mono, exactly eq_stage.render_window but with a choosable DSP.

    Engines are cached per source because compiling the 128-partial bank costs half a
    minute, which would otherwise dominate a run that renders it three times.
    """
    r = _WIN_ENGINES.get(dsp)
    if r is None:
        r = _WIN_ENGINES[dsp] = PadRenderer(n_voices=24, dsp=dsp)
        r.set_notes(chord.notes_upto())
        r.set_bend(bend_curve(int(chord.WIN_T1 * SR) + SR))
    r.set_params(params)
    return r.render(chord.WIN_T1)[:, chord.win_slice()].mean(0)


def test_amplifier(params: dict[str, float]) -> dict[str, object]:
    """The one-ulp oscillator difference, before and after the fitted filter."""
    fc, q = params["cutoff"], params["reso"]
    bodies = {
        "oscillator": "sawLeg(freq)",
        "oscillator_then_filter": f"(sawLeg(freq) : fi.resonlp({fc:.6f}, {q:.6f}, 1.0) "
                                  f": fi.lowpass(2, {fc:.6f}))",
    }
    r_pole = float(np.exp(-2.0 * np.pi * fc / SR))
    rows: dict[str, object] = {"cutoff": fc, "pole_radius": r_pole,
                               "roundoff_gain_per_pole": 1.0 / (1.0 - r_pole)}
    for tag, body in bodies.items():
        out = []
        for m in ('m = hslider("m", 0, 0, 1, 0.001);', "m = 0.0;"):
            eng = PadRenderer(n_voices=8, dsp=_PROBE % {"m": m, "body": body})
            eng.set_notes(_PROBE_NOTE)
            if "hslider" in m:
                eng.set_params({"m": 0.0})
            out.append(eng.render(1.5).astype(np.float64))
        d = out[0] - out[1]
        rows[tag] = {"max_abs": float(np.abs(d).max()),
                     "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(out[1])),
                     "peak": float(np.abs(out[1]).max())}
    return rows


def test_window(params: dict[str, float]) -> dict[str, object]:
    """WindowScore of the identity render, then of the two stages actually engaged."""
    sc = chord.WindowScore()
    warm = json.load(open(WARM_PATH))
    rows: dict[str, object] = {"baseline_from_brief": BASELINE_WINDOW,
                               "eq_warm_start_measured_post_hoc": warm["score"]}
    # Every render passes a complete parameter set, because the engines are reused and
    # set_params only writes the names it is handed: a row that named just its own
    # parameter would inherit the previous row's.
    base = {**{p.name: p.default for p in PARAMS}, **params}

    def row(tag: str, extra: dict[str, float], dsp: str = DSP) -> None:
        t0 = time.time()
        y = render_window({**base, **extra}, dsp)
        rows[tag] = {"score": sc(y), "cos": sc.cos_theta(y), "sec": time.time() - t0}

    row("identity", {})
    row("legacy_dsp", {}, legacy_dsp())
    row("eq_warm_start", eq_stage.gain_dict(warm["gains"]))
    row("wtMorph_1", {"wtMorph": 1.0})
    return rows


def main() -> None:
    patch = json.load(open(PATCH))
    x = pad_normalized(np.asarray(patch["normalized"], dtype=float))
    if len(x) != len(PARAMS):
        raise AssertionError(f"padded vector is {len(x)}, PARAMS is {len(PARAMS)}")
    k = len(patch["params"])
    if not np.array_equal(x[k:], norm_defaults()[k:]):
        # 0 dB is normalized 0.5, not 0: the identity is the default, not the box floor
        raise AssertionError("appended coordinates are not at their defaults")

    out: dict[str, object] = {}
    out["sources"] = test_sources(patch)
    print("sources    PARAMS {n_params_before} -> {n_params}, eq source matches eq_stage "
          "{eq_source_matches_eq_stage}, table err {table_max_abs_err:.1e} at h={table_h}, "
          "legacy oscmix {legacy_has_old_oscmix} with {legacy_stage_tokens} stage tokens left"
          .format(**out["sources"]))

    notes = load_notes()
    ident = test_identity(x, notes)
    out["identity"] = ident
    print("identity   legacy {legacy_loss:.10f}  new {new_loss:.10f}  delta {loss_delta:+.2e}  "
          "(recorded {recorded_loss:.10f}, render.wav {reference_loss:.10f})".format(**ident))
    for tag in ("vs_legacy", "vs_reference", "legacy_vs_reference", "saw_build_vs_legacy"):
        print(f"identity   {tag:20s} max_abs {ident[tag]['max_abs']:.2e}  "
              f"rel_l2 {ident[tag]['rel_l2']:.2e}")
    print("cost       {sec_per_render_new:.1f} s per {dur:.1f} s render, against "
          "{sec_per_render_legacy:.1f} s without the bank, plus {sec_compile_new:.0f} s "
          "of one-time faust compile".format(dur=DUR, **ident))

    amp = test_amplifier(patch["params"])
    out["amplifier"] = amp
    print("amplifier  1/(1-r) {roundoff_gain_per_pole:.0f} per pole at cutoff "
          "{cutoff:.0f} Hz".format(**amp))
    for tag in ("oscillator", "oscillator_then_filter"):
        print(f"amplifier  {tag:22s} max_abs {amp[tag]['max_abs']:.2e}  "
              f"rel_l2 {amp[tag]['rel_l2']:.2e}")

    out["window"] = test_window(patch["params"])
    for tag in ("legacy_dsp", "identity", "eq_warm_start", "wtMorph_1"):
        r = out["window"][tag]
        print(f"window     {tag:14s} score {r['score']:.4f}  cos {r['cos']:.4f}  "
              f"({r['sec']:.1f} s)")

    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
