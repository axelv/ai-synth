"""Adversarial audit of the torch_synth surrogate against Faust.

verify_torch_synth.py reports one aggregate distance per pair of renders. A single
scalar can hide a large localised error, and the two headline defences of the port
are both arguments rather than measurements:

  * "the surrogate gap is AT the Faust-vs-Faust floor" compares a full-fx distance
    (0.9917) against a floor measured on the DRY renders (1.1769). Both sides are
    re-measured in one configuration here, and the "floor" turns out not to be one:
    it is the error of a specific assumption, that a render is the sum of its notes,
    and it shrinks when the score is split into two rather than twenty-nine.
  * "the gap is small" was measured at one point in a 27-dimensional box. The
    optimiser moves, so the gap is swept one parameter at a time across the whole
    PARAMS range and scored on what actually matters: does the surrogate loss track
    the Faust loss, and do the two agree on which way is downhill.

The audit's own working reference is the sum of 29 one-note Faust renders. That is
exactly the model torch_synth implements, rendered by the authoritative renderer, so
comparing against it isolates the port from the modelling assumption.

Everything is measured on rendered audio. Subcommands, because the renders are big:

    uv run python scripts/audit_fidelity.py floor    # is the claimed floor real
    uv run python scripts/audit_fidelity.py bands    # octave-band energy, 20 Hz - 20 kHz
    uv run python scripts/audit_fidelity.py regions  # per chord region, and the intro
    uv run python scripts/audit_fidelity.py stereo   # L/R decorrelation, incl. fully wet
    uv run python scripts/audit_fidelity.py core     # fx off: osc+filter core alone
    uv run python scripts/audit_fidelity.py fx       # fx chain, same input on both sides
    uv run python scripts/audit_fidelity.py metric   # what mrstft reads for near-identical audio
    uv run python scripts/audit_fidelity.py sweep    # 1-D sweeps: does the loss track
    uv run python scripts/audit_fidelity.py port     # 1-D sweeps of the port error alone
    uv run python scripts/audit_fidelity.py gain     # is voice gain velocity/127

Scalars to stdout, everything else to out/fidelity_audit.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import librosa
import numpy as np
import torch

import torch_fx
from metrics import mel_dist
from stage2 import load_notes
from synth import PARAM_INDEX, PARAMS, PadRenderer, denorm
from torch_common import SR, Patch, default_n_samples
from torch_synth import TorchPad
from verify_torch_synth import mrstft_pair, objective_loss, rel_l2

DUR = 17.904
FIX = "out/fixtures/torch_synth.npz"
OUT = "out/fidelity_audit.json"
SINGLES = "out/fixtures/faust_singles_wet.npy"
PATCH = "out/patch.json"
STAGE1 = "out/stage1_choice.json"

# the requested "effects disabled" configuration: only the three wet sends go to
# zero. tilt and outGain stay at the patch values because tilt = 0 is not a bypass
# (the Butterworth shelf pair has a zero at each crossover), so zeroing it would
# put notches at 300 and 1200 Hz into the reference instead of removing a stage.
FX_OFF = {"chDepth": 0.0, "dlyWet": 0.0, "revWet": 0.0}
FULLY_WET = {"chDepth": 0.0, "dlyWet": 0.0, "revWet": 1.0}
FX_SLIDERS = ("chRate", "chDepth", "dlyTime", "dlyFb", "dlyWet",
              "revSize", "revDamp", "revWet", "tilt", "outGain")

SWEEP_POINTS = (0.02, 0.25, 0.5, 0.75, 0.98)
SWEEP_PARAMS = (
    "cutoff", "reso", "envAmt", "detune", "revWet", "revSize",
    "uniMix", "subLvl", "aA", "aR", "aS", "tilt", "outGain", "chDepth",
    "kbdTrk", "fS", "dlyWet", "revDamp",
)


# ---------------------------------------------------------------- shared


def record(**vals) -> None:
    have = json.load(open(OUT)) if os.path.exists(OUT) else {}
    have.update(vals)
    with open(OUT, "w") as fh:
        json.dump(have, fh, indent=2, sort_keys=True)


def normalized() -> np.ndarray:
    return np.asarray(json.load(open(PATCH))["normalized"], dtype=float)


def load(name: str) -> np.ndarray:
    return np.load(FIX)[name]


def bend():
    from bend2 import bend_curve

    return bend_curve(int(DUR * SR) + SR)


def renderer(notes) -> PadRenderer:
    r = PadRenderer(n_voices=24)
    r.set_notes(notes)
    r.set_bend(bend())
    return r


_MR = None


def mrs(a: np.ndarray, b: np.ndarray) -> float:
    global _MR
    if _MR is None:
        _MR = mrstft_pair()
    n = min(a.shape[-1], b.shape[-1])
    with torch.no_grad():
        return float(_MR(torch.from_numpy(np.ascontiguousarray(a[..., :n])).float().view(1, 1, -1),
                         torch.from_numpy(np.ascontiguousarray(b[..., :n])).float().view(1, 1, -1)))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def torch_render(z: np.ndarray, device: torch.device, overrides: dict[str, float] | None = None,
                 pad: TorchPad | None = None) -> np.ndarray:
    pad = pad or TorchPad(load_notes(), default_n_samples(), device)
    p = {k: v.detach() for k, v in Patch(z).to(device).values().items()}
    for k, v in (overrides or {}).items():
        p[k] = torch.tensor(float(v), device=device)
    with torch.no_grad():
        return pad.render(p).cpu().numpy()


# ---------------------------------------------------------------- floor


def step_floor() -> None:
    """Is the claimed Faust-vs-Faust floor a property of Faust, or of the measurement?

    The dry voice chain looks linear in the note set: voices are independent and the
    whole fx chain is linear, so a render of note set A plus a render of note set B
    should equal a render of A+B. It does, to 8e-6, as long as no voice slot is reused.
    Once notes outlive a slot's availability the wrapper hands a freed slot to a later
    note WITHOUT clearing its oscillator phase, so the render stops being a function of
    the notes alone and becomes a function of the allocation order. That is the real
    mechanism behind the "floor", and it is not a floor: a two-way split of the score
    disagrees with the full render by half as much as a 29-way split does.

    The number that matters is not the waveform distance but the loss, so the
    note-independent model is also evaluated in Faust itself: the sum of 29 one-note
    renders is exactly what the surrogate assumes, rendered by the authoritative
    renderer. Its loss separates a port error from a voice-allocation model error.
    """
    notes = load_notes()
    z = normalized()
    vals = denorm(z)
    n = default_n_samples()
    target, _ = librosa.load("data/original.wav", sr=SR, mono=True)

    def render(sub, params) -> np.ndarray:
        r = renderer(sub)
        r.set_params(params)
        return r.render(DUR).mean(axis=0)[:n].astype(np.float64)

    dry = {**vals, **FX_OFF}
    full_a = render(notes, dry)
    full_b = render(notes, dry)
    even = render(notes[0::2], dry)
    odd = render(notes[1::2], dry)
    halves = even + odd

    # the same split with the effects ON, so the floor can be quoted against the
    # full-render gap of 0.9917 instead of against a dry-only number
    wet_full = render(notes, vals)
    wet_even = render(notes[0::2], vals)
    wet_odd = render(notes[1::2], vals)
    wet_halves = wet_even + wet_odd

    # exact superposition where no slot is reused: two notes only
    pair = render([notes[0], notes[5]], dry)
    pair_sum = render([notes[0]], dry) + render([notes[5]], dry)

    # what verify_torch_synth.step_voices built: 29 single-note renders off ONE reused
    # engine. Also with the fx on, which is the configuration the 0.9917 gap was
    # measured in, so the floor is finally quoted against the right thing.
    r_dry = renderer(notes)
    r_dry.set_params(dry)
    r_wet = renderer(notes)
    r_wet.set_params(vals)
    singles = np.zeros(n, dtype=np.float64)
    singles_wet = np.zeros(n, dtype=np.float64)
    first = None
    for note in notes:
        r_dry.set_notes([note])
        a = r_dry.render(DUR).mean(axis=0)[:n].astype(np.float64)
        singles += a
        r_wet.set_notes([note])
        singles_wet += r_wet.render(DUR).mean(axis=0)[:n].astype(np.float64)
        if first is None:
            first = a
    fresh_first = render([notes[0]], dry)

    # where in time does note-independence stop holding
    d = halves - full_a
    onset = None
    for t0 in range(0, int(DUR), 1):
        i0, i1 = int(t0 * SR), min(int((t0 + 1) * SR), n)
        if rms(d[i0:i1]) > 0.02 * rms(full_a[i0:i1]):
            onset = t0
            break

    # the note-independent model, rendered by Faust itself: the reference the surrogate
    # should actually be judged against, since that is the model it implements
    np.save(SINGLES, singles_wet.astype(np.float32))

    res = {
        "determinism_max_abs_diff": float(np.abs(full_a - full_b).max()),
        "two_note_superposition_rel_l2": rel_l2(pair_sum, pair),
        "halves_dry_rel_l2": rel_l2(halves, full_a),
        "halves_dry_mrstft": mrs(halves, full_a),
        "halves_wet_rel_l2": rel_l2(wet_halves, wet_full),
        "halves_wet_mrstft": mrs(wet_halves, wet_full),
        "halves_divergence_onset_sec": onset,
        "singles_sum_dry_rel_l2": rel_l2(singles, full_a),
        "singles_sum_dry_mrstft": mrs(singles, full_a),
        "singles_sum_dry_rms_ratio": rms(singles) / rms(full_a),
        "singles_sum_wet_rel_l2": rel_l2(singles_wet, wet_full),
        "singles_sum_wet_mrstft": mrs(singles_wet, wet_full),
        "singles_sum_wet_rms_ratio": rms(singles_wet) / rms(wet_full),
        "single_note0_reused_vs_fresh_rel_l2": rel_l2(first, fresh_first),
        "loss_faust_poly": objective_loss(wet_full, target),
        "loss_faust_singles_sum": objective_loss(singles_wet, target),
        "faust_dry_rms": rms(full_a),
        "faust_wet_rms": rms(wet_full),
        "singles_sum_wet_bands": band_table(singles_wet, wet_full, "faust singles sum"),
    }
    record(floor=res)
    for k, v in res.items():
        if isinstance(v, (int, float)) or v is None:
            print(f"  {k}: {v}")


# ---------------------------------------------------------------- bands


OCTAVE_CENTRES = 31.25 * 2.0 ** np.arange(10)   # 31.25 Hz .. 16 kHz
OCTAVE_K = math.sqrt(2.0)


def band_energy(freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
    out = np.zeros(len(OCTAVE_CENTRES))
    for i, c in enumerate(OCTAVE_CENTRES):
        m = (freqs >= c / OCTAVE_K) & (freqs < c * OCTAVE_K)
        out[i] = power[m].sum()
    return out


def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.fft.rfft(x)
    return np.fft.rfftfreq(len(x), 1.0 / SR), np.abs(X) ** 2


def band_table(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    """Octave-band energy of a (torch) against b (faust), in dB."""
    fa, pa = spectrum(a)
    fb, pb = spectrum(b)
    ea = band_energy(fa, pa)
    eb = band_energy(fb, pb)
    err = 10.0 * np.log10((ea + 1e-30) / (eb + 1e-30))
    lvl = 10.0 * np.log10((eb + 1e-30) / (eb.max() + 1e-30))
    keep = lvl > -60.0
    centres = OCTAVE_CENTRES
    worst = int(np.argmax(np.where(keep, np.abs(err), 0.0)))
    print(f"[{label}] octave-band energy error torch - faust, dB "
          f"(bands within 60 dB of the faust peak):")
    print("  " + "  ".join(f"{c:.0f}Hz:{e:+.2f}" for c, e, kk in zip(centres, err, keep) if kk))
    print(f"[{label}] worst {err[worst]:+.2f} dB at {centres[worst]:.0f} Hz, "
          f"rms {np.sqrt((err[keep] ** 2).mean()):.2f} dB over {int(keep.sum())} bands")
    return {
        "centres_hz": centres.tolist(),
        "err_db": err.tolist(),
        "faust_level_db_re_peak": lvl.tolist(),
        "worst_db": float(err[worst]),
        "worst_hz": float(centres[worst]),
        "rms_db": float(np.sqrt((err[keep] ** 2).mean())),
        "n_bands_kept": int(keep.sum()),
    }


def step_bands() -> None:
    faust = load("faust").mean(axis=0).astype(np.float64)
    tor = load("torch_cpu").mean(axis=0).astype(np.float64)
    orig, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    singles = np.load(SINGLES).astype(np.float64)
    n = min(len(faust), len(tor), len(orig), len(singles))
    res = {
        "torch_vs_faust": band_table(tor[:n], faust[:n], "torch vs faust poly"),
        # the same table for the port error alone, with the voice-allocation model
        # error divided out
        "torch_vs_singles": band_table(tor[:n], singles[:n], "torch vs faust singles"),
        # how big is any of that next to the error both renders already have against
        # the target the loss is fitted to
        "faust_vs_original": band_table(faust[:n], orig[:n].astype(np.float64),
                                        "faust vs original"),
    }
    record(bands=res)


# ---------------------------------------------------------------- regions


def regions() -> list[tuple[str, float, float]]:
    """The frozen chord regions, plus the intro glide, plus whatever release tail exists.

    The last chord's note-off is at 17.900 s and the render is 17.904 s long, so there
    is no release tail to speak of: the final chord's 0.77 s release is truncated by the
    render itself. A tail region is only emitted if it is at least 0.5 s long.
    """
    d = json.load(open(STAGE1))
    out = [("intro", 0.0, d["regions"][0]["t0"])]
    for r in d["regions"]:
        out.append((r["label"], r["t0"], r["t1"]))
    last_off = max(s + du for _, _, s, du in load_notes())
    if DUR - last_off >= 0.5:
        out.append(("release_tail", last_off, DUR))
    return out


def step_regions() -> None:
    """Per chord region, and the intro that nothing models.

    Three distances per region, because the aggregate cannot tell them apart: the
    surrogate against the polyphonic Faust render, the Faust note-independent render
    against the same (the model error the surrogate inherits by construction) and the
    surrogate against that note-independent render, which is the port error alone.
    """
    faust = load("faust").mean(axis=0).astype(np.float64)
    tor = load("torch_cpu").mean(axis=0).astype(np.float64)
    singles = np.load(SINGLES).astype(np.float64)
    orig, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    n = min(len(faust), len(tor), len(orig), len(singles))
    faust, tor, orig, singles = faust[:n], tor[:n], orig[:n].astype(np.float64), singles[:n]

    rows = []
    for label, t0, t1 in regions():
        i0, i1 = int(t0 * SR), min(int(t1 * SR), n)
        f, t, o, s = faust[i0:i1], tor[i0:i1], orig[i0:i1], singles[i0:i1]
        row = {
            "label": label, "t0": t0, "t1": t1,
            "mrstft_torch_vs_faust": mrs(t, f),
            "mrstft_singles_vs_faust": mrs(s, f),
            "mrstft_torch_vs_singles": mrs(t, s),
            "rms_ratio_torch_faust": rms(t) / (rms(f) + 1e-20),
            "rms_ratio_singles_faust": rms(s) / (rms(f) + 1e-20),
            "rms_ratio_torch_singles": rms(t) / (rms(s) + 1e-20),
            "mel_torch_vs_faust": mel_dist(f, t),
            "mel_torch_vs_singles": mel_dist(s, t),
            "mrstft_faust_vs_orig": mrs(f, o),
            "mrstft_torch_vs_orig": mrs(t, o),
            "rms_faust": rms(f), "rms_torch": rms(t), "rms_orig": rms(o),
            "corr_torch_faust": float(np.corrcoef(t, f)[0, 1]),
            "corr_torch_singles": float(np.corrcoef(t, s)[0, 1]),
        }
        rows.append(row)
        print(f"  {label:10s} {t0:5.2f}-{t1:5.2f}s  mrstft t/poly {row['mrstft_torch_vs_faust']:.3f}"
              f"  singles/poly {row['mrstft_singles_vs_faust']:.3f}"
              f"  t/singles {row['mrstft_torch_vs_singles']:.3f}"
              f"  rms t/poly {row['rms_ratio_torch_faust']:.3f} t/singles "
              f"{row['rms_ratio_torch_singles']:.3f}  corr t/singles {row['corr_torch_singles']:+.3f}"
              f"  vs orig f {row['mrstft_faust_vs_orig']:.3f} t {row['mrstft_torch_vs_orig']:.3f}")
    worst = max(rows, key=lambda r: r["mrstft_torch_vs_singles"])
    worst_lvl = max(rows, key=lambda r: abs(math.log(r["rms_ratio_torch_singles"] + 1e-20)))
    print(f"  worst region by port error: {worst['label']} {worst['mrstft_torch_vs_singles']:.3f}; "
          f"by level: {worst_lvl['label']} x{worst_lvl['rms_ratio_torch_singles']:.3f}")
    record(regions=rows,
           whole_clip={"mrstft_torch_vs_singles": mrs(tor, singles),
                       "rel_l2_torch_vs_singles": rel_l2(tor, singles),
                       "rms_ratio_torch_singles": rms(tor) / rms(singles),
                       "mel_torch_vs_singles": mel_dist(singles, tor),
                       "corr_torch_singles": float(np.corrcoef(tor, singles)[0, 1])})
    print(f"  whole clip torch vs faust-singles: mrstft {mrs(tor, singles):.4f}  "
          f"rel L2 {rel_l2(tor, singles):.4f}  rms ratio {rms(tor) / rms(singles):.4f}  "
          f"corr {float(np.corrcoef(tor, singles)[0, 1]):+.4f}  mel {mel_dist(singles, tor):.4f}")


# ---------------------------------------------------------------- stereo


def lr_corr(a: np.ndarray) -> float:
    return float(np.corrcoef(a[0].astype(np.float64), a[1].astype(np.float64))[0, 1])


def step_stereo(device_name: str = "cpu") -> None:
    device = torch.device(device_name)
    z = normalized()
    faust = load("faust")
    tor = load("torch_cpu")
    res = {
        "faust_lr_corr": lr_corr(faust),
        "torch_lr_corr": lr_corr(tor),
        "faust_mid_side_db": 20.0 * math.log10(rms(faust.mean(axis=0))
                                               / (rms(faust[0] - faust[1]) / 2.0 + 1e-20)),
        "torch_mid_side_db": 20.0 * math.log10(rms(tor.mean(axis=0))
                                               / (rms(tor[0] - tor[1]) / 2.0 + 1e-20)),
    }

    # fully wet: the reverb surrogate with nothing to hide behind. revWet = 0.1193 at
    # patch.json, so the reverb stage is only ever seen through a 12 percent send.
    notes = load_notes()
    r = renderer(notes)
    r.set_params({**denorm(z), **FULLY_WET})
    wet_f = r.render(DUR)
    wet_t = torch_render(z, device, overrides=FULLY_WET)
    n = min(wet_f.shape[1], wet_t.shape[1])
    res.update(
        faust_wet_lr_corr=lr_corr(wet_f),
        torch_wet_lr_corr=lr_corr(wet_t),
        wet_mrstft=mrs(wet_t.mean(axis=0)[:n], wet_f.mean(axis=0)[:n]),
        wet_rel_l2=rel_l2(wet_t.mean(axis=0)[:n], wet_f.mean(axis=0)[:n]),
        wet_rms_ratio=rms(wet_t[:, :n]) / (rms(wet_f[:, :n]) + 1e-20),
        wet_bands=band_table(wet_t.mean(axis=0)[:n].astype(np.float64),
                             wet_f.mean(axis=0)[:n].astype(np.float64), "revWet=1"),
    )
    record(stereo=res)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------- dry core


def step_core(device_name: str = "cpu") -> None:
    """Blame split: the osc+filter+env core with the three wet sends at zero."""
    device = torch.device(device_name)
    z = normalized()
    notes = load_notes()
    r = renderer(notes)
    r.set_params({**denorm(z), **FX_OFF})
    core_f = r.render(DUR)
    core_t = torch_render(z, device, overrides=FX_OFF)
    n = min(core_f.shape[1], core_t.shape[1])
    fm = core_f.mean(axis=0)[:n].astype(np.float64)
    tm = core_t.mean(axis=0)[:n].astype(np.float64)
    # the same core under the note-independence the surrogate assumes, so the core port
    # error is not read through the voice-allocation model error again
    singles = np.zeros(n, dtype=np.float64)
    for note in notes:
        r.set_notes([note])
        singles += r.render(DUR).mean(axis=0)[:n].astype(np.float64)
    target, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    res = {
        "core_mrstft": mrs(tm, fm),
        "core_rel_l2": rel_l2(tm, fm),
        "core_rms_ratio": rms(tm) / (rms(fm) + 1e-20),
        "core_mel": mel_dist(fm, tm),
        "core_loss_faust": objective_loss(fm, target),
        "core_loss_torch": objective_loss(tm, target),
        "core_loss_faust_singles": objective_loss(singles, target),
        "core_vs_singles_mrstft": mrs(tm, singles),
        "core_vs_singles_rel_l2": rel_l2(tm, singles),
        "core_vs_singles_rms_ratio": rms(tm) / (rms(singles) + 1e-20),
        "core_vs_singles_corr": float(np.corrcoef(tm, singles)[0, 1]),
        "core_vs_singles_mel": mel_dist(singles, tm),
        "core_bands": band_table(tm, fm, "fx off vs poly"),
        "core_bands_vs_singles": band_table(tm, singles, "fx off vs singles"),
    }
    record(core=res)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------- metric floor


def step_metric() -> None:
    """What does mrstft read for signals that are almost identical?

    Needed to read every other number in this audit. The tilt-only fx case agrees to a
    relative L2 of 1e-4 and still measures mrstft 0.13, because the log-magnitude term
    compares near-silent bins where a float32 rounding difference is a large ratio. So
    mrstft has a floor well above zero, and a "gap" of 0.18 has to be read against it.
    """
    x = load("faust").mean(axis=0).astype(np.float64)
    gen = np.random.default_rng(7)
    noise = gen.standard_normal(len(x))
    noise /= np.linalg.norm(noise)
    rows = []
    for eps in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        y = x + eps * np.linalg.norm(x) * noise
        rows.append({"rel_l2": eps, "mrstft": mrs(y, x), "mel": mel_dist(x, y)})
        print(f"  rel L2 {eps:.0e} -> mrstft {rows[-1]['mrstft']:.4f}  mel {rows[-1]['mel']:.4f}")
    # and a pure gain error, which the loss does see
    for g in (1.01, 1.1, 1.25):
        print(f"  gain x{g} -> mrstft {mrs(g * x, x):.4f}  mel {mel_dist(x, g * x):.4f}")
        rows.append({"gain": g, "mrstft": mrs(g * x, x), "mel": mel_dist(x, g * x)})
    record(metric_floor=rows)


# ---------------------------------------------------------------- fx chain


FX_MARKER = "// ---------------- shared effects ----------------"


def effect_dsp() -> str:
    """The real `effect` chain from synth.DSP as a standalone 2-in 2-out probe.

    Sliced out of the authoritative source rather than retyped, so it cannot drift.
    """
    from synth import DSP

    return 'import("stdfaust.lib");\n' + DSP.split(FX_MARKER)[1] + "\nprocess = effect;\n"


def step_fx(device_name: str = "cpu") -> None:
    """The effects port measured with the SAME input on both sides.

    verify_torch_synth.step_dry fed Faust's fx-off render into torch_fx and blamed the
    resulting 0.5414 on the reverb, but that render was made with tilt = 0, which is a
    Butterworth shelf pair with a zero at each crossover rather than a bypass. Its input
    was therefore already notched at 300 and 1200 Hz while the reference was not, so the
    number measures the notch, not the reverb. Feeding one signal through both chains
    removes the question.
    """
    from faust_probe import render_fx

    device = torch.device(device_name)
    z = normalized()
    vals = denorm(z)
    pad = TorchPad(load_notes(), default_n_samples(), device)
    p = {k: v.detach() for k, v in Patch(z).to(device).values().items()}
    with torch.no_grad():
        mono = pad.voice_output(p).sum(dim=0)
    x = torch.stack([mono, mono])
    xnp = x.cpu().numpy().astype(np.float32)
    dsp = effect_dsp()

    res = {}
    cases = {
        "patch": {},
        "reverb_only": {"chDepth": 0.0, "dlyWet": 0.0, "revWet": 1.0, "tilt": 0.0, "outGain": 1.0},
        "chorus_delay_tilt_only": {"revWet": 0.0},
        "tilt_gain_only": {"chDepth": 0.0, "dlyWet": 0.0, "revWet": 0.0},
    }
    for label, over in cases.items():
        params = {**vals, **over}
        fa = render_fx(dsp, {k: v for k, v in params.items() if k in FX_SLIDERS}, xnp)
        pt = dict(p)
        for k, v in over.items():
            pt[k] = torch.tensor(float(v), device=device)
        with torch.no_grad():
            ta = torch_fx.effects(x, pt, SR).cpu().numpy()
        n = min(fa.shape[1], ta.shape[1])
        fm, tm = fa.mean(axis=0)[:n].astype(np.float64), ta.mean(axis=0)[:n].astype(np.float64)
        res[label] = {
            "mrstft": mrs(tm, fm),
            "rel_l2": rel_l2(tm, fm),
            "rms_ratio": rms(tm) / (rms(fm) + 1e-20),
            "corr": float(np.corrcoef(tm, fm)[0, 1]),
            "lr_corr_faust": lr_corr(fa[:, :n]),
            "lr_corr_torch": lr_corr(ta[:, :n]),
            "bands": band_table(tm, fm, f"fx {label}"),
        }
        print(f"  {label:24s} mrstft {res[label]['mrstft']:.4f}  rel L2 {res[label]['rel_l2']:.4f}"
              f"  rms ratio {res[label]['rms_ratio']:.4f}  corr {res[label]['corr']:+.4f}"
              f"  LR corr faust {res[label]['lr_corr_faust']:+.3f} torch "
              f"{res[label]['lr_corr_torch']:+.3f}")
    record(fx=res)


# ---------------------------------------------------------------- sweep


def step_sweep(device_name: str = "cpu", params: tuple[str, ...] = SWEEP_PARAMS) -> None:
    """One parameter at a time across its full normalized range, both renderers.

    The gap at a single point is weak evidence. What decides usability is whether the
    surrogate loss moves the way the Faust loss moves, so both losses are recorded at
    every point and scored on bias spread and on argmin agreement per parameter.
    """
    device = torch.device(device_name)
    z0 = normalized()
    notes = load_notes()
    r = renderer(notes)
    target, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    pad = TorchPad(notes, default_n_samples(), device)

    rows = []
    t_start = time.perf_counter()
    for name in params:
        i = PARAM_INDEX[name]
        for v in SWEEP_POINTS:
            z = z0.copy()
            z[i] = v
            r.set_params(denorm(z))
            fa = r.render(DUR).mean(axis=0)
            ta = torch_render(z, device, pad=pad).mean(axis=0)
            n = min(len(fa), len(ta))
            fa, ta = fa[:n].astype(np.float64), ta[:n].astype(np.float64)
            rows.append({
                "param": name, "z": v,
                "real": denorm(z)[name],
                "faust_loss": objective_loss(fa, target),
                "torch_loss": objective_loss(ta, target),
                "mrstft": mrs(ta, fa),
                "rms_ratio": rms(ta) / (rms(fa) + 1e-20),
            })
        done = [x for x in rows if x["param"] == name]
        bias = [x["torch_loss"] - x["faust_loss"] for x in done]
        fi = int(np.argmin([x["faust_loss"] for x in done]))
        ti = int(np.argmin([x["torch_loss"] for x in done]))
        print(f"  {name:8s} bias {min(bias):+.3f}..{max(bias):+.3f}  "
              f"mrstft max {max(x['mrstft'] for x in done):.3f}  "
              f"argmin faust z={done[fi]['z']} torch z={done[ti]['z']}"
              f"{'' if fi == ti else '  MISMATCH'}  ({time.perf_counter() - t_start:.0f}s)")

    fl = np.array([x["faust_loss"] for x in rows])
    tl = np.array([x["torch_loss"] for x in rows])
    bias = tl - fl
    rank_f = np.argsort(np.argsort(fl))
    rank_t = np.argsort(np.argsort(tl))
    spearman = float(np.corrcoef(rank_f, rank_t)[0, 1])
    mism = []
    for name in params:
        done = [x for x in rows if x["param"] == name]
        fi = int(np.argmin([x["faust_loss"] for x in done]))
        ti = int(np.argmin([x["torch_loss"] for x in done]))
        if fi != ti:
            mism.append({"param": name, "faust_argmin_z": done[fi]["z"],
                         "torch_argmin_z": done[ti]["z"],
                         "faust_loss_at_torch_argmin": done[ti]["faust_loss"],
                         "faust_loss_at_faust_argmin": done[fi]["faust_loss"]})
    worst = rows[int(np.argmax([x["mrstft"] for x in rows]))]
    worst_bias = rows[int(np.argmax(np.abs(bias)))]
    res = {
        "points": rows,
        "n_points": len(rows),
        "bias_mean": float(bias.mean()),
        "bias_std": float(bias.std()),
        "bias_min": float(bias.min()),
        "bias_max": float(bias.max()),
        "pearson_loss": float(np.corrcoef(fl, tl)[0, 1]),
        "spearman_loss": spearman,
        "worst_mrstft": worst,
        "worst_bias": worst_bias,
        "argmin_mismatches": mism,
        "n_argmin_mismatch": len(mism),
        "seconds": time.perf_counter() - t_start,
    }
    record(sweep=res)
    print(f"  bias mean {bias.mean():+.4f} std {bias.std():.4f} "
          f"range {bias.min():+.4f}..{bias.max():+.4f}")
    print(f"  loss agreement: pearson {res['pearson_loss']:.4f} spearman {spearman:.4f}")
    print(f"  worst gap: {worst['param']}={worst['z']} mrstft {worst['mrstft']:.3f} "
          f"(rms ratio {worst['rms_ratio']:.3f})")
    print(f"  worst bias: {worst_bias['param']}={worst_bias['z']} "
          f"torch {worst_bias['torch_loss']:.4f} faust {worst_bias['faust_loss']:.4f}")
    print(f"  argmin mismatches: {len(mism)}/{len(params)} "
          f"({', '.join(m['param'] for m in mism) or 'none'})")


# ---------------------------------------------------------------- voice gain


def step_gain() -> None:
    """Is a voice's gain really velocity/127? One note, two velocities, measured."""
    z = normalized()
    note = load_notes()[0]
    r = renderer([note])
    r.set_params({**denorm(z), **FX_OFF})
    out = {}
    for vel in (127, 100, 50, 25):
        r.set_notes([(note[0], vel, note[2], note[3])])
        out[vel] = rms(r.render(DUR).mean(axis=0))
    res = {"rms": {str(k): v for k, v in out.items()},
           "ratio_100_50_measured": out[100] / out[50],
           "ratio_100_50_expected": 2.0,
           "max_rel_error_vs_vel_over_127": max(
               abs(out[v] / out[127] - v / 127.0) / (v / 127.0) for v in (100, 50, 25))}
    record(voice_gain=res)
    for v in (127, 100, 50, 25):
        print(f"  vel {v:3d}: rms {out[v]:.6f}  rms/rms127 {out[v] / out[127]:.6f}  "
              f"vel/127 {v / 127.0:.6f}")
    print(f"  ratio rms(100)/rms(50) = {res['ratio_100_50_measured']:.6f} (expected 2)")
    print(f"  worst relative error against velocity/127: "
          f"{res['max_rel_error_vs_vel_over_127']:.2e}")


# ---------------------------------------------------------------- port sweep


def step_port(device_name: str = "cpu", n_notes: int = 6,
              params: tuple[str, ...] = SWEEP_PARAMS) -> None:
    """Sweep the PORT error alone, with the voice-allocation model error removed.

    On the full score every torch-vs-Faust distance is swamped by the fact that Faust
    reuses voice slots without clearing their phase from about 4.3 s in, so a sweep of
    that distance cannot see a port problem. On a short prefix of the score no slot is
    ever reused, Faust is then exactly the sum of its notes, and the surrogate models
    the same thing, so what is left is only the port. The precondition is asserted, not
    assumed: the prefix is checked against its own sum of one-note renders first.
    """
    device = torch.device(device_name)
    z0 = normalized()
    notes = load_notes()[:n_notes]
    r = renderer(notes)
    r.set_params(denorm(z0))
    poly = r.render(DUR).mean(axis=0).astype(np.float64)
    singles = np.zeros_like(poly)
    for note in notes:
        r.set_notes([note])
        singles += r.render(DUR).mean(axis=0).astype(np.float64)
    superpose = rel_l2(singles, poly)
    print(f"  precondition: {n_notes} notes, poly vs sum of singles rel L2 {superpose:.2e}"
          f"{'' if superpose < 1e-3 else '   NOT NOTE-INDEPENDENT, sweep is contaminated'}")
    r.set_notes(notes)

    pad = TorchPad(notes, default_n_samples(), device)
    rows = []
    t_start = time.perf_counter()
    for name in params:
        i = PARAM_INDEX[name]
        for v in SWEEP_POINTS:
            z = z0.copy()
            z[i] = v
            r.set_params(denorm(z))
            fa = r.render(DUR).mean(axis=0)
            ta = torch_render(z, device, pad=pad).mean(axis=0)
            n = min(len(fa), len(ta))
            fa, ta = fa[:n].astype(np.float64), ta[:n].astype(np.float64)
            rows.append({
                "param": name, "z": v, "real": denorm(z)[name],
                "mrstft": mrs(ta, fa),
                "rel_l2": rel_l2(ta, fa),
                "rms_ratio": rms(ta) / (rms(fa) + 1e-20),
                "corr": float(np.corrcoef(ta, fa)[0, 1]),
                "mel": mel_dist(fa, ta),
            })
        done = [x for x in rows if x["param"] == name]
        bad = max(done, key=lambda x: x["mrstft"])
        print(f"  {name:8s} mrstft {min(x['mrstft'] for x in done):.3f}..{bad['mrstft']:.3f} "
              f"(worst at z={bad['z']}, rel L2 {bad['rel_l2']:.3f}, rms ratio "
              f"{bad['rms_ratio']:.3f}, corr {bad['corr']:+.3f})  "
              f"({time.perf_counter() - t_start:.0f}s)")

    worst = rows[int(np.argmax([x["mrstft"] for x in rows]))]
    worst_lvl = rows[int(np.argmax([abs(math.log(x["rms_ratio"] + 1e-20)) for x in rows]))]
    res = {
        "n_notes": n_notes,
        "superposition_rel_l2": superpose,
        "points": rows,
        "mrstft_median": float(np.median([x["mrstft"] for x in rows])),
        "mrstft_max": worst["mrstft"],
        "worst": worst,
        "worst_level": worst_lvl,
        "seconds": time.perf_counter() - t_start,
    }
    record(port_sweep=res)
    print(f"  port error over {len(rows)} points: median mrstft {res['mrstft_median']:.3f}, "
          f"max {worst['mrstft']:.3f} at {worst['param']}={worst['z']} "
          f"(real {worst['real']:.4g}, rel L2 {worst['rel_l2']:.3f}, corr {worst['corr']:+.3f})")
    print(f"  worst level error: {worst_lvl['param']}={worst_lvl['z']} "
          f"rms ratio {worst_lvl['rms_ratio']:.3f}")


# ---------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["floor", "bands", "regions", "stereo", "core", "fx", "metric", "sweep", "port", "gain"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--params", default=None, help="sweep: comma separated parameter names")
    args = ap.parse_args()
    if args.step == "floor":
        step_floor()
    elif args.step == "bands":
        step_bands()
    elif args.step == "regions":
        step_regions()
    elif args.step == "stereo":
        step_stereo(args.device)
    elif args.step == "core":
        step_core(args.device)
    elif args.step == "fx":
        step_fx(args.device)
    elif args.step == "metric":
        step_metric()
    elif args.step == "gain":
        step_gain()
    elif args.step == "port":
        names = tuple(args.params.split(",")) if args.params else SWEEP_PARAMS
        step_port(args.device, params=names)
    else:
        names = tuple(args.params.split(",")) if args.params else SWEEP_PARAMS
        step_sweep(args.device, names)


if __name__ == "__main__":
    main()
