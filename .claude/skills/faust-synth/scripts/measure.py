"""Measure a Faust patch on the axes that cannot be checked by listening once.

Whoever writes the DSP cannot hear it, and the person who can hear it auditions one
pattern in one register at default settings. Everything outside that window ships
unchecked. Spike 2 shipped five patches that way and the defects it hid were all of the
same kind: a macro that clips at an extreme nobody moved it to, a control that changes
level rather than timbre, a patch whose voices never decorrelate.

The division of labour this assumes: the ear judges whether it is the right sound, and
this judges whether the patch is internally sound. It deliberately does not score
similarity to anything. There is no reference audio here, only a description, and the
one similarity metric this project has calibrated is a weak ranker even with a reference
(cos theta 0.68 for deliberately shuffled audio against 0.84 for a best-possible oracle).

Most of what comes back is a report, not a verdict, because most of it needs intent to
read: whether 6 dB of level change on `brightness` is a defect depends on what the macro
was meant to do. Only checks that are wrong under ANY intent are raised as failures.

Usage:
    uv run python <skill>/scripts/measure.py patch.dsp bass    # one patch, human report
    uv run python <skill>/scripts/measure.py --check           # the whole example set
    uv run python <skill>/scripts/measure.py --update          # re-record what it expects

`--check` is the regression pass over `references/examples/`. It used to be a paragraph
in `measured.md` asking whoever changed this file to re-run six patches and diff the
output by eye, which is a claim nothing enforced: expectations written as prose drift
from the code the moment someone does not do it by hand. What each patch is expected to
report now lives in `references/examples/expected.json`, and changing it is a diff.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

from faust_render import PATTERNS, Instrument

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references", "examples")
EXPECTED = os.path.join(EXAMPLES, "expected.json")

# Ten octave edges from 31.25 Hz, giving nine bands up to 16 kHz. Octave rather than
# third-octave because this compares gross spectral shape between two renders, and
# narrow bands mostly resolve individual partials moving, which is not the question.
BAND_EDGES = np.array([31.25 * 2.0 ** k for k in range(10)])

# A band this far below the loudest one is empty space between partials, where small
# absolute differences are large in dB and mean nothing.
BAND_FLOOR_DB = 60.0


@dataclass
class Finding:
    severity: str            # "fail" is wrong under any intent; "warn" needs a human
    message: str
    macro: str | None = None


@dataclass
class Report:
    name: str
    findings: list[Finding] = field(default_factory=list)
    macros: list[dict] = field(default_factory=list)
    voice: dict = field(default_factory=dict)
    register: list[dict] = field(default_factory=list)
    release: dict = field(default_factory=dict)
    fingerprint: dict = field(default_factory=dict)

    def fail(self, msg: str, macro: str | None = None) -> None:
        self.findings.append(Finding("fail", msg, macro))

    def warn(self, msg: str, macro: str | None = None) -> None:
        self.findings.append(Finding("warn", msg, macro))


# ---------------------------------------------------------------- primitives

def mono(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.mean(axis=0) if a.ndim == 2 else a


def rms_db(a: np.ndarray) -> float:
    x = mono(a)
    return float(20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-12)))


def peak(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a)))) if np.asarray(a).size else 0.0


def power_spectrum(a: np.ndarray, sr: int, n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Average power spectrum over Hann frames. Returns (freqs, power)."""
    x = mono(a)
    hop = n_fft // 2
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft)
    frames = 1 + (len(x) - n_fft) // hop
    acc = np.zeros(n_fft // 2 + 1)
    for i in range(frames):
        seg = x[i * hop: i * hop + n_fft] * win
        acc += np.abs(np.fft.rfft(seg)) ** 2
    return np.fft.rfftfreq(n_fft, 1 / sr), acc / max(frames, 1)


def band_profile(a: np.ndarray, sr: int) -> np.ndarray:
    """Energy per octave band, in dB. Carries overall level."""
    freqs, power = power_spectrum(a, sr)
    out = np.empty(len(BAND_EDGES) - 1)
    for i in range(len(out)):
        sel = (freqs >= BAND_EDGES[i]) & (freqs < BAND_EDGES[i + 1])
        out[i] = 10 * np.log10(max(float(power[sel].sum()), 1e-20))
    return out


def shape_profile(a: np.ndarray, sr: int) -> np.ndarray:
    """Octave-band profile with overall level removed, so only spectral SHAPE remains.

    This is what separates a timbre control from a volume control. A macro that moves
    band_profile but leaves shape_profile flat is changing loudness and nothing else,
    whatever its label claims.
    """
    b = band_profile(a, sr)
    return b - b.mean()


def shape_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Mean absolute octave-band difference after level normalisation, in dB."""
    sa, sb = shape_profile(a, sr), shape_profile(b, sr)
    ba, bb = band_profile(a, sr), band_profile(b, sr)
    live = (ba > ba.max() - BAND_FLOOR_DB) | (bb > bb.max() - BAND_FLOOR_DB)
    if not live.any():
        return 0.0
    return float(np.mean(np.abs(sa[live] - sb[live])))


def side_db(a: np.ndarray) -> float:
    """Side energy relative to mid, in dB. Negative is narrow, 0 is fully decorrelated.

    Everything else here folds to mono, which cannot see a width control at all: a
    correct mid/side widener leaves the mono sum untouched by construction, so without
    this the sweep reports it as a macro that does nothing. That is the same mono
    blindness already recorded for the stage-2 objective, rediscovered here.
    """
    x = np.asarray(a, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return -np.inf
    mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
    m = float(np.sqrt(np.mean(mid ** 2)))
    s_ = float(np.sqrt(np.mean(side ** 2)))
    # floored: a perfectly mono render has zero side energy, and -240 dB against a
    # normal -20 dB makes every delta involving it unreadable
    return float(max(20 * np.log10(max(s_, 1e-12) / max(m, 1e-12)), -80.0))


def centroid_hz(a: np.ndarray, sr: int) -> float:
    freqs, power = power_spectrum(a, sr)
    total = power.sum()
    return float((freqs * power).sum() / total) if total > 0 else 0.0


def centroid_track(a: np.ndarray, sr: int, n_fft: int = 2048,
                   hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Spectral centroid frame by frame, for movement within a single note."""
    x = mono(a)
    win = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    times, cents = [], []
    for i in range(0, max(len(x) - n_fft, 0), hop):
        p = np.abs(np.fft.rfft(x[i:i + n_fft] * win)) ** 2
        tot = p.sum()
        # skip frames that are essentially silence, whose centroid is meaningless
        if tot < 1e-10:
            continue
        times.append(i / sr)
        cents.append((freqs * p).sum() / tot)
    return np.array(times), np.array(cents)


# ---------------------------------------------------------------- measurements

def tail_energy_db(inst: Instrument, pitch: int, params: dict[str, float]) -> float:
    """Energy well after note-off, where a release control is the only thing acting."""
    a = inst.render([(pitch, 100, 0.0, 0.5)], 9.0, params)
    x = mono(a)
    seg = x[int(1.0 * inst.sr):int(2.5 * inst.sr)]
    return float(20 * np.log10(max(float(np.sqrt(np.mean(seg ** 2))), 1e-12)))


def macro_sweep(inst: Instrument, pattern: str, pitch: int, rep: Report) -> None:
    """Render each macro at min, default and max, and compare level against shape."""
    notes, dur = PATTERNS[pattern]
    for m in inst.macros:
        label, lo, hi = m["label"], m["lo"], m["hi"]
        at = {}
        for tag, val in (("min", lo), ("default", m["default"]), ("max", hi)):
            at[tag] = inst.render(notes, dur, {label: val})

        r = {t: rms_db(a) for t, a in at.items()}
        p = {t: peak(a) for t, a in at.items()}
        rms_delta = max(r.values()) - min(r.values())
        shape_delta = shape_distance(at["min"], at["max"], inst.sr)
        c_lo, c_hi = centroid_hz(at["min"], inst.sr), centroid_hz(at["max"], inst.sr)
        w = {t: side_db(a) for t, a in at.items()}
        width_delta = abs(w["max"] - w["min"])

        rep.macros.append({
            "macro": label, "range": (lo, hi), "default": m["default"],
            "rms_delta_dB": round(rms_delta, 2),
            "shape_delta_dB": round(shape_delta, 2),
            "centroid_ratio": round(c_hi / c_lo, 3) if c_lo > 0 else 0.0,
            "width_delta_dB": round(width_delta, 2),
            "peak_min": round(p["min"], 3), "peak_max": round(p["max"], 3),
        })

        # Clipping is wrong regardless of what the macro means, and a range the patch
        # declares is a range a player will turn it to.
        for tag in ("min", "max"):
            if p[tag] > 1.0:
                rep.fail(f"peaks {p[tag]:.3f} at {label}={lo if tag=='min' else hi:g} "
                         f"({tag} of its declared range)", label)

        # Neither of the next two needs to know what the macro was FOR.
        if rms_delta < 0.3 and shape_delta < 0.5 and width_delta < 0.5:
            # The audition pattern may simply not expose this macro. A release control
            # is invisible on a pattern whose chords overlap and whose render ends
            # before the tail does, which is how `warm-pad`'s working `tail` first got
            # reported as dead. Re-test on one short note with a long silence after it
            # before calling anything inert.
            tail_delta = abs(tail_energy_db(inst, pitch, {label: hi})
                             - tail_energy_db(inst, pitch, {label: lo}))
            if tail_delta < 1.0:
                rep.fail(f"does nothing: {rms_delta:.2f} dB level, {shape_delta:.2f} dB "
                         f"shape, {width_delta:.2f} dB width, and {tail_delta:.2f} dB "
                         f"in the release tail", label)
            else:
                rep.warn(f"is inert on the {pattern} pattern but moves the release tail "
                         f"by {tail_delta:.1f} dB, so the audition pattern does not "
                         f"exercise it", label)
        elif rms_delta > 6.0 and shape_delta < 1.5 and width_delta < 1.5:
            rep.fail(f"is a volume control: {rms_delta:.1f} dB of level for "
                     f"{shape_delta:.2f} dB of spectral shape", label)
        elif rms_delta > 6.0 and rms_delta > shape_delta:
            rep.warn(f"moves level more than shape ({rms_delta:.1f} dB against "
                     f"{shape_delta:.1f} dB); correct for an envelope-length control, "
                     f"a defect for a timbre control", label)


def voice_coherence(inst: Instrument, pitch: int, rep: Report) -> None:
    """One note against four of the same note.

    Faust instantiates N identical copies of `process`, so voices that contain noise or
    free-running LFOs produce the same samples and sum coherently. +12.04 dB means the
    voices are bit-identical; +6.02 dB means they are independent. Which one is CORRECT
    depends on the patch, so this reports rather than judges: a deterministic patch is
    legitimately coherent, but one that claims per-voice drift or analog movement is
    broken if it is.
    """
    one = inst.render([(pitch, 100, 0.0, 1.5)], 3.5)
    four = inst.render([(pitch, 100, 0.0, 1.5)] * 4, 3.5)
    delta = rms_db(four) - rms_db(one)
    rep.voice = {
        "pitch": pitch,
        "unison_gain_dB": round(delta, 2),
        "verdict": ("bit-identical voices" if delta > 10.5 else
                    "independent voices" if delta < 7.5 else "partly decorrelated"),
    }


def register_sweep(inst: Instrument, rep: Report) -> None:
    """The same note across five octaves. Audition patterns only cover one."""
    for pitch in (36, 48, 60, 72, 84):
        a = inst.render([(pitch, 100, 0.0, 1.2)], 3.0)
        rep.register.append({
            "pitch": pitch,
            "rms_dB": round(rms_db(a), 2),
            "centroid_Hz": round(centroid_hz(a, inst.sr), 1),
            "peak": round(peak(a), 3),
        })
    levels = [r["rms_dB"] for r in rep.register]
    spread = max(levels) - min(levels)
    if spread > 12.0:
        detail = ", ".join(f"{r['pitch']}:{r['rms_dB']:.0f}" for r in rep.register)
        rep.warn(f"level varies {spread:.1f} dB across MIDI 36-84 ({detail})")
    for r in rep.register:
        if r["peak"] > 1.0:
            rep.fail(f"peaks {r['peak']:.3f} at MIDI {r['pitch']}, outside the audition register")


def release_fit(inst: Instrument, pitch: int, rep: Report) -> None:
    """Whether the tail finishes inside the render, or is being cut off mid-decay."""
    dur = 6.0
    a = inst.render([(pitch, 100, 0.0, 1.0)], dur)
    x = mono(a)
    tail = x[-int(0.05 * inst.sr):]
    tail_db = 20 * np.log10(max(float(np.sqrt(np.mean(tail ** 2))), 1e-12))
    pk_db = 20 * np.log10(max(peak(a), 1e-12))
    rel = tail_db - pk_db
    rep.release = {"tail_below_peak_dB": round(rel, 1), "render_s": dur}
    if rel > -60.0:
        rep.fail(f"still sounding at {rel:.0f} dB below peak when the {dur:g} s render "
                 f"ends, so the tail is being truncated rather than decaying")


def note_trajectory(inst: Instrument, pitch: int, rep: Report) -> None:
    """How fast the timbre actually moves inside one note.

    A macro can declare a 0.3 s decay and deliver most of its travel in 30 ms, because an
    exponential envelope through an exponential cutoff mapping collapses. The declared
    unit and the audible one are different numbers.
    """
    a = inst.render([(pitch, 100, 0.0, 1.5)], 3.0)
    t, c = centroid_track(a, inst.sr)
    if len(c) < 4:
        return
    start, end = c[0], c[-1]
    travel = end - start
    pct = {}
    if abs(travel) > 1.0:
        for frac in (0.5, 0.9):
            target = start + travel * frac
            hit = np.where((c - target) * np.sign(travel) >= 0)[0]
            pct[f"t{int(frac*100)}_s"] = round(float(t[hit[0]]), 3) if len(hit) else None
    rep.fingerprint = {
        "centroid_start_Hz": round(float(start), 1),
        "centroid_end_Hz": round(float(end), 1),
        "bands_dB": [round(float(v), 1) for v in band_profile(a, inst.sr)],
        **pct,
    }


# ---------------------------------------------------------------- driver

def measure(dsp_path: str, pattern: str) -> Report:
    with open(dsp_path) as fh:
        dsp = fh.read()
    inst = Instrument(dsp)
    rep = Report(name=dsp_path.split("/")[-1])

    notes, dur = PATTERNS[pattern]
    pitch = int(np.median([n[0] for n in notes]))

    a = inst.render(notes, dur)
    if not np.all(np.isfinite(a)):
        rep.fail("render contains non-finite samples")
    dc = float(np.mean(mono(a)))
    if abs(dc) > 0.01:
        rep.fail(f"DC offset of {dc:+.3f}")

    macro_sweep(inst, pattern, pitch, rep)
    voice_coherence(inst, pitch, rep)
    register_sweep(inst, rep)
    release_fit(inst, pitch, rep)
    note_trajectory(inst, pitch, rep)
    return rep


def render_report(rep: Report) -> str:
    out = [f"=== {rep.name}"]
    fails = [f for f in rep.findings if f.severity == "fail"]
    warns = [f for f in rep.findings if f.severity == "warn"]
    out.append(f"{len(fails)} fail, {len(warns)} warn")

    if fails or warns:
        out.append("")
        for f in fails + warns:
            tag = "FAIL" if f.severity == "fail" else "warn"
            who = f"{f.macro} " if f.macro else ""
            out.append(f"  {tag}  {who}{f.message}")

    out.append("")
    out.append(f"  {'macro':<14}{'level':>8}{'shape':>8}{'width':>8}{'centroid':>10}"
               f"{'peak lo':>9}{'peak hi':>9}")
    for m in rep.macros:
        out.append(f"  {m['macro']:<14}{m['rms_delta_dB']:>7.1f}d"
                   f"{m['shape_delta_dB']:>7.1f}d{m['width_delta_dB']:>7.1f}d"
                   f"{m['centroid_ratio']:>10.2f}x"
                   f"{m['peak_min']:>9.3f}{m['peak_max']:>9.3f}")

    out.append("")
    out.append(f"  voices    {rep.voice['unison_gain_dB']:+.2f} dB for 4x unison "
               f"at MIDI {rep.voice['pitch']}: {rep.voice['verdict']}")
    reg = "  ".join(f"{r['pitch']}:{r['rms_dB']:.0f}dB/{r['centroid_Hz']:.0f}Hz"
                    for r in rep.register)
    out.append(f"  register  {reg}")
    out.append(f"  release   tail is {rep.release['tail_below_peak_dB']:.0f} dB below "
               f"peak at {rep.release['render_s']:g} s")
    if rep.fingerprint:
        fp = rep.fingerprint
        line = (f"  note      centroid {fp['centroid_start_Hz']:.0f} -> "
                f"{fp['centroid_end_Hz']:.0f} Hz")
        if fp.get("t50_s") is not None:
            line += f", 50% of that travel by {fp['t50_s']:.3f} s"
        if fp.get("t90_s") is not None:
            line += f", 90% by {fp['t90_s']:.3f} s"
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------- regression pass

# Renders here are deterministic, so in principle every digit should reproduce. These
# tolerances are for a different machine or a moved library version, where the last
# place or two moves without anything being wrong. A drift larger than this is a real
# change and should be read, not absorbed.
TOL = {"dB": 0.5, "ratio": 0.05, "peak": 0.02}


def finding_kind(message: str) -> str:
    """The message with its numbers removed.

    Findings are compared on this rather than on their text, because the text carries
    the measured value: `peaks 1.277 at width=1` would differ from `peaks 1.281` and
    report a regression where the only thing that moved is the last digit. The numbers
    are still compared, through the macro table.
    """
    return re.sub(r"[-+]?\d*\.?\d+", "#", message)


def summarize(rep: Report, pattern: str) -> dict:
    """The part of a report worth holding still, as plain data."""
    return {
        "pattern": pattern,
        "findings": [
            {"severity": f.severity, "macro": f.macro,
             "kind": finding_kind(f.message), "message": f.message}
            for f in rep.findings
        ],
        "macros": {
            m["macro"]: {
                "level_dB": round(m["rms_delta_dB"], 2),
                "shape_dB": round(m["shape_delta_dB"], 2),
                "width_dB": round(m["width_delta_dB"], 2),
                "centroid_ratio": round(m["centroid_ratio"], 3),
                "peak_lo": round(m["peak_min"], 3),
                "peak_hi": round(m["peak_max"], 3),
            }
            for m in rep.macros
        },
        "voices_dB": round(rep.voice["unison_gain_dB"], 2),
        "register_dB": [round(r["rms_dB"], 1) for r in rep.register],
        "release_below_peak_dB": round(rep.release["tail_below_peak_dB"], 0),
    }


def _diff_numbers(want: dict, got: dict, path: str, tol: float) -> list[str]:
    out = []
    for k in sorted(set(want) | set(got)):
        if k not in want:
            out.append(f"{path}.{k}: appeared, now {got[k]}")
        elif k not in got:
            out.append(f"{path}.{k}: gone, was {want[k]}")
        elif abs(float(want[k]) - float(got[k])) > tol:
            out.append(f"{path}.{k}: {want[k]} -> {got[k]}")
    return out


def compare(want: dict, got: dict) -> list[str]:
    """Every way the two summaries disagree, as one line each."""
    out: list[str] = []

    wf = {(f["severity"], f["macro"], f["kind"]): f for f in want["findings"]}
    gf = {(f["severity"], f["macro"], f["kind"]): f for f in got["findings"]}
    for key in gf.keys() - wf.keys():
        out.append(f"new finding: {gf[key]['severity']} {gf[key]['macro'] or ''} {gf[key]['message']}")
    for key in wf.keys() - gf.keys():
        out.append(f"finding gone: {wf[key]['severity']} {wf[key]['macro'] or ''} {wf[key]['message']}")

    for name in sorted(set(want["macros"]) | set(got["macros"])):
        if name not in want["macros"]:
            out.append(f"macro {name}: appeared")
            continue
        if name not in got["macros"]:
            out.append(f"macro {name}: gone")
            continue
        w, g = want["macros"][name], got["macros"][name]
        for field_, tol in (("level_dB", TOL["dB"]), ("shape_dB", TOL["dB"]),
                            ("width_dB", TOL["dB"]), ("centroid_ratio", TOL["ratio"]),
                            ("peak_lo", TOL["peak"]), ("peak_hi", TOL["peak"])):
            if abs(w[field_] - g[field_]) > tol:
                out.append(f"macro {name}.{field_}: {w[field_]} -> {g[field_]}")

    if abs(want["voices_dB"] - got["voices_dB"]) > TOL["dB"]:
        out.append(f"voices_dB: {want['voices_dB']} -> {got['voices_dB']}")
    if len(want["register_dB"]) != len(got["register_dB"]):
        out.append("register: different number of pitches")
    else:
        for i, (w, g) in enumerate(zip(want["register_dB"], got["register_dB"])):
            if abs(w - g) > TOL["dB"]:
                out.append(f"register[{i}]: {w} -> {g}")
    return out


def run_set(update: bool) -> int:
    """Measure every patch the expectations name, and either diff or re-record."""
    with open(EXPECTED) as fh:
        doc = json.load(fh)
    patches = doc["patches"]
    fresh, bad = {}, 0
    for name in sorted(patches):
        pattern = patches[name]["pattern"]
        got = summarize(measure(os.path.join(EXAMPLES, name), pattern), pattern)
        fresh[name] = got
        if update:
            print(f"  recorded  {name} ({pattern})")
            continue
        diffs = compare(patches[name], got)
        if diffs:
            bad += 1
            print(f"  CHANGED   {name}")
            for d in diffs:
                print(f"              {d}")
        else:
            f = sum(1 for x in got["findings"] if x["severity"] == "fail")
            w = len(got["findings"]) - f
            print(f"  ok        {name:<22} {f} fail, {w} warn")

    if update:
        doc["patches"] = fresh
        with open(EXPECTED, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {os.path.relpath(EXPECTED)}")
        return 0
    print(f"\n{len(patches) - bad} of {len(patches)} unchanged")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dsp", nargs="?", help="path to a .dsp file")
    ap.add_argument("pattern", nargs="?", choices=sorted(PATTERNS),
                    help="which measurement pattern to play")
    ap.add_argument("--check", action="store_true",
                    help="measure every patch in references/examples/ and diff against "
                         "expected.json; exits nonzero if anything moved")
    ap.add_argument("--update", action="store_true",
                    help="re-record expected.json from what the patches measure now")
    a = ap.parse_args()

    if a.check or a.update:
        sys.exit(run_set(a.update))
    if not a.dsp or not a.pattern:
        ap.error("give a .dsp and a pattern, or --check")
    print(render_report(measure(a.dsp, a.pattern)))
