"""Harmonic-number envelope oscillator: a fitted single-cycle spectrum, summed additively.

This is the SECONDARY axis of the timbre fit. On top of an absolute-frequency gain
curve it takes the partial-amplitude residual from 4.20 dB to 2.21 dB, so it earns a
place, but it is not the primary fix and it must not displace the existing texture.
Hence the scope: it replaces the single `oscmix(f)` waveform inside the unison, and
nothing else. The 7-voice detune, the sub, the chorus, the delay and the reverb stay.
That restraint is measured, not stylistic: a bank of clean sinusoids at exactly the
fitted partial amplitudes scores 1.949 on the window against the current render's
1.561, because the log-magnitude term punishes the silent gaps between partials that
detune and reverb are currently filling.

Why an explicit additive sum and not a pitched rdtable. A table read at an arbitrary
pitch aliases, and since the amplitudes are fitted the fit would quietly absorb that
aliasing into the table. An additive sum with a per-partial Nyquist gate is exactly
alias-free and realises the fitted amplitudes exactly. The gate is load-bearing rather
than decoration: the clip reaches MIDI 77, where harmonic 64 sits at 44.7 kHz, so
half the series has to switch itself off.

Why the amplitudes are a table and not CMA-ES parameters: they enter the spectrum
linearly, so they are least squares, exactly the argument chord.py makes.

Sign and phase convention. Partial h contributes -a[h]*sin(2*pi*h*f*t), which is the
Fourier series of Faust's own `os.sawtooth` when a[h] = 2/(pi*h). That makes morph=0
a genuine drop-in identity for the waveform being replaced, sign included, so the
integration agent can prove the change is inert before turning it on.
"""

from __future__ import annotations

import json
import sys
import time

import librosa
import numpy as np
import soundfile as sf

import chord
import synth
from bend2 import bend_curve
from faust_probe import render_gen

SR = chord.SR
H = chord.H_TABLE
CYCLE = 2048                      # single-cycle export length; 2048 holds 1024 harmonics
TAIL_H = 128                      # see with_saw_tail: 128 recovers the truncation deficit
WT_PATH = "out/wavetable.wav"
NPZ_PATH = "out/wt_osc_verify.npz"

# %(amps)s is the fitted table, %(h)d its length, %(gate)s the Nyquist factor (empty
# only for the negative control that proves the gate is what suppresses the aliasing).
FRAGMENT_TEMPLATE = """
// ---- harmonic-number envelope oscillator (wavetable stage) ----
// Additive rather than a pitched table read: exactly alias-free, and it realises the
// fitted amplitudes exactly instead of letting the fit absorb a table's aliasing.
// wtSaw is the Fourier series of os.sawtooth, so wtMorph=0 is that waveform truncated
// at harmonic %(h)d and wtMorph=1 is the fitted spectrum.
wtMorph = hslider("wtMorph", 1, 0, 1, 0.001);
wtFit(i) = ba.take(i+1, (%(amps)s));
wtSaw(i) = 2.0 / (ma.PI * float(i+1));
wtGain(i) = wtSaw(i) + wtMorph * (wtFit(i) - wtSaw(i));
wtPartial(f, i) = os.osc(f * float(i+1)) * wtGain(i)%(gate)s;
// Negated because 2*frac(x)-1 = -(2/pi)*sum sin(2*pi*h*x)/h: the minus is what makes
// the sawtooth end identical in sign to the os.sawtooth it stands in for.
wtosc(f) = 0.0 - (par(i, %(h)d, wtPartial(f, i)) :> _);
"""
GATE = " * (f * float(i+1) < 0.5 * ma.SR)"

PROBE_TEMPLATE = """
import("stdfaust.lib");
%s
freq = hslider("freq", 220, 20, 8000, 0.001);
process = %s;
"""


# ---------------------------------------------------------------- the table

def saw_table(h: int = H) -> np.ndarray:
    """The morph=0 end: harmonic amplitudes of a unit sawtooth, 2/(pi*h)."""
    return 2.0 / (np.pi * np.arange(1, h + 1, dtype=float))


def fit_harmonic_table(h: int = H, pitches=chord.PITCHES, iters: int = 40) -> dict:
    """Fit a[1..h] from the window's measured partial amplitudes.

    Model: amp[note k, harmonic h] = level_k * a[h], i.e. one shared harmonic-number
    envelope with a free level per note. Solved additively in the log domain, because
    dB is how the residual is reported and how the ear weights it, and weighted by
    linear amplitude so that partials down in the noise cannot steer the table.

    Rank one is deliberately all this returns. The absolute-frequency curve is the
    other agent's axis and the two are meant to be apportioned by a joint fit; a table
    fitted here alone will carry some of the frequency curve's job inside it.
    """
    amp = chord.harmonic_readout(chord.target(), pitches=pitches)
    tab = chord.partial_table(pitches)
    m = np.full((len(pitches), h), np.nan)
    for (k, hh, _), a in zip(tab, amp):
        if hh <= h:
            m[int(k), int(hh) - 1] = a
    if not np.isfinite(m).all():
        raise RuntimeError("window pitches do not all reach harmonic %d below 16 kHz" % h)

    lg = np.log(np.maximum(m, 1e-12))
    w = m / m.max()                       # amplitude weighting, so the fit follows the loud partials
    lvl = np.zeros(len(pitches))
    a_log = np.average(lg, axis=0, weights=w)
    for _ in range(iters):
        lvl = np.average(lg - a_log, axis=1, weights=w)
        a_log = np.average(lg - lvl[:, None], axis=0, weights=w)
    resid = lg - lvl[:, None] - a_log
    a = np.exp(a_log)
    # Match the sawtooth's total partial power so that wtMorph does not double as a
    # level control: a morph that changes loudness would fight outGain during the fit.
    a *= np.linalg.norm(saw_table(h)) / np.linalg.norm(a)
    db = 20.0 / np.log(10.0)
    return {
        "amps": a,
        "resid_rms_db": float(np.sqrt(np.average(resid ** 2, weights=w)) * db),
        "resid_max_db": float(np.abs(resid).max() * db),
        "levels": np.exp(lvl),
    }


def with_saw_tail(amps: np.ndarray, h_total: int) -> np.ndarray:
    """The fitted table below len(amps), the sawtooth's own 2/(pi h) above it.

    Why a tail is wanted at all. 64 harmonics cover the band the fit can see (99% of the
    window's energy is below 2.46 kHz), but stopping there empties everything above
    64*f0, and the loss's log-magnitude term notices an empty band even 40 dB down: the
    same sensitivity that makes PCM_16 cost 0.089 here. The tail leaves the top octaves
    exactly as loud as the waveform being replaced and lets only the fitted band change.
    Since the tail equals wtSaw there, wtMorph keeps acting on the fitted band alone.

    Measured on the window, sawtooth table at morph 0 against the 1.5605 baseline:
    64 harmonics 1.6640, 128 1.5780, 256 1.5762. So the whole 0.104 deficit is
    truncation, 128 buys back 0.086 of it and 256 buys 0.002 more for 3x the render
    time. Hence TAIL_H = 128.
    """
    a = np.asarray(amps, dtype=float)
    if h_total < len(a):
        raise ValueError("h_total must be at least len(amps)")
    return np.concatenate([a, saw_table(h_total)[len(a):]])


# ---------------------------------------------------------------- Faust source

def faust_fragment(amps: np.ndarray, gate: bool = True) -> str:
    """The Faust fragment for a given harmonic table. Defines wtosc(f) and wtMorph.

    gate=False exists only to render the negative control for the anti-aliasing check;
    never ship it.
    """
    a = np.asarray(amps, dtype=float)
    if a.ndim != 1:
        raise ValueError("amps must be 1-D")
    return FRAGMENT_TEMPLATE % {
        "amps": ", ".join(f"{v:.9g}" for v in a),
        "h": len(a),
        "gate": GATE if gate else "",
    }


def probe_dsp(amps: np.ndarray, gate: bool = True) -> str:
    """One bare oscillator, no envelope, no filter, no effects: what gets measured."""
    return PROBE_TEMPLATE % (faust_fragment(amps, gate), "wtosc(freq)")


SAW_DSP = PROBE_TEMPLATE % ("", "os.sawtooth(freq)")

_OSCMIX = "oscmix(f) = os.sawtooth(f) * (1.0 - sqrMix) + os.square(f) * sqrMix;"
_SQRMIX = 'sqrMix   = hslider("sqrMix", 0.0, 0, 1, 0.001);       // blend saw -> square'


def wire_into(amps: np.ndarray, dsp: str = synth.DSP) -> str:
    """synth.DSP with the one waveform swapped, for measuring cost and score in situ.

    Not an integration: it returns a string and leaves synth.py alone. sqrMix becomes a
    literal because the fitted table subsumes it (a square is 4/(pi*h) on odd h only),
    and because Faust prunes an hslider that nothing reads, which would then trip
    PadRenderer's missing-parameter check. Removing its declaration is what that check
    is written to tolerate.
    """
    for anchor in (_OSCMIX, _SQRMIX):
        if dsp.count(anchor) != 1:
            raise RuntimeError(f"anchor not found exactly once in dsp: {anchor[:40]!r}")
    return (dsp.replace(_SQRMIX, "sqrMix   = 0.0;   // subsumed by the fitted harmonic table")
               .replace(_OSCMIX, faust_fragment(amps) + "\noscmix(f) = wtosc(f);"))


# ---------------------------------------------------------------- wavetable export

def single_cycle(amps: np.ndarray, n: int = CYCLE) -> np.ndarray:
    """One cycle of exactly what the Faust fragment emits: -sum a[h] sin(2 pi h t).

    Not chord.single_cycle, which uses cosine phase. Phase is invisible to the
    objective, but a wavetable exported with a different phase spectrum is a different
    waveform, and this file's whole claim is that the exported table IS the oscillator.
    """
    a = np.asarray(amps, dtype=float)
    spec = np.zeros(n // 2 + 1, dtype=complex)
    spec[1:len(a) + 1] = 1j * a[: n // 2]
    return np.fft.irfft(spec, n=n) * (n / 2.0)


def export_wavetable(amps: np.ndarray, path: str = WT_PATH, n: int = CYCLE,
                     peak: float = 0.999) -> dict:
    """Write the single cycle as a mono PCM_24 wav, portable to a real wavetable synth."""
    y = single_cycle(amps, n)
    scale = peak / float(np.abs(y).max())
    synth.write_render(path, y * scale)
    return {"path": path, "n": n, "peak_scale": scale}


# ---------------------------------------------------------------- measurement

def dft_amps(x: np.ndarray, freqs: np.ndarray, sr: int = SR) -> np.ndarray:
    """Amplitude of x at each of freqs, by windowed DFT at that exact frequency.

    chord.harmonic_readout does this for the bench's 5-pitch harmonic grid; the folded
    alias frequencies are not on any harmonic grid, so they need the frequency list
    form. Same Hanning normalisation, so the two agree where they overlap.
    """
    w = np.hanning(len(x))
    xw = np.asarray(x, dtype=np.float64) * w
    t = np.arange(len(x)) / sr
    out = np.empty(len(freqs))
    for i, f in enumerate(np.asarray(freqs, dtype=float)):
        ph = 2 * np.pi * f * t
        out[i] = np.hypot((xw * np.cos(ph)).sum(), (xw * np.sin(ph)).sum()) / (w.sum() / 2.0)
    return out


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[0] if x.ndim > 1 else x


def _best_shift(a: np.ndarray, b: np.ndarray, span: int = 512) -> tuple[int, np.ndarray]:
    """Integer circular shift of b that best aligns it with a, searched over +-span.

    os.sawtooth is a PTR waveform, not a phasor, so it need not sit on the same sample
    phase as os.osc. A time-domain L2 is meaningless until that is removed. span must
    exceed one period of the highest pitch tested, or the search cannot reach the
    alignment and reports its own boundary.
    """
    best = (np.inf, 0)
    for s in range(-span, span + 1):
        e = float(np.linalg.norm(a - np.roll(b, s)))
        best = min(best, (e, s))
    return best[1], np.roll(b, best[1])


def folded(freqs: np.ndarray, sr: int = SR) -> np.ndarray:
    """Where a sinusoid at each frequency actually lands after sampling."""
    f = np.asarray(freqs, dtype=float)
    return np.abs(f - sr * np.round(f / sr))


# ---------------------------------------------------------------- verification

def main(tail: bool = False) -> None:
    """tail=True adds the saw-tail variants, which cost minutes of Faust compile time
    for 256 partials per unison voice. The three required checks do not need them."""
    fit = fit_harmonic_table()
    a = fit["amps"]
    saw = saw_table()
    print(f"fit: rank-1 log residual rms {fit['resid_rms_db']:.2f} dB, "
          f"max {fit['resid_max_db']:.2f} dB over {len(chord.PITCHES) * H} partials")
    print(f"table: a[1] {a[0]:.4f}  a[64] {a[-1]:.6f}  "
          f"|a|/|saw| {np.linalg.norm(a) / np.linalg.norm(saw):.4f}  min {a.min():.2e}")

    wt = export_wavetable(a)
    # Read the file back rather than trusting the array that went in: the deliverable is
    # the wav, so what has to carry the table is the wav.
    back, _ = sf.read(wt["path"], dtype="float64")
    spec = np.fft.rfft(back) / (len(back) / 2.0)
    wav_db = 20 * np.log10(np.abs(spec[1:H + 1]) / (a * wt["peak_scale"]))
    print(f"wavetable: {wt['path']} ({wt['n']} samples, mono {synth.RENDER_SUBTYPE}, "
          f"peak scale {wt['peak_scale']:.4f}); round-trip harmonic error "
          f"max {np.abs(wav_db).max():.4f} dB")

    dur = 1.0
    out: dict[str, np.ndarray] = {"amps": a, "saw": saw}

    # ---- 1. saw identity -------------------------------------------------
    ident = {}
    for pitch in (29, 77):
        f0 = float(librosa.midi_to_hz(pitch))
        mine = _mono(render_gen(probe_dsp(saw), {"freq": f0, "wtMorph": 0.0}, dur))
        ref = _mono(render_gen(SAW_DSP, {"freq": f0}, dur))
        shift, aligned = _best_shift(mine, ref)
        raw = float(np.linalg.norm(mine - ref) / np.linalg.norm(ref))
        rel = float(np.linalg.norm(mine - aligned) / np.linalg.norm(aligned))
        # The floor: os.sawtooth carries harmonics up to Nyquist, this carries 64, so
        # the missing tail is a legitimate residual. Analytic from the 2/(pi h) series,
        # and measured off the reference's own spectrum above harmonic 64.
        hn = int(0.5 * SR / f0)
        k = np.arange(1, hn + 1, dtype=float)
        floor_a = float(np.sqrt((1 / k[H:] ** 2).sum() / (1 / k ** 2).sum())) if hn > H else 0.0
        spec = np.abs(np.fft.rfft(ref * np.hanning(len(ref))))
        fbin = np.fft.rfftfreq(len(ref), 1 / SR)
        hi = spec[fbin > (H + 0.5) * f0]
        floor_m = float(np.sqrt((hi ** 2).sum() / (spec ** 2).sum()))
        am, ar = (chord.harmonic_readout(z, pitches=(pitch,), fmax=(H + 0.5) * f0)
                  for z in (mine, ref))
        nb = min(H, hn)
        d = 20 * np.log10(am[:nb] / np.maximum(ar[:nb], 1e-12))
        # Which of the two is wrong. os.sawtooth is saw2ptr, only 2nd-order
        # alias-suppressed, so at high pitch the REFERENCE is the one that departs from
        # 2/(pi h); this says by how much, so the identity residual can be attributed.
        dref = 20 * np.log10(np.maximum(ar[:nb], 1e-12) / saw[:nb])
        print(f"saw identity MIDI {pitch} (f0 {f0:.2f} Hz, {nb} live harmonics): "
              f"rel L2 {raw:.4f} raw, {rel:.4f} at shift {shift:+d}; "
              f"floor {floor_a:.4f} analytic {floor_m:.4f} measured; "
              f"per-harmonic vs os.sawtooth max {np.abs(d).max():.3f} dB "
              f"rms {np.sqrt((d ** 2).mean()):.3f} dB; "
              f"os.sawtooth itself vs 2/(pi h) max {np.abs(dref).max():.3f} dB")
        ident[f"ident_db_{pitch}"] = d
        ident[f"ident_ref_db_{pitch}"] = dref
        ident[f"ident_l2_{pitch}"] = np.array([raw, rel, floor_a, floor_m, shift])
    out.update(ident)
    out["wav_roundtrip_db"] = wav_db

    # The morph end must not depend on which table is loaded, or "prove the identity
    # case" would only prove it for one table. wtGain = wtSaw + m*(wtFit - wtSaw) says
    # these two renders are the same signal graph, so bit-identical is the right bar.
    f0 = float(librosa.midi_to_hz(45))
    z0 = _mono(render_gen(probe_dsp(saw), {"freq": f0, "wtMorph": 0.0}, 0.2))
    z1 = _mono(render_gen(probe_dsp(a), {"freq": f0, "wtMorph": 0.0}, 0.2))
    print(f"morph algebra: max |fitted table at morph 0 - saw table at morph 0| "
          f"{np.abs(z0 - z1).max():.2e}")

    # ---- 2. amplitude fidelity ------------------------------------------
    for pitch in (29, 45, 57):
        f0 = float(librosa.midi_to_hz(pitch))
        x = _mono(render_gen(probe_dsp(a), {"freq": f0, "wtMorph": 1.0}, dur))
        got = chord.harmonic_readout(x, pitches=(pitch,), fmax=(H + 0.5) * f0)
        nb = min(H, int(0.5 * SR / f0))
        d = 20 * np.log10(got[:nb] / a[:nb])
        print(f"amplitude fidelity MIDI {pitch} ({nb} harmonics below Nyquist): "
              f"max {np.abs(d).max():.3f} dB, rms {np.sqrt((d ** 2).mean()):.3f} dB, "
              f"gated harmonics {H - nb}")
        out[f"amp_db_{pitch}"] = d

    # ---- 3. anti-aliasing ------------------------------------------------
    pitch = 77
    f0 = float(librosa.midi_to_hz(pitch))
    hn = int(0.5 * SR / f0)
    alias_f = folded(np.arange(hn + 1, H + 1) * f0)
    for gate, tag in ((True, "gated"), (False, "control")):
        x = _mono(render_gen(probe_dsp(a, gate), {"freq": f0, "wtMorph": 1.0}, dur))
        alias = dft_amps(x, alias_f)
        live = dft_amps(x, np.arange(1, hn + 1) * f0)
        r = float(np.sqrt((alias ** 2).sum() / (live ** 2).sum()))
        print(f"anti-aliasing MIDI {pitch} ({H - hn} partials above Nyquist) {tag}: "
              f"folded/live energy {20 * np.log10(r):.1f} dB, "
              f"worst alias {20 * np.log10(alias.max() / live.max()):.1f} dB below peak")
        out[f"alias_{tag}"] = alias
        out[f"live_{tag}"] = live
    out["alias_freqs"] = alias_f

    # ---- 4. cost and score in situ --------------------------------------
    notes = chord.notes_upto()
    with open("out/patch.json") as f:
        x0 = synth.normalize(json.load(f)["params"])
    sc = chord.WindowScore()
    ws = chord.win_slice()
    variants = [("baseline", synth.DSP, None),
                (f"H={H} morph 0", wire_into(a), 0.0),
                (f"H={H} morph 1", wire_into(a), 1.0)]
    if tail:
        t256 = with_saw_tail(a, TAIL_H)
        variants += [(f"H={H}+tail{TAIL_H} morph 0", wire_into(t256), 0.0),
                     (f"H={H}+tail{TAIL_H} morph 1", wire_into(t256), 1.0)]
    for tag, dsp, morph in variants:
        t = time.time()
        r = synth.PadRenderer(n_voices=12, dsp=dsp)
        comp = time.time() - t
        r.set_notes(notes)
        r.set_params(synth.denorm(x0))
        if morph is not None:
            r.set_params({"wtMorph": morph})
        r.set_bend(bend_curve(int(chord.WIN_T1 * SR) + SR))
        t = time.time()
        y = r.render(chord.WIN_T1)[:, ws]
        el = time.time() - t
        print(f"in situ {tag}: score {sc(y):.4f}  cos {sc.cos_theta(y):.4f}  "
              f"compile {comp:.1f} s, render {el:.1f} s for {chord.WIN_T1:.2f} s of audio")

    np.savez_compressed(NPZ_PATH, **out)
    print(f"wrote {NPZ_PATH}")


if __name__ == "__main__":
    main("--tail" in sys.argv)
