"""Candidate synth: detuned supersaw pad + sub, ladder-ish LPF, chorus/delay/reverb.

Architecture chosen from the stage-0 analysis: complete harmonic series (saw),
closely-spaced partial clusters at high harmonics (unison detune), deep octave-1
fundamentals (sub), long smeared tails (reverb).

Two stages were added later to attack the two mismatches the first fit could not
close: `drive`, a tanh waveshaper before the filter, for the 250-900 Hz energy a
lowpassed saw cannot produce, and `spread`, per-note constant-power panning, for the
L/R decorrelation a mono voice sum into a stereo reverb cannot produce. Both are
appended to PAD_PARAMS with identity defaults, so an old 27-parameter vector padded with
them renders bit-identically.

Two timbre stages were added after that, both for the same finding: an oracle ladder
said the recoverable error in the render is a STATIC spectral envelope (a best-possible
fixed EQ buys 5x what a best-possible per-frame volume envelope buys), so the envelope
is already right and the timbre is not. Fitting the window's 382 audible partials then
said the envelope has two axes: a 34-parameter curve in absolute FREQUENCY leaves 4.20
dB of residual, a 177-parameter curve in HARMONIC NUMBER leaves 4.47 dB, and both
together reach 2.21 dB. So neither axis subsumes the other and the synth needs one
stage per axis.

`eq0`..`eq25` are the absolute-frequency axis: a cascade of 26 fi.svf.bell peaking
sections at third-octave centres from 40 Hz to 16 kHz, in the shared effect chain. The
curve the render needs is non-monotonic (+7.1 dB at 250-900 Hz, +3.3 at 900-2000, -4.6
at 2000-6000, +2.1 above), and fi.resonlp + fi.lowpass(2) + tiltEQ is monotonic above
its cutoff at every setting, so that shape was outside the reachable set entirely:
7500 CMA-ES renders moving the loss by 0.012 is what that looks like from the inside.
Negative results that shaped it. fi.peak_eq, the obvious primitive, is unusable at this
loss's sensitivity: ONE band of it flat costs 0.69 of loss while leaving cos theta
untouched, because its Lfx=0 numerator and denominator are equal in exact arithmetic
but not in float32, and at a pole radius of 0.998 the mismatch survives as a real -52
dB resonance in the near-empty bins between partials where the log-magnitude term
does its work. And band count is not a free choice: the reachable score plateaus at 26
(8 bands 1.4167, 12 bands 1.3571, 26 bands 1.2978, 32 bands 1.2970), so the 9 to 12
bands a human would draw cannot express the fitted curve at any gain setting.

The harmonic-number axis was tried and does NOT pay, which is worth recording because
the partial-amplitude evidence pointed the other way. Fitting the 382 audible partials
of one held chord, a 177-parameter harmonic-number envelope left 4.47 dB residual
against 4.20 dB for a 34-parameter absolute-frequency curve, and both together reached
2.21 dB, so the harmonic axis looked like a real secondary. Solved jointly through the
real loss after the 26 bands, its deltas came out identically 0 dB: it could lower the
score only by lowering cos theta, and a step-length scan put the optimum at zero. The
amplitude statement survives; it just does not survive translation into this loss once
the frequency curve is fitted properly first. dsp_source(amps) still builds the bank
for anyone who wants to re-measure that, but DSP does not, because it cost 23x render
time to contribute nothing.

The negative result that shaped the oscillator, and the reason the bank was additive
rather than a pitched table read: a pure-tone bank at EXACTLY the fitted partial
amplitudes scores 1.949, WORSE than the 1.561 of the render it was meant to replace,
because the loss punishes the empty gaps between partials that clean sinusoids leave
and the target fills. The unison detune, chorus and reverb are doing real work filling
those gaps, so nothing here replaces them.

The EQ bands are appended to PAD_PARAMS with identity defaults (0 dB) and the identity is
exact in algebra, so the pre-timbre patch padded with them renders to the same audio:
measured 5.9e-08 relative, i.e. 0.0 of loss against a recorded 1.5446351990.

"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import dawdreamer as daw
import numpy as np
import soundfile as sf

SR = 44100
BLOCK = 512
RENDER_SUBTYPE = "PCM_24"
FAUST_LIBS = os.path.join(os.path.dirname(daw.__file__), "faustlibraries")

NVOICE = 7

# ---------------- timbre stage 1: absolute-frequency EQ ----------------
# Geometry fixed by eq_stage.select_bands (band count and Q both chosen where the
# reachable score plateaus, and where the bank stops being redundant: 0.508 octave
# measured bandwidth against 0.346 octave spacing, design-matrix condition number 6.6).
# Duplicated here rather than imported so that the authoritative renderer does not
# depend on a fitting module; verify_timbre_stages asserts eq_faust() is the same string
# as eq_stage.EQ_FAUST, which is the source that was verified.
N_EQ = 26
EQ_F_LO = 40.0          # below F1 at 43.65 Hz
EQ_F_HI = 16000.0
EQ_Q = 2.8
EQ_LIMIT = 18.0         # dB; linearity in dB holds to 0.25 dB out to +-12 and 0.9 at the rail


def eq_band_freqs(n: int = N_EQ, lo: float = EQ_F_LO, hi: float = EQ_F_HI) -> np.ndarray:
    return np.geomspace(lo, hi, n)


def eq_faust(n: int = N_EQ, q: float = EQ_Q, name: str = "eqCurve",
             lo: float = EQ_F_LO, hi: float = EQ_F_HI) -> str:
    """One mono peaking-EQ cascade plus its sliders, for `par(i, 2, eqCurve)`.

    Every gain defaults to 0 dB, which is the identity: standalone, 2.5 s of the real
    render through the flat cascade differs from its own input by 0.0 relative, and in
    the effect chain the whole 26-band stage costs 5.9e-08 relative and 0.0 of loss.
    """
    f = eq_band_freqs(n, lo, hi)
    sliders = "\n".join(
        f'eq{i} = hslider("eq{i}", 0, {-EQ_LIMIT:g}, {EQ_LIMIT:g}, 0.001);'
        f'   // {f[i]:.0f} Hz' for i in range(n))
    chain = " : ".join(f"fi.svf.bell({f[i]:.4f}, {q:g}, eq{i})" for i in range(n))
    return f"{sliders}\n{name} = _ : {chain};\n"


# ---------------- timbre stage 2: harmonic-number table ----------------
# The rank-1 log fit of wt_osc.fit_harmonic_table: amp[note, h] = level_note * a[h] over
# the window's five held pitches, solved in dB and weighted by linear amplitude so
# partials in the noise cannot steer it. 4.53 dB weighted rms residual, i.e. the whole
# harmonic-number axis and nothing of the frequency axis, which is stage 1's job. A
# starting point, not an answer: the two axes overlap and want re-solving jointly, and
# harmonic_readout cannot separate the partials that coincide between these pitches
# (F3 is two octaves above F1, C3 sits 0.13 Hz from F1's third harmonic), which inflates
# h = 3, 4, 6, 8 and their multiples.
WT_FIT: tuple[float, ...] = (
    0.689125924, 0.276788863, 0.251869792, 0.174115265,
    0.0421351856, 0.0451091684, 0.0125780014, 0.0693530149,
    0.0357541193, 0.0123315794, 0.0115346719, 0.0366587878,
    0.00639596183, 0.00481358135, 0.0188381915, 0.0424431911,
    0.00405356262, 0.00547751692, 0.0120678127, 0.00928840815,
    0.00671297542, 0.00202965541, 0.0028392292, 0.0195488684,
    0.00313558453, 0.00186003975, 0.00530448944, 0.00378004563,
    0.00162665994, 0.00546451456, 0.00156746631, 0.0125858228,
    0.00223993352, 0.00202219371, 0.00202022351, 0.0112216644,
    0.00192795882, 0.00354276023, 0.00090535969, 0.00168972651,
    0.000724653266, 0.000679206745, 0.00132797164, 0.000550997138,
    0.00180618701, 0.00114891959, 0.000801709783, 0.00260754741,
    0.000477679784, 0.00154543807, 0.00477734371, 0.000351419936,
    0.000504581614, 0.00100412324, 0.000600252038, 0.000499243865,
    0.00398122665, 0.00143899355, 0.00132541683, 0.00279774415,
    0.00257173788, 0.000598050669, 0.000389384516, 0.00362910833,
)
WT_H = 128              # fitted band plus a sawtooth tail; 256 buys 0.002 for 3x the cost


def wt_table(fit: tuple[float, ...] = WT_FIT, h: int = WT_H) -> np.ndarray:
    """The fitted amplitudes below len(fit), the sawtooth's own 2/(pi h) above them.

    The tail is what makes wtMorph a pure timbre control: above harmonic 64 the table
    equals the waveform it is crossfading against, so the morph changes only the band
    the fit could see and never the top two octaves' level.
    """
    a = np.asarray(fit, dtype=float)
    if h < len(a):
        raise ValueError(f"h={h} is shorter than the fitted table ({len(a)})")
    tail = 2.0 / (np.pi * np.arange(len(a) + 1, h + 1, dtype=float))
    return np.concatenate([a, tail])


def wt_faust(amps: np.ndarray) -> str:
    """The oscillator's saw leg: os.sawtooth at wtMorph=0, the table's bank at 1.

    A crossfade against the REAL os.sawtooth, not against the bank's own 2/(pi h)
    setting, because only the former is the identity: the bank truncated at 128 harmonics
    is 0.018 of loss away from os.sawtooth, and truncated at 64 it is 0.104 away. `* 0.0`
    and `* 1.0` are exact in float, so both ends of the morph are the thing they claim to
    be, to the one ulp the module docstring accounts for.
    """
    a = np.asarray(amps, dtype=float)
    if a.ndim != 1:
        raise ValueError("amps must be 1-D")
    return f"""wtMorph = hslider("wtMorph", 0, 0, 1, 0.001);
wtGain(i) = ba.take(i+1, ({", ".join(f"{v:.9g}" for v in a)}));
// Hard Nyquist gate per partial, which is what makes the bank exactly alias-free
// (measured -125.6 dB of folded energy against -32.1 dB with the gate removed). It is a
// comparison, so a partial switches on and off instantly; during the measured bend the
// top partials cross it, at under -48 dB of the fundamental.
wtPartial(f, i) = os.osc(f * float(i+1)) * wtGain(i) * (f * float(i+1) < 0.5 * ma.SR);
// Negated because 2*frac(x)-1 = -(2/pi)*sum sin(2*pi*h*x)/h: the minus is what makes
// the bank's polarity the same as the os.sawtooth it crossfades with.
wtosc(f) = 0.0 - (par(i, {len(a)}, wtPartial(f, i)) :> _);
sawLeg(f) = os.sawtooth(f) * (1.0 - wtMorph) + wtosc(f) * wtMorph;"""


_DSP_TEMPLATE = f"""
import("stdfaust.lib");

freq = hslider("freq", 440, 20, 8000, 0.001);
gain = hslider("gain", 0.5, 0, 1, 0.001);
gate = button("gate");

// ---- voice params ----
detune   = hslider("detune", 20, 0, 80, 0.01);        // cents spread across unison
uniMix   = hslider("uniMix", 0.7, 0, 1, 0.001);       // centre vs spread balance
subLvl   = hslider("subLvl", 0.5, 0, 1, 0.001);       // sub sine at -1 oct
sqrMix   = hslider("sqrMix", 0.0, 0, 1, 0.001);       // blend saw -> square
cutoff   = hslider("cutoff", 2000, 60, 12000, 0.01);  // base LPF cutoff (Hz)
reso     = hslider("reso", 1.0, 0.5, 12, 0.001);      // filter Q
envAmt   = hslider("envAmt", 0.0, -4000, 8000, 1);    // filter env depth (Hz)
kbdTrk   = hslider("kbdTrk", 0.3, 0, 1, 0.001);       // cutoff keyboard tracking
fA       = hslider("fA", 0.3, 0.001, 4, 0.001);
fD       = hslider("fD", 0.5, 0.001, 4, 0.001);
fS       = hslider("fS", 0.6, 0, 1, 0.001);
aA       = hslider("aA", 0.4, 0.001, 4, 0.001);
aD       = hslider("aD", 0.5, 0.001, 4, 0.001);
aS       = hslider("aS", 0.8, 0, 1, 0.001);
aR       = hslider("aR", 1.2, 0.01, 6, 0.001);
lfoRate  = hslider("lfoRate", 4.0, 0.05, 12, 0.001);
lfoAmt   = hslider("lfoAmt", 0.0, 0, 50, 0.001);      // vibrato depth (cents)
drive    = hslider("drive", 0.0, 0, 1, 0.001);        // pre-filter saturation, 0 = bypass
spread   = hslider("spread", 0.0, 0, 1, 0.001);       // per-note stereo pan, 0 = mono

// Pitch bend, driven by sample-accurate automation rather than optimised.
// Multiplies the played frequency, so a note held at the glide's target pitch
// with bend ramping 0.63 -> 1.0 reproduces an upward glide onto that pitch.
bend = hslider("bend", 1.0, 0.25, 4.0, 0.000001);

ratio(i) = pow(2.0, (i - (({NVOICE}-1))/2.0) * detune / 1200.0);
vib = pow(2.0, os.osc(lfoRate) * lfoAmt / 1200.0) * bend;

// ---- saw leg: os.sawtooth, or the fitted harmonic table, or a morph between them ----
%(wt)s
oscmix(f) = sawLeg(f) * (1.0 - sqrMix) + os.square(f) * sqrMix;

centre = oscmix(freq * vib);
uni    = par(i, {NVOICE}, oscmix(freq * vib * ratio(i))) :> _ / {NVOICE};
sub    = os.osc(freq * 0.5 * vib) * subLvl;

osc = (centre * (1.0 - uniMix) + uni * uniMix) + sub;

// Waveshaper between the oscillators and the filter, added to test the hypothesis that
// the render's light 250-900 Hz band was missing saturation. It was not: the render is
// simultaneously light there and +6.7 dB hot at 2-6 kHz, so the defect is a tilt and a
// memoryless shaper deepens it. Measured worse at every setting, gain-matched and with
// the brightness controls refitted (drive_probe.py), so the fit leaves it at 0. Kept
// because a measured negative result is worth more than an untested suggestion.
// Normalised by tanh(dgain) so the peak stays at unity (small signals are boosted,
// loud ones compressed), and mixed by drive itself so drive=0 is algebraically the
// identity. Faust folds it away when drive is a literal 0; through the slider it does
// not, and verify_new_stages measures what that costs.
dgain = 1.0 + drive * 12.0;
shaped = osc + drive * (ma.tanh(osc * dgain) / ma.tanh(dgain) - osc);

fenv = en.adsr(fA, fD, fS, aR, gate);
aenv = en.adsr(aA, aD, aS, aR, gate);

trackedCut = cutoff * pow(freq / 261.6255, kbdTrk);
fc = max(30.0, min(16000.0, trackedCut + envAmt * fenv));

filtered = shaped : fi.resonlp(fc, reso, 1.0) : fi.lowpass(2, fc);

// Constant-power pan per note, so the voices are decorrelated BEFORE the shared
// effects instead of a mono sum being handed to a stereo reverb. The voice DSP has
// no voice index, so the pan position is a deterministic function of the note's own
// pitch: semitones from C4 times an irrational multiplier, which scatters adjacent
// notes across the image instead of drawing a smooth low-to-high ramp. Scaled by
// sqrt(2) so spread=0 gives unity in both channels, i.e. exactly the old `<: _,_`.
panPos = 0.5 + 0.5 * spread * sin(2.0 * ma.PI * 0.6180339887 * 12.0 * log(freq / 261.6255) / log(2.0));
panL = sqrt(2.0 * (1.0 - panPos));
panR = sqrt(2.0 * panPos);

process = filtered * aenv * gain <: *(panL), *(panR);

// ---------------- shared effects ----------------
chRate  = hslider("chRate", 0.6, 0.05, 6, 0.001);
chDepth = hslider("chDepth", 0.0, 0, 1, 0.001);
dlyTime = hslider("dlyTime", 0.35, 0.02, 1.2, 0.001);
dlyFb   = hslider("dlyFb", 0.3, 0, 0.85, 0.001);
dlyWet  = hslider("dlyWet", 0.0, 0, 1, 0.001);
revSize = hslider("revSize", 0.85, 0.1, 0.99, 0.001);
revDamp = hslider("revDamp", 0.4, 0, 1, 0.001);
revWet  = hslider("revWet", 0.35, 0, 1, 0.001);
tilt    = hslider("tilt", 0.0, -1, 1, 0.001);   // <0 darker, >0 brighter
outGain = hslider("outGain", 0.6, 0, 2, 0.001);

chorus = _,_ : ef.stereo_width(0.8) : par(i, 2, de.fdelay(4096, 220.0 + 200.0*chDepth*os.osc(chRate + 0.13*i)))
       : par(i, 2, _*chDepth) :> _,_ ;

pingpong = par(i, 2, (+ : de.fdelay(65536, dlyTime * ma.SR)) ~ (*(dlyFb)));

tiltEQ = _ : fi.highshelf(2, tilt * 12.0, 1200.0) : fi.lowshelf(2, -tilt * 6.0, 300.0);

// The 26-band cascade, immediately after the tilt it makes redundant. tilt stays in the
// chain and in PARAMS because PARAMS is append-only and the delivered patch's fitted
// tilt of +0.255 is part of the render this stage has to reproduce at its identity; a
// refit is free to zero it, since any tilt is also a set of band gains.
%(eq)s
wetdry(w, fx) = _,_ <: (par(i,2,_*(1.0-w))), (fx : par(i,2,_*w)) :> _,_;

effect = _,_
       : wetdry(chDepth, chorus)
       : wetdry(dlyWet, pingpong)
       : wetdry(revWet, re.zita_rev1_stereo(0, 200, 1200.0 + (1.0-revDamp)*8000.0, revSize*8.0, revSize*4.0, ma.SR))
       : par(i, 2, tiltEQ)
       : par(i, 2, eqCurve)
       : par(i, 2, _*outGain);
"""


def dsp_source(wt_amps: np.ndarray | None = None) -> str:
    """The Faust source. wt_amps=None omits the additive bank entirely.

    With the bank omitted, wtMorph becomes a literal 0.0 and Faust prunes the slider,
    which PadRenderer's missing-parameter check is written to tolerate and set_params
    silently skips. That build is the pre-wavetable oscillator exactly, and it renders
    14x to 26x faster than the bank does, so it is what a macro search over the filter
    and the envelopes should use; the bank is a least-squares problem, not a search one.
    """
    if wt_amps is None:
        wt = ("wtMorph = 0.0;   // no additive bank in this build\n"
              "sawLeg(f) = os.sawtooth(f);")
    else:
        wt = wt_faust(wt_amps)
    return _DSP_TEMPLATE % {"wt": wt, "eq": eq_faust()}


# The default build has NO additive bank. Fitting one measured its contribution at
# exactly zero (the fitted harmonic-number deltas came out identically 0 dB once a
# 26-band absolute-frequency curve was solved first), while it made an 18 s render 23x
# slower. dsp_source(amps) still builds it, so that result stays reproducible, but
# nothing should pay for it by default.
DSP = dsp_source(None)
DSP_SAW = DSP


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    default: float
    log: bool = False

    def denorm(self, v: float) -> float:
        """One coordinate of the [0,1] search vector as a real parameter value."""
        v = float(np.clip(v, 0.0, 1.0))
        if self.log:
            return float(np.exp(np.log(self.lo) + v * (np.log(self.hi) - np.log(self.lo))))
        return float(self.lo + v * (self.hi - self.lo))

    def normalize(self, value: float) -> float:
        """Inverse of denorm for a single parameter."""
        if self.log:
            return float((np.log(value) - np.log(self.lo)) / (np.log(self.hi) - np.log(self.lo)))
        return float((value - self.lo) / (self.hi - self.lo))


@dataclass(frozen=True)
class Architecture:
    """A Faust source and the parameter vocabulary that addresses it, as one value.

    These were separate: the source travelled as a `dsp=` argument while the names
    lived in module globals. That made a second architecture look supported when it was
    not. A foreign DSP compiled, exposed none of the pad's slider names, and
    `set_params` skipped every one of them in silence, so the render came out at the
    Faust defaults and scored as if the parameters simply did not help. Fitting a
    second architecture means constructing one of these, not passing another string.

    `index` is derived rather than given, and this is frozen, hence the
    object.__setattr__ that computing it needs. It is excluded from comparison so that
    the class stays hashable with a dict on it.
    """

    name: str
    dsp: str
    params: tuple[Param, ...]
    index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", {p.name: i for i, p in enumerate(self.params)})

    def denorm(self, x: np.ndarray) -> dict[str, float]:
        """Map a normalized [0,1] vector to real parameter values."""
        return {p.name: p.denorm(x[i]) for i, p in enumerate(self.params)}

    def normalize(self, values: dict[str, float]) -> np.ndarray:
        """Inverse of denorm: real parameter values back to the [0,1] search vector."""
        return np.clip([p.normalize(values[p.name]) for p in self.params], 0.0, 1.0)

    def norm_defaults(self) -> np.ndarray:
        return self.normalize({p.name: p.default for p in self.params})

    def padded(self, x: np.ndarray) -> np.ndarray:
        """Extend a normalized vector fitted before `params` grew, using the new defaults.

        `params` is append-only, so the leading coordinates of an old patch.json still
        mean what they meant; the appended parameters have identity defaults, hence a
        padded old vector must render to the same audio. verify_new_stages asserts that
        for drive and spread, verify_timbre_stages for the EQ and the wavetable morph.
        """
        x = np.asarray(x, dtype=float)
        if len(x) > len(self.params):
            raise ValueError(f"vector of {len(x)} is longer than {self.name} ({len(self.params)})")
        out = self.norm_defaults()
        out[: len(x)] = x
        return out

    def with_dsp(self, dsp: str) -> "Architecture":
        """The same vocabulary over a variant build: the wavetable bank, a legacy
        chain a verification script wants to score against. Same parameter order, so
        the same normalized vector means the same thing in both."""
        return replace(self, dsp=dsp)


# Search space for stage 2. Order defines the CMA-ES vector.
PAD_PARAMS: tuple[Param, ...] = (
    Param("detune", 2.0, 60.0, 22.0),
    Param("uniMix", 0.0, 1.0, 0.75),
    Param("subLvl", 0.0, 1.0, 0.45),
    Param("sqrMix", 0.0, 1.0, 0.0),
    Param("cutoff", 120.0, 9000.0, 1600.0, log=True),
    Param("reso", 0.5, 8.0, 1.2),
    Param("envAmt", -2000.0, 6000.0, 600.0),
    Param("kbdTrk", 0.0, 1.0, 0.3),
    Param("fA", 0.005, 3.0, 0.4, log=True),
    Param("fD", 0.02, 4.0, 0.8, log=True),
    Param("fS", 0.0, 1.0, 0.6),
    Param("aA", 0.005, 3.0, 0.35, log=True),
    Param("aD", 0.02, 4.0, 0.8, log=True),
    Param("aS", 0.05, 1.0, 0.85),
    Param("aR", 0.05, 5.0, 1.4, log=True),
    Param("lfoRate", 0.05, 10.0, 4.0),
    Param("lfoAmt", 0.0, 30.0, 0.0),
    Param("chRate", 0.05, 5.0, 0.6),
    Param("chDepth", 0.0, 1.0, 0.0),
    Param("dlyTime", 0.05, 1.0, 0.35),
    Param("dlyFb", 0.0, 0.8, 0.3),
    Param("dlyWet", 0.0, 0.8, 0.0),
    Param("revSize", 0.15, 0.98, 0.85),
    Param("revDamp", 0.0, 1.0, 0.4),
    Param("revWet", 0.0, 0.9, 0.35),
    Param("tilt", -1.0, 1.0, 0.0),
    Param("outGain", 0.05, 1.8, 0.6),
    # Appended, never inserted: every index above is what patch.json was fitted with,
    # and both defaults are the identity so an old 27-vector padded with them renders
    # bit-identically. drive=0 bypasses the waveshaper, spread=0 keeps the mono sum.
    Param("drive", 0.0, 1.0, 0.0),
    Param("spread", 0.0, 1.0, 0.0),
    # The two timbre stages, appended under the same contract: 0 dB per band is the
    # cascade's identity and wtMorph=0 is the untouched os.sawtooth, so a 29-vector padded
    # with them renders to the same audio (see the module docstring for the one ulp it is
    # not). The 26 gains are nearly separable in the loss, so a coordinate-wise search
    # (Powell) is the efficient way at them, not CMA-ES.
    *(Param(f"eq{i}", -EQ_LIMIT, EQ_LIMIT, 0.0) for i in range(N_EQ)),
)

# The pad: the architecture this project fitted, and PadRenderer's default.
PAD = Architecture(name="pad", dsp=DSP, params=PAD_PARAMS)

# Module-level shorthand for the pad, and nothing else. These are the same objects
# under another name, not a second way to reach them: live code takes an Architecture
# explicitly. They stay because the superseded scripts import them by name, and a
# measured negative result that no longer imports is worth less than one that runs.
PARAMS = PAD.params
PARAM_INDEX = PAD.index
denorm = PAD.denorm
normalize = PAD.normalize
norm_defaults = PAD.norm_defaults
pad_normalized = PAD.padded
normalize_one = Param.normalize          # normalize_one(p, v) is p.normalize(v)


def write_render(path: str, audio: np.ndarray, sr: int = SR) -> None:
    """Save a (2, n) render so that the file reproduces the loss that was reported.

    Not soundfile's default PCM_16: on this material 16-bit quantisation costs 0.089 of
    the stage-2 loss, because the log-magnitude STFT term sees the quantisation floor in
    the 6-16 kHz band where the render itself only sits at -41 dB. That is an order of
    magnitude more than the differences the fit is arguing about. PCM_24 measured
    identical to float to four decimals.
    """
    sf.write(path, np.asarray(audio).T, sr, subtype=RENDER_SUBTYPE)


class PadRenderer:
    """Reusable dawdreamer engine; MIDI is set once, params change per render."""

    def __init__(self, arch: Architecture = PAD, sr: int = SR, n_voices: int = 24) -> None:
        self.arch = arch
        self.sr = sr
        self.engine = daw.RenderEngine(sr, BLOCK)
        self.proc = self.engine.make_faust_processor("pad")
        self.proc.faust_libraries_path = FAUST_LIBS
        self.proc.num_voices = n_voices
        self.proc.release_length = 4.0
        self.proc.group_voices = True
        if not self.proc.set_dsp_string(arch.dsp):
            raise RuntimeError("faust compile failed")
        desc = self.proc.get_parameters_description()
        self.pidx = {d["label"]: d["index"] for d in desc}
        # set_automation needs the full Faust path, not the label
        self.ppath = {d["label"]: d["name"] for d in desc}
        # a variant DSP may replace a slider with a constant; only the sliders it still
        # declares have to come out the other side of the compiler
        missing = [p.name for p in arch.params
                   if f'hslider("{p.name}"' in arch.dsp and p.name not in self.pidx]
        if missing:
            raise RuntimeError(f"params not exposed by faust: {missing}")
        self.engine.load_graph([(self.proc, [])])
        self._notes: list[tuple[int, int, float, float]] = []

    def set_notes(self, notes: list[tuple[int, int, float, float]]) -> None:
        """notes = [(pitch, velocity, start_sec, dur_sec)]"""
        self._notes = list(notes)
        self.proc.clear_midi()
        for p, v, s, d in notes:
            self.proc.add_midi_note(int(p), int(v), float(s), float(d))

    def set_params(self, values: dict[str, float]) -> None:
        for name, val in values.items():
            if name in self.pidx:
                self.proc.set_parameter(self.pidx[name], float(val))

    def set_bend(self, curve: np.ndarray | None) -> None:
        """Audio-rate pitch-bend automation (frequency multiplier), or None for no bend."""
        if curve is None:
            self.proc.set_parameter(self.pidx["bend"], 1.0)
            return
        self.proc.set_automation(self.ppath["bend"], np.asarray(curve, dtype=np.float32))

    def render(self, dur: float) -> np.ndarray:
        self.engine.render(dur)
        return self.engine.get_audio()  # (2, n)


def render_with(x: np.ndarray, notes, dur: float, renderer: PadRenderer | None = None) -> np.ndarray:
    r = renderer or PadRenderer()
    r.set_notes(notes)
    r.set_params(r.arch.denorm(x))
    return r.render(dur)
