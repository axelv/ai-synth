# Polyphonic Faust for a playable instrument

Read before writing any DSP. Everything here either fails silently or was measured
failing in a batch of five patches written without it.

- [The skeleton](#the-skeleton)
- [Two declarations that fail silently](#two-declarations-that-fail-silently)
- [Every voice is a bit-identical copy](#every-voice-is-a-bit-identical-copy)
- [Playing across the keyboard](#playing-across-the-keyboard)
- [Effects belong in the shared chain](#effects-belong-in-the-shared-chain)
- [Primitives that compile and then misbehave](#primitives-that-compile-and-then-misbehave)
- [Errors actually observed](#errors-actually-observed)

## The skeleton

`process` is ONE voice. The host instantiates it N times and sums them. `effect` is the
shared chain applied once to that sum.

```faust
import("stdfaust.lib");
declare name "my-patch";                          // required, see below

freq = hslider("freq", 440, 20, 8000, 0.001);     // driven by MIDI note
gain = hslider("gain", 0.5, 0, 1, 0.001);         // driven by MIDI velocity
gate = button("gate");                            // driven by note on/off

// ... macros as hsliders ...

process = /* one voice, must output 2 channels */ <: _,_;
effect  = _,_;                                    // required even when it does nothing
```

`process` must produce stereo. `effect` is stereo in, stereo out.

## Two declarations that fail silently

Omitting either is not a compile error. Both change the parameter path that the host
addresses controls by, and `set_dsp_string` returns `True` regardless, logging only
`ERROR : undefined symbol : effect` to stderr where it is easy to miss.

```
declare name + effect  ->  /Sequencer/DSP1/Polyphonic/Voices/my-patch/brightness
no effect              ->  /Polyphonic/Voices/my-patch/brightness
no declare name        ->  /Sequencer/DSP1/Polyphonic/Voices/dawdreamer/brightness
```

Always declare both. Never trust the return value of `set_dsp_string` alone; read back
`get_parameters_description()` and check the labels are all there. Address parameters by
LABEL, never by path, because the path is not stable across these two declarations.

## Every voice is a bit-identical copy

Faust instantiates N copies of `process` with identical initial state. `no.noise` and
free-running `os.osc` LFOs therefore generate **the same samples in every simultaneously
gated voice**. Measured: four voices on one pitch is +12.04 dB over one voice, and the
four-voice render equals four times the one-voice render sample for sample. Independent
voices would give +6.02 dB.

This silently defeats any patch whose character depends on per-voice variation. Unison
detune inside a single voice still works. Two voices on the same note phase-add instead
of beating, so there is no chorusing, no analog drift, and no stereo decorrelation from
noise. Four of five patches written without this rule were affected, and one agent
measured the symptom, misattributed it to a filter normalisation bug, and rebalanced the
whole patch around the wrong cause.

**Fix: derive anything that should differ per voice from `freq`.** It is the only thing
that differs between simultaneously gated voices.

```faust
// A per-voice constant in [0,1), stable for the note, different for every pitch.
vseed = ma.frac(freq * 0.0177);

// Decorrelated noise: same generator, different position in it per voice.
vnoise = no.noise : de.delay(4096, int(vseed * 4000));

// Decorrelated LFO: rate and phase both offset per voice.
vlfo(rate) = os.osc(rate * (0.93 + 0.14 * vseed));
```

Two voices on the SAME pitch are still identical. That is unavoidable, and it matters
much less: a player rarely holds the same note twice, and a chord is what the rule is
about.

**There is one legitimate exception, and it is a whole class of instrument.** A DCO
synth locks its oscillators to a digital reset, so its voices really are phase-coherent
and `+12.04 dB` is the machine rather than a defect. `references/examples/juno-106.dsp`
is the case: its chorus exists precisely because its oscillators do not drift, and
decorrelating them would remove the reason the instrument sounds the way it does. Before
treating a `+12.04` reading as something to fix, decide whether the thing being modelled
drifts at all.

`measure.py` reports this as `voices +12.04 dB ... bit-identical voices`. Whether that is
a defect depends on the patch. A purely deterministic patch is legitimately coherent. A
patch that claims drift, air, breath, or width is broken if it reads +12.

## Playing across the keyboard

The measurement pattern covers one register. The instrument has to work over five.

**Keyboard tracking.** A filter at a fixed cutoff turns muddy low and thin high. Track it
at partial strength so the top stays soft:

```faust
ktrack = (freq / 261.6256) ^ 0.5;                 // half-power: an octave up moves a 5th
cut = cutoff * ktrack;
```

**Index and brightness scaling.** FM index and any brightness term needs the same
treatment or the top octave turns to fizz: `index * (110 / freq) ^ 0.28`.

**Velocity.** `gain` arrives as velocity. Mapping it straight to amplitude is rarely
right. A player expects harder to mean brighter more than louder:

```faust
vel = gain ^ 0.6;                                 // compress the loudness range
// then drive both amplitude AND a brightness or index term with vel
```

**Highpass with care.** A highpass placed to remove mud eats the fundamental at the
bottom of the keyboard. A 2-pole at 38 Hz starts taking the fundamental below MIDI 28.

Check the `register` line of the report. Levels should sit within about 12 dB across
MIDI 36 to 84, and centroid should rise with pitch.

## Effects belong in the shared chain

Reverb, delay and chorus go in `effect`, never inside `process`. Per-voice reverb costs N
times the CPU and gets louder with polyphony instead of denser.

Any saturation or limiting belongs per-voice, inside `process`. A `ma.tanh` ceiling on
the shared bus makes timbre depend on how many notes are held: measured at 4 voices it
compressed by 1.7 dB, so the patch sounds different in a chord than alone.

Long tails need render headroom. A note whose decay outlasts the host's release window is
truncated mid-ring. `measure.py` fails on this as `still sounding ... when the render
ends`.

## Primitives that compile and then misbehave

Compiling is not the bar. Each of these returned a working DSP and then produced audio
that was wrong, in a way no exception reports.

**`ve.moog_vcf` returns non-finite samples above a cutoff that depends on resonance.**
Not a soft degradation: the render fills with NaN. Measured on a saw at 44.1 kHz, peak
of the render, `NaN` where it blew up:

| res | 5 kHz | 6 kHz | 6.5 kHz | 7 kHz | 7.5 kHz | 8 kHz | 8.8 kHz |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.827 | 0.838 | 0.842 | 0.846 | 1.034 | 1.361 | NaN |
| 0.40 | 0.404 | 0.427 | 0.632 | NaN | NaN | NaN | NaN |
| 0.92 | 0.276 | NaN | NaN | NaN | NaN | NaN | NaN |

Two things to read off it. The safe ceiling falls as resonance rises, so a fixed cutoff
clamp is not enough; the clamp has to be a function of the resonance control. And at low
resonance it clips before it breaks, so a patch can be over the line and merely sound
loud. Clamp against `ma.SR` rather than a constant, because the boundary scales with
sample rate and a page built for the browser may run at 48 kHz:

```faust
fcMax = ma.SR * (0.142 - 0.036 * resonance);
fc    = max(60, min(fcMax, /* whatever the envelope and tracking ask for */));
```

**`ve.moogLadder` is not the way out of that.** It is stable everywhere and it is also
inert. Measured against `fi.lowpass(4, 1000)`, which gives a 548 Hz spectral centroid on
the same saw, `ve.moogLadder(1000/ma.SR, 1)` gives 2 Hz and `ve.moogLadder(1000/(ma.SR/2), 1)`
gives 3 Hz, so neither reading of its normalised frequency argument is the one it wants.
Sweeping Q from 0.5 to 25 at a fixed cutoff moved the peak from 0.022 to 0.022.
`ve.moogHalfLadder` is better and still wrong, at 125 Hz. Do not spend an afternoon on
which normalisation is intended; the resonance does not work either way.

**An odd saturator makes DC out of an asymmetric mix.** `sat(x) = x / sqrt(1 + k*x*x)` is
an odd function, so it cannot make DC from a symmetric wave, and it is easy to conclude
it can never make DC at all. A sawtooth plus a narrow pulse is not symmetric about zero,
and an odd curve applied to it has a nonzero mean. Measured +0.039 on a patch whose only
nonlinearity was that `sat`, against a `measure.py` threshold of 0.01, and exactly 0.000
with the saturator bypassed. Put `fi.dcblocker` after the nonlinearity. `warm-pad.dsp`
uses the same saturator and escapes this only because a `fi.highpass(2, 30)` happens to
sit after its filter, which is luck rather than design.

**A ladder's resonance makeup multiplies, it does not divide.** A resonant lowpass loses
passband level as it resonates, so compensation has to boost. Writing the obvious divider
turned a `resonance` macro into 13.2 dB of level for 7.4 dB of shape, which is the
volume-control failure from `patch-design.md` with the sign flipped. The same control
with `makeup = 1 + k*resonance` measured 1.1 dB of level for 7.7 dB of shape.

## Errors actually observed

Faust fluency is not the bottleneck. Across ten render invocations from five independent
patches there was exactly one compile error:

- `BoxIdent[tanh] is defined here : maths.lib:782` — bare `tanh` collides with the
  library. Use `ma.tanh`. The same applies to any name the standard library also defines.
- `redefinition of symbols are not allowed : mi` — a helper named `mi` collides with
  `mi.lib`, the modal instrument library, which `stdfaust.lib` imports under that
  prefix. The message names the symbol and nothing else, so it does not point at the
  cause. **Every two-letter library prefix is taken**, and `stdfaust.lib` has about
  thirty of them: `aa an ba co de dm dx en fd fi ho it ma mi ne no os pf pl pm qu rm
  ro si sf so sp sy ve vl wa wd`. `mi` for a modulation index and `dx` for anything
  DX7-shaped are the two that read as natural abbreviations, which is exactly why
  they get written.

Everything else compiled first try, including `re.zita_rev1_stereo`'s six arguments,
`ve.moog_vcf(res, freq)` argument order, `en.adsre`, `ba.take` being 1-indexed, and
`select2` polarity. Do not spend effort on an API cheat sheet; spend it on the rules
above, which is where the real failures were.

That `ve.moog_vcf` compiled first try is the point of the section above it. Getting the
arguments right is not the same as getting audio out, and the failures that cost real
time here were all on the far side of the compiler.
