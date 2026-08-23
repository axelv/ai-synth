# Polyphonic Faust for a playable instrument

Read before writing any DSP. Everything here either fails silently or was measured
failing in a batch of five patches written without it.

- [The skeleton](#the-skeleton)
- [Two declarations that fail silently](#two-declarations-that-fail-silently)
- [Every voice is a bit-identical copy](#every-voice-is-a-bit-identical-copy)
- [Playing across the keyboard](#playing-across-the-keyboard)
- [Effects belong in the shared chain](#effects-belong-in-the-shared-chain)
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

`measure.py` reports this as `voices +12.04 dB ... bit-identical voices`. Whether that is
a defect depends on the patch. A purely deterministic patch is legitimately coherent. A
patch that claims drift, air, breath, or width is broken if it reads +12.

## Playing across the keyboard

The audition pattern covers one register. The instrument has to work over five.

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

## Errors actually observed

Faust fluency is not the bottleneck. Across ten render invocations from five independent
patches there was exactly one compile error:

- `BoxIdent[tanh] is defined here : maths.lib:782` — bare `tanh` collides with the
  library. Use `ma.tanh`. The same applies to any name the standard library also defines.

Everything else compiled first try, including `re.zita_rev1_stereo`'s six arguments,
`ve.moog_vcf(res, freq)` argument order, `en.adsre`, `ba.take` being 1-indexed, and
`select2` polarity. Do not spend effort on an API cheat sheet; spend it on the rules
above, which is where the real failures were.
