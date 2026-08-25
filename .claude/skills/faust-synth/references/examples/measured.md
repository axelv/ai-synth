# Six measured patches

Complete working instruments. Five were written one-shot from a plain-language
description with no reference material, one per architecture family; the sixth models a
named machine and is laid out as that machine's panel. All six compile and all six
play. They serve two purposes:

1. **Worked examples.** Read the one whose family matches the sound being built. The
   families differ in how predictable the mapping from parameters to timbre is, and the
   three where it is emergent rather than specified are the ones worth reading before
   writing rather than after.
2. **A regression set**, which is now `measure.py --check` rather than this file. What
   each patch is expected to report lives in `expected.json`, and a change there is a
   diff rather than a paragraph asking someone to re-run six patches by eye. A finding
   that disappears is either a fix or a regression, and the difference matters.

The findings quoted below are illustration for the prose around them. They are not the
expectations the check compares against, so they cannot silently disagree with the code:
`expected.json` is the record, and it is regenerated with `--update`.

Every defect in the first five is deliberate and left in place. They are the cases the
harness exists to catch. `juno-106` used to be the other half of that argument and is
not any more: modelling the hardware panel gave it 24 controls and, with them, a
deliberate failure and two warnings, both named in its entry below. It is still
the one entry to copy the *shape* of, because its architecture is the sound one here, but
it is no longer the one that shows what a clean report looks like. Nothing does, which is
a gap worth closing. **Do not copy any patch here without reading its defects first.**

## The families

| patch | family | mapping | pattern |
|---|---|---|---|
| `warm-pad.dsp` | subtractive | direct: cutoff and envelope do what they say | `pad` |
| `bell-lead.dsp` | additive, inharmonic | direct: the partial ratios *are* the sound | `lead` |
| `fm-bass.dsp` | phase modulation | emergent: sidebands from index and ratio interaction | `bass` |
| `breathy-texture.dsp` | noise-sourced with resonators | emergent: depends on decorrelation | `pad` |
| `acid-lead.dsp` | resonant per-note sweep | emergent: envelope through a nonlinear mapping | `bass` |
| `juno-106.dsp` | subtractive, DCO | direct, and modelled on a specific machine | `pad` |

Reproduce one row, or check them all:

```bash
uv run python <skill>/scripts/measure.py <skill>/references/examples/warm-pad.dsp pad
uv run python <skill>/scripts/measure.py --check
```

`--check` measures all six and exits nonzero if anything moved. Run it after changing
`measure.py`, after changing a rule that the patches are written against, and before
adding a seventh.

## Expected findings

### warm-pad.dsp — 0 fail, 1 warn

```
warn  tail is inert on the pad pattern but moves the release tail by 23.4 dB
```

Not a patch defect. `tail` works; the `pad` pattern holds chords with 0.2 s gaps so the
release is always masked. This is the harness reporting its own coverage limit, and it is
the reason the inert check retries on an isolated note. It is also why a patch is
auditioned by playing it, never by listening to one of these renders.

### fm-bass.dsp — 1 fail, 3 warn

```
FAIL  width peaks 1.277 at width=1 (max of its declared range)
warn  body, decay, drive all move level more than shape
```

**Real defect, left in.** `spread(x) = (x+s, x-s)` is not gain-compensated, so the widener
clips at the top of a range the patch itself declares usable. `references/patch-design.md`
carries the energy-preserving form. `drive` at 10.6 dB level against 2.9 dB shape is the
canonical volume-knob-labelled-drive case; `body` and `decay` are envelope controls and
their level change is correct.

Also the only patch here that omits `effect`, which is why its parameter paths come back
shaped differently. See `references/faust-poly.md`.

### bell-lead.dsp — 1 fail, 1 warn

```
FAIL  decay peaks 1.116 at decay=14 (max of its declared range)
warn  decay moves level more than shape (15.5 dB against 8.4 dB)
```

**Real defect, left in.** Long decays overlap and there is no headroom management. The
warn on the same macro is *not* a defect: a longer decay genuinely holds more energy.

Its `sparkle` is the best macro in the set, 0.0 dB level against 10.3 dB shape, and worth
reading as the shape of a correct timbre control.

### breathy-texture.dsp — 0 fail, 1 warn

```
warn  air moves level more than shape (8.0 dB against 5.3 dB)
voices +10.72 dB for 4x unison
```

**The voice reading is the interesting one.** Every other patch measures +12.04. This one
reads +10.72 because a `ma.tanh` ceiling sits on the shared summed bus, so polyphony
changes timbre: the ceiling is compressing by 1.7 dB at four voices. Per-voice saturation
is the correct placement.

Its noise is identical across voices, so a chord sums coherently instead of forming a
texture, which is the defect the voice-decorrelation rule exists to prevent.

### juno-106.dsp — 1 fail, 2 warn

```
  cutoff            1.4d   25.0d    0.5d      2.34x    0.588    0.588
  level           221.6d   17.6d    5.7d      0.00x    0.000    0.706
  saw               4.3d    2.2d    0.3d      1.06x    0.427    0.565
  sustain           8.1d    0.8d    0.1d      1.05x    0.565    0.622
  voices    +12.04 dB for 4x unison at MIDI 62: bit-identical voices
```

**The panel model, and the one place this skill's macro rule is deliberately broken.**
Every other patch here exposes 4 to 7 macros, each driving several destinations. This one
exposes 24 controls in one-to-one correspondence with the machine's own faders, across
all seven of its sections, because the deliverable is an instrument a Juno player already
knows how to operate. `patch-design.md` still describes what to do when building from a
description. This is what modelling a named panel costs instead.

The failure and both warnings are that cost, and they are one shape: a hardware panel
has controls that legitimately move level.

- **`sustain` fails as a volume control**, 8.1 dB of level for 0.84 dB of shape. It is the
  sustain *level* of an ADSR, so that is precisely what it is. The harness excepts
  envelope-*length* controls and not this one, and it is right to be blunt: nothing
  visible to it separates a sustain fader from a fader someone mislabelled.
- **`level` warns at 221.6 dB.** That is the VCA fader reaching digital silence at zero,
  so the number is an artefact of comparing silence against signal rather than a
  measurement of anything.
- **`hpf` warns**, 6.9 dB level against 6.7 dB shape. A high-pass filter removes energy;
  taking the bottom out of a pad is a level change as well as a timbral one.

**This file no longer carries a clean report, and the corpus now has none.**
`warm-pad.dsp` is the nearest thing, at 0 fail and one warn that is a harness coverage
limit rather than a patch defect. That is a real loss, because a corpus of only defective
patches cannot show what passing looks like. It should be repaid by adding a clean
seventh patch, not by walking this one back.

**Its `+12.04 dB` is correct, and it is the only patch here where that is true.**
`warm-pad` reads the same number against its own stated design; this one reads it because
a Juno's DCOs lock to a digital reset and genuinely do not drift. The chorus is the
compensation the hardware shipped for exactly that. See the exception noted in
`references/faust-poly.md` before decorrelating anything here.

Six defects were measured and fixed during the build rather than left in, because this
entry is here to be copied. Three are written up in `references/faust-poly.md`: the cutoff
ceiling `ve.moog_vcf` needs before it returns NaN, the DC an odd saturator makes from an
asymmetric oscillator mix, and the direction a ladder's resonance makeup has to go. The
fourth arrived with the panel: **filter modulation has to be multiplicative.** Written as
a sum, `1 + env*depth*pol`, the polarity switch subtracted more than the standing cutoff
and pinned it to the 60 Hz floor, measuring 33.3 dB of level swing and a peak of 0.012
against 0.598, which is the voice vanishing rather than a filter sweep. As a ratio,
`pow(1 + env*depth, pol)`, INV became the mirror of NORM and the same change moved the
VCF's LFO amount from 0.8 dB level / 6.0 dB shape to 0.2 / 10.7. A cutoff lives in
octaves, so modulate it in octaves.

Two more came out of probing the waveform switches with the filter wide open, and both
are the same lesson: **a control that used to be part of a crossfade has to be re-checked
when it becomes a switch of its own.**

- **The pulse arrived 6 dB hot.** Normalised by `sin(pi*w)`, a pulse train's fundamental
  is 4/pi where a sawtooth's is 2/pi, so the pulse sat 6.02 dB above the saw. Measured
  5.71 dB. With both switched on, the pulse drowned the saw and turning SAW off moved the
  spectrum by 0.9 dB, a switch that did nothing. This was invisible while a single `tone`
  macro crossfaded the pair across a 15 dB range, and it only became a defect when they
  became two independent switches. A leading 0.5 in the normaliser matches the two
  fundamentals; SAW then measures 4.3 dB level / 2.2 dB shape against PULSE's 3.1 / 2.1.
- **The PWM fader ran off the end of its own normaliser.** The divisor is floored at
  `max(0.34, sin(pi*w))` so it cannot approach zero. The old macro's widest sweep reached
  w=0.22, where `sin(pi*w)` is 0.64 and the floor never engaged; the panel's PWM fader
  reached w=0.05, where it does. Past that point the compensation stops and the control
  becomes a fader: measured, the fundamental fell 12.1 dB across the range and the RMS
  4.3 dB. Capping the depth at 0.36, so the narrowest width is 0.14 and `sin(pi*w)` is
  0.43, brings that back to 2.0 dB and 0.6 dB. **A guard sized for one range is not a
  guard for a wider one**, and nothing errors when it is exceeded.

Probing the mix also confirmed what should be true and is: with both waveforms off, only
the sub's odd harmonics remain, at f0/2, 3f0/2, 5f0/2, with the integer harmonics 75 dB
down; and at the default width of 0.331 the pulse's third harmonic collapses by 30 dB,
which is the null a duty of exactly 1/3 has on harmonic 3.

**Not yet re-auditioned.** The six-macro version of this patch was played by a person and
reported fine. The panel version is a different instrument in the ways that matter to the
ear: the DCO mixer is no longer normalised to constant energy, the one `swell` macro has
become four ADSR faders, and the standing cutoff no longer moves with the envelope
amount. The defaults were chosen to land on the old voicing, and the measurements say the
architecture is intact, but neither of those is an audition. Play it before believing it.

### What is a hardware fact rather than a measurement

Everything else in this skill was measured in this repo. These were not. They come from
what the machine is, they are what make the patch a Juno rather than a generic
subtractive pad, and **they have not been verified against hardware**:

- One DCO per voice, digitally reset, so voices phase-lock rather than drift.
- **One** ADSR, wired to both the VCF and the VCA. Not two envelopes.
- A 1-pole HPF ahead of the VCF, on a 4-position switch whose lowest position is a bass
  boost rather than a bypass.
- A 4-pole resonant VCF with keyboard tracking.
- The sub oscillator is a square an octave down, not a sine.
- PWM sweeps down from square toward a narrow pulse, rather than around 50%.
- Chorus is two buttons: mode I near 0.5 Hz, mode II near 0.83 Hz, both up is bypass. The
  hardware inverts one wet output, which is why a Juno through a mono desk loses the
  chorus entirely. `juno-106.dsp` modulates two lines in antiphase instead, which keeps
  the mono sum intact at the cost of that particular authenticity.
- No reverb anywhere in the machine.

What is NOT reference material, and is fitted gain staging for this patch alone:
`cutBase = 120 * pow(35, brightness)`, `makeup = 1 + 0.90 * resonance`, and the master
`0.30`. Re-fit those for any patch that changes the oscillator mix.

### acid-lead.dsp — 0 fail, 1 warn

```
warn  drive moves level more than shape (10.9 dB against 4.5 dB)
note  centroid 229 -> 235 Hz, 50% of that travel by 0.012 s, 90% by 0.012 s
```

**Two real defects that produce no failure**, which is the point of including it. `decay`
declares 0.30 s and delivers 90% of the filter travel in 12 ms, because an exponential
envelope through an exponential cutoff mapping collapses. And `drive` has a centroid ratio
of 0.48x, so it darkens rather than opening up, the opposite of what the name implies.

Neither is a `FAIL`, because neither is wrong without knowing what the macro was for. They
are exactly the class of thing that needs a human reading the report.

## What they share

Every patch reads `+12.04 dB` for 4x unison except `breathy-texture`, meaning voices are
bit-identical. For the deterministic patches that is correct, and for `juno-106` it is the
instrument being modelled. For `warm-pad`, whose own comments call per-voice drift "the
whole trick behind an analog ensemble", it defeats the stated design, and for
`breathy-texture` it removes the decorrelation the texture depends on. Neither was fixed
here.

Worth knowing before weighting that rule too heavily: `warm-pad` was judged as sounding
right anyway. The defect is real and measured, and it is not what decides whether a patch
works.
