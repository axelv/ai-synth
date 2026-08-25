# Six measured patches

Complete working instruments. Five were written one-shot from a plain-language
description with no reference material, one per architecture family; the sixth models a
named machine and is the only one here with a clean report. All six compile, all six
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
harness exists to catch, and a corpus of only clean patches cannot test it. `juno-106` is
the other half of that argument: a corpus of only defective patches cannot show what
passing looks like, and it is the one entry to copy the shape of rather than read for
warnings. **Do not copy any of the other five without reading its defects first.**

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

### juno-106.dsp — 0 fail, 0 warn

```
  brightness        4.0d   26.4d    0.7d      2.65x    0.561    0.791
  voices    +12.04 dB for 4x unison at MIDI 62: bit-identical voices
```

**The clean reference.** Every macro moves more shape than level, every extreme of every
declared range peaks between 0.47 and 0.84, and the register spread is 4 dB from MIDI 36
to 84 with the centroid rising monotonically. This is what the harness is asking for.

**Its `+12.04 dB` is correct, and it is the only patch here where that is true.**
`warm-pad` reads the same number against its own stated design; this one reads it because
a Juno's DCOs lock to a digital reset and genuinely do not drift. The chorus is the
compensation the hardware shipped for exactly that. See the exception noted in
`references/faust-poly.md` before decorrelating anything here.

`swell` at 1.3 dB level against 1.7 dB shape is the weakest control in the file. It is an
envelope macro on a pattern whose chords overlap, which is the same coverage limit that
made `warm-pad`'s `tail` look dead, and it is a reason to play the patch rather than read
this row.

Three defects were measured and fixed during the build rather than left in, because this
entry is here to be copied. All three are written up in `references/faust-poly.md`: the
cutoff ceiling `ve.moog_vcf` needs before it returns NaN, the DC an odd saturator makes
from an asymmetric oscillator mix, and the direction a ladder's resonance makeup has to
go.

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
