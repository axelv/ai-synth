# Five measured patches

Complete working instruments, one per architecture family, each written one-shot from a
plain-language description with no reference material. All five compile, all five play,
and all five have been listened to. They serve two purposes:

1. **Worked examples.** Read the one whose family matches the sound being built. The
   families differ in how predictable the mapping from parameters to timbre is, and the
   three where it is emergent rather than specified are the ones worth reading before
   writing rather than after.
2. **A regression set.** The expected findings below are what `measure.py` reported. Change
   the harness or a rule, re-run all five, and diff. A finding that disappears is either a
   fix or a regression, and the difference matters.

Every defect listed is deliberate and left in place. They are the cases the harness exists
to catch, and a corpus of only clean patches cannot test it. **Do not copy a patch without
reading its defects first.**

## The families

| patch | family | mapping | pattern |
|---|---|---|---|
| `warm-pad.dsp` | subtractive | direct: cutoff and envelope do what they say | `pad` |
| `bell-lead.dsp` | additive, inharmonic | direct: the partial ratios *are* the sound | `lead` |
| `fm-bass.dsp` | phase modulation | emergent: sidebands from index and ratio interaction | `bass` |
| `breathy-texture.dsp` | noise-sourced with resonators | emergent: depends on decorrelation | `pad` |
| `acid-lead.dsp` | resonant per-note sweep | emergent: envelope through a nonlinear mapping | `bass` |

Reproduce any row with:

```bash
uv run python <skill>/scripts/measure.py <skill>/references/examples/warm-pad.dsp pad
```

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

## What all five share

Every patch reads `+12.04 dB` for 4x unison except `breathy-texture`, meaning voices are
bit-identical. For the deterministic patches that is correct. For `warm-pad`, whose own
comments call per-voice drift "the whole trick behind an analog ensemble", it defeats the
stated design, and for `breathy-texture` it removes the decorrelation the texture depends
on. Neither was fixed here.

Worth knowing before weighting that rule too heavily: `warm-pad` was judged as sounding
right anyway. The defect is real and measured, and it is not what decides whether a patch
works.
