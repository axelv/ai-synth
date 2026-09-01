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
| `dx7-fm.dsp` | 6-operator phase modulation | emergent, and modelled on a specific machine | `pluck` |

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

### juno-106.dsp — 3 fail, 1 warn

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

One of the two failures and the warning are that cost, and they are one shape: a
hardware panel has controls that legitimately move level. **The other failure is the
harness describing itself**, and is written up below the panel findings.

- **`sustain` fails as a volume control**, 8.1 dB of level for 0.84 dB of shape. It is the
  sustain *level* of an ADSR, so that is precisely what it is. The harness excepts
  envelope-*length* controls and not this one, and it is right to be blunt: nothing
  visible to it separates a sustain fader from a fader someone mislabelled.
- **`level` warns at 221.6 dB.** That is the VCA fader reaching digital silence at zero,
  so the number is an artefact of comparing silence against signal rather than a
  measurement of anything.
- **`delay` fails as inert, and is not.** See *What the harness cannot see* below: the
  LFO's routing defaults to almost nothing, exactly as the panel's does.

`hpf` no longer warns, and the way it stopped is worth recording. Modelled as a
continuous fader it moved 6.9 dB of level against 6.7 dB of shape, which is the profile
of a volume control wearing a filter's name. Modelled from the actual jack board, four
one-pole paths with a bypass and a bass boost, it moves **7.0 dB of level against 8.6 dB
of shape** and reads as the timbre control it is. Getting the circuit right moved a
warning without anyone tuning for it, which is about as good a signal as this harness
gives that a model is closer to the machine.

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

**The chorus LFO is a triangle, and the third failure is the cost of saying so.** It was
`os.osc`, a sine, while both the 106 service notes and the Juno-60 chorus board draw a
triangle. The shape is not cosmetic: a triangle sweeping the BBD clock holds the detune at
two nearly constant values per cycle where a sine lingers at the turnarounds. Making the
change moved every peak in the report slightly, because the chorus colours every
measurement, and that tipped `rate` from 0.31 dB level / 0.50 shape to 0.30 / 0.44, across
the "does nothing" threshold. **The failure is exposed, not caused.** `rate` was already
one hundredth of a decibel above the line, and `delay` at 0.11 / 0.36 already fails with
the identical finding. They are one defect counted twice: the voice LFO does almost
nothing in this pad voicing. Recorded rather than tuned away, because a threshold that
only passes by luck was never passing.

**Not yet re-auditioned.** The six-macro version of this patch was played by a person and
reported fine. The panel version is a different instrument in the ways that matter to the
ear: the DCO mixer is no longer normalised to constant energy, the one `swell` macro has
become four ADSR faders, and the standing cutoff no longer moves with the envelope
amount. The defaults were chosen to land on the old voicing, and the measurements say the
architecture is intact, but neither of those is an audition. Play it before believing it.

### What the harness cannot see

- **The release tail does not exist in an offline render.** The poly engine stops a voice
  within about 0.2 s of note-off whatever the envelope says, and the signal goes to exact
  digital silence rather than decaying: R declaring a 12 s T60 delivered 0.19 s, R
  declaring 1.5 s delivered 0.07 s, and with `group_voices` off the fader had no effect at
  all. `release_length` changes none of it. The same patch in the browser, sampled through
  an AnalyserNode, decays over 1.5 s at R=0.5 and extrapolates to about 14 s at R=1.0,
  which is what the fader claims. The control is correct and the measurement is not. This
  is now in `SKILL.md` under the harness's limits. `release` currently reads 1.0 dB and
  does not trip the inert check, but it is measuring nothing either way.
- **`delay does nothing`.** The LFO DELAY sets how fast the LFO ramps in, and at the
  panel's defaults the LFO has almost nowhere to go: VCF LFO is at 0 and DCO LFO at 0.04,
  which is where a Juno's own defaults sit. Measured, moving DELAY from 0 to 3 s shifts
  the spectral centroid of the first quarter-second by 26 Hz. That is a real control with
  nothing routed to it, not a broken one, and turning up either LFO amount makes it
  audible immediately.

Both were already almost inert before the panel rewrite, at 0.57 and 0.19 dB of level,
sitting just under the threshold. Nothing changed about them; small shifts elsewhere
pushed them across it and back, which is worth knowing about any finding this close to a
limit.

One more thing the harness stopped seeing, and it is a consequence of getting the HPF
right. The `note` line used to report the filter travelling 301 to 1219 Hz within one
note; it now reports 258 to 258. Nothing broke: `vcfEnv` still measures 19.9 dB of shape
and a 1.68x centroid ratio, so the envelope still opens the filter. The default HPF
position is the bypass, so the patch now carries much more low end, and a spectral
centroid is an energy-weighted mean that the bass pins in place. **A fatter patch hides
its own filter sweep from this metric.**

### Hardware facts, and where each one comes from

This entry models a specific machine, so some of what is in the DSP is not a measurement
made here. Those claims used to be listed together as unverified. Most are now sourced,
from two documents:

- **The panel board schematic**, which carries the switches, their latching logic and the
  lamps, and no audio at all.
- **The Roland JUNO-106 Service Notes**, First Edition, 31 July 1984, scanned at
  `archive.org/details/synthmanual-roland-juno-106-service-notes`. The audio sheets are
  the jack board and the module board; component values are on the page images and not in
  the OCR text.

**Verified from the service notes:**

- **The HPF is one pole, four positions, and global**, on the jack board after the
  six-voice sum. A 4052 selects one of four paths into a 47K virtual earth with 47K
  feedback: 4700 pF for 720 Hz, 0.015 uF for 226 Hz, direct for flat, and a shelf network
  for about +10 dB below 72 Hz. **The bypass is position 1 and the lowest position is a
  bass boost**, which is the opposite of what the name suggests and was modelled wrong
  here twice. Modelled correctly it measures 7.0 dB level against 8.6 dB shape, where the
  continuous-fader version measured 6.9 against 6.7 and warned. It is still per voice
  here rather than global, which is wasteful but keeps the saturator's input unchanged.
- **The VCF and VCA are one hybrid per voice**, the A1QH80017A, whose printed internal
  block diagram is an IR3109 plus two BA662s: four cascaded transconductance stages, so
  **4-pole, 24 dB per octave, and it self-oscillates** (the calibration procedure trims
  each voice to a 4.8 Vp-p sine while oscillating). `ve.moog_vcf` here is therefore the
  wrong nonlinearity, a ladder standing in for an OTA design. Cutoff spans 5 Hz to 50 kHz.
- **Key follow is 1:1 at full**, from the calibration: self-oscillation trimmed to 248 Hz
  holding C4 and 992 Hz holding C6, exactly 4x over two octaves. The `kybd` fader's
  exponent mapping gives that at 1.0 without having been designed to.
- **One envelope per voice, shared by VCF and VCA**, and it is not analog: the panel
  sliders are read by the CPU, and the envelope is computed and emitted as a stair-stepped
  digital CV on a sample-and-hold. Declared ranges are attack 1.5 ms to 3 s, decay and
  release 1.5 ms to 12 s. The cubic mappings invented here give 1 ms to 3 s and 5 ms to
  12 s, which is close enough to be luck.
- **The sub is a square an octave down**, divided by a flip-flop inside the waveshaper,
  with its level set by the DAC driving the switching transistor's collector supply.
- **The LFO runs 0.1 to 30 Hz with a 0 to 3 s delay.**
- **The chorus is two MN3009 bucket brigades with their own MN3101 clocks**, fed by one
  triangle LFO **in opposite polarity**, at TP3 and TP4. Antiphase modulation is what
  makes the width. An earlier claim here, that the hardware inverts one wet output, is
  **wrong**: both output mixers are identical inverting summers with the same ratio. The
  rates used here, 0.553 and 0.898 Hz, are solved from the 106's own integrator; the
  0.513 and 0.863 that stood here before are measurements of a Juno-60.
- **There is one noise generator for the whole instrument**, shared by all six voices and
  low-passed around 4.8 kHz. An earlier claim here that each voice board carries its own
  was wrong. The per-voice noise in this patch is now a deliberate departure, kept because
  Faust voices are otherwise bit-identical and a shared source would sum coherently across
  a chord instead of forming a texture.
- **The DCOs are digitally reset and phase-locked**, which is why chords sit still and why
  the machine needs a chorus at all. The 16'/8'/4' switch selects 399K/200K/100K into a
  1 nF integrator, a 4:2:1 ratio.

**From a second machine, and NOT verified for the 106:**

A Juno-**60** chorus board schematic (sheet dated 10 April 1983) was read for this entry.
It corroborates the antiphase mechanism above from an independent machine: panel board B
sends two anti-phase triangles, drawn as the symbols at pins 40->9 and 39->10, one into
each MN3101 clock VCO, and both output mixers are inverting summers of the same polarity.
So the 60 and the 106 agree, and the "inverts one wet output" claim is now wrong twice.

The rest of what that sheet gives is component values, and this repo has already been
burned once importing Juno-60 figures into a 106 patch (the chorus rates). They are
recorded here as leads to check against the 106 jack board, **not** as facts about the
106, and none of them is in the DSP:

- **The wet is louder than the dry, and by a stated ratio.** Both mixers on the 60 are
  inverting summers with 100K feedback, the dry through 47K and the wet through 39K. That
  puts the wet 1.205x above the dry, +1.62 dB. The patch has `dry = 0.72, wet = 0.62`,
  which is the wet 1.3 dB *below* the dry, so the sign is opposite and the magnitude is
  about 2.9 dB out.
- **Engaging the chorus does not duck the dry.** CHORUS OFF (pin 38, 1 = off) drives TR21
  into the wet mute FETs TR8 and TR16 only; the dry path is untouched, so the real machine
  gets louder when the chorus comes on. The `dry = 1 - 0.28 * chOn` here is a level
  compensation the board does not do. The mute is also soft, C44 4.7uF through 47K/470K,
  rather than the hard switch modelled here.
- **The wet path is much darker than one 2-pole at 7.2 kHz, and the dry is not filtered at
  all.** Tracing the sheet properly: at the node after R78 the signal splits, and only one
  branch is filtered. DIRECT SIG goes straight down the page to both mixers, unfiltered.
  The other branch runs R83/R84 22K with C39 820 pF and C38 680 pF into emitter follower
  TR19, then R81/R80 22K with C37 1800 pF and C36 270 pF into TR18, and **that** is what
  feeds both bucket brigades. Solved, those two Sallen-Key sections are 9.69 kHz at Q
  0.549 and 10.38 kHz at Q 1.291; a 4th-order Butterworth wants Q 0.541 and 1.307, so it
  is a Butterworth anti-alias filter and Roland designed it as one. The identical circuit
  appears again after each BBD as the reconstruction filter (R22/R21/C9/C8 then
  R24/R26/C11/C3, around TR6 and TR7). The wet therefore carries nine poles to the dry's
  none: 4-pole Butterworth shared, the 7.23 kHz R15/C7 pole per channel, a 45.4 kHz
  clock-rejection pole, and a second 4-pole Butterworth. An earlier version of this note
  said the dry ran through the TR19/TR18 filter. It does not, and the asymmetry is the
  whole character of the effect.
- **VR1 and VR2 are per-channel BBD bias trims**, set for minimum distortion, which is why
  no two units match and why the wet path distorts asymmetrically. Not modelled.
- **A weak bound on the clock, which does not close the open question above.** Designing
  the anti-alias filter at 7.23 kHz implies a clock whose Nyquist sits above it, so
  something above roughly 15 kHz, so a delay below roughly 8.5 ms. That trims the top of
  the datasheet's 0.64 to 12.8 ms and no more. The 60's clock frequency is set by the
  panel board, which is not on this sheet.

**Still not verified, and unlikely to be:**

- **The saw-to-pulse mixing ratio.** The waveshaper is a Roland/Matsushita MC5534A custom
  chip, and the service notes print its internals: saw and pulse are summed **through
  resistors inside the resin**, with no values given, onto one output pin. There is no
  external level control for either; the panel switches are on/off only. Two independent
  public teardowns of the Juno DCO stop at the same wall. The 0.5 factor used here, which
  matches the two fundamentals by putting 4/pi over 2/pi, is a derivation and not a fact,
  and only measuring a real 106 would settle it.
- **The chorus delay time.** The MN3009 is 256 stages, so delay is 128 over the clock
  frequency, and the datasheet range is 0.64 to 12.8 ms. The 106's actual clock frequency
  is not stated. The base of 4.60 ms and deviations of 2.55 and 3.30 ms used here match no
  source found.
- **The HPF corner frequencies are sourced but the boost shape is approximated**, as a
  first-order shelf rather than the actual two-path summing network.

### dx7-fm.dsp — 1 fail, 1 warn

```
  algo              1.1d    8.8d    4.1d      0.19x    0.684    0.554
  index             0.5d   12.6d    5.5d      8.67x    0.582    0.712
  op1               0.3d    7.3d    0.8d      2.43x    0.672    0.609
  fbMode            0.1d    0.9d    0.0d      0.98x    0.667    0.666
  sustain           6.7d    3.3d    0.7d      1.41x    0.659    0.705
  voices    +12.04 dB for 4x unison at MIDI 64: bit-identical voices
  register  36:-30dB/436Hz  48:-30dB/690Hz  60:-30dB/1291Hz  72:-31dB/2395Hz  84:-31dB/4153Hz
```

**The second panel model, and the closest thing this corpus has to a clean report.**
Twenty-five controls in correspondence with the DX7's parameter grid rather than with a
handful of intents, for the same reason `juno-106.dsp` has 24: the deliverable is an
instrument a DX7 programmer already knows how to operate. Unlike juno-106 it does not pay
for that with a volume control, because it exposes no master LEVEL fader; the master gain
is a measured constant instead.

Neither remaining finding is a defect in the patch:

- **`sustain` warns**, 6.7 dB of level against 3.3 dB of shape. It is the sustain level of
  an envelope, so that is what it is, and it is the same reading juno-106 gets for the
  same reason.
- **`pmd` fails as inert on `pluck`**, at 0.28 dB of shape, and it is not inert. The same
  macro on the same patch measures **1.6 dB of shape on `lead`**. Vibrato is a
  modulation in time and the shape metric averages spectra over the render, so on
  fast repeated notes that barely complete an LFO cycle there is nothing for it to
  see. This is the clearest case in the corpus of the pattern bounding the
  measurement rather than the patch being wrong, and it is worth more than the FAIL:
  before concluding a modulation macro is dead, re-measure it on material that lasts
  longer than a cycle of it.

`index` is now the best macro in the set at 0.5 dB level against 12.6 dB shape and a
centroid ratio of 8.67x, beating `bell-lead`'s `sparkle`. That is not craft, it is what
FM is: one number moves the whole sideband structure and nothing else.

**Its `+12.04 dB` is correct, and for the DX7's own reason rather than the Juno's.** The
hardware time-multiplexes one datapath across 96 operator slots at a fixed rate and its
oscillators are not free-running against each other, so coherent voices are the machine.
There is no chorus to compensate with, because the machine has none.

#### What the hardware research changed about the patch

Three things here come from the reverse-engineering literature rather than from
listening, and each one is checkable:

- **Operator feedback averages the previous TWO outputs, it does not delay by one.**
  The OPS chip holds two previous outputs in shift registers, sums and halves them,
  and modulates with that. Yamaha's patent (Tomisawa, US 4,249,447) gives the reason:
  a single-sample loop makes a large modulation produce a small output and the
  signal alternates every sample, which is an oscillation at Nyquist; the mean of two
  consecutive samples is a 2-tap FIR with a zero exactly there. `F.LOOP` switches
  between the two so the difference can be measured. Measured on the built page,
  algorithm 32 at feedback 1.0, note 62, through an AnalyserNode: mean per-bin
  difference **6.9 dB**, and the top quarter of the band carries **4.4 dB more energy
  in 1TAP** than in AVG (-65.3 against -69.7 dB). That is the predicted direction and
  it is why faustlibraries' own `dx7.lib`, which writes the single tap, carries the
  open TODO "artifacts that sound like aliasing for high feedback values".
- **Carrier-count compensation is real and is usually skipped.** The algorithm ROM
  stores an output count per operator. Six carriers at unity is 15.6 dB hotter than
  one, so without it the ALG switch is mostly a volume control. Dexed computes the
  count in `n_out()` under `#ifdef VERBOSE` and never applies it at render time. With
  `com` trimmed from measured peaks the switch reads 1.1 dB of level against 8.8 dB of
  shape.
- **The audible quantisation is the phase, not the output word.** The OPS truncates
  phase to 12 bits before the log-sine ROM. The first `GRAIN` control here quantised
  the OUTPUT instead, 15 bits down to 7, and measured 0.34 dB of shape: it failed as
  inert, because quantisation noise sits under a bright FM spectrum that already has
  energy in every band. Phase truncation moves the partials themselves and measures
  0.8 dB. A negative result worth keeping, because output-word depth is the obvious
  thing to reach for and it is the wrong one.

#### What is deliberately not modelled

Each of these is a simplification, not an oversight:

- **Six independent 4-rate/4-level operator envelopes**, which is 48 controls. One ADSR
  biased across the stack by `CONTOUR` stands in, geometric in the operator index.
- **28 of the 32 algorithms.** The four here span the shapes: a two-carrier stack, three
  2-op pairs, one carrier under three modulators, and six carriers in parallel. All four
  put feedback on operator 6, as the ROM does for most of the set.
- **Keyboard level scaling breakpoints and curves.** `LV.SCALE` is a single exponent.
- **Feedback is taken from the raw sine**, where the hardware takes it from the operator's
  enveloped output, so on hardware the feedback dies with the note.

#### One Faust collision worth adding to the list

`mi` is `mi.lib`, the modal instrument library, so a helper called `mi` fails with
`redefinition of symbols are not allowed : mi` and not with anything that points at the
name. Same class as the bare `tanh` collision in `references/faust-poly.md`: any
two-letter name that is a standard library prefix is taken.

### juno-106-bbd.dsp — 1 fail, 2 warn

```
  chorus            2.1d    1.0d   76.3d      1.08x    0.426    0.649
  sustain           7.9d    0.9d    0.4d      1.09x    0.658    0.713
warn  delay is inert on the pad pattern but moves the release tail by 45.3 dB
```

**The same instrument with the chorus BOARD in place of a chorus sketch**, so the two
files differ in their effect chain and nowhere else: lines 1 to 230 are a byte copy of
`juno-106.dsp`. The board is traced from a Juno-**60** CHORUS BOARD sheet of 10 April
1983, and the provenance is kept split inside the file, because this repo has already
been burned once for blurring it. The rates are the 106's. Every component value is the
60's and is **not** verified for a 106.

What the circuit changes, measured against `juno-106.dsp` on the same four-note chord:

| | juno-106 | juno-106-bbd |
|---|---|---|
| level, chorus OFF to I | -0.58 dB | **+2.95 dB** |
| side/mid, chorus I | -5.3 dB | **-3.6 dB** |
| DC on the output | 4.9e-06 | 7.4e-09 |
| energy below 20 Hz | -55 dB | -66 dB |

The level and width both move for one reason: the board's mixers are 100K feedback with
the dry through 47K and the wet through 39K, so the **wet sits 1.62 dB above the dry** and
nothing ducks the dry when the chorus engages. `juno-106.dsp` has that relationship
inverted, and its compensation makes the machine slightly quieter when the chorus comes
on. A real one gets louder. The extra width is the same fact seen from the other side:
more wet in the mix is more of the only decorrelated thing in the instrument.

**The filter modelling is correct and contributes nothing here, which is the result worth
keeping.** The wet path carries nine poles to the dry's none: a 4-pole Butterworth shared
ahead of both bucket brigades, a 7.23 kHz pole in each BBD input, a clock-rejection pole,
and a second 4-pole Butterworth on the way out. Driven with pink noise the standalone
board measures -8.9 dB at 8-12 kHz and -37.1 dB at 12-20 kHz against its own dry. Inside
this instrument, level-normalised, the whole difference is -0.4 dB at 8-12 kHz and -0.6 dB
at 12-20 kHz, because the pad's VCF has already removed everything up there. Nine poles
cannot darken a band that holds no energy. On a bright patch it would pay; on this one the
audible change is entirely the mixer ratio.

Two things needed care. The board does not compensate its own level, so at the inherited
0.95 output gain the patch peaked at 1.178 and the harness failed seven macros for
clipping at the top of their range; the headroom is taken at the output instead, leaving
the sourced 47K/39K ratio untouched. And the report's `centroid 285 -> 5 Hz` is a
measurement artifact on silence, not a defect: the release tail is 228 dB below peak, and
the patch is measurably cleaner down low than the original.

The circuit on its own, with its derivation and the arithmetic, is
`references/circuits/juno60-chorus.dsp`.

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
