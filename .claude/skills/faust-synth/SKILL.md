---
name: faust-synth
description: Build a playable synthesizer patch in Faust from a plain-language sound description, verify it by measurement, and ship it as a self-contained web page that runs in a browser and accepts MIDI. Use when the user wants to (1) create or design a synth sound or patch, for example "make me a warm analog pad", "build a plucky FM bass", "a 303-style acid line", "a glassy bell lead"; (2) write, debug, or review polyphonic Faust DSP for an instrument; (3) turn an existing .dsp file into something playable from a MIDI keyboard or in the browser; or (4) check a synth patch for defects such as macros that clip, do nothing, or only change loudness.
---

# Faust synth building

Turn a description of a sound into a patch someone can play and keep dialling. The
deliverable is the patch: a DSP architecture plus macro controls a person could turn, not
a fixed sound with sliders bolted on.

Scripts referenced below live beside this file. Set `SK` to the skill directory once:

```bash
SK=.claude/skills/faust-synth
```

## Setup

Two dependencies, both one-time:

```bash
uv add dawdreamer numpy soundfile      # offline rendering and measurement
npm install @grame/faustwasm           # Faust to wasm; ships libfaust, no faust CLI needed
```

The native `faust` binary is NOT required. `faustwasm` runs the compiler as wasm under
node. Install it only if the user wants VST or AU output later.

## Workflow

### 1. Read the conventions before writing DSP

Read `references/faust-poly.md` first, every time. It carries the polyphonic skeleton and
three things that fail silently rather than erroring: the two declarations that change
the parameter path, the fact that every voice is a bit-identical copy, and how to make
a patch work outside the register it gets measured in.

Read `references/patch-design.md` when choosing which macros to expose, their ranges, and
their defaults.

### 2. Write the patch

`references/examples/` holds eight complete working instruments, with `measured.md` naming
what each one gets right and what it gets wrong. Read the one whose family matches the
target sound.

Most of them carry deliberate defects and are read for warnings. **Two model a named
machine rather than a description, and they are the ones to copy the shape of:
`juno-106.dsp` and `dx7-fm.dsp`.** Read `dx7-fm.dsp` for the version of that pattern
that measures well, 1 fail and 1 warn with neither a patch defect, and for what a
panel model costs when the machine has no panel: its 25 controls are the DX7's
parameter grid, and its `measured.md` entry separates what the hardware research
changed from what was a taste decision. Read `juno-106.dsp` for the same exercise on a
real fader panel, and it is the worked example of modelling a named machine
rather than a description: its `measured.md` entry separates what was measured here from
what is a fact about the hardware. It does **not** measure clean. Laying it out as the
machine's own panel put 24 controls on it, and a hardware panel has faders that
legitimately move level, so it carries a deliberate failure and a warning for that,
plus one more failure that is the harness describing itself rather than the patch.
All three are named in its entry, which also records which of its hardware claims
come from the service notes and which are still guesses. No patch in the corpus currently shows a clean report.

Read one **especially** for FM, noise-sourced texture, and per-note filter sweeps. In
those three the mapping from parameters to timbre is emergent rather than specified, and
they are where writing blind goes wrong. Subtractive and additive are predictable enough
to write directly.

Every listed defect in those files is deliberate, so they double as the harness's
regression set. Do not copy a patch without reading its defects first.

Write to a `.dsp` file. Expose 4 to 7 macros named for what a musician would call them,
with defaults set so the patch sounds right untouched.

Do not pretend to iterate on timbre before rendering. Write one considered version, then
measure it.

### 3. Measure

```bash
uv run python $SK/scripts/measure.py patch.dsp bass
```

This compiles and renders internally, so it is also the check that the patch works at
all. Fix every `FAIL` before going further. Read every `warn` against what the macro was
meant to do; they are warnings precisely because they need that judgement.

The last argument picks the pattern it measures on, which must suit the instrument. A
bass measured on a held pad chord is being measured on material it was never written
for, and the pattern bounds what can be seen at all.

If the change was to `measure.py` itself, or to a rule the examples are written against,
run the regression pass over every one of them:

```bash
uv run python $SK/scripts/measure.py --check
```

It exits nonzero if any of them now measures differently, and prints what moved.
`--update` re-records `references/examples/expected.json` once the change is understood
and intended.

| pattern | material | exposes |
|---|---|---|
| `pad` | slow four-note chords, long holds | swell, tail, movement |
| `bass` | single line, short notes, wide register | attack, note-off clicks |
| `lead` | sustained melodic phrase plus a held tail | vibrato, legato |
| `pluck` | fast repeats then a five-note chord | decay length, voice stealing |

### 4. Build the playable page

```bash
uv run python $SK/scripts/build_page.py patch.dsp patches/patch.html "Display Name"
```

One self-contained HTML file: the Faust wasm, its metadata, the runtime, an on-screen
keyboard, a computer-keyboard mapping, MIDI input, and a slider per macro generated from
the DSP's own metadata. Typically under 300 KiB.

The computer keyboard is mapped by **physical position** (`e.code`), never by the
character a layout produces (`e.key`), so the piano keys fall under the same fingers on
AZERTY, QWERTZ and QWERTY. The printed legend is relabelled from
`navigator.keyboard.getLayoutMap()` where the browser reports it, which is Chromium
only; Firefox and Safari keep the QWERTY labels while the keys themselves still play
correctly.

#### Panel skins

`--skin <name>` picks how the controls are drawn. A skin supplies CSS and a control
renderer and nothing else, so the keyboard, the MIDI handling and the boot path stay in
one place however many skins exist. Skins live in `assets/skins/<name>.html`.

| skin | use |
|---|---|
| `plain` | the default. One labelled horizontal slider per macro |
| `juno` | a Juno-106 style panel: vertical faders in red-banded sections, discrete controls as lit buttons |
| `dx7` | a DX7 style panel: emerald-banded sections of slim horizontal slots, membrane buttons with amber lamps, an olive character display that reads the last-touched parameter, and a live algorithm diagram |

**Reach for `juno` when the user asks for a Juno, a Roland-style polysynth, or a
juno-ish pad**, and for `dx7` when they ask for a DX7, a 6-operator FM instrument, or
anything whose controls are operators and algorithms rather than filters and envelopes.
Both are the case where the machine's own panel is the layout the person
already has in their head. It is a homage to the panel's visual grammar, drawn from the
public-domain photograph at `commons.wikimedia.org/wiki/File:Roland-Juno-106.jpg`. No
maker's mark is reproduced.

**Give the page a descriptive display name, not the machine's name.** The display name
is what titles the page and sets its wordmark, so it is trademark use in a way that a
reference file inside this skill is not. `build_page.py` puts the homage in visible page
text on every juno-skinned page, via `SKIN_NOTICES`, so the attribution is stated where a
reader can check it rather than only in a source comment. That notice is per-skin and
automatic; there is no flag to remember and no call site that can drop it.

```bash
uv run python $SK/scripts/build_page.py juno.dsp patches/juno.html "Chorus Polysynth" --skin juno
```

The skin reads four optional metadata keys off each slider label. They are inert
everywhere else: `measure.py` addresses macros by label and Faust strips metadata out of
the label, so adding them does not move a single measurement.

| key | effect |
|---|---|
| `[panel:VCF]` | which panel section the control sits in. Sections are laid out in the machine's own order: LFO, DCO, HPF, VCF, VCA, ENV, CHORUS, then anything else. A section with no controls is not drawn |
| `[idx:2]` | position within the section. **Required if order matters**: Faust emits controls alphabetically, not in source order |
| `[cap:RES]` | a shorter panel caption. Defaults to the macro name, which is usually the better label |
| `[positions:OFF\|I\|II]` | names for a discrete control's steps. A control with 2 to 4 steps is drawn as buttons rather than a fader. `OFF\|ON` becomes one latching button with a lamp above it, the way each waveform switch is on the panel; three or more positions become a row of buttons each with its own lamp; any other two-position control becomes a plain pair, and gets no lamp, because the hardware uses an unlit slide switch there |

```faust
brightness = hslider("brightness[panel:VCF][idx:1]", 0.44, 0, 1, 0.001) : si.smoo;
chorus     = hslider("chorus[panel:CHORUS][positions:OFF|I|II]", 0.5, 0, 1, 0.5);
```

`references/examples/juno-106.dsp` carries the full set and is the one to copy. The
`dx7` skin reads the same four keys and adds one convention of its own: it treats
`positions` as a value-label map at any step count, not only at two to four. A control
with more positions than it will draw as buttons stays a slot and shows the position's
NAME as its readout, which is how `dx7-fm.dsp`'s operator ratios read `1.41` instead of
step `2`. The DSP stays the authority on what its numbers mean.

#### Two things that fail silently when verifying a page

Both were found by driving the built page from a browser, and both make a control look
right while being wrong:

- **Set a range input's `step` before its `value`.** A range input snaps its value to the
  step on assignment and the step defaults to 1, so assigning `value` first rounded 0.44
  to 0 while the readout printed beside it still said 0.44. The fader sat at the bottom,
  the DSP kept its own default, and nothing errored. The two disagreed until first touch.
- **`getParamValue` on the poly node reads one call stale.** Setting a parameter and
  reading it back in the same tick returns the *previous* value, so a working control
  reports as dead and the next check inherits the answer to the one before it. It cost a
  wrong conclusion here. Let a tick pass before reading, and confirm against a value the
  previous call did not already write.

### 5. Audition by playing it, not by rendering it

**Build the page before auditioning. Do not judge a patch from a rendered wav.**

Measured the hard way: five patches were judged from offline renders of a fixed pattern
and three of the five were called wrong. The same five, played, were all fine. A canned
pattern misrepresents an instrument, because it fixes the register, the velocities, the
note lengths and the voice count at whatever one guess was baked into the pattern, and
an instrument is the thing that has to hold up when none of those is fixed.

So hand over the page and let a person play it. Ask two questions: is it recognisably the
described sound, and does anything sound broken. Revise from that answer.

Rendering a wav is still useful for sending someone a fixed example, and it is what
`measure.py` does internally. It is not the audition.

```bash
uv run python $SK/scripts/faust_render.py patch.dsp patches/patch.wav pad
```

## Reading the measurement report

```
  macro            level   shape   width  centroid  peak lo  peak hi
  sparkle           0.0d   10.3d    0.0d      1.00x    0.701    0.712
```

| column | meaning |
|---|---|
| `level` | dB change in loudness across the macro's full range |
| `shape` | dB change in **level-normalised** spectral shape; this is the timbral work |
| `width` | dB change in side-versus-mid energy |
| `centroid` | ratio of spectral centroid at max to at min; below 1.0 means darker |
| `peak lo`/`hi` | absolute peak at each end of the declared range |

A good timbre macro has `shape` well above `level`. A macro with large `level` and near
zero `shape` is a volume control whatever its name says.

The other lines:

- `voices` — `+12.04 dB` for 4x unison means the voices are bit-identical. Correct for a
  deterministic patch, broken for one that claims drift, air, breath or width.
- `register` — level and centroid at MIDI 36 to 84. Levels should sit within about 12 dB
  and centroid should rise with pitch.
- `release` — whether the tail finishes inside the render or is being cut off.
- `note` — spectral centroid travel within one note, and how fast. Compare against what
  any time-valued macro claims; an exponential envelope through an exponential mapping
  can deliver 90% of its travel in a twentieth of its declared time.

### Three limits, two of the harness and one of measuring around it

The first two were found by the harness reporting a working control as broken. Do not
trust a `does nothing` verdict without checking them:

- **The measurement pattern bounds what can be seen.** A release control is invisible on a
  pattern whose chords overlap. The harness retries a suspected-inert macro on an
  isolated note and says so, but any macro the pattern does not exercise is unmeasured.
- **Everything except `width` folds to mono.** A correct mid/side widener leaves the mono
  sum untouched by construction.
- **The offline renderer truncates the release tail, so the tail is not measurable here
  at all.** The poly engine stops a voice within about 0.2 s of note-off whatever the
  patch's envelope says, and the output goes to exact digital silence rather than
  decaying. It is not `release_length`, which changes nothing; it does move with
  `group_voices`, and with grouping off an ADSR's R fader has no effect on the render
  whatsoever. Measured on juno-106: R declaring a 12 s T60 delivered 0.19 s offline,
  and R declaring 1.5 s delivered 0.07 s. The **same patch in the browser is correct** —
  R=0.5 decays 1.5 s and R=1.0 extrapolates to about 14 s, sampled through an
  AnalyserNode on the running page. So any `release does nothing` verdict, and the
  report's `release` line, describe the harness. Confirm a tail on the page, never here.

The third is about probes written alongside `measure.py` rather than about `measure.py`,
which windows correctly:

- **An RMS window shorter than the note's period measures waveform phase, not level.** A
  hand-rolled probe using 128 samples, 2.9 ms, on a MIDI 62 note whose period is 3.4 ms
  reported 4.7 to 8.9 dB of level swing on a held note. The same signal at 2048 samples,
  46 ms, reads 0.23 to 0.47 dB. The first number is an artefact and it is convincing
  enough to get a patch changed over it, which is what happened. Window several periods
  of the lowest note being measured, and confirm any swing by switching off the thing
  that supposedly causes it.

## Delivery

The generated page serves two roles from one file:

- **Published as an artifact**, for sharing and for hearing it immediately. Build it with
  `--fragment`: the artifact wrapper supplies `<!doctype html>` and the `<head>` itself,
  and a second shell inside the body would be a duplicate. Web MIDI does **not** work
  there: the viewer embeds the page in an iframe without the `midi` permission, so
  `requestMIDIAccess` throws `SecurityError` and no user action fixes it. The on-screen
  and QWERTY keyboards work.
- **Served locally**, where Web MIDI works and a hardware keyboard can play it. This is
  the default build, a whole document, because a fragment served over HTTP parses in
  quirks mode: measured on a 375 px viewport, `document.compatMode` reads `BackCompat`
  and the layout viewport comes out 980 px wide, so a skin's width media query never
  fires and the panel is too small to play. Nothing shows at desktop widths, which is
  how it went unnoticed. Check `compatMode` at a phone viewport, not by eye.

```bash
python -m http.server 8777 --directory patches
```

The page degrades quietly, reporting which case it is in. Do not add a MIDI failure path
that throws.
