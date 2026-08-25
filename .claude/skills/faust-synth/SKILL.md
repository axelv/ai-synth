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

`references/examples/` holds six complete working instruments, with `measured.md` naming
what each one gets right and what it gets wrong. Read the one whose family matches the
target sound.

Five of them carry deliberate defects and are read for warnings. `juno-106.dsp` is the
one that measures clean, so it is the one to copy the shape of, and it is also the worked
example of modelling a named machine rather than a description: its `measured.md` entry
separates what was measured here from what is a fact about the hardware.

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
keyboard, a QWERTY mapping, MIDI input, and a slider per macro generated from the DSP's
own metadata. Typically under 300 KiB.

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

- **Published as an artifact**, for sharing and for hearing it immediately. Web MIDI does
  **not** work there: the viewer embeds the page in an iframe without the `midi`
  permission, so `requestMIDIAccess` throws `SecurityError` and no user action fixes it.
  The on-screen and QWERTY keyboards work.
- **Served locally**, where Web MIDI works and a hardware keyboard can play it.

```bash
python -m http.server 8777 --directory patches
```

The page degrades quietly, reporting which case it is in. Do not add a MIDI failure path
that throws.
