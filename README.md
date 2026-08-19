# ai-synth

Reverse-engineering a playable synth patch from a recording.

Given 18 seconds of a synth pad, recover the notes that were played and the synth
settings that produced the sound, such that rendering them reproduces the recording.

The deliverable is **the patch**, not the audio. Matching the waveform by any means is a
different and easier problem; the constraint here is that the answer has to be something
a person could dial into a synth. That is why the renderer is a real Faust DSP driven
through dawdreamer rather than a neural vocoder or a direct optimisation over
spectrograms, and it is what makes some otherwise attractive shortcuts inadmissible.

## The pipeline

**Stage 1, transcription.** Recover 29 notes and a measured pitch-bend curve from the
audio. This is frozen: everything downstream treats the note list and bend automation as
given inputs, so that any change in output is attributable to the synth rather than to
the notes.

**Stage 2, patch fitting.** Fit synth parameters against a multi-resolution STFT loss.
The synth is a detuned supersaw pad with a sub oscillator, a resonant lowpass, two ADSR
envelopes, a 26-band EQ, chorus, ping-pong delay and a Zita reverb. 55 parameters.

## Results

| | target | before | after |
|---|---|---|---|
| loss | 0 | 1.5446 | **1.3823** |
| spectral centroid | 1052 Hz | 1362 | **1071** |
| rolloff 95% | 3752 Hz | 4397 | **3795** |
| mel distance | 0 | 7.048 | **5.163** |
| chroma agreement | 1.0 | 0.9129 | **0.9211** |
| envelope L1 | 0 | 0.1041 | **0.0865** |

"Before" is the best result from conventional parameter search: roughly 7500 CMA-ES
renders plus a differentiable PyTorch surrogate of the whole synth. "After" adds one
element, a fitted 26-band EQ at fixed frequencies, and is worth about **fourteen times**
everything the search achieved on its own.

Brightness now matches the reference for the first time, and every metric improved
including the ones the fit never optimised.

## What the project actually taught

The interesting results here are mostly about method, and several are negative.

**The reachable set was the constraint, not the optimiser.** Thousands of renders moved
the loss almost nothing because the spectral shape the target needs is non-monotonic and
a two-pole lowpass cannot produce it at any setting. The shape was not hard to find, it
was absent. Fitting an unconstrained oracle in the same domain first would have shown
this in minutes, and it now does: it separates "cannot represent" from "cannot find".

**Timbre here is set by absolute frequency, not harmonic number.** Fitting the 382
audible partials of one chord, a 34-parameter frequency curve left 4.20 dB residual
against 4.47 dB for a 177-parameter wavetable. The wavetable was built anyway, measured a
contribution of exactly zero once a proper frequency curve was in place, and removed.

**The objective is a weak ranker, and knowing how weak matters.** Calibrated against
deliberately wrong controls, spectral cosine similarity on this material reads 0.68 for
the original with its frames shuffled, 0.79 for the original delayed a full second, and
0.84 for the best possible shaping correction. Every plausible answer is compressed into
a narrow band, which explains why so much optimisation bought so little. Calibrate a
similarity metric against a wrong answer before reading any value as good or bad.

**A lower loss is not sufficient evidence.** One fit improved the aggregate loss while
being clearly worse on a different chord and regressing the envelope metric from 0.104 to
0.189. Acceptance now requires per-window checks plus metrics the fit never saw.

**The recording is two instruments.** Stereo width in the target rises about 16 dB from a
dead-centre low end to a fully decorrelated midrange: a mono bass under a wide pad. A
single voice path through one stereo chain is flat across the same span whatever its
parameters. A two-layer synth was built to address this and rejected on measurement; a
band-wise mid/side EQ does the same job better and at no cost to the mono spectrum.

## Running it

```
PYTHONPATH=scripts uv run python scripts/<script>.py
```

| script | does |
|---|---|
| `synth.py` | the Faust DSP, parameter definitions, the renderer |
| `stage2.py` | the objective and the CMA-ES driver |
| `fit_eq_full.py` | fits the EQ on the whole clip with a curvature penalty |
| `promote_eq.py` | acceptance gate: per-window scores plus unseen metrics |
| `diagnose.py` | measures the fit never optimised, including stereo structure |
| `chord.py` | single-window bench for quick diagnosis |

Current patch and render are `out/patch.json` and `out/render.wav`.

Dependencies are managed with [uv](https://docs.astral.sh/uv/): `uv sync`.

## Caveats

The remaining error is mostly stereo. The objective scores a mono downmix by default, so
it cannot see stereo width at all and actively prices the correct reverb depth as a
regression; stereo scoring exists but is opt-in. Note timing is also weak, and the loss
can barely distinguish correct timing from shuffled frames, so that gap will not close
without a different objective.
