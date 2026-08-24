# ai-synth

## What this is

Two tracks under one rule.

**The rule, which governs both: the deliverable is the patch, not the audio.** Anything
that matches a waveform but does not correspond to controls a person could dial in is
off-topic. That rules out learned vocoders, direct spectrogram optimisation, and
sample-level tricks, and it is why Faust stays the authoritative renderer on both
tracks: whatever is built has to be something a synth can actually do.

### Track A, building patches. Active.

A description of a sound goes in, a Faust patch that plays in a browser comes out,
checked by measurement rather than by ear. The skill lives in
`.claude/skills/faust-synth/` and owns its scripts; reach for those when the task is
building an instrument.

This is the half of `VISION.md` that was known-buildable. It exists because the full
app, matching a patch to a clip automatically, was judged too long a shot to attempt
first, and because the playable half is a prerequisite for it anyway. Nothing here is
throwaway if track B resumes.

### Track B, fitting a patch to a recording. Paused.

Reverse-engineering a patch from 18 seconds of a synth pad (`data/original.wav`) into a
note list plus a parameter vector. Two stages, in dependency order:

1. **Stage 1, transcription.** Notes and a measured pitch-bend curve. **Frozen.** Treat
   `data/transcription.mid` and `bend2.bend_curve` as inputs, not parameters. Re-opening
   this is a separate project and it confounds any judgement of stage 2.
2. **Stage 2, patch fitting.** Synth parameters against a multi-resolution STFT loss.
   **Paused, not abandoned**, at loss 1.544635 with `out/patch.json` as the incumbent.

Paused means the code still runs, every finding below still holds, and none of it is
retracted. It is not the place to start work without being asked to. Do not delete or
"tidy" stage-2 code on the assumption it is dead.

## Running things

Always from the repo root, always through uv. Use the uv project workflow (`uv add`),
never `uv pip install` into a bare venv. Node is here only for `@grame/faustwasm`, which
ships libfaust as wasm; `npm install` at the root, and the native `faust` binary is not
required for anything currently in the repo.

### Track A

```
uv run python .claude/skills/faust-synth/scripts/<script>.py
```

Paths below are relative to `.claude/skills/faust-synth/`. **Track A never uses the
repo-root `scripts/`, which belongs to track B.**

| file | role |
|---|---|
| `SKILL.md` | the workflow. Read it before touching the rest |
| `scripts/faust_render.py` | offline audition renderer, `Instrument`, the four patterns |
| `scripts/measure.py` | the verification pass. Macro sweeps, voice coherence, register |
| `scripts/build_page.py` | one `.dsp` to one self-contained playable page |
| `references/faust-poly.md` | poly conventions and the failures that are silent |
| `references/patch-design.md` | what makes a macro a macro; ranges and defaults |

### Track B

```
PYTHONPATH=scripts uv run python scripts/<script>.py
```

| file | role |
|---|---|
| `scripts/synth.py` | the Faust DSP, `PARAMS`, `PadRenderer`. The authoritative renderer |
| `scripts/stage2.py` | `Objective` (the loss), `load_notes`, CMA-ES driver |
| `scripts/eq_stage.py` | the 26-band EQ cascade and its gain fitting |
| `scripts/fit_eq_full.py` | full-clip EQ fit; the commutation trick and curvature penalty |
| `scripts/promote_eq.py` | acceptance gate: per-window plus unseen metrics |
| `scripts/diagnose.py` | measures the fit never optimised, including stereo structure |
| `scripts/chord.py` | single-window bench for quick diagnosis |
| `scripts/metrics.py` | shared metrics; reuse these rather than writing new ones |
| `scripts/bend2.py` | the measured pitch-bend lane. Stage-1 output, consumed as an input |
| `scripts/faust_probe.py` | renders isolated Faust sub-chains. Still the way to hear one stage alone |

Current state lives in `out/patch.json` and `out/render.wav`.

## Traps that have already cost real time

- **`data/original.wav` is 48 kHz.** Always `librosa.load(path, sr=44100)`. Reading it
  with `soundfile.read` and no resample silently compares audio 8.8% wrong in time and
  pitch. It has produced at least one bogus results table.
- **Write renders with `synth.write_render`** (PCM_24). soundfile's default PCM_16 costs
  0.089 of loss here, because the log-magnitude term sees the 16-bit floor in the
  6-16 kHz band. Score from the file on disk, not the array in memory.
- **`synth.PARAMS` is append-only.** `out/patch.json` depends on the order.
- **`eq_stage`'s engine cache keys on the signal**, not just its length. It bakes the
  playback buffer in at construction, so a length-only key silently returns the first
  signal filtered.
- **`dawdreamer.set_automation` needs the full Faust path**, not the slider label.
- Faust oscillators are free-running, so splitting notes across processor instances
  changes voice allocation and therefore phase. Compare **spectra, not waveforms**.
- **A DSP with no `effect` still returns `True` from `set_dsp_string`**, logging only
  `ERROR : undefined symbol : effect` to stderr, and the parameter path changes shape.
  `declare name` changes it independently: with both, `/Sequencer/DSP1/Polyphonic/
  Voices/<name>/x`; without `effect`, `/Polyphonic/Voices/<name>/x`; without the name,
  `.../Voices/dawdreamer/x`. Declare both, always, and address parameters by **label**.
- **Faust poly voices are bit-identical.** Every voice is a copy of `process` with the
  same initial state, so `no.noise` and free-running `os.osc` produce the same samples
  in every simultaneously gated voice: 4 voices measure +12.04 dB over 1, not +6.02, and
  the 4-voice render equals 4x the 1-voice render sample for sample. Unison detune
  inside one voice still works; two voices on one note phase-add instead of beating, so
  there is no chorusing and no decorrelation. Derive anything that must vary per voice
  from `freq`, the only thing that differs between them.

## Settled by measurement; do not re-litigate

- **The reachable set is usually the constraint, not the optimiser.** Three rebuilds
  ended this way. Before building, fit an unconstrained oracle in the same domain and
  compare parameter counts; it distinguishes "cannot represent" from "cannot find" in
  minutes.
- **Timbre here is set by absolute frequency, not harmonic number.** A 34-parameter
  frequency curve beat a 177-parameter wavetable. A fitted wavetable measured a
  contribution of exactly zero and was removed.
- **Fit on the whole clip, not one chord.** A single-chord fit produced a comb tuned to
  that chord's partials: better on its own chord, 0.153 worse on another. Windows are
  for diagnosis.
- **Penalise curvature when fitting EQ gains.** Without it the fit walks into the
  bank's near-singular alternating direction.
- **A lower loss is not sufficient evidence.** Always check per-window and the metrics
  the fit never saw (`mel_dist`, `env_l1`, `onset_f`, centroid). `env_l1` has caught
  overfitting twice.
- **The objective is a weak ranker.** Calibrated on this material, cos theta is 0.68 for
  frame-shuffled audio, 0.79 for the original delayed a second, 0.84 for the best
  possible shaping oracle. Calibrate any similarity metric against a deliberately-wrong
  control before reading a value as good or bad.
- **`stage2.Objective.loss_of` is mono** and cannot see stereo width at all. Stereo
  scoring is opt-in via `loss_parts`, which reports the mono term alongside so historical
  numbers stay comparable. `measure.py` rediscovered the same blindness from scratch and
  reported a working mid/side widener as a dead macro. Any new metric here needs a
  stereo term stated explicitly or it will make the same mistake a third time.
- **Writing Faust is not the constraint; verifying it is.** Five patches written one-shot
  from plain-language descriptions, with no reference material, produced 5 of 5 compiling
  and exactly one compile error across ten render invocations. The defects were all
  silent: macros clipping at extremes nobody had turned them to, controls that moved
  loudness rather than timbre, voices that never decorrelated. Spend effort on
  measurement, not on a Faust API reference.
- **Separate a macro's effect on level from its effect on level-normalised spectral
  shape.** It is the one check that does not need to know what the macro was for: large
  level with near-zero shape is a volume knob whatever the label says. It is also where
  ears are weakest, since louder reads as better. Envelope-length macros are the
  exception and legitimately change level.
- **Artifacts run wasm and AudioWorklets but not Web MIDI.** Measured in the published
  sandbox: WebAssembly compiles, `addModule` from a blob URL works, a `WebAssembly.Module`
  survives `postMessage` into the worklet, `baseLatency` 5.33 ms at 48 kHz. Web MIDI
  throws `SecurityError` from a permissions policy set by the embedding document, which
  no user action fixes. One generated page therefore serves both roles: published for
  sharing with an on-screen keyboard, served locally for a hardware keyboard.
- **A self-contained Faust instrument is small.** DSP wasm 12.4 KiB, effect 22.6 KiB, the
  `@grame/faustwasm` runtime-only bundle 190.6 KiB, whole page 268 KiB against a 16 MB
  cap. The 5.4 MB libfaust compiler never ships because the DSP is precompiled. Do not
  design around a size budget that is not binding.

## Style

No emdash. Comments explain **why**, not what. Prefer refactoring existing helpers over
adding new ones. Avoid `getattr`. Record measured negative results rather than deleting
them; several here are more valuable than the wins.

## What is superseded

`scripts/` holds 58 files, of which ten are load-bearing. **The track B table above lists
all ten**; everything below this line is dead, so a file in `scripts/` named here and not
there is safe to ignore. The rest is stage-1 work that is now frozen, or experiments that
were measured and abandoned. Do not read them for guidance on current approach; they are
kept because a measured negative result is worth more than an untested suggestion, and
each one closes a door someone would otherwise reopen.

None of this list is superseded by track A. The two tracks share the rule at the top of
this file and nothing else: track A builds a patch from a description, track B fits one
to a recording, and no result from either transfers to the other untested.

- **Stage 1, frozen.** `transcribe.py`, `pitch_track.py`, `pitch_probe.py`,
  `build_midi.py`, `refine_midi.py`, `refine2.py`, `polish_midi.py`, `finalize_midi.py`,
  `final_transcription.py`, `segment.py`, `chords_nnls.py`, `octave_check.py`,
  `try_db_major.py`, `velocity_balance.py`, `detune_vs_vibrato.py`, `intro_glide.py`,
  `late_bend.py`, `bend.py`, `analyze.py`. Output is `data/transcription.mid` plus
  `bend2.bend_curve`. These scripts still read and write inside `out/`, deliberately: a
  re-run produces a candidate to diff against the frozen copy rather than overwriting it,
  and both `finalize_midi.py` and `write_final.py` write `out/transcription.mid`.
  `bend.py` is the predecessor of `bend2.py`, which is live and in the table above.
- **Differentiable surrogate, abandoned.** `torch_*.py` (7 files), `verify_env.py`,
  `verify_filter.py`, `verify_fx.py`, `verify_osc.py`, `verify_torch_synth.py`,
  `audit_gradients.py`, `audit_fidelity.py`. A full PyTorch port of the synth, accurate
  to better than Faust-versus-itself. At the optimum the surrogate gradient had cosine
  -0.063 against the true finite-difference gradient and every step size made the real
  loss worse; CMA-ES beat 24 gradient steps in 8 renders. It is also blind to every
  stage added since.
- **Wavetable, zero contribution.** `wt_osc.py`, `fit_chord.py`. The fitted
  harmonic-number deltas came out identically 0 dB once the frequency curve was solved
  first. `synth.dsp_source(amps)` still builds the bank if anyone wants to re-measure.
- **Two-layer synth, rejected.** `layers.py`, `fit_layers.py`, `verify_layers.py`. Real
  stereo gains (bass width -7.68 to -22.01 dB against a target of -23.63) bought with
  reverb, which costs the mono term. Failed 4 of 10 acceptance checks. A band-wise
  mid/side EQ does the same job better at no mono cost, and is not built yet.
- **Probes and one-offs.** `drive_probe.py`, `spread_scan.py`, `defect_check.py`,
  `quality_test.py`, `compare.py`, `make_report.py`, `write_final.py`,
  `promote_patch.py`, `verify_new_stages.py`, `verify_timbre_stages.py`.
