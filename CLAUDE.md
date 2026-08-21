# ai-synth

## What this is

Reverse-engineering a **playable synth patch** from a recording. Input is 18 seconds of
a synth pad (`data/original.wav`); output is a note list plus a parameter vector that,
rendered through a synth, reproduces it.

The deliverable is the patch, not the audio. Anything that matches the waveform but does
not correspond to controls a person could dial in is off-topic. That rules out learned
vocoders, direct spectrogram optimisation, and sample-level tricks, and it is why Faust
through dawdreamer stays the authoritative renderer: whatever is fitted has to be
something a synth can actually do.

Two stages, in dependency order:

1. **Stage 1, transcription.** Notes and a measured pitch-bend curve. **Frozen.** Treat
   `data/transcription.mid` and `bend2.bend_curve` as inputs, not parameters. Re-opening
   this is a separate project and it confounds any judgement of stage 2.
2. **Stage 2, patch fitting.** Synth parameters against a multi-resolution STFT loss.
   This is where the work happens.

## Running things

Always from the repo root, always through uv:

```
PYTHONPATH=scripts uv run python scripts/<script>.py
```

Use the uv project workflow (`uv add`). Never `uv pip install` into a bare venv.

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
  numbers stay comparable.

## Style

No emdash. Comments explain **why**, not what. Prefer refactoring existing helpers over
adding new ones. Avoid `getattr`. Record measured negative results rather than deleting
them; several here are more valuable than the wins.

## What is superseded

58 scripts, of which ten are load-bearing. **The table above lists all ten**; everything
below this line is dead, so a file named here and not there is safe to ignore. The rest is
stage-1 work that is now frozen, or experiments that were measured and abandoned. Do not
read them for guidance on current approach; they are kept because a measured negative
result is worth more than an untested suggestion, and each one closes a door someone
would otherwise reopen.

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
