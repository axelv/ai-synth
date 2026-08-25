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
| `scripts/faust_render.py` | offline renderer and the four patterns. `Instrument` backs measure.py |
| `scripts/measure.py` | the verification pass. Macro sweeps, voice coherence, register. `--check` runs the whole example set |
| `scripts/build_page.py` | one `.dsp` to one self-contained playable page |
| `references/faust-poly.md` | poly conventions and the failures that are silent |
| `references/patch-design.md` | what makes a macro a macro; ranges and defaults |
| `references/examples/*.dsp` | six working instruments. `juno-106` is the clean one |
| `references/examples/measured.md` | what each gets right and wrong, and why |
| `references/examples/expected.json` | what each is expected to measure. The regression set, enforced by `--check` |

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
| `scripts/selfgen.py` | Tier A: self-rendered targets with recorded parameters |
| `scripts/selfrecover.py` | self-recovery bench; also the sensitivity and truth-start controls |
| `scripts/basin.py` | loss against distance from truth, and the attraction-radius probe |
| `scripts/losses.py` | candidate objectives behind one factory interface, plus baselines |
| `scripts/losscorpus.py` | the cached render corpus the bake-off screens against |
| `scripts/nuisance_probe.py` | the unfittable-phase sample, the sharpest screen there is |
| `scripts/bakeoff.py` | the four screens: discrim, pedestal, nuisance, near_wins |
| `scripts/earpanel.py` | retrieval calibration: can a fingerprint see a near neighbour |
| `scripts/widecrowd.py` | crowd diversity and the empirical rank test against a real-size bank |

Current state lives in `out/patch.json` and `out/render.wav`.

## Traps that have already cost real time

- **`PluginProcessor.load_state` accepts a `.vital` file, reports success, and does
  nothing.** It wants dawdreamer's own state blob, which starts `VC2!` and wraps base64 in
  VST3 XML, not Vital's preset JSON. There is no error. The renders come back at rms 0.1177
  and peak 0.3572 whatever preset you "load", and a kick drum is bit-comparable to a vowel
  pad. Always verify a preset load by rendering two presets that should sound nothing alike
  and comparing, never by the absence of an exception.

- **Vital's preset keys do not map to its plugin parameters by name or by order.** The JSON
  holds 771 scalar settings named `chorus_cutoff`; the plugin exposes 2983 parameters named
  `Chorus Filter Cutoff`. Exact name matches: 0 of 771. The orders nearly align, which is the
  trap: `chorus_delay_1` through `chorus_on` line up at a constant offset of one, then
  `chorus_spread` meets `Chorus Sync` and the offset changes with nothing to signal it. Take
  the mapping from Vital's open source, and gate it on the two-preset render check above.
  Arturia's Analog Lab V will not load headlessly at all, so its several thousand presets
  across ~30 architectures are out of reach without solving activation first.

- **`data/original.wav` is 48 kHz.** Always `librosa.load(path, sr=44100)`. Reading it
  with `soundfile.read` and no resample silently compares audio 8.8% wrong in time and
  pitch. It has produced at least one bogus results table.
- **Write renders with `synth.write_render`** (PCM_24). soundfile's default PCM_16 costs
  0.089 of loss here, because the log-magnitude term sees the 16-bit floor in the
  6-16 kHz band. Score from the file on disk, not the array in memory.
- **`synth.PAD.params` is append-only.** `out/patch.json` depends on the order.
- **A DSP and its parameter names travel together, as a `synth.Architecture`.**
  `PadRenderer(arch)` and `Objective(notes, arch=...)` take one. Passing a foreign
  Faust source alone used to compile fine and then set nothing, because the names
  came from a module global. `synth.PAD` is the fitted pad; `arch.with_dsp(src)`
  is a variant build over the same vocabulary.
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
  minutes. **But do not read a stuck loss as a reachability statement any more**: at a
  loss of 1.3 the fit is chance-distance from the answer even when the answer is inside
  the search space, see the self-recovery result below. The oracle comparison still
  works; the inference from "CMA-ES stopped moving" does not.

- **CMA-ES cannot recover a patch this synth rendered itself.** 8 Tier A targets, same
  renderer and frozen notes and bend curve, so reachable by construction. 0 of 8
  recovered: mean final loss 1.3096 against a mean floor of 0.0006, mean normalised
  parameter rms 0.328. READ THE BASELINE CAREFULLY, the obvious one is wrong and was
  believed for weeks: 1/3 is E|u-v|, the mean ABSOLUTE error between two uniforms, while
  the reported figure is an RMS, whose uniform value is sqrt(1/6) = 0.408 and whose
  measured value against these truth vectors is 0.376. So the fit beats a random guess by
  13%. It LOSES by 36% to setting every knob to the middle of its range, which scores
  0.242. That constant is the null that matters and no run has ever beaten it. The obvious escapes are all
  closed by measurement. Not an early stop: t00 spent its full 140-generation budget and
  the last-20-generation slope across targets runs +0.0017 to -0.0018 per generation, so
  t00 would need ~770 more generations to reach its floor. Not unidentifiable
  parameters: 40 to 55 of the 55 move the loss by >= 0.01 at +-0.05 normalised, and
  restricting the error to those changes it from 0.283-0.357 to 0.293-0.356. Not a bad
  minimum: started AT truth with sigma 0.05, all 8 sat still for 992 renders each and
  found nothing below the floor, so truth is a strict local minimum the search simply
  never reaches. The bottleneck is the search, not the reachable set.

- **The log-magnitude STFT term makes the objective a spike function of the parameters.**
  This is the mechanism behind the recovery failure, measured directly. Perturb one
  parameter from truth by 1e-9 normalised and the render is bit-identical, because
  float32 parameter quantisation absorbs it. Perturb by 1e-8, the next representable
  step, and the spectrum moves by a relative 5.4e-5, about -85 dB, while the loss jumps
  +0.194. That is more than the 0.162 the whole EQ stage bought, for a change nothing
  can hear. Isolating the terms on that same perturbation: spectral convergence
  +0.000047, linear magnitude +0.000162, log-magnitude +0.194056. So it is entirely the
  log term, punishing the near-empty bins between partials. `fi.peak_eq` costing 0.69
  while leaving cos theta untouched was the first sighting of this; it is not a quirk of
  that primitive, it is the objective everywhere.

- **The objective's broad minimum is not at the true patch.** Perturb truth by rms 0.001,
  0.01 or 0.05 and run a local polish: 0 of 8 return, at every radius. Worse, the
  endpoint does not depend on the start. Mean final rms is 0.015, 0.018, 0.018 and mean
  final loss 0.944, 0.921, 0.919 for starts fifty times apart. The polish forgets where
  it began and converges to an attractor about 0.017 away from truth. Truth is an
  isolated needle beside that attractor, one float32 grid point wide, and the landscape
  contains no route to it. `basin.py --profile` maps this for a few hundred renders and
  is the cheapest way to re-check it after any change to the loss.

- **CMA-ES makes the 26 EQ bands WORSE than not touching them.** Measured per group on
  the 8-target bench against `seeded_start`, as start-to-truth versus fit-to-truth rms:
  core 0.423 -> 0.379 (+10.5%), fx 0.405 -> 0.387 (+4.3%), eq 0.169 -> 0.247 (-46.0%).
  The EQ starts close because the seed sits at 0 dB and `selfgen` draws truth as a smooth
  4-mode curve concentrated near it, and the free search then walks off that manifold. So
  the standing advice to fit the EQ with `fit_eq_full` and keep it out of CORE is now
  measured rather than argued, and the `full` phase freeing all 55 is actively harmful to
  26 of them.

- **Dimensionality is not the binding constraint.** Same measurement: the 8-parameter FX
  group improves LESS than the 21-parameter CORE group (4.3% against 10.5%). Beware the
  raw per-group rms, which says EQ is recovered best (0.247 against 0.379 and 0.387): that
  is entirely a starting-point artefact, and the per-group chance baselines differ too
  (core 0.400, fx 0.420, eq 0.333) because the EQ truth is manifold-drawn and concentrated.
  Always compare against start-to-truth, not against the fit's rms alone.

- **Invariance to the phase nuisance is cheap and mostly buys blindness.** Of the 8
  objectives that charge under 0.15 for it, 6 come out blinder than the incumbent: they
  cannot separate a frame-shuffled target from the target. Across all 32, spearman between
  nuisance sensitivity and discrimination is +0.50, so the parts that see the phase are
  largely the parts that see real differences. Two escape the trade-off (graduated_blur at
  0.051 nuisance and 1.06 discrim, nuisance_whitened at 0.146 and 0.98) and BOTH still fail
  Screen C, so invariance is not the binding constraint either. Note the incumbent is
  already magnitude-only and therefore already blind to pure time translation; the nuisance
  is a change in the beating pattern, not a shift, so translation invariance does not touch it.

- **43% of the incumbent objective's dynamic range is unfittable phase.** Push every
  note 0.37 s later, render, trim the lead-in back off: same 55 parameters, same notes,
  same relative timing, but the free-running oscillators, LFO and chorus are at different
  phases, and the spectrum moves by a relative 0.16 to 0.35. Nothing in PARAMS reaches
  it. The incumbent charges 0.432 of the gap between truth and a patch rms 0.10 away for
  that difference, against -0.0055 for the entire attractor-versus-near-truth margin. So
  the fit is chasing phase roughly 80 times harder than it is chasing the answer.
  `nuisance_probe.py` builds the sample; changing `n_voices` does NOT work, because
  allocation is identical below the voice limit and the render comes out bit for bit the
  same.

- **Deleting the log-magnitude term makes things worse, not better.** The obvious
  inference from the term isolation is to drop the term that causes 100% of the pedestal.
  Measured, `sc_only` is worse than the incumbent on both counts that matter: near@2e-3
  -0.0367 against -0.0055, and its pedestal is still 0.0372. Whatever the log term
  contributes, spectral convergence alone cannot carry the objective.

- **The loss is not the lever either. 32 objectives screened, 0 recover.** 6 baselines
  including the incumbent control, plus 26 drawn from six independent lenses plus a critic.
  A bake-off screened them on a cached render
  corpus, and put the six survivors through real CMA-ES on held-out targets at a budget
  matched to a control. Mean normalised parameter rms: graduated_blur 0.300, nmr_mask
  0.302, nuisance_whitened 0.309, band26_env 0.313, jtfs_lite 0.328, slope_conditioned
  0.332, incumbent 0.344, against 0.376 for a random guess, 0.242 for setting every knob mid-range, and a goal
  of 0.05. Nothing beat the constant. The
  spread between the best candidate and the incumbent is smaller than the spread the
  incumbent shows between budgets, so loss choice is not what is binding.

- **Random-direction screens cannot detect the degeneracy that matters.** This is the
  methodological result and it invalidates the cheap half of the bake-off. graduated_blur
  won every cheap screen (pedestal 0.0002, nuisance 0.051, spearman 0.943) and then
  scored its own Screen C answer, a patch rms 0.285 from truth, at 0.01199: LOWER than a
  random point at rms 0.02 (0.01266) and a third of a random point at rms 0.05 (0.03523).
  The degenerate directions are a vanishing fraction of a 55-dimensional sphere, so random
  sampling almost never finds them and an optimiser always does. A loss can therefore rank
  random perturbations almost perfectly and still be hollow along the exact directions the
  search will walk. To screen a loss you have to optimise against it; nothing cheaper
  substitutes.

- **Score a fit by metrics it did not see, never by its own loss.** graduated_blur called
  its own answer 0.012 against a floor of 0.000, near perfect, on a render whose mel_dist
  is 2.70 and env_l1 0.319 against roughly 0.104 for the incumbent's fit on the real
  target. A loss reporting success about its own output is worth nothing; only
  `metrics.report` and per-window numbers settle it.

- **The search's answer is not a metamer; the attractor partly is.** The obvious rescue for
  the recovery failure is that patches audio-identical to truth form a manifold, so distance
  to the one recorded representative is an arbitrary target. Measured on held-out audio
  metrics, mean over 4 targets, mel_dist / env_l1: truth 0.000/0.000, the SAME patch with
  only oscillator phase changed 1.529/0.142 (this is the unfittable floor), a random patch
  at rms 0.02 3.114/0.293, the attractor at rms 0.027 2.288/0.158, and the full fit's answer
  at rms 0.31 5.811/0.306. Two conclusions. The full fit is genuinely wrong audio, not an
  equivalent patch: its mel_dist of 5.811 is WORSE than the 4.167 of a deliberately wrong
  control, the target delayed one second, so the metamer rescue fails and the search really
  is lost. Note the comparison is mel-specific and does NOT hold on env_l1, where the fit's
  0.306 beats the control's 0.449; a delayed copy has the right spectrum and the wrong
  envelope, so it is a hard control on mel and a soft one on env.
  But the attractor IS closer in audio than a random patch at a third its distance (2.288
  at rms 0.027 against 3.114 at rms 0.02) and sits at 1.5x the phase floor, so a local
  polish does find a partly audio-equivalent patch. Parameter rms is the right yardstick
  for the global failure and an unfair one for the local attractor.

- **The unfittable floor is large in audio terms too, not just in loss.** Two renders of the
  SAME patch differing only in oscillator phase measure env_l1 0.142. The incumbent's fit on
  `data/original.wav` measures env_l1 0.104. Both numbers are scale-invariant by
  construction, so on that metric the real fit is already at or below the level at which one
  patch differs from itself. Treat any env_l1 improvement near 0.1 as unmeasurable.

- **A loss near 1.3 is not evidence of anything.** The mean Tier A final loss, 1.3096,
  is BELOW the incumbent's 1.3823 on `data/original.wav`, on targets the fit is
  chance-distance from. Any argument of the form "the loss came down, so the patch got
  closer" is unsupported in this range.
- **Retrieval fails its own calibration test before any bank is built.** SUPERSEDED in part
  by the rank measurement above: the conclusion below is right about rank-1 lookup and wrong
  about the method being dead, because a shortlist is what the pipeline actually needs. The plan was to
  replace the search with a lookup: fingerprint the target, fingerprint a bank of presets,
  rank by descriptor distance. That needs a genuinely close patch to land further below the
  crowd of unrelated patches than chance alone puts the crowd's own nearest member. Measured
  on the cached corpus at zero render cost, 7
  fingerprints times 2 level treatments, 0 of 14 pass. Best mean margin for a genuine
  rms-0.05 neighbour is 2.95 sd (mel_std) and the best worst-target margin is 1.86 sd
  (all four blocks concatenated), against a keep bar of 3 sd which is itself only worth a
  bank of about 460. The supported bank size is **4 to 20 patches**. Not a crowd-assembly
  artefact: the three crowd subgroups have
  means agreeing within 5%, and dropping the between-group variance entirely, the most
  generous reading available, lifts margins by 0.2 to 0.6 sd and changes no verdict. The
  test is also the best case by construction, since it plants a real perturbation of the
  actual truth; a real preset bank carries no guarantee that any such neighbour is in it.

- **Retrieval works as a shortlist and fails as a lookup. Measured on a bank of 1993.**
  This supersedes the day-1 verdict below, which was built on a Gaussian margin rather than
  a count. Rendering a real-size bank from the whole parameter box and ranking a planted
  rms-0.05 relative against it: rank 1 on **0 of 4** targets, but ranks of 22, 2, 3 and 6,
  so **top-25 on 4 of 4** and top-10 on 3 of 4. The relative's quantile in the crowd is
  fixed, so expected rank grows linearly with bank size: at 10k, top-100 holds on 3 of 4;
  by 200k nothing survives. So the usable regime is a shortlist of tens out of a bank up to
  roughly 10k, handed to something else to refine. That is worth having, because a local
  polish starting inside rms 0.05 reaches the attractor at mel_dist 2.288 against the global
  search's 5.811. What remains untested is the only thing that now matters: a planted
  relative is guaranteed to exist, and a real preset library carries no such guarantee.

- **More crowd diversity does not raise the margin. Fingerprint distance saturates.**
  The natural defence of retrieval is that a real preset library spans architectures and is
  far more spread out than a shell of perturbations, and the arithmetic supports it: scaling
  crowd distances by k gives margin (k*mu - near)/(k*sd), climbing to a ceiling of 1/CoV,
  measured at 3.8 to 4.9 against the 3.42 a bank of 2000 demands, needing only k = 1.3 to
  1.6. Measured, **k = 0.99**. Drawing from the entire parameter box instead of a shell at
  rms 0.33 moved the mean crowd distance by one percent, while CoV rose 0.230 -> 0.276, so
  the margin FELL from 2.67 to 2.21. Once two patches are unrelated, making them more
  unrelated adds no distance. Retrieval is decided by the near tail of the crowd, and
  diverse far-away presets pile mass where it cannot matter. Do not argue from crowd
  breadth again; it is measured and it is flat.

- **Score retrieval by rank, never by a Gaussian margin.** Both model-based estimates were
  wrong in opposite directions on the same data. The margin said the worst target should
  fail between N=17 and N=165; it ranked 1st at N=61. The k-model said diversity would
  rescue it; k came out 0.99. The rank is one line of code and assumes nothing, and the
  quantile it yields extrapolates to any bank size for free.

- **Do not use sqrt(2 ln N) for the best-of-N correction at these sizes.** It is the
  asymptotic expected extreme of N standard normals and it is the formula everyone reaches
  for, but it is badly wrong small: at N = 11 it claims 2.19 sd where the true expectation
  is 1.59, and at N = 100 it claims 3.03 against 2.50. Used to invert a margin into a bank
  size it understated the answer by about a factor of three, which is how the first write-up
  of the retrieval result reported a supported bank of 2 to 6 rather than 4 to 20. Blom's
  approximation, `norm.ppf((n - 0.375) / (n + 0.25))`, is accurate across the whole range
  and is what `earpanel.needed_margin` now uses. The verdict never depended on it, which is
  exactly why it survived a first pass unchecked.

- **Phase invariance turned out to be the easy half.** The same panel charges the 0.37 s
  phase twin 0.033 to 0.055 of what it charges a genuine rms-0.02 error, for the
  time-averaged blocks, against the incumbent objective spending 43% of its dynamic range
  there. So the nuisance that dominates the loss is straightforwardly designed away by
  averaging over time. It buys nothing, because discrimination fails independently, and the
  bake-off's +0.50 spearman between nuisance sensitivity and discrimination reappears here
  in a completely different framework: mel_std has the best margin (2.95 sd) and the worst
  nuisance (0.214), band26 and mel_mean the best nuisance (0.035, 0.033) and margins of
  2.11 and 1.65.

- **Time-averaged fingerprints rank a frame-shuffled target at or below an unrelated
  patch.** band26 -0.55 sd, mel_mean -0.91 sd, mel_mean+mel_std -0.97 sd from the crowd
  mean, on audio calibrated at cos theta 0.68 and unmistakably not the same patch. Only the
  loudness envelope places it clearly on the far side (+1.35 sd), and env has the worst
  nuisance figure of the four (0.349). Any retrieval built on a mean spectrum would happily
  return nonsense; this is the same blindness the bake-off found, reached by another route.

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
