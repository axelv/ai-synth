# How long does a fit actually take?

Research findings for [01-how-long-does-a-fit-take](../issues/01-how-long-does-a-fit-take.md).

**Nothing in the pipeline was changed.** Every number below came from scripts written
alongside this document, which import the existing modules and write only into this
directory:

| script | produces |
|---|---|
| [`bench_core.py`](bench_core.py) | `bench_core.json` — imports, Faust compile, render, loss, landmark losses |
| [`bench_cma.py`](bench_cma.py) | `bench_cma.json` — a wall-clock-bounded prefix of a real CMA-ES fit, every evaluation traced |
| [`bench_eq.py`](bench_eq.py) | `bench_eq.json` — the 26-band EQ fit, piece by piece |
| [`bench_staircase.py`](bench_staircase.py) | `bench_staircase.json` — does the cheap EQ work on a half-fitted patch |
| [`bench_par.py`](bench_par.py) | `bench_par.json` — render throughput against worker count |

All run as `PYTHONPATH=scripts uv run python .scratch/app-design-flow/research/<script>.py`
from the worktree root.

**Machine.** Apple M1, 8 cores, 16 GB, macOS 25.2 (Darwin), Python 3.11, torch 2.13.0,
`torch.get_num_threads() == 4`, CUDA unavailable, MPS available. A server would be
faster; the *ratios* are what should travel, not the seconds.

**Measured vs estimated.** Everything in a "measured" row was run. A full stage-2 fit is
~7 400 renders and was deliberately **not** run; where a full-fit number appears it is
marked *estimated* and says what measured prefix it was extrapolated from.

---

## 1. Wall clock for each stage that exists today

### One-off costs, paid once per fit

| thing | cold | warm | note |
|---|---|---|---|
| `import synth, stage2` | 45.0 s | 2.8–3.2 s | measured. Cold is the first run after `uv sync` (bytecode compile). A server process pays it once, ever. |
| `PadRenderer()` — Faust compile + engine | 1.04 s | 0.02–0.06 s | measured, 3 constructions in one process |
| `librosa.load(sr=44100)` of the 18 s clip | 2.61 s | — | measured. Includes the 48 → 44.1 kHz resample. |
| `stage2.Objective()` — target load ×2 (mono + stereo) + MRSTFT setup + its own renderer | 14.9 s | 1.9–2.8 s | measured. Cold is resampy building its filter cache. |
| `stage2.load_notes()` | 5.6 ms | | measured. 29 notes, pitch 25–72, 0.02–17.90 s. |
| `bend2.bend_curve()` | 9–14 ms | | measured |
| `metrics.report()` (the acceptance metrics) | 2.53 s | 0.41–0.43 s | measured |

**Warm start-up to first evaluation is about 5 seconds.** That is the entire fixed cost
before the optimiser can begin.

### The inner loop

| thing | measured | share of an evaluation |
|---|---|---|
| `PadRenderer.render(17.904)` — 29 notes, 24 voices, full DSP | 1.041–1.083 s (median 1.06) | 75 % |
| `Objective.loss_of` — mono MRSTFT (4 resolutions) + envelope | 0.166–0.194 s | 13 % |
| `Objective.loss_parts` — the same plus the stereo side term | 0.353–0.395 s | — |
| `Objective.__call__` end to end | 1.41–1.75 s | 100 % |
| **sustained rate over 353 consecutive evaluations** | **1.402 s / evaluation** | |
| `render(2.0)` of a single held note | 0.055–0.068 s | |

The full-clip render runs at about **17× realtime**; a single 2 s note at about **30×
realtime**. That matters for ticket 02: this DSP is nowhere near the browser's realtime
budget.

### A CMA-ES generation

`stage2.run_cma` uses `popsize: 16`, so a generation is 16 evaluations.

**Measured: 21–24 s per generation** (22 generations, 353 renders, 495 s → 22.5 s/gen).

### The 26-band EQ fit

`fit_eq_full.py`. The commutation trick means one poly render serves the whole fit.

| step | measured |
|---|---|
| `FullScore()` construct | 2.39 s |
| `flat_render()` — one 17.9 s render with EQ at 0 dB | 1.14 s |
| `score()` — loss + optimal level + cos θ | 0.20–0.30 s |
| `eq_window()` — 18 s through the cascade | 0.079–0.087 s (0.244 s first call) |
| `oracle_start()` — closed-form band gains, **no search** | 0.06–0.27 s |
| one Powell evaluation | **0.335 s** (300 evals in 100.5 s) |

`out/eq_full_lam*.json` records **1 200 evaluations per lambda**, and the default run
does **six** lambdas.

- *Estimated*, from the measured 0.335 s/eval: **one lambda ≈ 6.7 min**, the default
  six-lambda sweep ≈ **40 min**. The six lambdas are independent of each other.

### A full stage-2 fit as currently run — ESTIMATED, NOT RUN

`stage2.main()` defaults are `--restarts 2`, `--core-gens 90`, `--full-gens 140`,
popsize 16 → **7 360 evaluations** if nothing plateaus.

- At the measured 1.402 s/evaluation: **10 320 s ≈ 2 h 52 min**, single process.
- The plateau rule (stop when the best has improved < 1 % over 20 generations) can only
  fire on 20-generation boundaries, so the floor for a 2-restart run is
  2 × (20 + 20) × 16 = 1 280 evaluations ≈ **30 min**.
- **Realistic band: 30 min to 2 h 52 min.** Add ~40 min for the EQ sweep and ~3 min for
  stage 1's machine parts.

For calibration, the repo's own historical logs (`out/log_run27.txt`,
`out/log_runext.txt`, from a different machine) record 697 renders in 680 s, i.e.
**0.98 s/render** and ~12 s per 16-render generation — that machine was about 1.4×
faster than this one. Same shape.

### Stage 1, transcription (frozen, but timed)

| step | measured |
|---|---|
| `librosa.load` | 2.61 s |
| `transcribe.fit_region` on one of six regions | **27.2 s** (295 cached 2 s candidate renders) |
| all six regions | ≈ 163 s *estimated* — an upper bound, `main()` shares the render cache across regions |
| `pitch_track`: STFT + harmonic-sum salience + Viterbi | 0.21 + 0.87 + 3.23 = **4.3 s** |

**So the machine part of stage 1 is ~3 minutes.** But it is not the pipeline that
produced `out/transcription.mid`, and the design should not assume it is:

> Run today on region 7.45–10.40 s, the greedy chord search returns MIDI
> `[25, 26, 27, 37]` — C♯1, D1, D♯1, C♯2, three adjacent semitones stacked in the bass.
> The delivered transcription for that region is D♭ minor, bass 25 with offsets
> `[12, 19, 24, 27, 31]`. `final_transcription.py`'s own docstring says why: *"the
> author wrote the part in GarageBand and cannot export MIDI, so the chord qualities
> come from them."*

Stage 1 as delivered has a human in the loop. Timing its automatic half is honest, but
**there is no measured per-request transcription time, because there is no per-request
transcription.** That is a real gap for the app and it is not this ticket's to close.

---

## 2. What each stage leaves behind, and whether you can play a note with it

| stage | artefact | playable? |
|---|---|---|
| stage 1, chord search | `out/chords.json` — region → MIDI pitches | no. Notes, no sound. |
| stage 1, pitch track | `out/f0_track.npy`, `out/intro_f0.npy` → `bend2.bend_curve` | no |
| stage 1, delivered | `out/transcription.mid` — 29 notes | no |
| stage 2, seed | `stage2.seeded_start()` — a 55-number vector, **in memory only, never written to disk** | **yes** |
| stage 2, CMA-ES | `out/patch.json` (`normalized` 55-vector + `params` dict + loss + metrics), `out/render.wav`, `out/loss_history.npy` | **yes** |
| EQ fit | `out/eq_full_lam*.json` (26 gains + loss + `env_l1` + curvature), then `out/patch_eq_full.json`; `promote_eq.py` merges into `out/patch.json` | **yes**, as a delta on an existing patch |

**The structural finding: the only artefact that ever reaches a player is a 55-number
parameter vector, and a valid one exists before any fitting happens at all.**
`synth.denorm(x)` turns any point in the search box into a complete slider dict, the
Faust source is a fixed string, and both appended stages have exact identity defaults.
There is no moment in this pipeline where a patch does not exist. The staircase is
**not** blocked by artefact availability.

Note also that `out/transcription.mid` never reaches the player. The user plays their own
keys; the note list exists only so the fit has something to render against the target.

### How bad is it, early?

Measured mono `loss_of` against `data/original.wav`, all in one process:

| patch | loss | reading |
|---|---|---|
| target against itself | 0.000 | floor |
| **delivered `out/patch.json`** | **1.382** | the answer |
| delivered patch, 26 EQ bands zeroed (= converged CMA, pre-EQ) | 1.558 | |
| CMA prefix, 8 min / 353 renders | 1.814 | measured today |
| `norm_defaults()` — the middle of the box | 3.091 | |
| **`seeded_start()` — the hand-seeded start** | **3.605** | |
| white noise at the target's RMS | 7.489 | ceiling control |
| the target attenuated to silence | 4.865 | **worse than nothing is 4.87** |

The unfitted seed sits at 3.61 against 4.87 for silence and 1.38 for the answer. It is
closer to silence than to the answer.

And the loss is a weak ranker, so it needs its controls. Spectral cosine θ, measured in
`fit_eq_full.FullScore`'s own domain (2048-point magnitude STFT, optimal level solved in
closed form):

| | cos θ |
|---|---|
| target against itself | 1.000 |
| target delayed a full second | 0.806 |
| **delivered patch** | **0.794** |
| converged CMA, pre-EQ | 0.720 |
| **target with its own frames shuffled** | **0.698** ← the deliberately-wrong control |
| CMA prefix + closed-form EQ | 0.625 |
| `seeded_start()` | 0.543 |
| CMA prefix, 8 min / 353 renders | 0.514 |
| white noise | 0.187 |

**Every state short of a converged fit scores below the frame-shuffled control.** The
delivered patch itself only just beats the target-delayed-one-second control. So the
usable range of this metric is 0.70 to 0.81, and the whole staircase before convergence
lives underneath it.

---

## 3. Loss against wall clock through a fit

**There is no recorded curve for the delivered fit.** `out/loss_history.npy` is 80
generations with a minimum of 1.5564 — it never reaches the delivered 1.3823 or even the
pre-EQ 1.5446, so it is a leftover from some variant run, not a trace of the answer. That
is why one was measured.

Measured: `bench_cma.py 480` — CMA-ES from `stage2.seeded_start()` over `stage2.CORE`
(21 parameters, σ 0.22, popsize 16, seed 100, exactly what `stage2.main()` does on its
first restart), run for a fixed 480 s budget. Every evaluation is traced in
`bench_cma.json`.

Journey defined as seed 3.605 → delivered 1.382, total Δ = 2.223.

| elapsed | generation | renders | best loss | Δ recovered |
|---|---|---|---|---|
| 0 s | seed | 1 | 3.605 | 0 % |
| **24 s** | 1 | 17 | **2.148** | **65.5 %** |
| 45 s | 2 | 33 | 1.958 | 74.1 % |
| 91 s | 4 | 65 | 1.854 | 78.8 % |
| 203 s | 9 | 145 | 1.835 | 79.6 % |
| 364 s | 16 | 257 | 1.833 | 79.7 % |
| 452 s | 20 | 321 | 1.814 | 80.6 % |
| 495 s | 22 | 353 | 1.814 | 80.6 % |
| ~30 min – 2 h 52 min *(estimated)* | converged | ~1 300–7 400 | 1.558 | 92.1 % |
| + EQ fit | | | 1.382 | 100 % |

**The loss curve is exactly the shape the staircase bet wants.** Two thirds of the total
improvement lands in the first generation, 24 seconds in. Eighty per cent lands in the
first ninety seconds. The remaining 20 % costs somewhere between half an hour and three
hours.

**And the cos θ curve says the loss curve is lying about what improved.** Same
measurement, `bench_staircase.py`:

| | loss | cos θ |
|---|---|---|
| seed | 3.674 | 0.543 |
| after 8 minutes of CMA-ES | 1.816 | **0.514** |
| converged (delivered, EQ flat) | 1.546 | 0.720 |

Eight minutes of CMA-ES cut the loss by half and moved spectral agreement **slightly
backwards**. Level, broadband balance and the envelope term are what the early generations
buy. The thing a listener would call "the same instrument" arrives with convergence, not
before it. This is the repo's own warning — *"a lower loss is not sufficient evidence"* —
showing up exactly where the product bet needs it not to.

### The one cheap step that is genuinely worth taking early

The 26-band EQ is the largest single win the project found (0.162 of loss), and its
closed-form warm start is essentially free. Measured, applied to three patches at
different stages of convergence:

| starting patch | loss flat | + `oracle_start` (0.06 s) | + 150 Powell evals (~50 s) | cos θ flat → oracle |
|---|---|---|---|---|
| seed, no fitting at all | 3.674 | 2.891 | 2.763 | 0.543 → **0.487** |
| CMA prefix, 8 min | 1.816 | 1.688 | 1.642 | 0.514 → 0.625 |
| delivered, EQ flat | 1.546 | 1.426 | 1.443 | 0.720 → 0.806 |

Read this carefully, because two of the three rows are traps.

- **The closed-form EQ costs one render plus 0.06 s and delivers ~73 % of the full EQ
  win** (1.546 → 1.426 against a full-fit 1.382). That is real and it is the cheapest
  thing in the pipeline.
- **On the raw seed it lowers the loss by 0.78 while lowering cos θ from 0.543 to
  0.487.** It buys loss by bending the spectrum toward a level and tilt match on top of a
  patch whose shape is wrong. Running the EQ before the core is converged makes the number
  better and the sound worse.
- **Truncating the Powell refinement is not monotone.** On the delivered patch, 150 evals
  gives 1.443 — worse on the pure loss than its own 0.06 s warm start at 1.426 — while 300
  evals gives 1.418 and 1 200 gives 1.382. Powell minimises `loss + λ·curvature`, and the
  first sweeps spend themselves flattening the curve (the oracle start has curvature
  4 066). A staircase that publishes whatever the EQ fit currently holds will show the
  user a step *backwards*.

---

## 4. What is parallelisable, and what is GPU-bound

**Measured** (`bench_par.py`, N worker processes, each with its own Faust engine, 6
full-clip renders each):

| workers | renders/s | speed-up | s per render inside a worker |
|---|---|---|---|
| 1 | 0.51 | ×1.00 | 1.10 |
| 2 | 0.97 | ×1.92 | 1.30 |
| 4 | 1.54 | ×3.05 | 1.60 |
| 6 | 1.81 | ×3.58 | 1.80 |
| 8 | 2.05 | ×4.06 | 2.23 |

Near-linear to 2 workers, ×4.06 at 8 on an M1's 4 performance + 4 efficiency cores. A
generation would drop from 22.5 s to about 5.5 s, and the estimated full fit from
2 h 52 min to roughly **42 min**.

What is parallel and what is not:

- **A CMA-ES generation: yes, perfectly.** 16 independent renders, no coupling until
  `es.tell`. This is the whole lever.
- **The two restarts: yes.** Independent by construction.
- **The six EQ lambdas: yes.** Independent fits over the same flat render.
- **Powell inside one EQ fit: no.** A coordinate line search is strictly sequential. But
  it barely matters: `oracle_start` solves the same 26 numbers in closed form in 0.06 s
  and gets three quarters of the way there.
- **Generations against each other: no.** CMA-ES is sequential by nature; the only way to
  spend more cores is more restarts or a bigger population.

**Nothing here is GPU-bound.** The render is Faust through dawdreamer on CPU, single
threaded per engine, and it is 75 % of an evaluation; there is no GPU path for it at all.
The loss is the only torch code. Measured, the four-resolution MRSTFT on this clip:

- CPU: 0.178–0.190 s
- MPS: 0.094–0.100 s (identical value to 7 significant figures)

**1.9× on 13 % of the work — about 6 % off an evaluation.** A GPU is not the answer to
this pipeline; CPU cores are. The one place a GPU would matter is a batched objective
that scored many candidates at once, and that is only worth building if the renders can be
batched too, which Faust-through-dawdreamer does not do.

---

## What this means for the staircase

**The bet is half supported, and the half that fails is the half that matters.**

**Supported: a patch always exists, and the loss falls early.** There is no point in this
pipeline where there is nothing to play. A complete 55-parameter vector exists before the
first render, `denorm` turns it into every slider the Faust DSP declares, and warm start-up
to the first evaluation is about 5 seconds. Two thirds of the total loss improvement lands
24 seconds in and 80 % lands within 90 seconds. If the loss were the product, the
staircase would be nearly free and the tail — 30 min to 2 h 52 min, or ~42 min across 8
cores — could be spent improving a patch the user is already playing.

**Not supported: nothing before convergence is recognisably the same instrument.** The
repo's own bar is *"nobody should call it a different sound."* Against a
frame-shuffled control at cos θ 0.698, the seed measures 0.543, the eight-minute patch
0.514, and the eight-minute patch with the cheap EQ on top 0.625. The delivered patch
reaches 0.794 and the practical ceiling of this metric is around 0.81. **Every rung below
the top sits underneath a control built by destroying the audio on purpose.** Worse, the
early rungs are not even monotone in what a listener would hear: eight minutes of CMA-ES
halved the loss while moving cos θ *backwards*, because what the early generations buy is
level and envelope, not timbre.

**The earliest honest moment of playability**, on today's pipeline:

- **~5 seconds — the seed.** Honest as *"here is a pad while we work"*, and nothing more.
  It measures 3.61 against 4.87 for silence and 1.38 for the answer. It is a placeholder
  with a keyboard attached. Presenting it as "your sound, roughly" would be a lie the
  first note exposes.
- **~90 seconds — 80 % of the loss recovered.** Still cos θ 0.51, still below the
  shuffled control. Better as a *progress display* than as an instrument.
- **Convergence — 30 min to ~3 h single-threaded, ~40 min across 8 cores — is the first
  rung that clears the bar.** cos θ 0.720 pre-EQ, 0.794 with the EQ. That is the first
  moment the patch is defensibly the same instrument.

**What this does to the design.** The staircase cannot be a series of progressively
better versions of *the same claim*. Sold as "here is your sound, improving", the early
rungs are false and the user hears it. Three directions the map should consider, in
order of how much they cost:

1. **Re-label the rungs instead of re-timing them.** Rung one is explicitly a stand-in
   ("play something while we listen"), and only the converged patch is ever called *your
   sound*. This costs nothing and is honest. The wait is still 40 minutes, which
   `VISION.md` permits — *"minutes are acceptable, silence is not"* — but 40 minutes is at
   the far end of what "a few minutes later you are playing that sound" promised.
2. **Parallelise the generation.** Measured ×4.06 at 8 workers, and a server can have more
   cores than an M1. This is the only measured lever that shortens the wait, and it needs
   no research: 16 independent renders per generation, already independent in the code.
3. **Find a cheap early estimator that is not CMA-ES.** The evidence that one could exist
   is right here: `oracle_start` recovers 73 % of the largest win the project ever found,
   in 0.06 s, by solving in closed form instead of searching. It only works on a converged
   core today (on the raw seed it improves the loss while making cos θ *worse*), but it is
   proof that this problem has cheap structure the optimiser is currently rediscovering
   the hard way. A first rung worth playing probably comes from an estimator, not from a
   prefix of the search.

**Two things this ticket could not answer and the map should not assume.**

- **There is no per-request transcription.** `out/transcription.mid` came from a chain
  with the author in it; the automatic chord search, run today, returns three adjacent
  semitones stacked in the bass where the answer is a D♭ minor voicing. Every stage-2
  number above assumes correct notes are handed to it for free. On the app's critical
  path they are not.
- **The measured staircase is one run, on one seed, on one 18-second clip.** CMA-ES is
  stochastic and `CLAUDE.md` already records that it scores 0/8 at recovering targets the
  synth itself rendered. The shape of the curve — most of the loss early, all of the
  timbre late — is the finding; the individual seconds are not a contract.
