# How long does a fit actually take?

Type: research
Status: resolved
Blocked by: none

## Question

The staircase design bet depends on facts nobody has written down: what this pipeline
produces, in what order, and how long each part takes on the reference clip.

Measure and report, against `data/original.wav`:

- Wall-clock for each stage that exists today: transcription (frozen, but still timed),
  a single render through `PadRenderer`, one objective evaluation, the EQ fit, a CMA-ES
  generation, and a full stage-2 fit as currently run.
- What intermediate artefact each stage leaves behind, and whether it is enough to play
  a note with. The question behind this: how early can a patch be playable at all, and
  how bad is it at that point?
- How the loss improves against wall-clock through a fit. A curve, not a final number.
  If most of the gain lands in the first tenth of the time, the staircase is cheap. If
  it arrives at the end, the staircase is a lie and the design has to change.
- Which parts are parallelisable or GPU-bound, since the app runs this server-side per
  request and the wait is a product feature.

Do not change the pipeline. Measure it. Record findings on a `research/fit-timings`
branch and link them here.

## Answer

Full findings and five reproducible bench scripts: [fit-timings.md](../research/fit-timings.md).
Apple M1, 8 cores. Nothing tracked was modified.

**Inner loop, measured.** Full-clip render (17.9 s, 29 notes, 24 voices) 1.06 s. Mono
loss 0.18 s. One objective evaluation 1.402 s sustained over 353 consecutive renders.
Warm start to first evaluation ~5 s. A CMA-ES generation at popsize 16 is 22.5 s.

**Full fit, estimated not run.** `stage2.main()` defaults are 7360 evaluations, so 2 h
52 min single-process, floored at ~30 min by the plateau rule. The EQ sweep adds ~40 min.
Stage 1's automatic half is ~3 min. Parallelism measured at 4.06x on 8 workers, taking a
generation to ~5.5 s and the full fit to ~42 min. Nothing is GPU-bound: the render is 75%
of an evaluation and has no GPU path.

**The staircase bet fails.** The loss curve supports it and the quality curve destroys
it. Two thirds of the total loss improvement lands 24 s in and 80% within 90 s. But
spectral cos theta goes seed 0.543, after eight minutes of CMA-ES **0.514**, which is
backwards, converged 0.720, delivered 0.794. The frame-shuffled control on this material
is 0.698. **Every rung below convergence sits under a control built by destroying the
audio on purpose.** Early generations buy level and envelope, not timbre. This is the
project's own calibration rule doing its job: a lower loss is not sufficient evidence.

**Earliest playability is ~5 s**, and it is never blocked by artefact availability: a
complete 55-parameter vector exists before any fitting starts. But at that point it is a
stand-in, loss 3.61 against 4.87 for silence and 1.38 for the answer. The first rung that
clears "recognisably the same instrument" is convergence.

**Two traps for whoever builds a staged fit.** `fit_eq_full.oracle_start` recovers 73% of
the project's largest single win in 0.06 s, closed form, one render, but on the raw seed
it lowers the loss by 0.78 while lowering cos theta from 0.543 to 0.487: better number,
worse sound. And truncating the EQ Powell is non-monotone, since it minimises loss plus
a curvature penalty, so a staircase that publishes the current EQ state will show the
user a step backwards.

**A gap the map had not seen.** There is no per-request transcription. Run today, the
automatic chord search returns three adjacent semitones stacked in the bass where the
answer is a D flat minor voicing, and `final_transcription.py` records that the chord
qualities came from the author. Every stage-2 timing above assumes correct notes are
handed over for free. Raised as [Where do the notes come from?](12-where-do-the-notes-come-from.md).

Also noted: `out/loss_history.npy` is not a trace of the delivered fit, so no
loss-against-time curve existed in this repo before these benches.
