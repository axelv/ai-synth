# The stand-in and the swap

Type: grilling
Status: resolved
Blocked by: none

## Question

Rewritten after [How long does a fit actually take?](01-how-long-does-a-fit-take.md)
killed the staircase. There are two sounds, not a ladder: a stand-in available within
seconds that is honestly not the sound, and the fitted patch roughly forty minutes later.
This ticket designs the pair and the moment between them.

Settle:

- What the stand-in actually is. A complete parameter vector exists at ~5 seconds before
  any fitting, at loss 3.61 against 4.87 for silence and 1.38 for the answer. Is it that
  raw seed, something cheap and better, or a generic pad that makes no claim at all?
- How the app says "this is not the sound yet" without making you feel you were given
  something broken. This is the sentence the whole first impression hangs on.
- The quality gate. Measured cos theta at convergence is 0.720 against 0.698 for
  deliberately-shuffled audio, so the margin is thin and a gate is not a formality. What
  happens when a fit finishes and does not clear it.
- The swap. What happens to a held note, a Position you have moved, and anything you were
  playing when the real patch arrives. Parameters can be ramped over ~186 ms without
  clicking; the architecture is fixed at stage one so the graph never changes.
- Whether you can refuse the swap, or go back to the stand-in, and whether anyone ever
  would.
- Where the ear check from [Where do the notes come
  from?](12-where-do-the-notes-come-from.md) sits in this sequence. The stand-in needs no
  notes, so it is playable before transcription finishes, and the check then interrupts
  someone who is already playing. The fit starts optimistically alongside it.
- Whether the user is present for the swap at all, given forty minutes, and what the app
  does when they are not. This overlaps [Persistence and the
  exits](09-persistence-and-exits.md); decide the boundary.

## Answer

**The stand-in is a hand-made factory default for the chosen architecture.** Not the
optimiser seed, and not anything cheaply fitted. It has to be the chosen architecture
regardless, because the architecture is fixed for the life of a Patch and the swap must be
parameters only, so the only real choice is whether the first thing a user hears was
designed or was a byproduct of an optimiser's starting position. A seed exists to be
optimised from, not to be listened to.

Fitting something cheap was ruled out by measurement, not taste: `fit_eq_full.oracle_start`
recovers 73% of the project's largest single win in 0.06 s, but on the raw seed it lowers
the loss by 0.78 while lowering cos theta from 0.543 to 0.487. And the seed itself
(0.543) beats eight minutes of CMA-ES (0.514). Anything that is a fit in progress sounds
worse than something that never tried.

**It is named as a placeholder**, in words, and it is not your sound. Framing it as an
activity ("play while we work") is evasive, and showing it as your patch in a fitting
state is a lie by implication. The credibility of this product is the sound, so the claim
has to be explicit and the warmth has to live in the wording.

**The stand-in is not a Version.** The Patch has no versions until the fit lands. It
cannot be pinned, linked or returned to, which also settles this ticket's "can you refuse
the swap" bullet: there is nothing to go back to. This **corrects the amendment on**
[What is a Patch?](04-what-is-a-patch.md), which had recorded the stand-in as the first
Version.

**The gate is a per-clip control, and it labels rather than blocks.** For each Clip, score
a deliberately-wrong version of that same Clip, frame-shuffled, and require the fit to
beat its own Clip's control by a margin. `CLAUDE.md` states the rule: calibrate any
similarity metric against a deliberately-wrong control before reading a value as good or
bad. A fixed threshold assumes every clip scores on the same scale and the calibration
numbers say they do not. The control needs a score but no render, so it is nearly free.

A failing fit **still swaps in, labelled as a poor match**. Converged cos theta was 0.720
against 0.698 for shuffled, a margin of 0.022, and a metric that thin should not overrule
a person who can simply listen. `VISION.md` puts the bar at "recognisably the same
instrument", which is a human judgement. A poor fitted patch is also strictly more useful
than a factory default, because it is at least about your Clip. What to offer at that
point is [When the fit comes back wrong](08-when-the-fit-comes-back-wrong.md).

**Design for absence.** Forty minutes is a walk, a coffee, another task. The arrival has
to work when it is discovered later; presence is the lucky case. How the user learns it is
ready, and how they get back, is [Persistence and the exits](09-persistence-and-exits.md).

**If they are present, the swap happens under their fingers**, ramped over ~186 ms, which
the research measured at 0.1x the signal's own motion against 1.3x for an instant jump.
The morph is the best moment this product has and it costs nothing. One detail the spec
must carry: the Position rides through as offsets, which is why
[What is a Patch?](04-what-is-a-patch.md) made them offsets, but the fitted patch declares
its own macro excursions, so the same offset means a different amount either side of the
swap and must be ramped with everything else.

**An unanswered ear check is dropped, not deferred.** Its whole purpose was to save the
forty minutes, and once they are spent it has no job. If the notes were wrong, the A/B
against the Clip shows it better than the check ever could. Asking someone to validate an
input after the output has arrived is work the finished patch already did for them.
