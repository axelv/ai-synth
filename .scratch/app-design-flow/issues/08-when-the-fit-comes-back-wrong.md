# When the fit comes back wrong

Type: grilling
Status: resolved
Blocked by: 03, 05

## Question

The bar is "recognisably the same instrument". Sometimes it will miss, and the most
likely gap in this flow is what happens next.

Settle:

- What the recovery paths are, and which of them exist in this design. Refit with a
  hint about what is off, correct it by hand with macros, retrim and try again, start
  over, give up cleanly.
- Whether the user can express what is wrong in words a musician uses ("too thin", "the
  wobble is too fast") and whether anything downstream could act on that. If nothing
  can, do not build a form that pretends otherwise.
- What to offer when the per-clip gate from [The stand-in and the
  swap](03-the-stand-in-and-the-swap.md) marks a fit as a poor match. The patch still
  arrives, labelled; this ticket owns what the label offers to do about it.
- What a refit costs the user. Waiting again, losing the macro positions they had found,
  losing the URL they had sent themselves.
- The honest floor: which clips this app should decline outright rather than fit badly.
  [Where do the notes come from?](12-where-do-the-notes-come-from.md) already creates one
  class of declined clip, where the user rejects the transcription and has no aligned MIDI
  file. Word that refusal here.

## Answer

**There is no refit.** Same Clip plus same architecture is the same reachable set, and
`CLAUDE.md` records the reachable set as the constraint rather than the optimiser, three
rebuilds in a row. Nor is the search reliably leaving value on the table for a second
attempt to collect: CMA-ES scored 0/8 recovering targets the synth itself had rendered,
because a loss around 1.3 carries no parameter information. A "try again" button therefore
sells forty minutes against a wall. The only "again" this design has is **a different
Clip**, by retrimming or by starting over, and that is a new Patch rather than a new
Version.

Worth recording because it is legal today and will matter later: the domain model says one
Clip per Fit and one Fit per Patch, but it never says a Clip has only one Fit. So "fit this
clip again, differently" already has a coherent shape, a second Patch standing beside the
first over the same Clip. It is not in this version because there is nothing to vary: a
single architecture, chosen by the app, no choice offered. When more than one architecture
exists, that is where it attaches.

**There is no hint channel, and no form.** Every complaint a musician would actually type
lands in one of two places. Either it names one of the five macros, "too thin" is Body,
"the wobble is too fast" is Movement, "too dull" is Brightness, in which case the fix is
already on screen, instant, and free. Or it names something the architecture cannot reach,
in which case a reweighted refit cannot reach it either. The axis a user is most likely to
complain about, width, is the one the objective is structurally blind to, since
`stage2.Objective.loss_of` is mono, which is precisely why Width is a macro. A form that
collects a complaint and does nothing is worse than no form; one that triggers a refit
spends forty minutes on a search that could not self-recover on its own renders.

**The poor-match label states the measurement, points at the A/B, offers one action, and
can be dismissed for good.** The measurement is the per-clip frame-shuffled control from
[The stand-in and the swap](03-the-stand-in-and-the-swap.md), and the converged margin was
0.022, on a metric `CLAUDE.md` calls a weak ranker. A number that thin has no business
out-arguing someone who can listen, and the comparison frame settled by [The play
surface](06-the-play-surface.md) is already on the screen, so the label's real job is to
send them to it. The single action offered is **try a different part of the clip**, which
returns to the trim step. Dismissal is final: once you have listened and disagreed, the app
stops saying it.

**The honest floor is very low, and deliberately so.** The app declines exactly two things:
the transcription dead end already worded in [Where do the notes come
from?](12-where-do-the-notes-come-from.md), and a clip with effectively no signal. It does
not decline full mixes, acoustic instruments or voices, because refusing on content needs a
classifier this app does not have in order to turn away clips that might have worked.
`VISION.md` says synths first and adds that nothing in the design should assume that
forever. Not knowing in advance which clips will fit is the true state of affairs, and the
app should not pretend otherwise. Files that will not decode at all belong to [When the
machinery fails](14-when-the-machinery-fails.md).

**Nothing is ever lost, because nothing is replaced.** A retrim produces a new Patch at a
new URL. The old Patch keeps its Versions, its Position and its link, so the user can walk
back to the fit they had. Macro Positions are deliberately **not** carried across: offsets
are relative to fitted defaults and each fit declares its own macro excursions, so the same
numbers mean a different amount on the other side, the same trap [The stand-in and the
swap](03-the-stand-in-and-the-swap.md) flagged for the swap itself. Carrying them would
silently change the sound the user had found.

### The shape of recovery, in full

1. **Correct it with the macros.** Instant, and already the whole surface.
2. **Try a different part of the clip.** Back to the trim step, a new Patch, the old one
   untouched.
3. **Start over with a different clip.** Always available.
4. **Leave with it anyway.** `VISION.md`: the patch can always leave. A poor fitted patch
   still exports, and is still strictly more useful than a factory default because it is at
   least about your clip.

There is no fifth path, and the value of this ticket is mostly in having established that
the four obvious-sounding others, refit, refit with a hint, refit with a different seed, and
a feedback form, are all measured dead ends rather than unbuilt features.
