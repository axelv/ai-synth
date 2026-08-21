# The progress rail

Type: prototype
Status: resolved
Blocked by: none

## Question

Rewritten after [The first screen](07-the-first-screen.md). This was "the wait screen"
until the prototype showed there is no wait screen: the rail from the setup flow stays on
after Start, beside the play surface, as a status display. Seven steps, the last four
ticking over while the user plays the stand-in.

"Minutes are acceptable, silence is not" now has to be true of a rail rather than a page.

Work out:

- What each running step shows beyond its name. Something concrete for "finding the
  notes", something for "fitting the patch". **Not** the current best attempt:
  [How long does a fit actually take?](01-how-long-does-a-fit-take.md) measured it as
  sounding worse than deliberately-shuffled audio for most of the run, and the EQ stage
  moves non-monotonically, so publishing progress-as-sound shows the user a step backwards.
- How progress is expressed across forty minutes when the remaining time is only loosely
  known, in the space a rail affords.
- How "check the notes" announces itself as actionable without yanking someone out of
  playing.
- What a step looks like when it is taking far longer than usual, and where the line is
  before the app has to admit something is wrong. Overlaps
  [When the machinery fails](14-when-the-machinery-fails.md); decide the boundary.
- Whether the rail persists once everything is done, or retires.
- Whether it survives a reload and a return an hour later, which is
  [Persistence and the exits](09-persistence-and-exits.md)'s to answer for state and this
  ticket's to answer for what is shown.

Extend the existing prototype rather than starting a new one.

## Answer

Prototype: [10-progress-rail.prototype.html](../prototypes/10-progress-rail.prototype.html),
a sibling of the setup-flow prototype, which was already at four variants. Four
placements mounted beside a stubbed keyboard and macro set, with a phase selector walking
all five states. **Variant A wins**: the rail keeps a column.

**The rail stays as a full seven-step column beside the keyboard**, exactly where it was
during setup. Nothing moves when Start is pressed, which is the whole reason the rail
survived into the play surface. The alternatives, and why they lost:

- **B, a top strip.** Compact and readable, but the check breaks out as a full-width
  banner directly above the instrument, which reads as the app grabbing your wrist.
- **C, an ambient pill in the corner.** Most room for the instrument, but it breaks
  continuity, collapsing the thing you were following at the moment you press Start, and
  risks being too quiet against "silence is not acceptable".
- **D, A's position with C's pill**, built at request and then rejected on sight. The
  column collapsed to a pill that expanded in place. It gave the instrument more room and
  lost what makes A work, which is that the machine's state is legible without asking.

**Progress on the forty-minute step is a bar driven by attempts against budget**, "2 480
of 7 360", with elapsed time beside it. Grounded in something the fitter counts, monotone,
and it can finish early because of the plateau rule, which reads as a bonus rather than a
broken estimate. What must **never** drive it is loss improvement: 80% of the improvement
lands in the first 90 seconds, so a loss-driven bar would sit at 80% for thirty-eight
minutes. Time estimates with a range were rejected as inviting the app to be wrong out
loud, repeatedly.

**A step that is running long says so**, past a threshold that becomes a number in the
spec: "longer than usual", still running. Acknowledging a delay is the cheapest thing that
keeps trust, and the alternative is a bar that has visibly stopped while the app pretends
otherwise. This makes the boundary with [When the machinery
fails](14-when-the-machinery-fails.md) clean: this ticket owns *slow*, that one owns *dead*.

**When the fit lands the rail collapses to one line**, "fitted in 38 minutes", expandable
back to the full seven steps. It is the only thing that knows what happened, and an hour
later or on a shared link that is the difference between a patch and a patch you can
trust. Retiring it entirely loses the provenance; leaving it open forever leaves a
completed wizard sitting next to an instrument.

**The check reaches an absent user through the browser tab, not through the page.** The
on-page treatment is the pill-to-card change inside the rail, which is a large visual
change in a quiet layout and needs no help. The failure that actually happens is that the
user is in another tab, and no on-page prominence addresses that. So the tab title and
favicon change, which makes the tab title part of the design and something the spec must
state: "Check the notes", not the app's name.

**What a running step shows** is the phase name plus one concrete fact: note events found
so far while transcribing, attempts against budget while fitting. Never the current best
attempt as sound. [How long does a fit actually take?](01-how-long-does-a-fit-take.md)
measured it as sounding worse than deliberately-shuffled audio for most of the run, and
the EQ stage moves non-monotonically, so publishing progress-as-sound shows the user a
step backwards.
