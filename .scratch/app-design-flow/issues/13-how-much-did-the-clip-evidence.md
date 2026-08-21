# How much of the keyboard did the clip actually evidence?

Type: grilling
Status: resolved
Blocked by: none

## Question

Graduated from the map's fog by the discussion on [Where do the notes come
from?](12-where-do-the-notes-come-from.md). `VISION.md`: a patch that only works in the
register it was fitted to is not a patch, it is an impression of one. A Clip may be a
single held note. The Patch is played across the whole keyboard. Something has to tell
the difference between where the fit is backed by evidence and where it is extrapolating.

Settle:

- What "evidenced" means, measurably. Pitch coverage is the obvious axis, but the fit is
  constrained by where the Clip put *energy*, not by which notes were played, and a single
  low note puts harmonics across most of the spectrum. Whether the useful measure is
  played range, spectral coverage, or both.
- Whether velocity and hold time are part of it. A Clip demonstrates one velocity and one
  hold time per note, and `VISION.md` names both as things the Patch must survive.
- What the user is shown, and where. Marking the fitted range on the keyboard is the
  obvious surface; whether it is a permanent marking, a warning on first play outside the
  range, or something in the visuals.
- What tone to take. This is the app admitting the limits of its own answer, and the
  difference between honest and discouraging is entirely in the wording.
- Whether anything can be done about it beyond disclosure: extrapolating deliberately,
  asking for a second Clip in another register, or nothing.
- Whether a Patch that was barely evidenced should be marked as such when returned to
  later, which touches [Persistence and the exits](09-persistence-and-exits.md).

## Answer

**Evidenced means the played range of fundamentals, marked exactly, with no margin.**
The span runs from the lowest to the highest note in the Transcription. Spectral coverage
is real and the fit uses it, but it is never shown, because it is always the more
flattering of the two numbers and it is flattering in exactly the case where it is least
earned. A single held C2 puts harmonics across four octaves of the EQ bank, so a
coverage-derived reading would report four octaves evidenced. What actually happened is
that each of those bands had its gain chosen to shape one quiet partial sitting between
other partials. Playing a loud fundamental into that band is precisely the untested case.
The measure that turns weak evidence into a wide claim is the wrong measure.

**Velocity and hold time get no surface.** Not because they are well evidenced but
because velocity is not fitted at all: `process = filtered * aenv * gain` and `gain` is
the Faust polyphonic voice slider, so velocity scales loudness linearly and touches
nothing else, no cutoff, no envelope, no drive. A velocity gauge would imply a response
the Patch does not have. The honest statement is not "your clip only demonstrated
velocity 90", it is "this patch gets louder when you play harder and does nothing else",
and that belongs in Known gaps, not on the play surface. Hold time is covered: the
amplitude ADSR sustains, so any hold length is defined behaviour and only the release
shape was fitted against the Clip.

**The surface is a permanent, passive marking of the span on the on-screen keyboard.**
Always visible, never interrupting, and it appears only once a Version exists, so the
stand-in makes no claim about a range it was never fitted to. The on-screen keyboard
mirrors incoming MIDI, so it serves hardware players without a second surface. Rejected:
a one-time note on first play outside the span, which puts a dismissible message into the
single moment the app exists to produce; and a marking that only surfaces on transgression,
which makes the app look like it is catching you out.

**Wording is a fact about the Clip, in the Clip's voice.** "Your clip played C2 to G3."
No hedge attached to the Patch, no warning colour, no word like unreliable or
extrapolated. The keys outside the span are fully playable and unmarked in any way that
reads as an error. The app describes what it saw rather than apologising for what it made.

**Nothing is done about it beyond disclosure, in the app.** Rejected: asking for a second
Clip in another register, which reopens one-Clip-per-Fit for no measured reason; and a
hidden key-tracking correction outside the span, which is an invisible parameter doing
something the fit never sanctioned, contradicts macros being the only surface, and could
not be checked against anything. What this ticket produces instead is a contract line for
the fitting research, recorded on the map.

**On return, the span is shown identically and nothing extra is added.** It is derived
from the Fit, so it is a property of the Patch and needs no separate storage decision in
[Persistence and the exits](09-persistence-and-exits.md). A barely-evidenced Patch gets no
badge; distinguishing strong from thin Patches at a glance is a library affordance and the
library is out of scope.

### Why this ticket found a fitting problem, not just a wording problem

The reachable architecture is the worst possible shape for keyboard generalisation, and
that is measured rather than incidental. The 26-band EQ sits post-voice at fixed absolute
centres, and `CLAUDE.md` records the harmonic-number alternative being fitted and removed
for contributing exactly zero: a 34-parameter absolute-frequency curve beat a
177-parameter harmonic-number one. So the fitted timbre is a fixed curve in Hz. Transpose
an octave and every harmonic slides into a different band, which is the same as saying the
instrument changes character with register in a way nothing ever fitted or checked. The
one thing that does generalise is `trackedCut = cutoff * pow(freq / 261.6255, kbdTrk)`,
a single fitted exponent, and `panPos`, which scatters pan by pitch through an irrational
multiplier and therefore puts unplayed notes at arbitrary positions in the image by design.

`VISION.md` says a patch that only works in the register it was fitted to is an impression
of one. On the current architecture that is not a risk, it is the default, and no amount of
honest labelling fixes it.
