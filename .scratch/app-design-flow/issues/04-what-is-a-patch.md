# What is a Patch?

Type: grilling
Status: resolved
Blocked by: none

## Question

The noun the entire app hands around, and it is not yet defined. Domain modeling, ending
in the first entries of `CONTEXT.md`.

Settle:

- What a Patch is made of. `VISION.md` says a synth architecture plus the settings that
  go with it. Does that include the notes recovered from the clip, the clip itself, the
  macro mappings, the fit's provenance?
- Whether a Patch is immutable. When you move a macro and like the result, is that the
  same Patch changed, a new Patch, or a Patch plus a separate layer of your edits?
- What a URL addresses. The Patch, the fit that produced it, or the session you are in.
- What a stage of the staircase produces. A different Patch each time, or one Patch
  getting better?
- The surrounding vocabulary and which word wins for each: clip, sample, recording,
  source; fit, match, search; preset, patch, sound, instrument; macro, knob, control.

Write the resolved terms into `CONTEXT.md` as they settle.

## Answer

**A Patch is an architecture, its parameter values, and its macro mapping. Nothing
else.** The clip and the recovered notes sit outside it, reachable through the Fit that
produced it. A patch is what a person could have dialled in, and nobody dials in a clip;
notes are transcription output, which `VISION.md` calls a means and not an end. Keeping
them out means a Patch can be exported, compared against another Patch, or played with
different music without dragging someone's audio behind it.

**A Patch is one identity that gains versions.** Each staircase stage writes a version.
Your macro moves do not, until you explicitly say keep this, which writes one. The live
unsaved state is the **position**, saved continuously but not versioned. This matches how
you would say it out loud, that your patch got better while you played it, and gives one
stable thing to name, link and return to.

**A macro position is an offset from the current version's fitted defaults**, not an
absolute value. It is the only model in which the staircase and the player's hands can
both move without one clobbering the other, and it makes "back to how it was fitted"
mean "all macros at zero". This does not decide what the staircase does at a stage
boundary, which is [The stand-in and the swap](03-the-stand-in-and-the-swap.md); it decides whether that ticket has the option.

**The architecture is fixed for the life of a Patch**, chosen at stage one before
anything is playable. Later versions move parameters only. Forced by [What can the
browser actually play?](02-what-can-the-browser-play.md): there is no in-place recompile
and no access to voice state, so held notes cannot survive a graph swap. A fit that
discovers mid-climb it needs a different architecture must announce a rebuild, which
visibly interrupts the player and should be rare. The rejected alternative was one
global superset architecture with identity defaults: it makes swapping vanish, but inert
modules still burn cycles, and at 4.3% of a core per voice against a budget of eight
voices, a superset voice costs as much for a thin pluck as for the fattest patch
possible. It also caps how different two patches can ever be.

**Constraint handed to [The stand-in and the swap](03-the-stand-in-and-the-swap.md):** the
architecture decision happens before first playability, on the least information the
fitter will ever have.

**A URL addresses the Patch**, with an optional version pin. The unpinned form is what
you send yourself and what survives the staircase still climbing; the pinned form costs
almost nothing and pays for itself the first time you want to say it was better before.

**Exactly one Clip per Fit, one Fit per Patch.** A deliberate narrowing, not an
oversight: it guarantees there is always something to compare against, which is the only
way to judge "recognisably the same instrument". Confirmed as a two-way door. Making the
Fit optional later only widens the constraint and migrates no data; the cost is empty
states for the A/B panel and notes playback, not a rebuild.

**A Patch always has a name**, auto-set from the clip's filename and editable at any
time. Nothing in the flow ever stops to ask you to name a sound before you have heard it.

**Vocabulary settled**, written into `CONTEXT.md`: Clip, Fit, Patch, Version, Position,
Macro, Architecture. "Sample" is banned, since it means a playable audio instrument in
this domain and `VISION.md` rules that off the table permanently. "Fit" over match or
search, because it names an optimisation against a target rather than a lookup. "Patch"
over preset, because a preset is something you were given and a patch is something that
was made for you.

## Amendment

Resolved while the map still assumed a staircase of several fitted stages. [How long does
a fit actually take?](01-how-long-does-a-fit-take.md) replaced that with one stand-in and
one fitted patch, and [The stand-in and the swap](03-the-stand-in-and-the-swap.md) then
settled that **the stand-in is not a Version at all**: it is a hand-made factory default
that never claimed to be your sound, so the Patch has no versions until the fit lands.
Read "each staircase stage writes a version" as "the fit writes a version when it lands".
An earlier draft of this amendment said the stand-in was the first Version; that was
wrong. Left as written rather than rewritten, since what was believed at the time is part
of why the model looks like this.
