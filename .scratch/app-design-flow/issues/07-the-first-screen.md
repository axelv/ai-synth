# The first screen

Type: prototype
Status: resolved
Blocked by: 01

## Question

Arrival, drop, trim, start. The stretch before anything is playing.

Work out:

- What an empty app says for itself. Someone arriving with a clip and no idea what this
  is has to understand it in one screen.
- The drop. File picker, drag and drop, paste. What is accepted and what is refused.
- The trim. How you choose the part of the clip that gets fitted, and what guidance the
  app gives about what makes a good clip. Constrained by what the pipeline actually
  needs, from [How long does a fit actually take?](01-how-long-does-a-fit-take.md). Note
  that [Where do the notes come from?](12-where-do-the-notes-come-from.md) decided the app
  prefers wide pitch coverage, monophonic where possible, and accepts anything including a
  single held note. This screen has to make that case without turning it into a gate.
- Whether anything is asked of the user before the fit starts. A hint about the kind of
  sound, a name, nothing at all.
- The moment of commitment: what the button says, and what happens in the first second
  after it is pressed.

Link the prototype from this ticket as an asset.

## Answer

Prototype: [07-first-screen.prototype.html](../prototypes/07-first-screen.prototype.html).
Four variants on `?variant=`: A a minimal single surface, B a dense workbench, C a
one-question-per-step conversation, D the winner. Kept whole rather than trimmed to the
winner, because the losing variants are the record of what was considered. Styling is
deliberately flat, so this prototype decides nothing about visual language, which stays in
the map's fog.

**Variant D: one question per step, with a rail carrying every phase.** Seven steps in one
list, setup and machine together: choose a sound, choose the part, start, finding the
notes, check the notes, fitting the patch, play it. Times sit on the machine steps.

**The whole cost is disclosed on the first screen.** "About 40 minutes" is visible on the
rail before the user has invested anything, which is earlier than C managed by giving the
wait its own step, and earlier is more honest. The consequence, recorded deliberately: the
commitment step shrank to a heading and two lines, so the wording on the rail is now
load-bearing, and a grey subtitle may be too quiet for the single largest ask in the
product. Flagged for the spec rather than solved here.

**The rail stays after Start, beside the play surface.** This is the decision the
prototype earned. A linear stepper implies a queue you wait in, and from Start onward you
are not waiting: you are playing the stand-in while the fit runs, and the note check
interrupts you rather than summoning you. So after Start the rail stops being a wizard and
becomes a status display next to the keyboard. Steps tick over while you play, and "check
the notes" lights up as something to do, not a screen you are sent to.

Two consequences handed on. [The progress rail](10-the-progress-rail.md) is no longer a
screen; it is rewritten as the rail's content. And [The play surface](06-the-play-surface.md)
inherits a layout constraint: it has to host a seven-step rail next to a keyboard and five
macros without crowding either.

**Nothing is asked before the fit but the clip and the trim.** No name, no hint about the
kind of sound, no account. The Patch is auto-named from the file, per
[What is a Patch?](04-what-is-a-patch.md).

**Clip guidance is one sentence at the trim step**, not a gate and not a readout: "a
stretch where the pitch moves works best, a single held note is fine, it just tells us
less". Variant B's live coverage readout was the alternative and was not chosen; it turns
choosing a clip into a small optimisation problem, and
[How much of the keyboard did the clip actually evidence?](13-how-much-did-the-clip-evidence.md)
is the honest place to deal with thin material, after the fact rather than as a hurdle.

**Which files are refused, and what happens when one is**, belongs to
[When the machinery fails](14-when-the-machinery-fails.md).
