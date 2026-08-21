# When the machinery fails

Type: grilling
Status: resolved
Blocked by: none

## Question

Graduated from the map's fog once [The stand-in and the
swap](03-the-stand-in-and-the-swap.md) settled the normal path. Distinct from [When the
fit comes back wrong](08-when-the-fit-comes-back-wrong.md), which is about a fit that
succeeded and does not sound right. This is about the fit not happening at all.

Settle:

- The file that will not open. Wrong format, corrupt, silent, far too long, far too short.
  Which are refused at the door and which are accepted with a warning.
- The fit that dies on the server. Forty minutes of compute, no result, and a user who may
  not be watching. What they are told, what it costs them, and whether it retries itself.
- The connection that drops mid-fit, and the tab that closes. Overlaps [Persistence and
  the exits](09-persistence-and-exits.md); decide the boundary rather than answering both.
- The browser that cannot play. No Web MIDI in Safari at all, and the whole API behind a
  permission prompt in Chrome since 124. What an unsupported browser is told, and how
  early.
- The audio that will not start. Autoplay policy means the first sound needs a gesture, so
  there is at least one unavoidable click before anything is audible.
- Whether any of these should be caught before the clip is uploaded rather than after.

## Answer

**Files are checked in the browser, before anything is uploaded, and there are exactly two
refusals: it will not decode, and it has effectively no signal.** The decode is free
because the app has to decode the file anyway to draw the waveform for the trim step, so
the check is the same code path and the failure lands before any promise is made. Length is
never a refusal in either direction: too long is what the trim step exists for, and too
short is caught later and more honestly by the transcription finding no notes, which [Where
do the notes come from?](12-where-do-the-notes-come-from.md) already handles by never
accepting silently. These two are the same two [When the fit comes back
wrong](08-when-the-fit-comes-back-wrong.md) named as the honest floor; this ticket only
moves them to the door and puts them client-side.

**A fit that dies on the server retries once, visibly.** The rail says so, in the same
vocabulary it already uses for a step running long, because a silent retry doubles a wait
the rail put a number on. If the second attempt dies too, the step enters a failed state
that says what happened and offers to start over from the trim.

This needs a distinction [When the fit comes back wrong](08-when-the-fit-comes-back-wrong.md)
did not draw. **"No refit" means no second attempt at a fit that completed**, because the
reachable set is the constraint and a completed fit has already explored it. A fit that died
explored nothing, so retrying it is a different act entirely and none of that ticket's
reasoning applies.

It also names a state nobody had: **a Patch whose fit died is a Patch with zero Versions**,
permanently on the stand-in. It is not deleted and it is not hidden. It stays in the list,
it is playable, and it says what happened, which is consistent with [The stand-in and the
swap](03-the-stand-in-and-the-swap.md) making the stand-in real but never a Version.

**A dropped connection and a closed tab are not failures.** [Persistence and the
exits](09-persistence-and-exits.md) put the running Fit on the server and made a reload
rejoin it, so the boundary between these two tickets is clean: 09 owns what persists, this
one owns what the screen says while the wire is down. The rail shows a reconnecting state
and never implies the fit stopped, because it has not. A fit that finishes while nobody is
connected is not an error, it is a result waiting to be found through the local list or the
link.

**No browser is refused.** The instrument does not depend on Web MIDI at all: faustwasm's
poly node exposes `keyOn` and `keyOff` on the node itself and MIDI is only one caller, so
anything with AudioWorklet plays through the on-screen keyboard. Chromium-only is a decision
about what gets designed and tested, not a lock, and shutting out a browser that can
genuinely play the sound would be inventing a limit that does not exist. Non-Chromium gets
one line: hardware keyboards need Chrome or Edge. That line is the whole accommodation.

**Web MIDI is asked for on an explicit user action, never on load.** The permission has been
in front of the entire API since Chrome 124 and cannot be pre-empted, and it is queryable,
so the app can tell never-asked from denied and avoid prompting into a wall. The trigger is
the play surface's device line, which reads as a button until granted. That one surface
absorbs all three states the research identified, never asked, denied, and granted with no
device present, and hot-plug is a live subscription to `statechange` rather than an
enumeration at load, so plugging a keyboard in at any point just works. **SysEx is never
requested**: the app has no use for it and asking makes the prompt scarier.

**The autoplay gesture is already in the flow.** Start unlocks the AudioContext on the way
in, and everything audible, the ear check and the stand-in, happens after it. The only case
needing thought is arriving cold at a Patch URL in a fresh tab, where there is no Start:
there the keyboard is the gesture, since pressing a key both unlocks and plays. There is
never an "enable sound" modal. The one thing this forecloses was never planned: the tab
title and favicon arrival from [The progress rail](10-the-progress-rail.md) cannot make a
sound.

### A research note this supersedes

`research/browser-audio.md` warns that a fixed on-screen velocity of 100 would hide exactly
the defect the user needs to find, since `VISION.md` requires the patch to hold up at
velocities the clip never showed. That is now moot: [How much of the keyboard did the clip
actually evidence?](13-how-much-did-the-clip-evidence.md) established that velocity reaches
nothing but the output gain in this architecture, so there is no velocity response for a
fixed value to hide. It becomes live again the moment the fit honours the contract line
about velocity, and the on-screen keyboard will need a velocity source then.
