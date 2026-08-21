# Persistence and the exits

Type: grilling
Status: resolved
Blocked by: 04

## Question

A patch survives a reload and is addressable by URL. What that actually buys, and how
the flow ends, is open.

Settle:

- What the URL gives someone who opens it. The patch playable, the clip alongside it,
  the fit's history, or a read-only view.
- What survives a reload and what does not. Macro positions, MIDI device, the original
  clip, a fit still running.
- How a user gets back to a patch they made last week without accounts existing. Browser
  history, a list in local storage, a mailed link.
- How the export exits are stubbed: readable Faust source and a plugin build are
  promised by `VISION.md` and are out of scope here, but the flow has to show where they
  attach and what the user is told about them today.
- Settled elsewhere, do not reopen: the evidenced span is derived from the Fit, so it is a
  property of the Patch and shown identically on every load, and a barely-evidenced Patch
  gets **no** badge on return. See [How much of the keyboard did the clip actually
  evidence?](13-how-much-did-the-clip-evidence.md).
- What the app does with the uploaded clip after the fit. Kept, discarded, and on what
  schedule. This is the one place a single-user design still owes an answer.

## Answer

**The URL is the credential, and it carries the audio.** Someone who opens a Patch link
gets exactly what the maker gets: the patch playable, the clip audible in the comparison
frame, the evidenced span, the version list. There is no second kind of session, because
with no accounts the app cannot tell a maker from a visitor without inventing a token that
is an account in disguise. The consequences are handled by being stated rather than by a
permissions model: ids are unguessable, and the app says once, plainly, that anyone with
the link can hear the clip you uploaded. Version pinning stays as [What is a
Patch?](04-what-is-a-patch.md) settled it, an optional suffix on the same URL.

**Server holds the Patch, its Versions, the Clip, and a running Fit. The browser holds the
Position.** Reloading mid-fit rejoins the rail rather than restarting anything, which is
what makes forty minutes survivable at all. The split on the Position is the deliberate
part:

- It gives **keep this** a real job. An explicit save is how a tweak becomes durable and
  travels to another machine, which is otherwise a distinction without a difference.
- It stops a visitor on a shared link from silently clobbering the macro positions the
  maker had found, which server-side Positions would allow and no one would notice.
- Ticket 04 called the Position saved continuously but not versioned, and local storage is
  precisely that.

MIDI is separate and needs no design: Chromium remembers the Web MIDI permission per
origin, so the device re-resolves on reload and the app never asks twice.

**Getting back is a list plus a link.** On the first screen, under the clip drop, a short
**your patches** list read from local storage on the machine that made them. Everything
else is the URL. When the fit lands while you are away, it arrives through the **tab title
and favicon**, the same channel [The progress rail](10-the-progress-rail.md) chose for the
note check, because the failure that actually happens is that you are in another tab.
Rejected: browser notifications, which put a second permission prompt on a page that has
already asked for Web MIDI, and mailing yourself a link, which needs an address and is an
account in disguise.

**The Clip lives as long as the Patch does.** This was decided elsewhere without anyone
noticing: [The play surface](06-the-play-surface.md) made the clip the permanent backdrop
of the comparison frame, so discarding it does not degrade that screen, it breaks it. The
only thing that removes a clip is deleting its Patch, which is the single destructive
action the app offers, and the confirmation says the clip goes with it. Rejected: a 30-day
expiry, which quietly turns the play surface into a worse screen on a timer nobody agreed
to, and keeping only a spectrum, which leaves a clip you can see but not hear.

**Both exits are on the play surface from the moment a Version exists, and one of them
works.** Faust source is a real download today: the server compiled the patch, so it can
hand over the text. `VISION.md` promises that if the service disappears your sounds do not,
and [The macro layer](05-the-macro-layer.md) leans on that export as the escape hatch for
the 55 parameters the app deliberately refuses to show, so it cannot be a stub without
making that decision dishonest. The plugin build is a named button that says what it will
do and that it does not do it yet. Hiding it would be worse: the promise is part of why
someone would invest forty minutes.

### The gap this leaves, named rather than solved

**The list is per browser profile.** Lose the profile, or move to another machine, and the
only way back to a Patch is a link you saved somewhere else. This is the honest shape of
single-user with no accounts, and accounts later replace the list with a synced one and
migrate nothing, so it is a two-way door. It is worth the spec saying so where the list
appears, rather than letting someone discover it.
