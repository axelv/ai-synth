# The app: flow, screens and contract

The end-to-end design of the web app described in `VISION.md`: what a person sees from
arriving with a clip to playing the fitted patch on a keyboard and pushing its macros.

Read `VISION.md` for what the product is and which arguments are already settled,
`CONTEXT.md` for the vocabulary, and `CLAUDE.md` for the fitting pipeline this contracts
with. Terms in **Clip**, **Fit**, **Patch**, **Version**, **Position**, **Architecture**,
**Transcription**, **Macro** are used exactly as `CONTEXT.md` defines them.

Every decision here was settled one ticket at a time under `.scratch/app-design-flow/`,
and the reasoning, including the rejected alternatives, lives in those tickets. This
document states the outcome; the index at the end says where each one was argued.

Scope of this version: hosted, one user, no accounts, desktop Chromium as the browser that
gets designed and tested for. One Clip per Fit, one Fit per Patch, no patch library.

---

## 1. The flow at a glance

Seven steps, in one list, visible before the user has invested anything:

| # | step | shown as | typical |
|---|---|---|---|
| 1 | Choose a sound | setup | |
| 2 | Choose the part | setup | |
| 3 | Start | setup | |
| 4 | Finding the notes | machine | about 3 minutes |
| 5 | Check the notes | yours to answer | seconds |
| 6 | Fitting the patch | machine | about 40 minutes |
| 7 | Play it | | |

The list is the **rail**. It is the same object throughout: a wizard before Start, a status
display after. Steps 1 to 3 are the setup surface, steps 4 to 7 all happen on the play
surface, because from Start onward the user is playing rather than waiting.

Disclosing "about 40 minutes" on the very first screen is deliberate. It is the single
largest ask in the product and the user meets it before uploading anything.

**Consequence flagged and not solved:** the commitment step is now a heading and two lines,
so the rail's own wording carries the whole disclosure. A grey subtitle may be too quiet
for it. The visual treatment of that one line needs deciding when the visual language is.

---

## 2. Setup surface

One question per step. Nothing is asked but the clip and the trim: no name, no hint about
the kind of sound, no account.

### S1 — Choose a sound

- A drop zone for one audio file, plus a file picker and an **use the example clip**
  affordance.
- Below it, **your patches**: a short list read from this browser's local storage, most
  recent first, each entry a name and when it was fitted. Empty on a first visit and
  absent rather than empty-stated.
- The file is decoded **in the browser**, before any upload. That decode is needed anyway
  to draw the waveform in S2, so the validity check is free and happens before any promise
  is made.
- Two refusals, and only two: **it will not decode**, and **it has effectively no signal**.
  Both are stated at the drop zone and leave the user on S1. Length is never a refusal:
  too long is what S2 is for, too short is caught honestly at step 4 by the Transcription
  finding no notes.

**Transitions.** Valid file → S2. Refused file → S1 with the reason. Clicking an entry in
your patches → the play surface for that Patch, at whatever state it is in.

### S2 — Choose the part

- The decoded waveform with two draggable trim handles. The trimmed span is the **Clip**.
- One sentence of guidance, not a gate and not a live readout: *a stretch where the pitch
  moves works best, a single held note is fine, it just tells us less.*
- No coverage meter. Turning clip choice into a small optimisation problem was rejected;
  thin material is dealt with after the fact, in §5.4, rather than as a hurdle.

**Transitions.** Back → S1. Forward → S3.

### S3 — Start

- A heading, two lines, and the Start button. The rail beside it is carrying the cost
  disclosure.

**Transitions.** Start →
1. the **Patch** is created and named from the file's name, editable at any time;
2. its URL is minted and pushed, so the address bar is already the thing to keep;
3. an entry is written to your patches in local storage;
4. the AudioContext is unlocked by this click, which is the only autoplay gesture the flow
   needs on the way in;
5. the Transcription starts;
6. the play surface replaces the setup surface. **The rail does not move.**

---

## 3. The rail

A 250px column on the left, present from S1, unchanged in position and content structure
when Start is pressed. It collapses to a single line once the fit lands.

### Per-step display

A running step shows its name plus **one concrete fact**:

- step 4: note events found so far.
- step 6: **attempts against budget**, for example `2 480 of 7 360`, with elapsed time
  beside it, and a bar driven by that ratio.

The bar is never driven by loss improvement. 80% of the loss improvement lands in the first
90 seconds, so a loss-driven bar would sit at 80% for thirty-eight minutes. Attempts against
budget is monotone, is grounded in something the fitter counts, and can finish early, which
reads as a bonus rather than a broken estimate. Time estimates with a range were rejected as
inviting the app to be wrong out loud, repeatedly.

**The rail never publishes progress as sound.** Every intermediate state of the fit measures
worse than deliberately-shuffled audio, and the EQ stage moves non-monotonically, so an
audible progress preview would show the user a step backwards.

### Rail states

| state | display |
|---|---|
| `R-setup` | steps 1-3 active, 4-7 ahead with their times |
| `R-transcribing` | step 4 running, note count |
| `R-check` | step 5 is a **card**, not a pill: the ear check, §4.2 |
| `R-fitting` | step 6 running, bar plus attempts plus elapsed |
| `R-slow` | as `R-fitting`, plus **longer than usual**, still running. Threshold: 1.5x the estimate |
| `R-reconnecting` | the wire is down. Says so; **never** implies the fit stopped |
| `R-done` | collapsed to one line, `fitted in 38 minutes`, expandable back to all seven |
| `R-failed` | step 6 failed, §6.2 |

`R-done` is expandable rather than retired because the rail is the only thing that knows what
happened, and an hour later or on a shared link that is the difference between a patch and a
patch you can trust.

### Reaching an absent user

The failure that actually happens is that the user is in another tab, and no on-page
prominence addresses it. So **the browser tab is the channel**: the tab title and the favicon
change when step 5 needs an answer and when the fit lands.

The tab title is therefore designed text, not the app's name. `Check the notes` while the
check is open; the Patch's name otherwise.

---

## 4. Play surface

Everything from Start onward. Layout settled by prototype:
[`06-play-surface.prototype.html`](.scratch/app-design-flow/prototypes/06-play-surface.prototype.html),
variant E.

Top to bottom, right of the rail:

1. **The comparison frame.** One frame holding the Clip's spectrum as a filled ghost with
   this Patch's live spectrum drawn inside it. A legend, and a segmented toggle choosing
   which one you hear.
2. **Five macro knobs** in a row: Brightness, Movement, Width, Body, Attack.
3. **The device line**, right-aligned and quiet.
4. **The keyboard**, roughly 120px, full width.
5. **The evidenced-span bracket** beneath it.

### 4.1 The comparison frame

The visuals are **diagnosis, not decoration**. There is one picture on the screen and its
whole job is the judgement `VISION.md` sets as the bar: whether anyone would call this a
different sound. Nothing is drawn that does not serve that.

A/B is structure, not a button. The Clip is on screen permanently, so comparing costs no
click and cannot be forgotten; the toggle only chooses what comes out of the speakers.

**The compare control is a segmented toggle and must never be a slider.** A crossfader
sitting above five macro sliders reads as a sixth macro, and in an app where macros are the
only surface, a control that looks like a macro and is not one is a defect. The macros are
knobs for the same reason: nothing else on the screen is round, so nothing else can be
mistaken for one.

During the stand-in the toggle names the stand-in, not "this patch".

### 4.2 The ear check (step 5)

Not a screen. The rail's step 5 becomes a card in place.

- It plays the Transcription on a **neutral tone, mixed with the Clip**, with a balance
  control. Played simultaneously, a wrong chord announces itself as beating and dissonance.
- Alternating playback was rejected: it makes the user hold two things in memory. Playing
  the stand-in Patch instead of a neutral tone was rejected: when it sounds wrong you cannot
  tell whether the notes or the timbre are at fault, and at that point the timbre definitely is.
- **No piano roll, no notation, anywhere in the flow.** The Transcription is machinery.
- **No confidence signal.** The user listens and judges. Some of this is provably
  undecidable, so a confident-looking score would sometimes be confidently wrong.
- **The fit has already started**, optimistically, at the moment the card appeared. Saying no
  kills it.
- The card may be answered an hour late, and must survive that.
- **An unanswered check is dropped, not deferred**, once the fit lands. Its whole purpose was
  to save the forty minutes; once they are spent it has no job, and the A/B shows wrong notes
  better than the check ever could.

**Yes** → the card returns to a completed step, nothing else changes.

**No** → state `S-reject`. The fit is killed. The only way forward is a **MIDI file already
aligned to the Clip's timeline**. There is no proceeding on notes you have said are wrong and
no playing them in. The app says plainly that this needs a DAW and that most people will not
have one, rather than dangling a path they cannot take. See the gap in §7.1.

### 4.3 The keyboard

- Roughly five octaves, C1 to C6, playable by mouse and by the computer keyboard's home row.
- **It is a real instrument, not a placeholder.** faustwasm's poly node takes `keyOn` and
  `keyOff` directly, so MIDI is one caller among several and the on-screen keyboard is a
  first-class input.
- It mirrors incoming MIDI, so it doubles as the device indicator.
- Voices: 8 is safe, 12 is the ceiling, 16 glitches. At 4.3% of a core per sounding voice
  plus 3.5% for the shared effects.

### 4.4 The evidenced span

The **played range of fundamentals** from the Transcription, marked exactly, no margin.

Rendered as a **bracket beneath the keys** with its sentence attached to the left end: *your
clip played C2 to G3*. It lives below the keys so it survives them lighting up over it.

Two markings were tried and rejected. **Tinting the keys in range reads as a stain**: the
keyboard looks damaged rather than annotated, which fails the never-a-warning rule while
using no warning colour at all. **Bare endpoint note names read as nothing** to anyone who
was not told what they are.

- Present only once a Version exists. The stand-in claims no range, because it was never
  fitted to one.
- Worded as a fact about the Clip. No hedge attached to the Patch, no warning colour, no word
  like unreliable or extrapolated. Keys outside the span are fully playable and unmarked.
- Spectral coverage is **never shown**, though the fit uses it. A single held C2 lights four
  octaves of bands with its harmonics, so a coverage-derived reading turns the weakest
  evidence into the widest claim: those bands had their gains chosen to shape quiet partials
  sitting between other partials, and a loud fundamental there is exactly the untested case.
- Velocity gets no surface, because it is not fitted at all. See §7.2.
- Identical on every load. A barely-evidenced Patch gets no badge on return.

### 4.5 The device line

One quiet line, right-aligned above the keyboard. Never a banner, never blocking.

Four states on one surface:

| state | line |
|---|---|
| granted, device present | the device name, e.g. `Roland A-49 connected` |
| granted, no device | `No MIDI keyboard. Play the keys below.` |
| never asked | reads as a button: `Connect a MIDI keyboard` |
| denied | says so, and does not ask again |

- **Web MIDI is requested only on an explicit user action, never on load.** The whole API has
  been behind a permission prompt since Chrome 124 and the prompt cannot be pre-empted.
  `navigator.permissions.query({name:"midi"})` separates never-asked from denied, so the app
  never prompts into a wall.
- **SysEx is never requested.** The app has no use for it and asking makes the prompt scarier.
- Hot-plug is a live subscription to `statechange`, not an enumeration at load, so plugging a
  keyboard in at any moment just works.

### 4.6 Play-surface states

| state | rail | macros | span | notes |
|---|---|---|---|---|
| `P-standin-transcribing` | `R-transcribing` | all disabled, *waiting for the fit* | absent | stand-in playable within ~5 s of Start |
| `P-standin-check` | `R-check` | all disabled | absent | |
| `P-standin-fitting` | `R-fitting` / `R-slow` | all disabled | absent | |
| `P-fitted` | `R-done` | live | shown | one Version exists |
| `P-fitted-poor` | `R-done` | live | shown | poor-match label, §6.1 |
| `P-failed` | `R-failed` | all disabled | absent | zero Versions, permanent stand-in |

The stand-in is a hand-made **factory default for the chosen architecture**, named in words as
a placeholder and explicitly not your sound. It is not a Version, cannot be pinned, linked or
returned to. Nothing cheaper was viable: the seed alone measures cos theta 0.543 against 0.514
after eight minutes of CMA-ES, and a fast EQ pre-solve lowers loss by 0.78 while lowering cos
theta to 0.487. Anything that is a fit in progress sounds worse than something that never tried.

**Consequence, stated because it follows from decisions rather than being chosen:** for the
forty minutes before the swap, the user can play and can change nothing. See §7.3.

### 4.7 The swap

When the fit lands and the user is present, the patch **morphs under their fingers**, every
parameter ramped over **~186 ms**. Measured at 0.1x the signal's own motion, against 1.3x for
an instant jump, which clicks. All 55 parameters can be replaced under a held note this way.

An Architecture change cannot preserve held notes at all, which is why the Architecture is
fixed for the life of a Patch and is chosen before first playability.

The Position rides through as offsets, but the fitted Patch declares its own macro excursions,
so the same offset means a different amount either side of the swap and must be ramped with
everything else.

Design for absence: forty minutes is a walk or another task, and presence is the lucky case.
The arrival has to work when it is discovered later.

---

## 5. The macros

Five, bipolar, zero is the sound as fitted.

| macro | from → to | typically moves |
|---|---|---|
| Brightness | dull → bright | filter cutoff and the EQ's spectral tilt |
| Movement | static → animated | LFO depth and rate, oscillator drift, chorus depth |
| Width | mono → wide | stereo spread, chorus and reverb width |
| Body | thin → fat | sub oscillator level, low shelf |
| Attack | soft → immediate | amplitude and filter envelope attack and release |

Named after complaints a musician actually voices. Fewer than five and "too thin" has nowhere
to go; more and you are facing a panel again.

- **The Fit declares each macro's excursion, per Patch**, chosen so the extreme is still a
  usable version of this sound. A fixed global mapping is unusable across architectures.
- **A macro the Patch cannot honour is shown disabled with a reason.** Requiring every
  architecture to honour all five was rejected: it puts a floor on architecture complexity and
  therefore on per-voice cost, against a budget of eight safe voices. A disabled macro must
  never read as broken.
- **Macros are the only surface. The full parameter set is not visible in the app.** The Faust
  source export is the escape hatch for anyone who wants the 55 values. This closes softly:
  showing them later is a widening, and editing them would mean direct edits write a Version,
  since they change the base the macros offset from.
- Controls are sampled once per 2.9 ms block and do not zipper.

**Width is doing more work than it looks.** The project's loss is mono and cannot see stereo
width at all: the reference render is 4.7 dB too narrow, and correcting it costs as much loss
as the entire EQ win. Width is the user's compensation for something the objective is
structurally blind to, and may be the most used control on the surface.

---

## 6. When it goes wrong

Two different failures, kept apart on purpose. §6.1 is a fit that happened and does not sound
right. §6.2 is a fit that did not happen.

### 6.1 The fit came back wrong

The quality gate is a **per-clip frame-shuffled control**: for each Clip, score a
deliberately-wrong version of that same Clip and require the fit to beat it by a margin. A
fixed threshold assumes every clip scores on the same scale, and the calibration says they do
not. The control needs a score but no render, so it is nearly free.

**A failing fit still swaps in, labelled.** Converged cos theta was 0.720 against 0.698
shuffled, a margin of 0.022 on what `CLAUDE.md` calls a weak ranker, and a number that thin has
no business out-arguing someone who can listen.

The label states what was measured, points at the comparison frame, and offers exactly one
action, **try a different part of the clip**, which returns to S2. **It is dismissible, and
dismissal is final.** Once you have listened and disagreed, the app stops saying it.

**There is no refit.** Same Clip plus same Architecture is the same reachable set, and the
reachable set is the constraint, not the optimiser, three rebuilds in a row. Nor is the search
leaving value for a second attempt: CMA-ES scored 0/8 recovering targets the synth itself had
rendered, because a loss around 1.3 carries no parameter information. A "try again" button
sells forty minutes against a wall.

**There is no hint channel and no form.** Every complaint a musician would type either names
one of the five macros, where the fix is already on screen and instant, or names something the
architecture cannot reach, where a reweighted refit cannot reach it either. The axis users
complain about most, width, is the one the objective cannot see.

The four recovery paths, in full:

1. **Push a macro.** Instant.
2. **Try a different part of the clip.** Back to S2. A new Patch; the old one is untouched.
3. **Start over with a different clip.** Always available.
4. **Leave with it anyway.** A poor fitted Patch still exports, and still beats a factory
   default by being about your Clip.

**Nothing is ever lost, because nothing is replaced.** A retrim makes a new Patch at a new URL.
Macro Positions are deliberately **not** carried across: offsets are relative to fitted
defaults and each fit declares its own excursions, so the same numbers would mean a different
amount on the other side and would silently change the sound the user had found.

### 6.2 The machinery failed

| failure | what happens |
|---|---|
| file will not decode | refused at S1, before upload, with the reason |
| file has no signal | refused at S1 |
| Transcription finds no notes | said at step 4, back to S2 |
| notes rejected, no MIDI | `S-reject`, §4.2, an honest dead end |
| fit dies server-side | **retries once, visibly**; the rail says so |
| fit dies twice | `R-failed` / `P-failed`; offers to start over from S2 |
| connection drops | `R-reconnecting`. **Not a failure**: the Fit is server-side and still running |
| tab closes | nothing is lost; return by the list or the link |
| fit finishes with nobody connected | not an error; it waits |
| non-Chromium browser | plays through the on-screen keyboard; one line saying hardware keyboards need Chrome or Edge |
| autoplay blocked | never seen: Start unlocks it inbound, first key press unlocks it on a cold URL. **There is never an enable-sound modal** |

A silent retry would double a wait the rail put a number on, so it is announced in the rail's
own vocabulary.

**"No refit" and "retry a dead fit" are not in conflict.** No refit means no second attempt at
a fit that *completed* and has therefore already explored its reachable set. A fit that died
explored nothing.

**A Patch whose fit died is a Patch with zero Versions**, permanently on the stand-in. It is not
deleted and not hidden: it stays in the list, it plays, and it says what happened.

**No browser is refused.** The instrument never needed Web MIDI. Chromium-only is a decision
about what gets designed and tested, not a lock, and shutting out a browser that can genuinely
play the sound would invent a limit that does not exist.

---

## 7. Persistence, the URL and the exits

**The URL is the credential, and it carries the audio.** Someone who opens a Patch link gets
exactly what the maker gets: the Patch playable, the Clip audible in the comparison frame, the
evidenced span, the version list. With no accounts the app cannot tell a maker from a visitor
without inventing a token that is an account in disguise. Handled by unguessable ids and one
plain sentence in the app, not by a permissions model. Version pinning is an optional suffix on
the same URL.

**Server holds** the Patch, its Versions, the Clip, and a running Fit. Reloading mid-fit rejoins
it, which is what makes forty minutes survivable.

**The browser holds the Position.** This gives *keep this* a real job, making a tweak durable and
portable, and it stops a visitor on a shared link from silently clobbering the macro positions
the maker had found.

**MIDI needs no persistence design.** Chromium remembers the permission per origin, so the
device re-resolves on reload and the app never asks twice.

**Getting back** is the local **your patches** list on S1, plus the URL. The fit landing while
you are away arrives through the tab title and favicon. Browser notifications were rejected as a
second permission prompt on a page that already asks for Web MIDI; mailing a link was rejected
as needing an address, which is an account in disguise.

**The Clip lives as long as the Patch.** This is forced rather than chosen: the comparison frame
makes the Clip the permanent backdrop of the play surface, so discarding it does not degrade
that screen, it breaks it. The only thing that removes a Clip is deleting its Patch, which is
the single destructive action the app offers, and the confirmation says the Clip goes with it.

**Two exits, both on the play surface from the moment a Version exists.**

- **Faust source. This works.** The server compiled the Patch, so it can hand over the text.
  `VISION.md` promises that if the service disappears your sounds do not, and §5 leans on this
  export as the escape hatch for the 55 parameters the app refuses to show, so it cannot be a
  stub without making that decision dishonest.
- **Plugin build.** A named button that says what it will do and that it does not do it yet.
  Hiding it would be worse: the promise is part of why someone invests forty minutes. VST and AU
  packaging is out of scope; this is where it attaches.

---

## 8. Gaps

Named rather than quietly resolved. Calling these out was the stated reason for the exercise.

### 8.1 The transcription escape hatch excludes the person it was built for

Rejecting the Transcription requires a MIDI file already aligned to the Clip's timeline, which
means a DAW and a sequenced part. The documented case in this repo is an author who wrote the
part in GarageBand, could not export MIDI at all, and typed the chord qualities into a Python
file by hand.

The considered alternative was to take pitches from any MIDI file in any timing, and alignment
from the app's own onset detection, which is exactly the split the evidence supports: stage 1
here got the timing exactly right and the chord qualities wrong. Rejected to keep the first
version simple. Accepted as a gap, not an oversight.

### 8.2 Velocity does nothing but change the volume

`process = filtered * aenv * gain` with `gain` as the Faust polyphonic voice slider, so playing
harder is a fader move: no brightness, no envelope change, no drive. `VISION.md` names velocity
as something the Patch must survive.

Deliberately given no surface, because a velocity readout would advertise a response that does
not exist. It is a contract line for the fit (§9), not a UI problem.

### 8.3 The stand-in is playable but not adjustable

The Fit declares each macro's excursion and the stand-in has no Fit, so all five macros are
disabled for the forty minutes before the swap. This follows from decisions already taken rather
than being chosen, and it had never been said out loud.

### 8.4 The patch list is per browser profile

Losing the profile or moving machines leaves only links saved elsewhere. The honest shape of
single-user with no accounts. A two-way door: accounts later replace the list with a synced one
and migrate nothing.

### 8.5 A Patch link hands over the Clip

Anyone with the URL can hear the audio that was uploaded. Accepted, and handled by saying so.

### 8.6 The rail says "Play it" is still to come while you are playing

The rail's seven steps read as a queue at exactly the point where the user is already playing the
stand-in. Wording problem, unresolved, and the play surface is where it becomes visible.

### 8.7 The cost disclosure rests on one grey subtitle

"About 40 minutes" is the largest ask in the product and it lives in a rail subtitle. Needs a
decision when the visual language is settled.

### 8.8 Visual language

Typography, colour, the look of the thing. In scope for a first design and not yet done. Both
prototypes are deliberately flat and decide nothing about it.

### 8.9 How many architectures there are, and who picks

This version has exactly one, chosen by the app, with no choice offered, which is what makes "no
refit" correct in §6.1. The domain model says one Clip per Fit and one Fit per Patch, but never
says a Clip has only one Fit, so "fit this clip again, differently" is already legal and produces
a second Patch beside the first. That is where a choice of architecture attaches when there is
more than one.

---

## 9. The contract the fitting side must satisfy

What the app needs back, in what order, within what time.

### 9.1 Sequence and timing

| # | the app needs | by when |
|---|---|---|
| 1 | an Architecture chosen and fixed | before first playability |
| 2 | a stand-in: a hand-made factory default for that Architecture, compiled | ~5 s after Start |
| 3 | a Transcription: notes on the Clip's timeline, plus a neutral-tone render for the ear check | ~3 min |
| 4 | attempts-so-far and attempts-budget, streamed | continuously during the fit |
| 5 | the fitted parameter vector | ~40 min |
| 6 | per-macro excursions, or an unsupported flag with a reason string | with 5 |
| 7 | the fit's score and its Clip's frame-shuffled control score | with 5 |
| 8 | elapsed time | with 5 |
| 9 | Faust source text for the Patch | on demand |

The evidenced span is derived from the Transcription and needs nothing extra.

### 9.2 Requirements on the fit itself

- **Pitch-dependent behaviour must generalise past the played range.** The current
  architecture's timbre is a 26-band EQ at fixed absolute centres, measured as beating the
  harmonic-number alternative so decisively that the wavetable was removed for contributing
  exactly zero. That makes register-dependence the default: transpose an octave and every
  harmonic lands in different bands. Only `kbdTrk`, one exponent on the filter cutoff, tracks
  pitch at all. `VISION.md`: a patch that only works in the register it was fitted to is an
  impression of one. No amount of honest labelling fixes it.
- **Velocity must reach something other than the output gain**, or the app ships a keyboard
  instrument that ignores how it is played.
- **A macro that cannot be honoured is declared unsupported with a reason**, never silently
  flattened.
- **The control score is per Clip**, not a global threshold.
- **Nothing intermediate is ever offered as sound.** Every rung below convergence measures worse
  than deliberately-shuffled audio.

### 9.3 Delivery and budget

- **Compile server-side.** 118 KB raw, 29 KB gzipped per patch. Client-side libfaust is 5.5 MB
  plus 966 ms and buys nothing.
- **8 voices safe, 12 the ceiling.** 4.3% of a core per sounding voice, 3.5% for the shared
  effects, ~2.8% for the 26-band EQ.
- **Parameters ramp over ~186 ms.** All 55 can be replaced under a held note.
- **No in-place recompile and no access to voice state.** An Architecture change cannot preserve
  held notes, which is why §4.7 fixes the Architecture for the life of a Patch.

### 9.4 Known defect in the current pipeline

The chorus delay is specified in samples and is therefore rate-dependent, while the delay and
reverb are not. Raised separately; it will bite when the browser runs at a rate the fit did not.

---

## 10. Out of scope, and where each would attach

| | where it attaches |
|---|---|
| **Source separation**, pointing at a sound inside a mix | before S2; would replace the trim step with a second research project |
| **Link import** from a streaming service | S1, beside the drop zone |
| **Accounts, sharing, paid plans** | replaces the local list in §7 with a synced one; migrates nothing |
| **Plugin export mechanics**, VST and AU packaging | behind the stubbed button in §7 |
| **A patch library**, browsing and organising | grows out of the your-patches list in S1 |
| **Patches that were never fitted**: import, duplicate, build from scratch | widens one-Fit-per-Patch; costs empty states for the comparison frame and nothing else |
| **Mobile and responsive layout** | not designed for; nothing forecloses it |

---

## 11. Where each decision was argued

Tickets under `.scratch/app-design-flow/issues/`, map at `.scratch/app-design-flow/map.md`.

| ticket | settles |
|---|---|
| 01 how long does a fit take | the 42 minutes, and why there is no staircase |
| 02 what can the browser play | voices, ramping, compile path, Web MIDI reality |
| 03 the stand-in and the swap | §4.6, §4.7, the quality gate in §6.1 |
| 04 what is a patch | the domain model, `CONTEXT.md`, the URL |
| 05 the macro layer | §5 |
| 06 the play surface | §4, prototype `06-play-surface.prototype.html` |
| 07 the first screen | §2, §1, prototype `07-first-screen.prototype.html` |
| 08 when the fit comes back wrong | §6.1 |
| 09 persistence and the exits | §7 |
| 10 the progress rail | §3, prototype `10-progress-rail.prototype.html` |
| 12 where do the notes come from | §4.2, gap 8.1 |
| 13 how much did the clip evidence | §4.4, gap 8.2 |
| 14 when the machinery fails | §6.2 |

Research findings behind the numbers: `.scratch/app-design-flow/research/fit-timings.md` and
`.scratch/app-design-flow/research/browser-audio.md`. One note in the latter is superseded: its
warning about a fixed on-screen velocity hiding a defect is moot while gap 8.2 stands, and
becomes live again the moment the velocity contract line is honoured.
