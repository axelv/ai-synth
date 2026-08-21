# Map: first design of the app and its user flow

Label: `wayfinder:map`

## Destination

**Reached.** The spec is `SPEC.md` at the repo root. This map is now the record of how it was
argued, not a live tracker.


A written product spec of the end-to-end flow, from arriving at the app to playing the
fitted patch on a keyboard and pushing its macros, with every screen, state and
transition named, and the gaps in the flow called out explicitly. Buildable-from, and
concrete enough that the fitting research knows what contract it has to satisfy.

## Notes

**Domain.** Product and interaction design for the web app described in `VISION.md`.
Not DSP research. The fitting pipeline is a black box with a contract, and where this
map needs a fact about it, that is a research ticket, not a redesign.

**Skills every session should consult.** `grilling` and `domain-modeling` by default.
Prototype tickets also call `prototype`. Research tickets call `research`.

**Read first.** `VISION.md` for the rules that settle arguments. `CLAUDE.md` for what
the current pipeline is and which doors are already closed.

**Plan, don't do.** One exception, carried into the map deliberately: the final ticket
writes the spec, because the spec is the destination.

**Settled while charting** (constraints on every ticket, not decisions to revisit):

- Input is a single audio file the user has already trimmed to an isolated part. The app
  provides a trim step. Full mixes and separation are out of scope.
- **One honest rung, not a staircase.** A playable stand-in within seconds, explicitly
  labelled as not the sound yet, while the real patch is fitted behind you and swaps in
  when it clears a quality gate. Revised from the original staircase constraint after
  [How long does a fit actually take?](issues/01-how-long-does-a-fit-take.md) measured
  every intermediate rung as sounding worse than deliberately-shuffled audio, and measured
  the full fit at ~42 minutes rather than the promised minutes.
- MIDI keyboard, plus an on-screen keyboard fallback so the result is audible the
  instant it lands.
- A fixed universal macro set, mapped onto each patch. Not per-patch macros.
- One patch per fit. No candidate gallery.
- Hosted, single user, no accounts. Nothing may foreclose accounts later.
- A patch survives a reload, addressable by URL.
- Desktop Chromium only. No responsive work.

## Decisions so far

<!-- one line per closed ticket: gist plus link -->

- [Assemble the spec](issues/11-assemble-the-spec.md): written to **`SPEC.md`** at the repo
  root. Eleven sections; two surfaces plus a rail rather than a linear walkthrough. Nine gaps,
  up from four. The assembly itself caught one collision: "no refit" and "a dead fit retries
  once" read as contradictory until both were on one page, and the spec now states the
  distinction where they meet.

- [When the machinery fails](issues/14-when-the-machinery-fails.md): files are checked **in
  the browser before upload**, reusing the decode the trim step needs, and only two things are
  refused, will-not-decode and no-signal. A dead fit **retries once, visibly**, which needs a
  distinction ticket 08 did not draw: no-refit means no second attempt at a fit that
  *completed*. A Patch whose fit died is a Patch with **zero Versions**, permanently on the
  stand-in, kept and honest. Dropped connections and closed tabs are not failures at all. **No
  browser is refused**, because the instrument never needed Web MIDI; MIDI is asked for on an
  explicit action, never on load, and never with SysEx. The autoplay gesture is Start, or the
  first key press on a cold URL, and there is never an enable-sound modal.

- [Persistence and the exits](issues/09-persistence-and-exits.md): **the URL is the
  credential and it carries the audio**, since with no accounts the app cannot tell a maker
  from a visitor; unguessable ids plus one plain sentence, not a permissions model. Server
  holds the Patch, its Versions, the Clip and a running Fit, so a reload mid-fit **rejoins**
  it; the browser holds the Position, which gives "keep this" a real job and stops a visitor
  clobbering the maker's macros. Getting back is a **local list on the first screen** plus the
  link, and the fit lands through the tab title and favicon. The Clip lives as long as the
  Patch, forced by the comparison frame. **Faust export works today**; the plugin build is a
  named, honest stub.

- [When the fit comes back wrong](issues/08-when-the-fit-comes-back-wrong.md): **there is no
  refit**, because same clip plus same architecture is the same reachable set and the search
  scored 0/8 recovering the synth's own renders. **No hint channel and no form**: every
  complaint a musician would type is either one of the five macros, instant and free, or
  something the architecture cannot reach either way. The poor-match label states the
  measurement, points at the A/B, offers **one** action, try a different part of the clip,
  and is **dismissible for good**, because a 0.022 margin on a weak ranker should not
  out-argue someone listening. The app declines almost nothing: no content check, since
  refusing needs a classifier it does not have. A retrim costs nothing because it makes a new
  Patch and replaces nothing, and macro Positions are deliberately not carried across.

- [The play surface](issues/06-the-play-surface.md): one frame holding **the clip's spectrum
  with this patch's drawn inside it**, then five knobs, then a 120px keyboard with the
  evidenced span bracketed beneath. A/B is structure rather than a button, so comparing costs
  no click, and the visuals are **diagnosis, not decoration**: the only picture on the screen
  exists to answer whether anyone would call this a different sound. The compare control is a
  segmented toggle and the macros are knobs, because a crossfader above five sliders read as
  a sixth macro. MIDI state is one quiet line, never a banner. Prototype:
  [06-play-surface.prototype.html](prototypes/06-play-surface.prototype.html).

- [How much of the keyboard did the clip actually evidence?](issues/13-how-much-did-the-clip-evidence.md):
  evidenced means the **played range of fundamentals**, marked exactly on the on-screen
  keyboard, permanently and passively, and only once a Version exists. Spectral coverage is
  never shown: a single held C2 lights four octaves of bands with harmonics, so a
  coverage-derived reading turns the weakest evidence into the widest claim. Wording is a
  fact about the clip, "your clip played C2 to G3", with no hedge on the patch and no
  warning colour. Velocity gets no surface because it is **not fitted at all**, only
  loudness. Nothing is done beyond disclosure in the app; what came out instead is a
  contract line for the fit, below.

- [The progress rail](issues/10-the-progress-rail.md): the rail keeps a **full seven-step
  column** beside the keyboard, unchanged from setup, and collapses to one line once the
  fit lands. Progress on the long step is a bar driven by **attempts against budget**,
  never by loss. A step running long says "longer than usual" and keeps going, so this
  ticket owns *slow* and [When the machinery fails](issues/14-when-the-machinery-fails.md)
  owns *dead*. The note check reaches an absent user through the **tab title and favicon**,
  not through more on-page prominence. Prototype:
  [10-progress-rail.prototype.html](prototypes/10-progress-rail.prototype.html).

- [The first screen](issues/07-the-first-screen.md): one question per step, with a rail
  carrying all seven phases, setup and machine together, so "about 40 minutes" is disclosed
  before the user invests anything. The rail **stays on after Start** beside the play
  surface as a status display, because a wizard implies a queue and the user is playing,
  not waiting. Nothing asked but the clip and the trim. Clip guidance is one sentence, not
  a gate. Prototype: [07-first-screen.prototype.html](prototypes/07-first-screen.prototype.html).

- [The stand-in and the swap](issues/03-the-stand-in-and-the-swap.md): the stand-in is a
  hand-made factory default for the chosen architecture, named as a placeholder, and **not
  a Version**, so a Patch has no versions until the fit lands. The quality gate is a
  per-clip frame-shuffled control that **labels rather than blocks**: a poor fit still
  arrives, marked. Design for absence over forty minutes; if the user is present the swap
  morphs under their fingers, ramped over ~186 ms. An unanswered ear check is dropped once
  the patch exists.

- [Where do the notes come from?](issues/12-where-do-the-notes-come-from.md): automatic
  transcription, then an ear check that plays a neutral-tone render **mixed with the clip**
  before the expensive part. No piano roll, no notation, no confidence signal. The fit
  starts optimistically alongside the check and is killed if you say no. The gate is hard:
  reject the notes and the only way forward is an aligned MIDI file. On clip guidance the
  app prefers **wide pitch coverage, monophonic where possible**, and accepts anything
  including a single held note, never silently.

- [The macro layer](issues/05-the-macro-layer.md): five bipolar macros, Brightness,
  Movement, Width, Body, Attack, at zero when freshly fitted. The fit declares each one's
  excursion per patch. A macro the patch cannot honour is shown disabled with a reason,
  rather than forbidden by requiring every architecture to support all five. Macros are
  the only surface; the full parameter set is not visible in the app, and the Faust export
  is the escape hatch.

- [How long does a fit actually take?](issues/01-how-long-does-a-fit-take.md): one objective
  evaluation is 1.4 s, a full fit ~42 min on 8 cores. Two thirds of the loss improvement
  lands 24 s in, **but** measured cos theta goes 0.543 seed, 0.514 after eight minutes,
  0.720 converged, against 0.698 for deliberately-shuffled audio. Every rung below
  convergence sounds worse than a control built by destroying the audio. Earliest
  playability is ~5 s and is never artefact-blocked. **This killed the staircase**, replaced
  in Notes above by one stand-in plus one swap. Also found: there is no working per-request
  transcription, raised as [Where do the notes come
  from?](issues/12-where-do-the-notes-come-from.md).

- [What is a Patch?](issues/04-what-is-a-patch.md): a Patch is architecture plus parameters
  plus macro mapping, and nothing else. One identity gaining **versions**, written by the
  fit and by your explicit saves; the live unsaved macro state is the **position**, held as
  offsets from the version's fitted defaults. Architecture is fixed
  for the life of a Patch, chosen before first playability, because the
  browser cannot swap a graph under a held note. URL addresses the Patch, optionally
  version-pinned. Exactly one Clip per Fit, one Fit per Patch. Vocabulary in `CONTEXT.md`.

- [What can the browser actually play?](issues/02-what-can-the-browser-play.md):
  Compile server-side, 29 KB gzipped per patch. 8 voices safe, 12 ceiling. Parameters do
  not zipper, and all 55 can be replaced under a held note if ramped over ~186 ms, but an
  **architecture** change cannot preserve held notes at all: no in-place recompile, no
  access to voice state, which is why [What is a Patch?](issues/04-what-is-a-patch.md)
  fixes the architecture for the life of a Patch. Web MIDI is Chromium and Firefox only, no Safari, and
  behind a permission prompt.

## Known gaps

<!-- named, accepted, and owed a mention in the spec. The destination requires gaps to be
     called out rather than quietly resolved, and they need somewhere to live. -->

- **The transcription escape hatch excludes the person it was built for.** Rejecting the
  Transcription requires an aligned MIDI file, so it needs a DAW and a sequenced part. The
  reference case in this repo is an author who could not export MIDI at all. Accepted to
  keep the first version simple; see [Where do the notes come
  from?](issues/12-where-do-the-notes-come-from.md).

- **Velocity does nothing but change the volume.** `process = filtered * aenv * gain` and
  `gain` is the Faust polyphonic voice slider, so playing harder is a fader move: no
  brightness, no envelope change, no drive. `VISION.md` names velocity as something the
  patch must survive. Accepted for the first version and deliberately given no surface,
  because a velocity readout would imply a response that does not exist. See
  [How much of the keyboard did the clip actually
  evidence?](issues/13-how-much-did-the-clip-evidence.md).

- **The stand-in is playable but not adjustable.** The fit declares each macro's excursion
  and the stand-in has no fit, so all five macros are disabled for the forty minutes before
  the swap. This follows from decisions already taken rather than being chosen, and it had
  never been said out loud. Surfaced by [The play surface](issues/06-the-play-surface.md).
- **The rail says "Play it" is still to come while you are playing the stand-in.** The
  rail's seven steps are settled and the play surface is where the contradiction becomes
  visible. Left for [Assemble the spec](issues/11-assemble-the-spec.md) to word.

- **The patch list is per browser profile.** With no accounts, losing the profile or moving
  machines leaves only links saved elsewhere. A two-way door: accounts later replace the list
  with a synced one and migrate nothing. See [Persistence and the
  exits](issues/09-persistence-and-exits.md).
- **A Patch link hands over the clip.** Anyone with the URL can hear the audio that was
  uploaded. Accepted as the honest shape of single-user with no accounts, and handled by
  saying so rather than by a permissions model.

## Contract for the fit

<!-- what this map hands back to the fitting research. The destination requires the spec to
     be concrete enough that fitting knows what it has to satisfy. -->

- **Pitch-dependent behaviour has to generalise past the played range.** The current
  architecture's timbre is a 26-band EQ at fixed absolute centres, measured as beating the
  harmonic-number alternative so decisively that the wavetable was removed for contributing
  zero. That makes register-dependence the default: transpose an octave and every harmonic
  lands in different bands. Only `kbdTrk`, one exponent on the filter cutoff, tracks pitch
  at all. A patch that changes character with register is what `VISION.md` calls an
  impression of a patch, and no labelling fixes it.
- **Velocity has to reach something other than the output gain**, or the app ships a
  keyboard instrument that ignores how it is played.
- **The fit declares each macro's excursion per patch**, and declares a macro unsupported
  rather than silently flattening it. From [The macro layer](issues/05-the-macro-layer.md).
- **The fit reports a per-clip frame-shuffled control score**, which is what lets a poor fit
  arrive labelled instead of blocked. From [The stand-in and the
  swap](issues/03-the-stand-in-and-the-swap.md).

## Not yet specified

- **How many architectures there are, and who picks.** This version has exactly one, chosen
  by the app, with no choice offered, which is what makes "no refit" the right answer in
  [When the fit comes back wrong](issues/08-when-the-fit-comes-back-wrong.md). A second Fit
  over the same Clip is already legal in the domain model, so a choice of architecture has
  somewhere to attach when there is more than one.

- **Visual language.** Typography, colour, the look of the thing. In scope for a first
  design, but downstream of knowing what the screens are.

## Out of scope

- **Source separation and pointing at a sound inside a mix.** Ruled out with the input
  decision above: it puts a second research project inside the first-run flow.
- **Link import**, pasting a URL from a streaming service instead of a file.
- **Accounts, sharing, paid plans.** The flow must not foreclose them; designing them is
  a separate effort.
- **Plugin export mechanics.** VST and AU packaging. The exit is stubbed in
  [Persistence and the exits](issues/09-persistence-and-exits.md); what happens past the
  stub is not this map.
- **A patch library**, browsing and organising past patches.
- **Patches that were never fitted.** Importing one, duplicating one, building one
  from scratch. Ruled out with the one-Fit-per-Patch decision, and confirmed as a
  two-way door: allowing them later widens a constraint and migrates no data.
- **Mobile and responsive layout.**
