# Where do the notes come from?

Type: grilling
Status: resolved
Blocked by: none

## Question

Surfaced by [How long does a fit actually take?](01-how-long-does-a-fit-take.md). The
whole flow assumes a Clip goes in and notes come out, silently and correctly. That
assumption does not currently hold. Stage 1 is frozen in this repo partly because a human
supplied the chord qualities; run automatically today, the chord search stacks three
adjacent semitones in the bass where the answer is a D flat minor voicing.

Settle:

- What the app does when transcription is wrong, given the user can hear that it is wrong
  immediately and the fit downstream is wasted effort if it is.
- Whether notes are ever shown to the user at all, or stay machinery. If they are shown,
  where, and can they be corrected.
- Whether a user could supply the notes instead, by playing them in or dropping a MIDI
  file, and whether that is a fallback or a first-class path.
- What this costs the promise that you drop a clip and play the sound. Every note-editing
  surface is a step between the two.
- Whether a wrong transcription is even detectable automatically, or the user is the only
  detector.

This is a design question about the flow, not a request to reopen stage 1.

## Answer

Three facts from the source shaped every decision here. The Transcription **never reaches
the player**: it exists only so the fit has something to render against the target, and
the user plays their own keys. Some of it is **provably undecidable**: `final_transcription.py`
records that the sus2-versus-major distinction in the reference clip cannot be resolved
by measurement, because Bb3 and Ab4 coincide with harmonics 5 and 9 of the Gb bass.
And transcription **is itself a fit**, roughly 163 s over six regions, not a cheap
preprocessing step.

**Automatic, with an ear check before the expensive part.** The app transcribes, then
plays its Transcription back on a neutral tone **mixed with your Clip**, with a balance
control, and asks whether those are the right notes. Played simultaneously a wrong chord
announces itself as beating and dissonance, and the failure that prompted this ticket,
three adjacent semitones stacked in the bass, would be unmissable inside a second.
Alternating was rejected because it makes the user hold two things in memory; playing the
stand-in patch instead of a neutral tone was rejected because when it sounds wrong you
cannot tell whether the notes or the timbre are at fault, and at that point the timbre
definitely is.

**No piano roll, no notation anywhere in the flow.** The Transcription is machinery.
Exposing it as something to read and edit is the research project `VISION.md` says kills
the moment.

**No confidence signal.** The user listens and judges. A confidence number the app cannot
justify makes people defer to it, and here the human is the better detector: the specific
failure in this repo was obvious to an ear and was not caught by the search that produced
it, and there is a documented case where the distinction is undecidable, so a
confident-looking score would sometimes be confidently wrong. Revisit if a real
confidence measure ever exists.

**The fit starts optimistically**, at the same moment the check appears, rather than
waiting for the answer. Saying no kills it and starts again. This wastes compute in the
minority case and wastes none of the user's time in the majority one, and it removes the
perverse incentive where the app must interrupt your playing to save you time. The spec
must handle the check being answered late, including an hour late, and saying no has to be
honest that it means starting over.

**The gate is hard.** Reject the notes and the only way forward is a MIDI file already
aligned to the clip's timeline. There is no proceeding on notes you have said are wrong,
and no playing them in.

**Wide pitch coverage, monophonic where possible. Revised, see below.** The app prefers a
line that moves across the register over a dense chord passage, and accepts anything,
including a single held note.

The original wording here was "rich clips, not simple ones", justified by `CLAUDE.md`'s
result that a fit made on a single chord produced a comb tuned to that chord's partials,
0.153 worse on another chord. That result is real but it was measured on a different
question, which window of an 18 s clip to fit on, and "rich" bundled four separable
things: distinct pitches spread across the register, which helps by mechanism; polyphony,
which hurts transcription; timbral movement, on which there is no evidence; and length,
which costs compute linearly. Only the first was wanted. A monophonic line spanning two
octaves wins on both axes at once, so the trade against transcription risk was invented,
not measured.

**Single-note clips are accepted, with the consequences shown, never silently.** A single
note is the extreme of the comb failure: fewer partials means more spectrum for the 26
fixed-frequency bands to fill freely, and `CLAUDE.md` already records that the EQ fit
walks into the bank's near-singular alternating direction unless curvature is penalised.
But a single low note is not sparse where it matters: a 130 Hz fundamental puts harmonics
every 130 Hz, dense above ~500 Hz against 26 log-spaced bands, so it constrains the
timbre reasonably and the bass badly. What it cannot constrain at all is variation across
the keyboard: filter key tracking, whether a fitted cutoff is absolute or relative,
envelope feel two octaves up, velocity response. Refusing such clips is unusable, since
someone recreating a pad from a track takes the seconds the track gives them. The honest
response is [How much of the keyboard did the clip actually
evidence?](13-how-much-did-the-clip-evidence.md), graduated from the map's fog by this
discussion.

**Confidence level, for the spec.** All of the above is a mechanism argument resting on
one measurement, not a result. It is worth a cheap experiment: fit the same patch against
a one-chord excerpt, a single note, and a wide-range excerpt, and compare how each holds
up outside its own range.

## Known gap, accepted deliberately

The escape hatch excludes the person it was built for. Rejecting the Transcription
requires an aligned MIDI file, which means a DAW, the clip imported and the part
sequenced in time. The documented case in this repo is an author who wrote the part in
GarageBand, **could not export MIDI**, and typed the chord qualities into a Python file
by hand.

The considered alternative was to take pitches from any MIDI file in any timing and
alignment from the app's own onset detection, which is exactly the split the evidence
supports: this repo's stage 1 got the timing exactly right, +3.00 semitones and a 13.25 s
midpoint both exact, and the chord qualities wrong. It was rejected to keep the first
version simple. Recorded here so it is a decision and not an oversight.

Consequence for the spec: rejecting the Transcription is a dead end for most users, and
the app has to say so honestly rather than dangling a path most people cannot take.
