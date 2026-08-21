# The macro layer

Type: grilling
Status: resolved
Blocked by: 02, 04

## Question

A fixed universal macro set is already decided. What is in it, and what it is allowed to
do, is not.

Settle:

- Which macros. `VISION.md` names brightness, movement, width as examples. Fix the list
  and defend each one: what musical intent does it serve, and what does a musician
  expect it to do.
- What each macro means in DSP terms, well enough that the fitting side can be held to
  it. Brightness is not one parameter, and the mapping has to survive different patch
  architectures.
- The default position. Does a freshly fitted patch sit at centre on every macro, or
  wherever the fit landed? Can you get back to the fitted sound after wandering?
- What happens when a patch cannot honour a macro, because its architecture has nothing
  to move. Hide it, disable it, or fake it?
- Whether the full parameter set is reachable at all, or macros are the only surface.
  `VISION.md` says "without facing fifty parameters", which is not the same as forbidding
  them.
- Whether macro positions are part of the patch, which depends on
  [What is a Patch?](04-what-is-a-patch.md).

## Answer

**Five macros**, each named after a complaint a musician actually voices about a fit that
came back close but not right:

1. **Brightness**: dull to bright. Filter cutoff and the EQ's spectral tilt, together.
2. **Movement**: static to animated. LFO depth and rate, oscillator drift, chorus depth.
3. **Width**: mono to wide. Stereo spread, chorus and reverb width.
4. **Body**: thin to fat. Sub oscillator level and the low shelf.
5. **Attack**: soft to immediate. Amplitude and filter envelope attack, and their release.

Fewer than five and "too thin" has nowhere to go. More and you are facing a panel again.

**Bipolar offsets, zero is the sound as fitted.** Inherited from [What is a
Patch?](04-what-is-a-patch.md), not re-decided here.

**The fit declares each macro's excursion, per patch.** When it produces a Patch it also
says how far each macro travels, chosen so the extreme is still a usable version of this
sound. A fixed global mapping is unusable across architectures, since the same cutoff is
bright on one patch and inaudible on another.

**A macro a patch cannot honour is shown disabled, with a reason.** The alternative
considered and rejected was forbidding the situation outright, by requiring every
architecture to honour all five. That would have put a floor on architecture complexity
and therefore on per-voice cost, against a measured budget of eight safe voices at 4.3%
of a core each. Disabling is honest and cheap; the cost is that the control surface is
universal in intent rather than in effect, and the spec has to say what a disabled macro
looks like so it never reads as broken.

**Macros are the only surface. The full parameter set is not visible in the app.** The
Faust source export is the escape hatch for anyone who wants the 55 values. This closes
softly: making parameters visible, or editable, is a later widening, and editing would
mean direct edits write a Version since they change the base the macros offset from.

**Worth flagging to whoever specs the fit.** This project's loss is mono and cannot see
stereo width at all, which is a known and measured limitation: the reference render is
4.7 dB too narrow and correcting it costs as much loss as the entire EQ win. Width as a
macro is therefore not just a nudge, it is the user's compensation for something the
objective is structurally blind to. It may be the most used control on the surface.
