# The play surface

Type: prototype
Status: resolved
Blocked by: 03, 05, 10

## Question

The screen you spend all your time on once the sound arrives. Build a rough, concrete
artefact to react to, not a written description.

Work out:

- The layout. Keyboard, macros, and whatever else earns its place, ranked by how often
  a person touches it.
- How you compare against the original clip. A/B against the source is the only way to
  judge "recognisably the same instrument", and it either has a place here or it does
  not exist.
- What the visuals show. `VISION.md` promises responsive visuals; decide whether they
  are decoration, feedback, or diagnosis.
- MIDI device state: connected, absent, hot-plugged. And the on-screen keyboard, which
  is a real instrument here, not a placeholder, and which mirrors incoming MIDI so it
  serves hardware players too.
- Marking the evidenced span. [How much of the keyboard did the clip actually
  evidence?](13-how-much-did-the-clip-evidence.md) settled it as a permanent, passive
  marking of the played range on the on-screen keyboard, present only once a Version
  exists, worded as a fact about the clip and never coloured as a warning. The look of
  that marking is this ticket's problem: it has to be legible without reading as an
  error, and it has to survive the keys lighting up while you play over it.
- What the surface looks like while you are on the stand-in and the real fit is still
  running behind it.
- Hosting the rail. [The progress rail](10-the-progress-rail.md) settled it as a full
  seven-step column beside the keyboard, roughly 250px, collapsing to one line once the fit
  lands. This layout has to carry that column alongside a keyboard and five macros without
  crowding any of them, and has to still look right once the column is one line.

Link the prototype from this ticket as an asset.

## Answer

**Variant E wins: C's comparison frame as the primary visual, A's bracket for the
evidenced span.** Prototype:
[06-play-surface.prototype.html](../prototypes/06-play-surface.prototype.html), five
variants on `?variant=A|B|C|D|E`, with a stage toggle for stand-in against fitted and a
MIDI toggle.

The screen, top to bottom: one frame holding the clip's spectrum as a filled ghost with
this patch's live spectrum drawn inside it, a row of five knobs, a quiet device line, a
120px keyboard, and the bracket beneath it. The rail sits to the left as a 250px column
while the fit runs and collapses to a single line above everything once it lands.

**The visuals are diagnosis, not decoration.** That was the open question in this ticket
and the comparison frame answers it by construction: there is only one picture on the
screen and its whole job is the judgement `VISION.md` sets as the bar, whether anyone
would call this a different sound. Nothing is drawn that does not serve that.

**A/B is structure, not a button.** The clip is on screen permanently, so comparing costs
no click and cannot be forgotten. The toggle only chooses which one you hear.

### Two choices the hybrid forced

**The compare control is a segmented toggle, not a slider.** In C it was a crossfader
sitting directly above five macro sliders, same shape and same row rhythm, and it read as
a sixth macro. Since macros are the only surface in this app, a control that looks like a
macro but is not one is a genuine defect, not a polish item.

**The macros are knobs.** Once the compare control is a toggle, knobs make the separation
structural rather than a matter of spacing: nothing else on the screen is round, so
nothing else can be mistaken for a macro.

### What the marking has to be, and what it cannot be

Ticket 13 settled that the evidenced span is marked permanently, passively, and never as a
warning. Building three different markings showed the band is narrower than it sounds.

- **Tinting the keys in range reads as a stain.** In B the evidenced black keys go
  brownish and the keyboard looks damaged rather than annotated. It fails the "never
  coloured as a warning" rule while using no warning colour at all.
- **Bare endpoint note names read as nothing.** In D, `C2` and `G3` printed under the
  keyboard are so quiet they carry no meaning to anyone who was not told what they are.
- **The bracket carries its own sentence and stays out of the way.** A rule under the keys
  spanning the span, with "your clip played C2 to G3" attached to its left end. It survives
  keys lighting up over it because it lives below the keys, not on them.

### Why the others lost

- **A, the instrument panel.** The layout is right and the bracket is its own, but its big
  spectrum is a spectacle with nothing to measure against, so it fills a third of the
  screen while answering no question. Its "hold to hear the clip" button also makes
  comparison an action you have to remember to take.
- **B, keyboard first.** Two vertical columns flanking the main area, the rail on the left
  and the macros on the right, squeezes the instrument from both sides, and the squeeze is
  worst in the stand-in state where the rail is widest. The macro column also could not
  hold five labels with disabled reasons without scrolling.
- **D, bare.** Putting the comparison behind a text link hides the one judgement the app
  exists to support. It is the right instinct about restraint applied to the wrong element.

### What the prototype exposed that nobody had decided

- **The stand-in is playable but not adjustable.** Ticket 05 says the fit declares each
  macro's excursion, and the stand-in has no fit, so all five macros come up disabled with
  "waiting for the fit". That follows from decisions already made, but it had never been
  stated: for the forty minutes before the swap you can play and you can change nothing.
- **The comparison frame labels the stand-in as "this patch".** During the stand-in the
  frame invites a comparison that is guaranteed to look bad against a sound explicitly
  labelled as not yours yet. The toggle needs to name the stand-in, or the frame needs to
  hold its ghost alone until a Version exists.
- **The rail says "Play it" is still to come while you are playing.** Ticket 10 settled the
  rail's seven steps, and this surface is where that reads as wrong. Not reopened here; it
  belongs to whoever writes the spec.

### The rest of the ticket's bullets

- **MIDI device state** is one quiet line, right-aligned above the keyboard: the device
  name, or "No MIDI keyboard. Play the keys below." Hot-plugging changes that line and
  nothing else. It is never a banner and never blocks.
- **The on-screen keyboard is a real instrument.** It mirrors incoming MIDI, so it is also
  the device indicator, and it is playable by mouse and by the computer keyboard's home row.
