# ai-synth

Reverse-engineering a playable synth patch from a recording, and the web app that lets a
musician do it with a clip and a keyboard. See `VISION.md` for what the product is and
`CLAUDE.md` for how the current pipeline works.

## Language

**Clip**:
The audio a user drops in, trimmed to an isolated part of a single instrument. The target
a Fit is measured against.
_Avoid_: sample, recording, source. "Sample" means a playable audio instrument in this
domain, which is explicitly off the table.

**Fit**:
The attempt to derive a Patch from a Clip. An optimisation against a target, not a lookup
in a library of existing sounds.
_Avoid_: match, search, analysis

**Patch**:
An Architecture, its parameter values, and its Macro mapping. What a person could have
dialled in on a synth, and can now keep dialling. The Clip and the recovered notes are
not part of it; they belong to the Fit.
_Avoid_: preset, sound, instrument

**Architecture**:
The shape of the synth a Patch is built from: which oscillators, filters and effects
exist and how they are wired. Fixed for the life of a Patch.
_Avoid_: graph, topology, engine

**Version**:
One saved state of a Patch's parameters. Written by each stage of a Fit, and by the user
explicitly keeping a Position they like. A Patch is one identity that gains versions.
_Avoid_: revision, snapshot, save

**Position**:
Where the Macros currently sit, held as offsets from the Version's fitted defaults, so
all-zero is the sound as fitted. Live and unsaved until it is kept as a Version.
_Avoid_: state, settings, tweak

**Transcription**:
The note list derived from a Clip, on the Clip's own timeline. Machinery, not output: it
exists so a Fit has something to render against the target, and it never reaches the
player, who plays their own keys.
_Avoid_: score, notation, the MIDI. MIDI is a file format, not this.

**Macro**:
One of a small fixed set of controls exposed on every Patch, each mapped onto whatever
parameters the Patch has. The whole control surface a player is asked to learn.
_Avoid_: knob, control, parameter. A parameter is a single value inside the Patch; a
macro is the musical intent that moves several of them.
