# Vision

## The frustration

You hear a sound. A hook in a series, a pad under a scene, a bass in a track someone
sent you. You want to play it, right now, on your own keyboard.

The notes are usually the easy part. Most people can work out the melody, the chords and
the bass by ear or with a bit of patience. The sound is the wall. Recreating a patch
means learning synthesis properly, and by the time you have read enough to try, the
moment that made you want to play it is gone. Inspiration does not survive a research
project.

## What this becomes

A web app. You drop in a clip, from anywhere, and a few minutes later you are playing
that sound on your MIDI keyboard in the browser. A handful of macros let you push it
around, brightness, movement, width, without facing fifty parameters. When the idea
survives past the moment, you export it as a plugin and use it in a real production.

What comes back is a patch: a synth architecture plus the settings that go with it. Not
a rendering of the clip, not a sample of it. Something a person could have dialled in,
and can now keep dialling.

The clip demonstrates a few notes. The patch has to hold up across the whole keyboard,
at velocities and hold times the clip never showed. A patch that only works in the
register it was fitted to is not a patch, it is an impression of one.

## Why this is buildable now, and by one person

Three things changed.

AI coding agents make a project like this tractable when the target is well defined and
cheap to check. Sound matching is exactly that shape: render, score, compare. The agent
can iterate against a number instead of against taste.

Browsers grew up. Web MIDI works, Faust compiles to WebAssembly and runs in the audio
thread, and FFT analysis with responsive visuals is ordinary front end work now. The
playable half of this needs no native app.

Faust is platform agnostic, so the same DSP that plays in the browser compiles to VST
and AU. Getting a fitted patch into a DAW is a packaging problem, not a research one.

## The rules that settle arguments

**The patch is the deliverable, not the audio.** Anything that reproduces the sound
without producing controls a person could turn is off topic, however good it sounds.

**Neural models are allowed as means, never as output.** A model may propose the DSP
graph, predict parameters, or judge similarity. What reaches the user is still a synth
made of real controls.

**Minutes are acceptable, silence is not.** The wait can be long enough to make coffee
if the app is visibly working and shows what it found. What kills the moment is not
duration, it is having to go and study something.

**Recognisably the same instrument is the bar.** Not identical. Nobody should call it a
different sound.

**Synths first.** Pads, leads, basses, plucks. Nothing in the design should assume that
forever, but acoustic instruments and voices are not the problem being solved.

**The patch can always leave.** Readable Faust source and a plugin export. If the
service disappears, your sounds do not.

Off the table permanently: slicing the source into a sampler instrument, returning the
nearest existing preset without fitting anything, and any design that traps patches in
the service.

## How the pieces sit

Fitting runs on a server, because it is heavy and because an agent designing DSP for a
specific sound is not a client side job. The browser holds the playable half: Faust in
WebAssembly, Web MIDI, the macros, the visuals.

Notes and source separation are means, not ends. If getting a clean look at the sound
means transcribing the part or pulling it out of a mix first, fine, but nobody is here
for a transcription.

## Where this repository fits

This is the reference and the testbed. One eighteen second pad, fitted properly against
a real Faust renderer, with the method written down and the negative results kept. It is
where a technique gets proven against a known hard target before it goes anywhere near
the app, and it is the ground truth for what the loss and the renderer should be.

Read `CLAUDE.md` for how the current pipeline works and which doors are already closed.

## Ambition

This is a side project, built first for its author. It should be free to grow: nothing
in the architecture should make it awkward if other musicians want it, or if some of it
ends up behind a paid plan one day.
