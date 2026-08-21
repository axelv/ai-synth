# What can the browser actually play?

Type: research
Status: resolved
Blocked by: none

## Question

The playable half runs Faust compiled to WebAssembly in an AudioWorklet, driven by Web
MIDI. `VISION.md` asserts this works. Establish what it actually costs, because the
answer bounds the play surface and the macro layer.

Find out:

- How a Faust DSP reaches the browser: compiled server-side to wasm and shipped, or
  compiled client-side with `@grame/faustwasm`. What each costs in bytes and in seconds
  before the first note sounds.
- Polyphony. How many voices of a supersaw-plus-sub patch of roughly this complexity a
  mid-range laptop sustains in an AudioWorklet before glitching.
- Parameter changes at audio rate. What happens when a macro is swept while notes are
  held: are parameters smoothed, do they zipper, is there a per-change cost.
- Whether the DSP graph can be swapped for a better-fitted one while the instrument is
  playing, without dropping held notes or clicking. The staircase needs this.
- Web MIDI reality: browser support, permission prompts, device hot-plug, and what the
  fallback path is when no device is present.

Record findings on a `research/browser-audio` branch and link them here.

## Findings

[What can the browser actually play?](../research/browser-audio.md)

## Answer

Full findings: [browser-audio.md](../research/browser-audio.md). This repo's actual
`synth.DSP` was exported, compiled with `@grame/faustwasm@0.16.7` and benchmarked offline
in Node at 128-frame blocks with the fitted `out/patch.json` values, on an Apple M1, so
every number below is an optimistic ceiling.

**Delivery.** Compile server-side. A precompiled patch is 118 KB raw, 29 KB gzipped.
Client-side compilation costs 5.5 MB for libfaust-wasm plus 966 ms to compile this DSP
as poly. The browser never needs libfaust.

**Polyphony.** 3.5% of one core fixed for the effect chain, plus 4.3% per sounding
voice. Idle voices are free. Real-time break-even is ~22 voices on an M1; derated for a
mid-range laptop with headroom, 8 is safe, 12 is the ceiling, 16 glitches. One thread,
no scaling out. The 26-band EQ alone costs ~2.8% of a core.

**Parameter changes.** Faust samples controls once per 2.9 ms block, nothing smoothed by
default. Macro sweeps do not zipper. Even a single parameter jumped to an extreme in one
block stays below the pad's own sample-to-sample motion.

**Swapping the patch under a held note.** Replacing all 55 parameters in one block
clicks; ramped over ~186 ms it does not. Changing the *architecture* is different: there
is no in-place recompile, no way to unregister a processor, and no API exposing
oscillator phase, filter memory or envelope state, so held notes cannot survive an
architecture change. The Faust IDE's own reference implementation disconnects, restores
parameters by path, and reconnects, and audio stops. Two workarounds exist: a two-node
crossfade, which doubles CPU, comb-filters, cuts the reverb tail and re-attacks held
notes; or an unbuilt but verified-feasible in-worklet module swap. The third option is
to avoid the problem: ship one superset architecture with identity defaults, the
append-only pattern this repo already uses for `PARAMS`, and send only parameter vectors.
Then held notes survive, nothing clicks, and each staircase step is kilobytes.

**Web MIDI.** Chromium and Firefox 108+ only. Safari has none. Since Chrome 124 the
entire API sits behind a permission prompt, not just SysEx. Hot-plug via `statechange`
works on Chromium. With the autoplay gesture requirement, the first screen has one
unavoidable click and one unavoidable permission modal.

**Trap found in passing, for the fitting side.** The chorus specifies its delay in
samples, so it is sample-rate dependent while the delay and reverb are not. An
AudioContext defaults to 48 kHz and the fit runs at 44.1 kHz, so the browser will not
play back what was fitted unless the context rate is forced or the chorus is made
rate-independent.
