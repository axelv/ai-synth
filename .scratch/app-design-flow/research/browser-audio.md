# What can the browser actually play?

Research for [02-what-can-the-browser-play](../issues/02-what-can-the-browser-play.md).

Everything below is either **specified** (a normative spec or first-party doc, cited),
**measured here** (I ran it, on this repo's real DSP, method and machine stated),
**widely reported** (first-hand engineering accounts, not specification), or
**unverified** (I could not find a source and did not measure it). Every claim is
labelled.

## How the numbers here were obtained

The DSP under test is `scripts/synth.py`'s `DSP` string exactly as the fitter uses it,
exported to a `.dsp` file (8,081 characters, 55 `PARAMS`), with `process` as the Faust
voice and `effect` as the shared effect chain, so `-poly` splits it the way Faust's own
polyphony expects. Parameters were set to the fitted values in `out/patch.json`.

Compilation and rendering ran through `@grame/faustwasm@0.16.7` (libfaust 2.86.2) under
Node v22.22.2 on an **Apple M1, 8 cores, macOS 26.2**, using
`FaustPolyDspGenerator.createOfflineProcessor(44100, 128, voices)`. Block size 128 was
chosen to match the browser's render quantum so per-block overhead is represented.

**Two caveats that apply to every measured number.** First, this is Node's V8 rendering
offline, not Chrome's audio rendering thread: there is no AudioParam marshalling per
block, no competition with the compositor, and no real-time deadline. Second, an M1 is
fast. A mid-range laptop should be assumed to be roughly half as quick, and the
WebAssembly-versus-native literature puts wasm at a mean **1.55x slowdown in Chrome**
against native code on SPEC CPU
([Jangda et al., USENIX ATC 2019](https://www.usenix.org/conference/atc19/presentation/jangda)),
which is already inside these numbers since they are wasm. Treat the measurements as an
optimistic ceiling and derate.

To reproduce: `npm i @grame/faustwasm`, export the DSP, then compile with
`FaustPolyDspGenerator` and render with `createOfflineProcessor`.

---

## 1. How a Faust DSP reaches the browser

There are two paths, and they are architecturally different, not just different in size.

### Path A: compile client-side with libfaust-wasm

The Faust compiler itself is shipped as WebAssembly. **Specified** by the FaustWasm
README: the compiler level "consists in 3 different files: `libfaust-wasm.wasm` : the
Faust compiler provided as a WebAssembly module, `libfaust-wasm.js` : a Javascript
loader, `libfaust-wasm.data` : a virtual file system containing the Faust libraries"
([README](https://github.com/grame-cncm/faustwasm)).

**Measured** payload of `@grame/faustwasm@0.16.7`:

| file | raw | gzip | brotli |
|---|---|---|---|
| `libfaust-wasm.wasm` | 3,102,755 B | 768,797 B | 457,324 B |
| `libfaust-wasm.data` | 2,263,644 B | 560,120 B | 391,109 B |
| `libfaust-wasm.js` | 177,063 B | 49,126 B | 42,283 B |
| `dist/esm/index.js` (runtime) | 196,246 B | 35,244 B | 28,722 B |
| **total** | **5,739,708 B** | **1,413,287 B** | **919,438 B** |

So client-side compilation costs roughly **0.9 MB brotli / 1.4 MB gzip** on the wire
before anything can be compiled.

**Measured** timings (M1, warm disk, Node):

| step | time |
|---|---|
| instantiate `libfaust-wasm` | 64 ms |
| compile this DSP, mono | 195 ms |
| compile this DSP, poly (voice + effect) | 966 ms |
| same via the `faust2wasm -poly` CLI | 1,556 ms |
| recompile identical source (sha-keyed cache hit) | 1 ms |

The poly path is five times the mono path because it compiles two separate DSPs: the
voice, and the 26-band EQ plus zita reverb plus delay plus chorus effect chain. The
compiler caches by sha key, so recompiling unchanged source is free.

In the browser add network transfer of ~0.9-1.4 MB and wasm compilation of a 3 MB
module. **Unverified**: I did not measure libfaust instantiation in Chrome, only in
Node; the browser figure will be larger than 64 ms because it includes streaming
compilation of the 3 MB compiler.

### Path B: precompile on the server, ship wasm

**Specified**: "The Faust Wasm and Audio Node levels make it possible to generate
instances from Faust dsp code as well as from pre-compiled WebAssembly modules. In the
latter case, it is not necessary to include the `libfaust-wasm.js` library, `index.js`
is sufficient to provide the required services. This allows to generate lighter and
faster-loading HTML pages." ([README](https://github.com/grame-cncm/faustwasm))

**Measured** output of `faust2wasm pad.dsp out -poly` for this exact DSP:

| file | raw | gzip |
|---|---|---|
| `dsp-module.wasm` (one voice) | 18,705 B | 6,162 B |
| `effect-module.wasm` (26-band EQ + reverb + delay + chorus) | 61,203 B | 18,597 B |
| `mixer-module.wasm` | 502 B | 301 B |
| `dsp-meta.json` | 14,330 B | 1,546 B |
| `effect-meta.json` | 23,617 B | 2,028 B |
| **per patch** | **118,357 B** | **28,634 B** |

Plus the faustwasm runtime, `dist/esm/index.js`, at 196 KB raw / 35 KB gzip, loaded
once.

**A patch is ~29 KB gzipped.** That is the number the staircase should be designed
around: shipping a new fitted DSP down the wire mid-session is a 29 KB transfer, not a
megabyte. The 5.5 MB compiler only earns its place if the browser has to compile source
it was not given a wasm build of.

**Recommendation implied by the numbers**, not a finding: fitting already runs on a
server (`VISION.md`), the server already has libfaust, so Path B is right for the
staircase and Path A buys nothing except the ability to compile without a round trip.

### Time to first note

**Specified**: an AudioContext will not start without user activation. MDN's autoplay
guide states Web Audio is "subject to autoplay rules" and that autoplay is allowed when
"the user has interacted with the site"
([MDN Autoplay guide](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Autoplay)).
So the first note is gated on a click regardless of how fast the DSP arrives.

**Measured/derived** for Path B: ~29 KB transfer, plus `audioWorklet.addModule` of a
generated blob, plus node construction. Instantiating the poly DSP itself is
sub-millisecond (see §4). The dominant term is the network and the user's click, not
the audio stack.

---

## 2. Polyphony

### What is specified

- The render quantum is **128 sample frames** (Web Audio API, `[[render quantum size]]`
  defaults to 128) — [spec](https://webaudio.github.io/web-audio-api/#rendering-loop).
  Chrome's own guidance: "The timing budget for the stable audio stream is quite
  demanding: it is only 3ms at the sample rate of 44.1Khz... If the code cannot finish
  the task within the timing budget of render quantum (~3ms at 44.1Khz), it will affect
  the onset timing of subsequent callback function and eventually cause glitches."
  ([Audio Worklet Design Pattern, Chrome](https://developer.chrome.com/blog/audio-worklet-design-pattern))
- All AudioWorkletProcessors run on **one rendering thread**
  ([spec](https://webaudio.github.io/web-audio-api/#AudioWorklet)). Voices cannot be
  spread across cores. Whatever the instrument costs, it costs it on one core.
- Faust computes only sounding voices. The Faust manual notes that a naive approach
  where "all voices would always be computed... could be too CPU costly"
  ([Faust MIDI/polyphony manual](https://faustdoc.grame.fr/manual/midi/)), and the
  faustwasm source confirms it: the per-block loop skips any voice whose `fCurNote` is
  `kFreeVoice`, and frees a releasing voice once its measured level drops below
  `VOICE_STOP_LEVEL` (`FaustWebAudioDsp.ts`).
- The effect chain is computed **once**, not per voice: "a single instance of the effect
  defined in effect will be created and shared by all voices"
  ([Faust MIDI manual](https://faustdoc.grame.fr/manual/midi/)).
- Voice stealing exists and is deterministic: faustwasm's `getFreeVoice` prefers a free
  voice, then steals the oldest releasing voice, then the oldest playing voice, logging
  "Steal release voice" / "Steal playing voice" (`FaustWebAudioDsp.ts`).

### What I measured

Single core, 44.1 kHz, float32, 128-frame blocks, fitted patch, notes held for the whole
10-second render (so "sounding" is the steady-state worst case, not an average):

| sounding voices | % of one M1 core |
|---|---|
| 0 (effect chain only) | 3.5 |
| 1 | 7.8 |
| 2 | 11.3 |
| 4 | 20.1 |
| 8 | 36.8 |
| 12 | 54.5 |
| 16 | 70.9 |
| 24 | 107.9 (over real time) |
| 32 | 140.3 (over real time) |

Fixed cost 3.5%, marginal cost **~4.3% of one M1 core per sounding voice**. Real-time
break-even is at about 22 voices.

Idle voices are genuinely free: `nvoices=32` with 8 sounding cost 36.8%, `nvoices=8` with
8 sounding cost 42.1% (the difference is run-to-run noise). Allocating a generous voice
table costs nothing.

The 26-band EQ cascade, running in stereo in the shared effect, costs about **2.8% of a
core**: with no notes sounding, 3.4% with the cascade and 0.6% without it.

### What that means for a mid-range laptop

**Derived estimate, not measured.** Take the M1 numbers, halve the machine, and target
50% render capacity rather than 100% (headroom for the UI, the visuals, the FFT display,
and the fact that a missed deadline is audible):

- **8 voices is safe.**
- **12 is the working ceiling.**
- **16 is where a mid-range machine starts to glitch** on this patch.
- 24+ is out of reach on one core at this DSP complexity.

Two things inflate the sounding-voice count beyond what the player thinks they are
holding: the fitted amplitude release `aR` (range up to 6 s) keeps released voices
computing until their level falls below the stop threshold, and a pad player holds
chords. A four-note chord change with a 2-second release is eight sounding voices for
two seconds.

**Widely reported, worth knowing**: AudioParam count per processor is itself a
performance trap in Chrome. One first-hand account traced clicking in a 16-voice
synth to `AudioWorkletProcessor::Process` marshalling 544 AudioParams (34 per node
across 16 nodes) every render quantum, and fixed it by cutting to 96
([cprimozic.net](https://cprimozic.net/blog/webaudio-audioworklet-optimization/)).
This DSP is **not** exposed to that: Faust's poly design puts all voices inside **one**
AudioWorkletNode, so the parameter set is ~52 AudioParams on a single node (55 params
minus `freq`/`gain`/`gate`, which faustwasm excludes from the descriptors in poly mode —
`analysePolyParameters` in `FaustAudioWorkletProcessor.ts`). Keep it that way; do not
build one node per voice.

---

## 3. Parameter changes while notes are held

### What is specified

Faust samples controls **once per block**, not per sample: "all control values are
_sampled_ once at the beginning of the `compute` method, so that to _keep the same value
during the entire audio buffer_"
([Faust architectures manual](https://faustdoc.grame.fr/manual/architectures/)). With a
128-frame quantum that is one update every **2.9 ms** at 44.1 kHz.

The faustwasm node's `setParamValue` does two things (`FaustAudioWorkletNode.ts`): it
posts a `param` message to the processor port, and it calls
`param.setValueAtTime(value, this.context.currentTime)` on the corresponding AudioParam.
The processor's `process()` reads only the **first sample** of each AudioParam block
(`const [paramValue] = parameters[path]`) and writes it into the DSP only when it differs
from a cached value (`FaustAudioWorkletProcessor.ts`). So:

- Every control is an AudioParam, so AudioParam automation (`linearRampToValueAtTime`,
  `setTargetAtTime`) works — but it is **sampled at block granularity**, giving a
  128-sample staircase, not a sample-accurate ramp.
- Per-change cost is one comparison plus one wasm memory write. Negligible. The
  per-block cost is a fixed loop over all descriptors regardless of whether anything
  moved.
- **Nothing is smoothed by default.** Faust offers smoothing (`si.smoo` in DSP code, or
  the `smoothing_dsp` decorator with linear and exponential variants, per the
  architectures manual), and `timed_dsp` for sample-accurate control via timestamped
  events, but none of that is in `scripts/synth.py`. Every slider in this repo's DSP is
  raw.

### What I measured

I rendered 2 s with four notes held, applied a parameter change at the 1-second block
boundary, and compared the sample-to-sample step across that boundary against the 99.9th
percentile of the signal's own sample-to-sample motion. A ratio near or below 1 means the
change is buried in the signal; well above 1 means a discontinuity a listener can hear as
a click.

| change | applied over | step at boundary | signal p99.9 | ratio |
|---|---|---|---|---|
| nothing | - | 9.6e-3 | 7.4e-2 | 0.1 |
| `cutoff` 258 → 6000 Hz | 1 block | 1.05e-1 | 2.05e-1 | 0.5 |
| `cutoff` 258 → 6000 Hz | 64 blocks (186 ms) | 1.0e-2 | 1.98e-1 | 0.1 |
| `outGain` 0.6 → 0 | 1 block | 3.3e-2 | 7.1e-2 | 0.5 |
| all 26 EQ gains 0 → ±12 dB | 1 block | 2.3e-2 | 7.3e-2 | 0.3 |
| all 26 EQ gains 0 → ±12 dB | 64 blocks | 3.4e-2 | 7.3e-2 | 0.5 |
| **whole 55-param patch replaced** | **1 block** | **4.4e-1** | **3.3e-1** | **1.3** |
| whole 55-param patch replaced | 64 blocks (186 ms) | 1.8e-2 | 3.4e-1 | 0.1 |

And a separate test stepping a parameter once per block across a 0.2 s sweep found no
elevated discontinuity at block boundaries versus block interiors for `cutoff`, `eq10`
and `outGain` (mean ratios 0.89-1.00); only `revWet` showed any boundary bias (2.07 on
the mean, but with a maximum step of 3.6e-3, i.e. tiny).

**Conclusions, measured:**

1. A macro swept by hand, even quickly, does **not** zipper on this DSP even without
   `si.smoo`. Moving one parameter in 128-sample steps produces steps smaller than the
   pad's own waveform motion.
2. A **single** parameter jumped to a wildly different value in one block also does not
   produce a step above the signal's own noise floor. The filter and the EQ absorb it.
3. Changing the **whole patch at once in one block** does produce a real discontinuity
   (ratio 1.3). This is the staircase's exact failure mode.
4. Ramping that same whole-patch change over **64 blocks, ~186 ms**, removes it
   completely (ratio 0.1).

**Unverified**: I measured waveform discontinuity, not perception. A step below the
signal's p99.9 is very likely inaudible but I did not listen, and I did not test
envelope-shape parameters (`aA`, `aD`, `aR`) whose effect on a held note is a change of
trajectory rather than a step.

---

## 4. Swapping the DSP graph while the instrument plays

This is the app's core bet, so this section is longer.

### The blunt facts

**Specified / found in source.** There is no supported way to replace the DSP inside a
running Faust node:

- `FaustAudioWorkletNode` has `setParamValue`, `keyOn`, `keyOff`, `start`, `stop`,
  `destroy` and nothing that installs a different DSP (`FaustAudioWorkletNode.ts`).
- `AudioWorkletGlobalScope.registerProcessor()` throws `NotSupportedError` when a
  constructor is already registered under that name
  ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletGlobalScope/registerProcessor)),
  and there is no unregister. faustwasm sidesteps this by naming the processor after the
  DSP's sha key (`processorName = factory?.shaKey || name`) and keeping a per-context
  `Set` of names it has already registered, generating a fresh Blob module and calling
  `addModule` for each new DSP (`FaustDspGenerator.ts`). A new fit is therefore a new
  processor name, a new module, and permanent residency in that
  AudioWorkletGlobalScope.
- The AudioWorkletGlobalScope runs **on the rendering thread**
  ([spec](https://webaudio.github.io/web-audio-api/#AudioWorklet)), so evaluating a new
  module's top-level script and constructing a new processor happens on the same thread
  that has a 2.9 ms deadline.

**The reference implementation does the crude thing.** The Faust IDE, which recompiles
while audio is running, works like this: `DspRunner` "Compiles a DSP, replaces the
previous node, restores saved parameters, and reconnects input/output graph state" —
`disconnectCurrentDsp()`, `restoreParams(node, options)` (which loops
`node.setParamValue(path, options.dspParams[path])` for every path present in both), then
reconnect (`grame-cncm/faustide`, `src/runtime/DspRunner.ts`). No crossfade. No held
notes. Audio stops and starts. **That is the state of the art shipped by the Faust
authors**, and it is not good enough for the staircase.

### What can and cannot carry across a swap

**Parameters can.** Faust exposes `getParams()` / `getParamValue(path)` /
`setParamValue(path, value)` keyed by UI path, and the IDE's parameter restore proves
this is the intended migration mechanism. Any parameter whose path exists in both the
old and the new DSP moves across intact.

**Internal state cannot.** Nothing in the faustwasm API exposes oscillator phase, filter
memory, envelope stage, or delay-line contents in an addressable form. The DSP's state
lives in the wasm instance's linear memory with a layout the compiler chose for that
program; a different program has a different layout. This repo already knows the
consequence from the other side: "Faust oscillators are free-running, so splitting notes
across processor instances changes voice allocation and therefore phase. Compare spectra,
not waveforms" (`CLAUDE.md`).

**Held notes must be re-issued.** The app is the MIDI source, so it knows which notes are
down and can call `keyOn` on the new node. But `keyOn` starts the envelopes from zero.
With the fitted `aA` of ~0.37 s, a re-key is an audible re-attack, and any note in its
release tail is simply gone.

### Four techniques, ranked

**A. Do not swap. Ship one architecture and only ever change parameters.**

This is the strongest option and this repo already invented the pattern. `synth.PARAMS`
is append-only, and every stage was added with identity defaults so that "an old
27-parameter vector padded with them renders bit-identically" and the 26-band EQ "costs
5.9e-08 relative and 0.0 of loss" at its identity (`scripts/synth.py`). If the app ships
the **superset** DSP once and the staircase delivers successive parameter vectors, then:

- No `addModule`, no new processor, no new wasm, no accumulation.
- Held notes keep their envelopes, their phase, their reverb tail.
- Measured above: ramp the vector over ~186 ms and there is no discontinuity at all.
- The transfer per staircase step is a JSON parameter vector, kilobytes, not 29 KB.

The cost is that the architecture is frozen before the fit starts, which is in tension
with `VISION.md`'s "an agent designing DSP for a specific sound". The honest framing is
that this is a **product choice about how much of the staircase is architecture search
versus parameter search**, and the browser strongly prefers the latter.

**B. Two nodes, equal-power crossfade through GainNodes.**

Ordinary Web Audio: build the new node, connect both through gains, ramp one down and
one up with AudioParam automation. Facts that bound it:

- CPU is **doubled for the duration of the fade**. At 8 voices that is 36.8% → ~74% on
  an M1, i.e. already past a mid-range laptop's safe budget. The crossfade must be
  short, or the voice cap must be lower than the steady-state cap.
- Both DSPs are free-running and phase-unrelated, so during the fade the two copies of
  the same chord comb-filter against each other. On a detuned pad this reads as
  flanging, not as a click. **Unverified**: I did not measure how audible it is.
- The old node's reverb tail (`revSize` up to 0.99, zita) is inside the old node. Fading
  the node out cuts its tail. Keeping it alive past the fade keeps its CPU too.
- Held notes must be re-keyed on the new node, with the re-attack described above,
  partially masked by the old node still sounding.

**C. One processor, two DSP instances, crossfade inside the worklet.**

Not provided by faustwasm, but the pieces exist and I verified each one:

- A compiled `WebAssembly.Module` is structured-cloneable — MDN: it "can be efficiently
  shared with Workers, and instantiated multiple times"
  ([MDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly/Module)) —
  and faustwasm already relies on this, passing `mixerModule: WebAssembly.Module`
  through `processorOptions` into the AudioWorkletGlobalScope
  (`FaustAudioWorkletProcessor.ts`). So a new DSP's module can cross into a **running**
  processor over its port.
- Synchronous instantiation on the audio thread is affordable. **Measured** on M1:
  `new WebAssembly.Module(bytes)` ~0.01 ms warm; `createSyncPolyDSPInstance` 0.36-0.85 ms
  for 4 to 32 voices, against a 2.9 ms budget. faustwasm already instantiates
  synchronously inside the processor constructor
  (`FaustWasmInstantiator.createSyncMonoDSPInstance`), so this is a supported shape.
- That buys a **sample-accurate** crossfade with one clock, no `addModule`, no
  accumulation of registered processor names, and the option of crossfading only the
  voice DSP while keeping a single shared effect chain and therefore a single continuous
  reverb tail.

It still cannot carry voice state. It is more code than B, and Chrome's compile of the
module should be done on the main thread and only the `Module` handed over. **Unverified**:
nobody appears to have published this for Faust; I found no implementation to point at.

**D. Recompile in place.** Does not exist. No mechanism in the Web Audio API, in
WebAssembly, or in faustwasm allows patching a running wasm instance's code.

### Prior art worth knowing

[Elementary Audio](https://www.elementary.audio/docs/in_depth) solves the general
problem: its renderer "carefully step[s] through the new graph to identify similarities
and differences between the new graph and the one that's currently making sound" and
"precisely applies _only_ the required changes to the underlying platform", inside a
single AudioWorkletNode. That is technique C generalised, and it is an existence proof
that live graph reconciliation in one worklet is viable. Its docs do **not** state
whether it fades to avoid clicks or what happens to node state across a change, so it is
a design reference, not a source of guarantees.

---

## 5. Web MIDI reality

### Support

**Specified**, from MDN's browser-compat data (`api/MIDIAccess.json`,
`api/Navigator.json#requestMIDIAccess`):

| browser | `navigator.requestMIDIAccess` |
|---|---|
| Chrome / Edge / Opera (desktop) | 43+ |
| Chrome Android, Samsung Internet | yes |
| Firefox desktop | 108+, gated behind a **site permission add-on** install |
| Firefox Android | no |
| **Safari, macOS and iOS** | **no** ([WebKit bug 107250](https://webkit.org/b/107250)) |

Secure context is required everywhere. The map already scopes the app to desktop
Chromium, which is exactly the set where this works properly.

### Permission

**Specified**: since Chrome 124 the **entire** API is behind a permission prompt, not
just SysEx. Chrome's announcement: "This change is rolling out gradually starting in
Chrome 124", and developers should request SysEx explicitly with
`navigator.requestMIDIAccess({ sysex: true })` "only if your website absolutely needs
this feature"
([Chrome for Developers](https://developer.chrome.com/blog/web-midi-permission-prompt)).
The blink-dev PSA frames the motivation as closing "drive-by access to client MIDI
devices" and notes both permissions are now requested in a single bundled prompt
([blink-dev PSA](https://groups.google.com/a/chromium.org/g/blink-dev/c/nz320H9J6bs)).
Denial surfaces as a rejected promise with `error.name === "SecurityError"`.

Consequences for the flow:

- **There is a modal permission prompt between the user and their keyboard.** It is not
  optional and it cannot be pre-empted.
- **Do not ask for SysEx.** This app has no use for it and asking makes the prompt
  scarier.
- The permission is queryable with `navigator.permissions.query({ name: "midi" })`, so
  the app can tell "never asked" from "denied" and avoid re-prompting into a wall.
- **Unverified**: whether Chrome requires transient user activation to show the prompt,
  and how long the grant persists. I found no normative statement. Asking on an explicit
  user action is the safe design regardless.

### Hot-plug

**Specified**: `MIDIAccess` fires `statechange` when a port connects or disconnects, and
each `MIDIPort` carries `state` (`connected` / `disconnected`) and `connection`
(`open` / `closed` / `pending`)
([Web MIDI API spec](https://webaudio.github.io/web-midi-api/)). Supported in Chrome 43+.
Firefox 108 has the handler but **the event never fires** (bug 1802149, per MDN compat) —
irrelevant under the Chromium-only decision, but it means "plug your keyboard in now"
silently fails to be noticed on Firefox.

So on Chromium the app can honestly promise: plug a keyboard in at any time and it
appears. Design the device state as a live subscription, not a one-time enumeration at
load.

### Fallback when there is no device

Nothing in the spec provides one; this is entirely the app's job, and the map already
requires it ("MIDI keyboard, plus an on-screen keyboard fallback so the result is
audible the instant it lands"). Three things the research bears on:

1. The instrument does not depend on Web MIDI at all. faustwasm's poly node exposes
   `keyOn(channel, pitch, velocity)` / `keyOff` directly on the node
   (`FaustAudioWorkletNode.ts`), and MIDI is just one caller. An on-screen keyboard or
   computer-keyboard mapping is a first-class input, not a shim.
2. Velocity has to come from somewhere. A mouse click has no velocity; the on-screen
   keyboard must synthesise one, and `VISION.md`'s requirement that the patch hold up
   "at velocities and hold times the clip never showed" means a fixed velocity of 100
   hides exactly the defect the user needs to find.
3. The permission prompt means the no-device path is also the **not-yet-granted** path
   and the **denied** path. Three distinct states, one fallback surface.

---

## What this bounds

Concrete limits the play surface and the macro layer have to be designed within.

**Voices**

1. Budget **8 simultaneously sounding voices as safe, 12 as the ceiling, 16 as the point
   where a mid-range laptop glitches** on a patch of this complexity. Derived from 4.3%
   of one M1 core per voice plus 3.5% fixed, halved for a slower machine and held to 50%
   render capacity.
2. Sounding is not the same as held. The fitted release keeps voices alive after key-up;
   a chord change with a long release doubles the count for the length of the release.
   The voice cap and the release time are one decision, not two.
3. Allocate a generous voice table anyway — idle voices measured free — and rely on
   Faust's stealing (oldest releasing voice first, then oldest playing voice).
4. Everything runs on **one** thread. There is no scaling out. If the fitted architecture
   gets heavier, the polyphony ceiling drops in the same breath, and the play surface
   should be honest about that rather than silently stealing voices.
5. Keep the whole instrument in **one** AudioWorkletNode. One node per voice is a known
   Chrome performance trap.

**Macros**

6. Macros move parameters at **block rate, one update per 2.9 ms**, and are not smoothed
   unless the emitted Faust asks for it. Measured: hand sweeps do not zipper on this
   patch, and even a single parameter jumped to an extreme in one block stays below the
   signal's own noise floor. So macros can be direct and do not need a smoothing layer
   bolted on top — but that is a property measured on **this** pad and should be
   re-checked for a percussive or clean-waveform patch, where the same step would be
   naked.
7. A macro can drive many parameters at once cheaply. Sweeping all 26 EQ gains together
   cost nothing measurable and produced no discontinuity. "Brightness is not one
   parameter" is affordable.
8. The 26-band EQ costs ~2.8% of a core in stereo. It is not free, but it is one third of
   a voice. Do not design it out for performance reasons.

**The staircase**

9. **Replacing an entire parameter vector in one block clicks** (measured, the only
   change that did). **Ramping the same change over ~186 ms does not** (measured).
   Whatever the staircase does, a new patch must arrive as a ramp of at least a couple
   of hundred milliseconds, not as an assignment.
10. If the staircase only ever changes **parameters** of a fixed superset architecture,
    held notes survive intact, nothing is dropped, nothing clicks, and each step is a
    kilobyte-scale JSON transfer. This repo's append-only `PARAMS` with identity defaults
    is already the design that makes this work. **This is the cheapest possible version
    of the app's core bet and the design should default to it.**
11. If the staircase changes the **architecture**, held notes cannot survive. Parameters
    migrate by path; oscillator phase, filter state, envelope stage and reverb tails do
    not, and there is no API that would let them. The best available outcome is a
    crossfade with the held notes re-keyed on the new DSP, which means a re-attack (~0.37 s
    on the fitted patch) partly masked by the old DSP still sounding.
12. A crossfade **doubles CPU for its duration**. The voice cap during a staircase step
    is roughly half the steady-state cap. Either cap voices lower, or crossfade fast, or
    schedule architecture steps for moments when nothing is held.
13. Each new architecture costs a new `addModule` and a permanently registered processor
    name in that AudioContext; there is no unregister. A long session with many
    architecture steps accumulates modules. Bound the number of architecture changes per
    session, or plan to rebuild the AudioContext (which stops all sound).
14. `addModule` and processor construction happen on the **rendering thread**. Build the
    new node and let it run silently for a few blocks before starting the crossfade, so
    any hiccup lands in silence. **Unverified**: I found no published measurement of how
    much `addModule` disturbs a running graph in Chrome. Measure it before promising
    seamlessness.
15. A patch is **~29 KB gzipped** as precompiled wasm (voice 6 KB, effect 19 KB, metadata
    4 KB). Client-side compilation instead costs **~0.9 MB brotli** for the compiler plus
    **~1 second** to compile this DSP as poly. Compile on the server; the browser should
    never need libfaust.

**MIDI and the first note**

16. Sound cannot start without a user gesture (autoplay policy), and MIDI cannot start
    without a permission prompt (Chrome 124+). The first screen has at least one
    unavoidable click and one unavoidable modal. Design them as one deliberate moment,
    not two interruptions.
17. Never request SysEx.
18. MIDI device state is a live subscription: `statechange` on Chromium reports hot-plug
    reliably. Absent, pending-permission and denied are three different states with three
    different messages, and all three land on the on-screen keyboard.
19. Safari has no Web MIDI at all and no sign of it. The Chromium-only scope in the map is
    load-bearing, not a convenience.

**One trap found in passing**

20. `scripts/synth.py`'s chorus specifies its delay in **samples**
    (`de.fdelay(4096, 220.0 + 200.0*chDepth*os.osc(chRate + 0.13*i))`), so it is
    sample-rate dependent, while the delay and reverb are specified in seconds and are
    not. An AudioContext defaults to the hardware rate, commonly 48 kHz, and the fit was
    done at 44.1 kHz. Either request `new AudioContext({ sampleRate: 44100 })` or make
    the chorus rate-independent, or the browser will not play back what was fitted. This
    is the same class of error as the `data/original.wav` 48 kHz trap already recorded in
    `CLAUDE.md`.

---

## Sources

Primary specifications and first-party documentation:

- [Web Audio API — rendering loop / render quantum](https://webaudio.github.io/web-audio-api/#rendering-loop)
- [Web Audio API — AudioWorklet](https://webaudio.github.io/web-audio-api/#AudioWorklet)
- [Web MIDI API specification](https://webaudio.github.io/web-midi-api/)
- [MDN — AudioWorkletGlobalScope.registerProcessor](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletGlobalScope/registerProcessor)
- [MDN — WebAssembly.Module](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly/Module)
- [MDN — Autoplay guide](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Autoplay)
- [MDN browser-compat-data](https://github.com/mdn/browser-compat-data) — `api/MIDIAccess.json`, `api/Navigator.json`, `api/AudioWorklet.json`, `api/AudioContext.json`
- [Chrome for Developers — Access to MIDI devices now requires user permission](https://developer.chrome.com/blog/web-midi-permission-prompt)
- [blink-dev — PSA: Web MIDI Permissions Prompt Change](https://groups.google.com/a/chromium.org/g/blink-dev/c/nz320H9J6bs)
- [Chrome for Developers — Audio Worklet Design Pattern](https://developer.chrome.com/blog/audio-worklet-design-pattern)
- [Faust — Deploying on the Web](https://faustdoc.grame.fr/manual/deploying/)
- [Faust — Architecture files (control sampling, smoothing, timed_dsp)](https://faustdoc.grame.fr/manual/architectures/)
- [Faust — MIDI and polyphony](https://faustdoc.grame.fr/manual/midi/)
- [grame-cncm/faustwasm](https://github.com/grame-cncm/faustwasm) — README, `src/FaustDspGenerator.ts`, `src/FaustAudioWorkletNode.ts`, `src/FaustAudioWorkletProcessor.ts`, `src/FaustWebAudioDsp.ts`
- [grame-cncm/faustide](https://github.com/grame-cncm/faustide) — `src/runtime/DspRunner.ts`

Secondary, peer-reviewed or first-hand:

- [Jangda et al., "Not So Fast: Analyzing the Performance of WebAssembly vs. Native Code", USENIX ATC 2019](https://www.usenix.org/conference/atc19/presentation/jangda)
- [Casey Primozic — Finding + Fixing an AudioWorkletProcessor Performance Pitfall](https://cprimozic.net/blog/webaudio-audioworklet-optimization/)
- [Elementary Audio — In Depth](https://www.elementary.audio/docs/in_depth)
