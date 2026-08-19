# Reverse-engineering `Which movie should have an opening like this - Kaymo.mp3`

A 17.9 s cinematic pad. Two stages: transcription first (frozen), then patch matching.

## What the clip actually is

Measured before assuming anything about the brief:

- **No drums.** Percussive energy is 0.1% of total after HPSS; spectral flatness 0.0013 (highly tonal).
- **One sustained pad**, slow chord changes, no second instrument.
- **Sawtooth-type spectrum**: a complete harmonic series over a deep fundamental (e.g. 43.6, 87.5, 131.9, 176.3, 220.7, 263.8, 351.3, 441.4, 528.9 Hz — every integer multiple present).
- **Detune + slow modulation.** High partials resolve into closely-spaced clusters; the offset partials sit a consistent **+1.38% (≈ 23 cents)** above the main series at every harmonic. Those are not extra notes. Measuring the cluster structure separates two mechanisms: cluster *width* grows with harmonic number (a fixed ratio → **static unison detune**, ~10-25 cents total, narrow) while *spacing between adjacent peaks inside* a cluster is a constant ~1.3 Hz at harmonics 10, 12 and 16 alike (fixed Hz → **slow modulation**, chorus/LFO). Both are present.
- **A pitch bend in the intro.** The first ~3.5 s is not static: every partial sweeps upward together. Tracking four of them gives frequency ratios **1 : 1.330 : 1.497 : 2.007 ≈ 6 : 8 : 9 : 12**, i.e. harmonics of one common fundamental — a single voice gliding, not a chord. The fundamental runs **F2 (88.3 Hz) → C#3 (138.9 Hz), +7.85 semitones over ~3.5 s**, a riser landing exactly on the drop at 3.45 s.
- **Standard A440.** Fitted fundamentals land on MIDI 29.00 / 25.00 / 27.00 / 30.00 — no global detune.
- **Heavy reverb**, long smeared tails that bleed each chord into the next.
- **Arrangement**: the first 3.45 s carry only 1-2% of their energy below 200 Hz; at 3.45 s the low end jumps to 25-33%. The bass drops in there.

## Stage 1 — transcription

`basic-pitch` was tried first and is unusable here: it shreds the sustained reverb-heavy chords into ~90 fragmentary ghost notes. Two better-suited methods were used instead and cross-checked:

1. **High-resolution partial analysis** (zero-padded FFT + parabolic interpolation) to pin each bass fundamental, with an explicit octave test — odd harmonics (3, 5, 7) are the discriminator between `f0` and `2*f0`.
2. **Greedy voicing search evaluated on rendered audio**: each candidate voicing is rendered and scored by chromagram agreement and mel distance, accepted only if the *full-clip* score improves (monotone by construction).

### Result

Chord qualities come from the author, who wrote the part in GarageBand and cannot export MIDI: **F > Dbm > Ebsus2, with the last chord pitched up at half its length**. Everything checkable was verified against the audio first (see below); what measurement could not resolve was taken on the author's word.

| region (s) | chord | bass | voicing |
|---|---|---|---|
| 3.45-4.30 | Eb(top) | D#2 | D#4 G4 C5 |
| 4.30-7.45 | F major | F1 | A2 C3 F3 A3 |
| 7.45-10.40 | Db major | C#1 | C#2 G#2 C#3 F3 G#3 |
| 10.40-16.05 | Ebsus2 | D#1 | D#2 A#2 D#3 F3 A#3 |
| 16.05-17.90 | F major | F1 | F2 C3 F3 A3 C4 |

Intro (0-3.53 s): **C#3 F3 held under a pitch bend**.

### Pitch bend

Two segments — intro glide 88.3 Hz (F2) -> 138.9 Hz (C♯3), +7.85 st over 3.67s; then +3 st at 13.35s, released at 16.05s:

1. **Intro riser.** One voice gliding, driven directly by the measured f0 trajectory rather than a fitted curve. The note is held at the glide's *target* pitch so the bend reaches unity exactly at the drop, leaving the following chords unbent.
2. **The Eb chord pitched up +3 semitones at half its length**, released at 16.05 s so the closing F chord sounds unbent.

Both are applied as sample-accurate automation on a `bend` control that multiplies oscillator frequency (DAWDreamer `set_automation`). The two large tuning excursions I had measured but not explained — around 3.45-4.30 s and 16.05-17.90 s (min -165 cents) — are exactly these two bend transitions, the second being the Eb notes' tail sliding back down as the bend releases.

> **Correction.** An earlier pass transcribed this section as a static chromatic cluster (F3 G3 A3 B3 C4 D4 E4 F4). That was an artefact of fixed-window analysis: a partial sweeping through a band is sampled at different frequencies in successive windows and looks like several simultaneous notes. Partial tracking shows one gliding voice.

Velocities were fitted rather than guessed (bass 100 / upper 75 / intro 80), after the long-term average spectrum showed the render 10-20 dB light between 250 and 900 Hz — the register the upper voicing occupies. Refitting them against the stage-2 loss improved it from 1.6175 to 1.5890 with the patch untouched.

Chord overlap was also tested (0-2 s) and measured best at 0.0 s: the fitted patch's long release already bridges chord boundaries, so no note overlap is needed.

### Verifying the author's structure

**The pitched-up last chord checks out exactly.** The Eb chord spans 10.40-16.10 s, so half its length is 2.85 s, putting the change at **13.25 s** — my independently measured chord boundary was **13.35 s**. The bass ratio across that boundary (38.89 -> 46.25 Hz) is **+3.00 semitones** to the digit. What I had transcribed as a separate sixth chord ("Gb major") is the same Eb chord bent up; that boundary was a phantom, and it accounts for the second set point in the author's automation lane.

**The sus2-vs-major distinction is not measurable on this material**, which is worth being explicit about. My metric preferred "Gb major" over the transposed Ebsus2 by 3.5% chroma, but every note that would distinguish them coincides with a sawtooth harmonic of the Gb bass:

| note | as harmonic of Gb1 (46.25 Hz) | offset |
|---|---|---|
| Bb3 233.08 Hz | harmonic 5 (231.25 Hz) | +13.7 cents |
| Ab4 415.30 Hz | harmonic 9 (416.24 Hz) | -3.9 cents |
| Db4 277.18 Hz | harmonic 6 (277.50 Hz) | -2.0 cents |

A saw on a low root already supplies its own 2nd (h9), 3rd (h5) and 5th (h6), so adding either note reinforces something present either way. Worse, the strong Bb2 (116.5 Hz) I had cited as evidence for Gb major is **harmonic 3 of Eb1 (116.67 Hz, 2 cents off)** — the previous chord's reverb tail. That evidence was doubly confounded, so the author's reading stands.

### The Db chord is not minor

The author labelled it `Dbm`, which would have been the one theoretically awkward chord: Db minor needs Fb (E natural), which is foreign to Bb minor and cross-relates with the F natural that is the root of the preceding chord and a member of the following one. Unlike the major third, the minor third **is** cleanly measurable — E natural sits 86 cents from the nearest harmonic of the Db bass, and neither neighbouring chord contains an E, so reverb cannot mask it. Measured over 8.5-10.2 s (normalised amplitude):

| | amplitude |
|---|---|
| Ab3, the 5th | 0.396 |
| Db3, the root | 0.251 |
| **E3 / E4, the b3 (Fb)** | **0.009 / 0.010** |
| Eb3, sus2 (clean) | 0.009 |

E natural is ~34 dB below the root: absent. The chord is **Db major**, which is also the diatonic III of Bb minor. Rendering confirms the ordering:

| Db-region voicing | loss | chroma |
|---|---|---|
| Db minor (current) | 1.5799 | 0.9137 |
| Db major | 1.5788 | 0.9147 |
| Db no third | 1.5741 | 0.9145 |

Db major beats Db minor on both. The no-third voicing scores marginally lower loss still, and cannot be separated from Db major by measurement — harmonic 5 of the Db bass supplies a major third at +13.7 cents whether one is written or not. Db major is delivered.

**And the author's structure measures better anyway**, on the same fitted patch:

| transcription | stage-2 loss | chroma |
|---|---|---|
| mine: 6 regions, all major triads | 1.5756 | 0.9130 |
| author: F / Dbm / Ebsus2 + bend | **1.5650** | **0.9138** |

It is also simpler: 4 chords and 29 notes instead of 6 chords and 41.

### Stage 1 metrics

Chroma agreement depends on which synth the MIDI is rendered through, so both readings are given:

| measured through | chroma | mel dist | env L1 | onset F |
|---|---|---|---|---|
| neutral reference saw patch | 0.8404 | 12.91 | 0.2201 | 0.220 |
| final matched patch | **0.9129** | **7.05** | **0.1041** | 0.327 |

The 0.90 chroma target is met by the delivered render (0.9129) and not by the neutral-reference reading (0.8404). The neutral patch is much darker than the target, which suppresses the upper harmonics chroma keys on.

**Two honest caveats:**

- *An octave-inflated chroma score was available and was rejected.* A voicing search free to pick bass octaves reached 0.9151 — but only by moving every bass note up an octave. Chroma is octave-invariant and cannot arbitrate that; direct spectral evidence can. In 5.10-6.45 s the 43.65 Hz component is the **single strongest peak in the spectrum**, and 131 Hz (odd harmonic 3) is stronger than 87.3 Hz. An F2 fundamental produces neither. The physically correct octave was kept and the lower score reported.
- *Onset F (0.22) is far below the 0.90 target and the target is not meaningful here.* The clip has ~6 real onsets in 17.9 s; librosa's detector fires ~30 times on both signals, triggered by reverb swell and beating rather than note attacks. Recall is high (most true onsets are found); precision is what collapses. Chord-change times were instead verified directly against the CQT-flux boundaries used for segmentation.

## Stage 2 — sound matching

Candidate architecture (Faust via DAWDreamer, 29 free parameters, all normalised to [0,1]), chosen from the stage-0 evidence rather than by default:

- 7-voice unison saw/square with detune spread + sub sine  *(complete harmonic series + detune clusters)*
- tanh waveshaper (`drive`)  *(added later to test the missing-midrange hypothesis; it failed, see below)*
- resonant lowpass with its own ADSR and key tracking  *(centroid ~1050 Hz, rolloff95 ~3750 Hz)*
- amp ADSR  *(slow swell, long tails)*
- per-note constant-power pan (`spread`)  *(added later, for the L/R decorrelation)*
- chorus -> ping-pong delay -> Zita reverb -> tilt EQ  *(smeared decay)*

Optimised with CMA-ES (population 16) against `auraloss` multi-resolution STFT loss (FFT 512/1024/2048/4096, log-magnitude + spectral convergence) plus a scale-invariant loudness-envelope L1 term. Hierarchical: oscillator/filter/amp core first, then all parameters, with restarts. Early stop on <1% improvement over 20 generations.

Optimisation history:

| step | loss |
|---|---|
| analysis-seeded start | 3.5710 |
| CMA-ES, 3 restarts (6242 renders) | 1.6175 |
| + velocities refitted (stage 1) | 1.5890 |
| + local polish | 1.5705 |
| + intro re-transcribed as a pitch bend (stage 1) | 1.6029 |
| + refit | 1.5765 |
| + intro voicing corrected to a third (author screenshot) | 1.5756 |
| + author structure: F/Dbm/Ebsus2 + bend at half length | 1.5650 |
| + Db region corrected to Db major (no Fb measured) | 1.5788 |
| + MIDI overlap defect fixed; rendered from the file | 1.5719 |
| + final polish | 1.5564 |
| + hybrid gradient / CMA-ES refit, still 27 parameters | 1.5482 |
| + `spread` added and fitted, 29 parameters | **1.5446** |

### The patch

- 7-voice unison sawtooth, 45 cents total detune spread, 12% spread / 88% centre
- sub oscillator essentially off (the saw's own fundamental carries the low end)
- resonant lowpass at 258 Hz, Q 0.51, key tracking 0.02
- filter envelope opening 5757 Hz (A 0.16s, D 0.18s, S 0.62)
- amp ADSR A 0.37s / D 1.62s / S 1.00 / R 1.21s
- vibrato 21.5 cents at 9.31 Hz
- chorus depth 0.26 at 2.55 Hz
- delay 369 ms, feedback 0.43, wet 0.03
- pre-filter saturation off (`drive` 0.00; measured to make the fit worse)
- per-note constant-power pan, spread 0.90
- reverb: size 0.35, damping 1.00, wet 0.12
- output tilt EQ brighter (+0.26)

Full parameter values are in `patch.json`.

### Stage 2 metrics (render vs original)

- MRSTFT + envelope loss **1.5446**
- mel distance **7.05 dB** (stage-1 reference synth: 12.91 dB)
- chromagram agreement 0.9129
- loudness-envelope L1 0.1041

### Architecture check: is the movement detune or modulation?

The fit put `uniMix` near zero (almost no unison spread) and leaned on vibrato and chorus instead, which looked like it contradicted the measured detune. It was tested rather than assumed: a second full optimisation was run with the unison **pinned wide** (`uniMix` 0.80, `detune` 55 cents, vibrato off).

| variant | loss | chroma | mel dist |
|---|---|---|---|
| free (delivered) | **1.5446** | **0.9129** | **7.05** |
| pinned wide unison | 1.6970 | 0.7668 | 8.35 |

The wide-unison variant is worse on every metric. That is consistent with the measurement rather than at odds with it: the observed clusters are only ~10-25 cents wide, so a 55-cent spread is far too much — it smears pitch classes badly (chroma 0.77). The delivered patch's narrow effective spread plus chorus movement is the correct reading. The earlier "75-cent spread" seen in the intro was different chromatic notes in the riser, not unison detune.

## Two stages added after the fit: `drive` and `spread`

The two largest remaining mismatches were attacked by changing the architecture rather than by re-searching it: a tanh waveshaper between the oscillators and the filter (`drive`) for the light 250-900 Hz band, and per-note constant-power panning (`spread`) for the missing L/R decorrelation. Both live in `scripts/synth.py`, which is the portable deliverable, and in the differentiable surrogate.

`drive` is `osc + drive*(tanh(osc*g)/tanh(g) - osc)` with `g = 1 + 12*drive`, i.e. Faust's `ma.tanh` normalised so the peak stays at unity, mixed in by `drive` itself so that `drive = 0` is algebraically the identity rather than tanh's own curvature (what that costs in the real render is measured just below). `spread` pans each note to `0.5 + 0.5*spread*sin(2*pi*0.618*semitones_from_C4)` with gains `sqrt(2*(1-pos))`, `sqrt(2*pos)`: the Faust voice DSP has no voice index to hash, so the pan position is a deterministic function of the note's own pitch, and the irrational multiplier scatters neighbouring notes instead of ramping low to high. At `spread = 0` both gains are exactly 1, which is the old `<: _,_`.

### Both are appended, and the old result still reproduces

`PARAMS` is append-only, so `patch.json`'s 27 coordinates keep their meaning and the vector is padded with the new defaults (`synth.pad_normalized`). Compiling the DSP with both new sliders replaced by the constant 0 lets Faust fold the two stages away again, and that build reproduces the recorded baseline to every digit: **1.5563665383** against the recorded **1.5563665383**.

The padded vector through the *new* DSP is not bit-identical, and that is worth stating rather than glossing: **1.5561847337**, a difference of -1.82e-04, from a render that differs by 5.1e-05 relative L2 (max sample error 6.0e-05). Both stages are algebraically the identity at their defaults, and each one added separately produces the same size of difference, so this is Faust's code generation for the voice changing when operations are added to it, amplified by the time-varying resonant filter recursion, not a semantic change. It moves the baseline down by 0.012%, which makes any later improvement marginally harder to claim, not easier.

### Surrogate parity of the two new stages

| stage | check | worst error |
|---|---|---|
| `drive` | Faust `ma.tanh` probe vs `torch.tanh`, 6 drive settings | 1.2e-07 max abs (float32 epsilon) |
| `spread` | per-channel RMS of a real one-note render vs `pan_gains` | 8.3e-06 on the R/L ratio |

Both stages are exact ports, unlike the reverb, so the gradient through them is the gradient Faust would give. It also points the right way, which is checkable here because a one-dimensional Faust scan of each parameter exists: at the grid best the surrogate reports `dL/dlogit` **-7.8e-03 for `spread`** (increase it, and the Faust scan below does fall until 0.90) and **+3.9e-06 for `drive`** (decrease it, and Faust rejects every nonzero setting). Whole-gradient magnitude is dominated by `outGain` at 1.65; the surrogate's own loss at this patch is 1.6244 against Faust's 1.5450, the familiar +0.079 bias.

### What each stage actually bought

A 30-render Faust grid over drive x spread at the best 27-parameter patch, with the loss also minimised over output level (rescaling the render is exactly equivalent to rescaling `outGain`, so a stage that changes level is not judged on the level change):

| drive | spread | loss | gain-matched loss | 1 - corr(L,R) | 250-900 Hz |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 1.5482 | 1.5482 | 0.0084 | -4.86 dB |
| 0.00 | 0.75 | 1.5450 | 1.5450 | 0.2408 | -4.86 dB |
| 0.08 | 0.00 | 1.5678 | 1.5622 | 0.0084 | -6.31 dB |
| 0.08 | 0.75 | 1.5608 | 1.5608 | 0.2116 | -6.31 dB |
| 0.16 | 0.00 | 1.5920 | 1.5801 | 0.0081 | -5.96 dB |
| 0.16 | 0.75 | 1.5822 | 1.5772 | 0.2302 | -5.98 dB |
| 0.30 | 0.00 | 1.6325 | 1.5903 | 0.0079 | -5.36 dB |
| 0.30 | 0.75 | 1.6155 | 1.5886 | 0.2324 | -5.39 dB |
| 0.50 | 0.00 | 1.7355 | 1.6095 | 0.0076 | -6.77 dB |
| 0.50 | 0.75 | 1.7064 | 1.6065 | 0.2129 | -6.77 dB |
| 0.75 | 0.00 | 1.9164 | 1.6357 | 0.0070 | -6.72 dB |
| 0.75 | 0.75 | 1.8734 | 1.6379 | 0.2024 | -6.74 dB |

`spread` helps at every setting it was tried at up to about 0.9, on the loss and on the defect it was added for. `drive` is rejected at every setting, and it makes its own target band **worse**. The band-by-band spectrum says why, and it corrects the earlier diagnosis:

| band | gain-aligned render error |
|---|---|
| 20-60 Hz | +2.14 dB |
| 60-250 Hz | +4.60 dB |
| 250-900 Hz | -4.86 dB |
| 900-2000 Hz | -1.72 dB |
| 2000-6000 Hz | +6.58 dB |
| 6000-16000 Hz | -2.22 dB |

The render is not short of harmonics. It is **+6.7 dB hot between 2 and 6 kHz** while being 4.9 dB light between 250 and 900 Hz: the defect is a spectral tilt, not a missing harmonic series. A memoryless waveshaper adds energy at every harmonic, so it deepens exactly the imbalance it would have to correct. That is measurement overturning the hypothesis that was written down here as the obvious next step.

`drive` was given one more chance in `scripts/drive_probe.py`, on the grounds that a shaper can only pay off if the brightness controls hand the extra treble back: the same CMA-ES search over cutoff, reso, envAmt, fS, sqrMix, tilt and outGain, run twice from the same base, once with `drive` free and once without it.

| arm | loss | fitted drive | 250-900 Hz | renders |
|---|---|---|---|---|
| no drive | 1.5597 | 0.000 | -5.36 dB | 192 |
| with drive | 1.5515 | 0.007 | -4.81 dB | 192 |

The arm with `drive` free lands -0.0082 from the arm without it, but the value it fitted is **0.007**: at that setting the shaper's pre-gain is 1.079 and it is mixed in at 0.7%, which is the identity to three decimals. The optimiser zeroed the parameter when it was free to use it, so the difference between the two arms is CMA-ES trajectory noise and not evidence for the shaper. Both arms also finished above the base they started from, which is the size of that noise.

`drive` is therefore delivered at 0, i.e. bypassed. It stays in the DSP because a measured negative result is worth more than an untested suggestion, and because the identity default costs nothing.

### The refit, reported as two separate numbers

Adding parameters to a fit lowers a loss almost by construction, so the 27-parameter number and the 29-parameter number are kept apart. Same driver, same render budget, same gradient schedule; the only difference is whether `drive` and `spread` are allowed to move.

| fit | free parameters | best true Faust loss | Faust renders |
|---|---|---|---|
| baseline `patch.json`, as delivered before | 27 | 1.5564 | - |
| refit, `drive` and `spread` pinned at 0 | 27 | 1.5482 | 697 |
| refit, both free | 29 | 1.5450 | 697 |
| **delivered**, from the `spread` scan below | 29 | **1.5446** | 8 |

Neither refit run improved on the patch it was warm-started from: 697 Faust renders each, 12 gradient steps each, every candidate rejected. The gradient polish was rejected at all three learning rates in both runs, which reproduces what the gradient audit measured at a converged patch. The whole 29-parameter gain comes from the two cheap directed scans, 30 renders of grid and 8 of spread sweep, and none of it from the 1394 renders of search.

### Did the two defects actually improve?

Neither defect is in the loss: it is mono, so width is invisible to it, and its spectral-convergence term only loosely constrains level, so a band deficit survives a lower loss. Both numbers therefore come from re-rendered audio (`out/defect_check.json`), never from the optimiser.

| render | loss | 1 - corr(L,R) | 250-900 Hz | 2-6 kHz | chroma | mel dist |
|---|---|---|---|---|---|---|
| original | - | **0.687** | - | - | - | - |
| delivered before | 1.5562 | 0.010 | -4.13 dB | +6.87 dB | 0.9154 | 7.10 |
| refit27 | 1.5482 | 0.008 | -4.86 dB | +6.58 dB | 0.9126 | 7.04 |
| extended | 1.5446 | 0.370 | -4.86 dB | +6.62 dB | 0.9129 | 7.05 |

### How much width does the loss want?

Barely any. The loss only sees `spread` through what the pan gains do to the mono sum, and the whole 0 to 1 range moves it by 0.0036. A one-dimensional Faust scan at the delivered patch (`out/spread_scan.json`), against the original's decorrelation of 0.687:

| spread | loss | 1 - corr(L,R) |
|---|---|---|
| 0.00 | 1.5482 | 0.008 |
| 0.20 | 1.5476 | 0.023 |
| 0.40 | 1.5468 | 0.069 |
| 0.55 | 1.5463 | 0.126 |
| 0.70 | 1.5456 | 0.207 |
| 0.80 | 1.5449 | 0.279 |
| 0.90 | 1.5446 | 0.370 |
| 1.00 | 1.5460 | 0.504 |

The loss is lowest at spread 0.90, which is what is delivered (decorrelation 0.370). The width that actually matches the original is spread 1.00 (decorrelation 0.504, loss 1.5460). The delivered patch is the lowest-loss one, because that is the rule this fit has been run under throughout, and the wider setting is recorded here so the choice is visible rather than silent.

## Known mismatches

- **The intro's highpass is still not modelled.** The glide itself is now reproduced, but in the original the first 3.45 s also has its low end removed (1-2% of energy below 200 Hz) and the bass drops in at 3.45 s. That is filter automation, which a static patch cannot express. Only the pitch trajectory is automated here; the filter is not.
- **The multi-resolution STFT loss could not arbitrate the glide**, which is worth stating plainly. Final loss is 1.5765 with the (correct) gliding intro versus 1.5705 with the (incorrect) static cluster — a difference of 0.4%, in favour of the wrong answer. Measured through the same patch, the pitch-sensitive metric separates them properly: intro-region chroma 0.78 gliding vs 0.61 static. This is exactly why pitch is frozen in stage 1 and not left to the spectral loss.
- **Upper voicing is partly under-determined.** A sawtooth's own harmonics already reproduce much of the perceived chord, and reverb tails bleed each chord into the next, so some upper notes cannot be separated from harmonics by spectrum alone. The voicings above are the best-measured hypotheses, not certainties. The bass line is solid — each fundamental was measured directly and passed an explicit octave test.
- **The midrange is still light, and the earlier explanation for it was wrong.** The delivered render is **-4.86 dB** across 250-900 Hz after gain alignment, and **+6.62 dB** across 2-6 kHz. The band-by-band long-term average spectrum shows the render is simultaneously light in the midrange and hot above 2 kHz, so this is a spectral tilt, not a missing harmonic series, and the `drive` stage added to test the saturation hypothesis makes it worse at every setting (see above). What the measurement points at instead is the filter shape: one resonant lowpass plus a second-order lowpass at the same cutoff, with a two-shelf tilt EQ, cannot produce a dip at 250-900 Hz and a rise at 2-6 kHz at the same time. A band-specific EQ, or a second filter path, is the next thing to try. It was not, because it is a third architecture change and the loss cannot distinguish tilt from level well enough for the change to be self-validating.
- **Stereo width is closer but still short.** Measured L/R decorrelation (`1 - corr(L,R)`) is **0.687 for the original**, was **0.010** when the voices summed to a mono pair, and is **0.370** with `spread` fitted. The structural cause is fixed: the voices are now decorrelated before the effects rather than a mono sum being handed to a stereo reverb. What remains is that the loss is mono and cannot ask for width, so `spread` is set by its second-order effect on the mono sum; the scan above shows a wider setting that is both closer to the original image and still better than the baseline loss. Closing the rest honestly needs a width term in the objective, which would change what the fit is optimising and was left alone.
- **The render sits -4.5 dB quieter** than the original (RMS 0.0816 vs 0.1367). The loss's spectral-convergence term is scale-sensitive but the envelope term is scale-invariant, so absolute level is only loosely constrained. Rescaling the render to match does not lower the loss, which was checked over a 0.5x-2x sweep while gridding the new stages.
- **A render-write defect was found and fixed late, and it was larger than the improvements being argued about.** `out/render.wav` was written with soundfile's default `PCM_16`. Scored as a file rather than in memory, the delivered render measured **1.6344 against the 1.5450 that was reported for it**: the loss's log-magnitude STFT term sees the 16-bit quantisation floor in the 6-16 kHz band, where the render itself only sits at -41 dB. That is 0.089 of loss, an order of magnitude more than the differences the last several refits have moved. Renders are now written as `PCM_24` through `synth.write_render`, measured identical to float to four decimals, so the delivered file reproduces the delivered number. Every loss quoted in this report is measured in memory or from a `PCM_24` file.
- **A MIDI-write defect was found and fixed late.** My measured region boundaries overlapped by ~50 ms (region 2 ended at 10.45 s while region 3 began at 10.40 s) and F3 occurs in both chords. Writing that to a MIDI file emits a note-off for the earlier F3 inside the later one, truncating it to 50 ms — so the delivered `transcription.mid` did not contain what had been rendered. Region ends are now snapped to the next region's start, the file resolution is 1920 ticks/beat (round-trip error 0.26 ms), and `render.wav` is produced from notes **reloaded from the file**, with both properties asserted. Absolute losses quoted before this fix were measured from in-memory notes and were ~1% optimistic relative to the artifacts.
- **The continuity-constrained f0 tracker is not trustworthy for pitch moves.** Its jump penalty hid a real pitch change twice: it drew a flat F1 through the gliding intro, and it reported the bass staying on D#1 across the +3 bend. Both were settled by direct spectral measurement instead (at 14.2-15.8 s the F#1 fundamental measures 1.000 normalised amplitude against D#1's 0.012). Window-based harmonic-comb fitting and NNLS were the reliable tools here; the tracker was only ever useful for spotting *that* something moved.
- **The filter envelope re-triggers on every chord**, producing a small transient at each change that the original does not have. It survives because it costs little under the loss.

## Files

| file | what |
|---|---|
| `out/transcription.mid` | frozen stage-1 MIDI |
| `out/patch.json` | fitted parameters + metrics |
| `out/render.wav` | transcription.mid through the matched patch |
| `out/comparison.png` | mel spectrograms, loudness envelopes, L/R correlation, loss curve |
| `out/new_stages.json` | drive/spread surrogate parity, append-only compatibility, coarse grid |
| `out/drive_probe.json` | the drive-free vs drive-pinned CMA-ES control |
| `out/defect_check.json` | stereo and per-band measurements, before and after |
| `out/spread_scan.json` | loss and L/R decorrelation against stereo width |
| `out/patch_baseline27.json` | the 27-parameter patch delivered before the two new stages |
| `out/baseline27_render.wav` | its render, for the before/after comparison |
| `out/spectrum_compare.png` | long-term average spectra |
| `out/stage1_final.wav` | stage-1 render (reference synth) |
| `out/pitch_track.png` | full-clip f0 trajectory |
| `out/intro_glide.png` | tracked intro partials, all sweeping together |
| `out/intro_f0.npy` | measured glide trajectory driving the bend |
| `scripts/synth.py` | the Faust synth + parameter space |
