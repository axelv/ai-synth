# What separates a patch from a parameter set

The deliverable is something a person can keep dialling, not a fixed sound with sliders
attached. Read when choosing which macros to expose and what their ranges and defaults
should be.

- [A macro must move shape, not level](#a-macro-must-move-shape-not-level)
- [Normalise anything that adds gain](#normalise-anything-that-adds-gain)
- [Envelope macros are the exception](#envelope-macros-are-the-exception)
- [A macro drives at least two destinations](#a-macro-drives-at-least-two-destinations)
- [Declared units must match audible behaviour](#declared-units-must-match-audible-behaviour)
- [Ranges and defaults](#ranges-and-defaults)
- [Measured examples](#measured-examples)

## A macro must move shape, not level

The test that does not need to know what a macro was for: compare its effect on overall
level against its effect on **level-normalised** spectral shape. A control that moves
band energy but leaves the normalised profile flat is a volume knob, whatever it is
called. `measure.py` fails this as `is a volume control`.

This is the one case where measurement genuinely beats listening. Louder reads as better,
which is why gain-matched comparison is standard practice in audio. Turning up a `drive`
that is secretly 10 dB of gain sounds exactly like drive working.

## Normalise anything that adds gain

A saturator written the obvious way is a volume control:

```faust
// wrong: 10.6 dB of level for 2.9 dB of spectral shape
shaped = ma.tanh(x * k);

// right: divide by the shaper's own response to unity, so small signals are boosted
// and the peak stays put
shaped = ma.tanh(x * k) / ma.tanh(k);
```

The same applies to width. A mid/side widener must preserve energy, or `width` becomes a
loudness control that also happens to clip:

```faust
// wrong: peaks 1.277 at width = 1
spread(l, r) = l + s, r - s with { s = (l - r) * w; };

// right: normalise by the gain the matrix applies
spread(l, r) = (l + s) / g, (r - s) / g
  with { s = (l - r) * w; g = sqrt(1 + w * w); };
```

The rule is really about energy, not about gain, so it reaches further than the controls
that obviously multiply. Pulse width modulation is the case that does not look like one:
narrowing a pulse moves its energy up into harmonics a lowpass then removes, so a PWM
sweep is partly a level sweep even though nothing in it multiplies anything. Normalise
by the fundamental of a pulse of that width, `(4/pi)*sin(pi*pw)`:

```faust
// wrong: the width sweep is audible as level
pul = os.pulsetrain(f0, pw);

// right: floored so the divisor cannot approach zero
pul = os.pulsetrain(f0, pw) / max(0.34, sin(ma.PI * pw));
```

Measured on a held note, sustain level swing went from 1.21 dB to 0.47 dB, against a
0.28 dB floor with the modulation switched off entirely.

## Envelope macros are the exception

A longer decay genuinely has more sustained energy. A higher sustain level is genuinely
louder. Do not gain-compensate these; the level change *is* the control working.

`measure.py` distinguishes the two by comparing level against shape and only warns when
level moves more than shape does. Read the warning with the macro's purpose in mind:

```
warn  decay moves level more than shape (15.5 dB against 8.4 dB)
```

For `decay` that is correct behaviour. For `drive` it is a defect.

## A macro drives at least two destinations

A macro that renames one DSP parameter is a parameter with a nicer label. In the measured
set, five of one patch's seven macros were single-destination renames: `brightness` to
cutoff, `movement` to sweep depth, `squelch` to resonance, `decay` to an envelope time,
`drive` to a saturation constant.

Macros that earned the name, and what they touched:

- `air` crossfaded between two synthesis layers
- `sparkle` drove three destinations at once
- `decay` rescaled all eight per-mode decay times together
- `bite` drove both an FM index and an attack click level

## Declared units must match audible behaviour

An exponential envelope through an exponential cutoff mapping collapses. One patch
declared `decay` in seconds with a default of 0.30 s; measured, 90% of the filter travel
was done in **12 ms**, a factor of 25 off what the number claims.

The `note` line of the report gives this directly:

```
note  centroid 229 -> 235 Hz, 50% of that travel by 0.012 s, 90% by 0.012 s
```

Either fix the mapping or stop declaring a unit the control does not deliver.

## Ranges and defaults

**A declared range is a range a player will turn to.** Two of five patches clipped at a
macro extreme they shipped as usable: `width` at 1 peaked 1.277, `decay` at 14 peaked
1.116. Render both ends of every macro before shipping. `measure.py` does this and fails
on it.

**Defaults are a finished sound, not a midpoint.** Six of one patch's seven defaults sat
between 0.45 and 0.62, which is a vector of shrugs rather than a chosen sound. Asymmetric
ranges with an off-centre default read as decided: `snap` at 0.07 within 0.01 to 0.40.

**Target peak -6 to -1 dBFS on the measurement pattern**, at the polyphony it uses.
Set the master level as a final measured step rather than guessing a constant. Three of
ten render invocations in the measured batch were spent purely trimming master gain.

## Measured examples

From five patches written one-shot from plain-language descriptions. `level` and `shape`
are dB across the macro's full range; a good timbre macro has shape well above level.

| macro | level | shape | verdict |
|---|---|---|---|
| `sparkle` | 0.0 | 10.3 | ideal: pure timbre, no level change |
| `brightness` (pad) | 6.1 | 26.4 | good: a real filter, level change is incidental |
| `movement` | 0.4 | 7.4 | good |
| `drive` | 10.6 | 2.9 | defect: a volume knob labelled drive |
| `body` | 10.6 | 6.2 | acceptable, it is a sustain level |
| `accent` | 0.0 | 1.8 | weak: barely moves, and slightly the wrong direction |
