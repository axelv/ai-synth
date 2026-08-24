declare name "acid-lead";
declare description "303-style acid line: sharp saw into a resonant ladder lowpass that sweeps on every note, with accent and post-filter overdrive.";

import("stdfaust.lib");

//======================================================================
// poly interface (one voice)
//======================================================================
freq = hslider("freq", 440, 20, 8000, 0.001);
gain = hslider("gain", 0.5, 0, 1, 0.001);
gate = button("gate");

//======================================================================
// macros: what a musician would reach for, not what the DSP touches
//======================================================================
// where the filter sits before the sweep starts: closed and dark vs open and nasal
brightness = hslider("brightness", 0.34, 0, 1, 0.001);
// how far the filter travels on each note. this is the "wow" of the acid line
movement   = hslider("movement", 0.72, 0, 1, 0.001);
// resonance. below ~0.5 it is just a lowpass, near 1 it whistles and squelches
squelch    = hslider("squelch", 0.80, 0, 1, 0.001);
// how long the sweep takes to fall back down, in seconds
decay      = hslider("decay", 0.30, 0.03, 1.5, 0.001);
// post-filter overdrive. also what stops the resonant peak from clipping
drive      = hslider("drive", 0.55, 0, 1, 0.001);
// how much harder a hard-hit note sounds: brighter, more resonant, louder
accent     = hslider("accent", 0.75, 0, 1, 0.001);
// stereo spread of the slapback (declared in the shared effect chain)
width      = hslider("width", 0.35, 0, 1, 0.001);

//======================================================================
// voice
//======================================================================
// A 303 has no velocity, it has an accent switch. Map incoming velocity through a
// knee so the loud end of a normal MIDI performance (~100-127) lands on the accented
// half of the range and everything below stays flat and unaccented.
acc = accent * min(1, pow(max(0, (gain - 0.45) / 0.55), 1.5));

// Filter envelope: near-instant attack, exponential fall to zero while the key is
// still held. Sustain level 0 is the whole point, the sweep must complete under a
// held note. Accented notes snap back faster.
fenv = en.adsre(0.004, decay * (1 - 0.3 * acc), 0.0, 0.06, gate);

// Cutoff tracks pitch only partially (exponent well under 1). Full tracking makes the
// top of the keyboard shrill and the bottom lifeless; none at all makes high notes
// vanish behind the filter.
keytrack = pow(freq / 261.63, 0.4);
cutBase  = 95 * pow(2, brightness * 5.2) * keytrack;
sweepOct = movement * 4.2 + acc * 2.2;
// upper clamp is a stability limit of the ladder approximation, not a taste choice:
// moog_vcf needs fr well under SR/6.3
cutoff = min(0.13 * ma.SR, max(35, cutBase * pow(2, sweepOct * fenv)));

reso = min(0.985, 0.70 + squelch * 0.26 + acc * 0.04);

// Sharp saw. The second saw is 6 cents up at low level: not a 303 feature, but it
// keeps single held notes from sounding static without softening the edge.
osc = os.sawtooth(freq) * 0.82 + os.sawtooth(freq * 1.0035) * 0.18;

// The ladder loses most of its level at DC as resonance rises (DC gain is 1/(1+4res)),
// so put it back, otherwise turning up squelch just turns the patch down.
resComp = 0.4 + 1.6 * reso;

// Overdrive after the filter, like an acid box into a distortion pedal. It is also the
// only thing keeping the resonant peak inside the rails, so it is not optional.
sat(x) = ma.tanh(x * (1.2 + drive * 9.0));

ampenv = en.adsre(0.003, 1.2, 0.85, 0.045, gate);

voice = osc * 0.5
      : ve.moog_vcf(reso, cutoff)
      : *(resComp)
      : sat
      : *(ampenv * (0.35 + 0.65 * gain) * 0.55);

process = voice <: _, _;

//======================================================================
// shared effect: slapback with different times per side, which is where the
// stereo comes from. The voice itself is mono, as an acid line should be.
//======================================================================
combL = 0.128 * ma.SR;
combR = 0.171 * ma.SR;
fbk   = 0.26;

echo(dt, x) = x : (+ : de.fdelay(65536, dt)) ~ *(fbk);

// keep the low end out of the echo so the bottom stays mono and tight
wetize(dt, x) = x : fi.highpass(2, 220) : echo(dt);

wet = 0.08 + width * 0.30;

effect(l, r) = l * (1 - 0.5 * wet) + wetize(combL, l) * wet,
               r * (1 - 0.5 * wet) + wetize(combR, r) * wet;
