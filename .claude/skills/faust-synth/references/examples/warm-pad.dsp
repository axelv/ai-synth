declare name "warm-pad";
declare description "Warm analog pad. Triple detuned saw plus PWM and sub octave, soft 4-pole filter, ensemble chorus and hall.";

import("stdfaust.lib");

//======================================================================
// Poly interface. Fixed by the Faust polyphonic convention.
//======================================================================
freq = hslider("freq", 440, 20, 8000, 0.001);
gain = hslider("gain", 0.5, 0, 1, 0.001);
gate = button("gate");

//======================================================================
// Macros, per voice
//======================================================================
brightness = hslider("brightness", 0.42, 0, 1, 0.001) : si.smoo;
detune     = hslider("detune", 0.38, 0, 1, 0.001) : si.smoo;
swell      = hslider("swell", 0.45, 0, 1, 0.001);
tail       = hslider("tail", 0.5, 0, 1, 0.001);

//======================================================================
// Voice
//======================================================================

// Small-signal unity, compresses above it. sqrt beats tanh here only because
// it is a Faust primitive everywhere, so this cannot fail to compile.
sat(x) = x / sqrt(1 + 0.8 * x * x);

// Velocity is deliberately gentle on level and stronger on the filter: on a pad
// a hard player should sound brighter, not much louder, or it will not sit still
// under a vocal.
vel = gain ^ 0.6;

// Envelope times. The swell is the whole character of the patch, so it gets the
// widest range of any macro.
att = 0.06 + swell * 3.2;
rel = 0.35 + tail * 4.0;
aenv = en.adsre(att, 1.8, 0.80, rel, gate);
// The filter opens later than the amplitude does, which is what reads as "swell"
// rather than "fade in".
fenv = en.adsre(att * 1.6 + 0.15, 2.6, 0.55, rel * 0.8, gate);

// Half-power key tracking: a full octave up moves the cutoff by a fifth, so the
// top of the keyboard stays soft instead of turning thin and glassy.
ktrack = (freq / 261.6256) ^ 0.5;
cutoff = max(70, min(15000,
    260 * (28 ^ brightness) * ktrack * (0.55 + 0.45 * vel) * (0.40 + 0.60 * fenv)));

// Free-running slow LFOs at mutually irrational-ish rates: the three saws never
// re-align, which is the whole trick behind an analog ensemble.
drift1 = os.osc(0.13);
drift2 = os.osc(0.19);
drift3 = os.osc(0.077);
common = os.osc(0.061);

spread = detune * 0.009;                      // up to ~15 cents either side
wob    = detune * 0.0018;
base   = freq * (1 + 0.0006 * common);
f1 = base * (1 - spread + wob * drift1);
f2 = base * (1 + wob * drift2 * 0.7);
f3 = base * (1 + spread + wob * drift3);

saw1 = os.sawtooth(f1);
saw2 = os.sawtooth(f2);
saw3 = os.sawtooth(f3);
// The Juno's pulse-width sweep. Slow, and narrow enough never to go hollow.
pwm  = os.pulsetrain(f2, 0.5 + 0.16 * os.osc(0.09));
// Sine sub rather than the classic square: it thickens the bottom without adding
// anything in the range a voice occupies.
sub  = os.osc(base * 0.5);

// The two outer saws are panned apart at the source. Chorus alone gives width
// that collapses in mono; detuned sources placed apart survive it.
dry_l = 0.40*saw1 + 0.26*saw2 + 0.16*saw3 + 0.20*pwm + 0.26*sub;
dry_r = 0.16*saw1 + 0.26*saw2 + 0.40*saw3 + 0.20*pwm + 0.26*sub;

// Saturate before the filter, so the harmonics drive adds are then removed again
// from the top. That is the order an analog voice uses, and why it sounds thick
// without sounding bright.
chan = *(1.25) : sat : fi.lowpass(4, cutoff) : fi.highpass(2, 30);

amp = aenv * (0.30 + 0.70 * vel) * 0.30;

process = (dry_l : chan : *(amp)), (dry_r : chan : *(amp));

//======================================================================
// Shared effect chain
//======================================================================
width = hslider("width", 0.55, 0, 1, 0.001) : si.smoo;
space = hslider("space", 0.34, 0, 1, 0.001) : si.smoo;

// Two modulated delays at unrelated rates. Juno-style: mostly dry, the wet path
// only widens and thickens.
chorus(l, r) = l*0.72 + wl*wetg, r*0.72 + wr*wetg
with {
    wetg  = 0.30 + 0.45 * width;
    depth = 2.0 + 4.0 * width;
    dl = ma.SR * 0.001 * (9.0  + depth * os.osc(0.29));
    dr = ma.SR * 0.001 * (11.5 + depth * os.osc(0.41));
    wl = de.fdelay(2048, dl, l);
    wr = de.fdelay(2048, dr, r);
};

// A broad, shallow dip through the vocal's home register. This is the difference
// between a pad that supports a voice and one that argues with it.
scoop(x) = x - 0.34 * (x : fi.lowpass(1, 3400) : fi.highpass(1, 1100));
// Nothing harsh on top.
soften = fi.lowpass(1, 6800);

verb = re.zita_rev1_stereo(40, 200, 5200, 3.4, 2.4, 48000);

reverb_mix = _,_ <: (par(i, 2, *(1 - 0.30 * space))),
                    (verb : par(i, 2, *(1.05 * space)))
             :> _,_;

effect = chorus : (scoop : soften), (scoop : soften) : reverb_mix : par(i, 2, *(0.95));
