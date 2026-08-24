// Plucky FM bass: DX7-style slap, modernised.
//
// Two phase-modulators into one sine carrier. The 1:1 operator supplies the growl
// and dies back within the note, so what is left holding the bottom is a plain sine
// and not a filtered saw; that is what keeps it tight instead of boomy. The 7:1
// operator is the metallic bite and is given its own, much faster envelope, because
// on the real instrument the clang is an attack transient and not a timbre.
// A short filtered noise burst rides the attack for the modern slap consonant.

import("stdfaust.lib");

// --- poly convention -------------------------------------------------------
freq = hslider("freq", 440, 20, 8000, 0.001);
gain = hslider("gain", 0.5, 0, 1, 0.001);
gate = button("gate");

// --- macros ----------------------------------------------------------------
brightness = hslider("brightness", 0.55, 0, 1, 0.001); // growl / harmonic reach
bite       = hslider("bite", 0.60, 0, 1, 0.001);       // metallic attack clang
snap       = hslider("snap", 0.070, 0.01, 0.40, 0.001);// how fast that clang goes
decay      = hslider("decay", 0.34, 0.05, 2.0, 0.001); // note length
body       = hslider("body", 0.16, 0, 0.80, 0.001);    // fundamental left holding
drive      = hslider("drive", 0.28, 0, 1, 0.001);      // saturation
width      = hslider("width", 0.35, 0, 1, 0.001);      // spread, highs only

// --- helpers ---------------------------------------------------------------
sn(p) = sin(2*ma.PI*p);
ph(f) = os.phasor(1.0, f);

kf = max(20.0, freq);

// Timbre here has to track absolute frequency, not harmonic number: a constant
// phase-mod index played two octaves up lands its sidebands two octaves higher
// and turns to fizz. Pull the index down as the note rises.
kscale = min(1.6, max(0.45, pow(110.0/kf, 0.28)));

// Harder playing is brighter as well as louder, which is most of what makes an
// FM bass feel responsive.
velIdx = 0.30 + 0.90*gain;
velAmp = 0.28 + 0.72*gain;

// A struck string sheds its high partials faster the shorter it is.
snapT = max(0.008, snap * min(1.4, pow(90.0/kf, 0.30)));

ampEnv   = en.adsre(0.003,  decay,        body,  0.10, gate);
idx1Env  = en.adsre(0.002,  decay*0.55,   0.14,  0.06, gate);
idx2Env  = en.adsre(0.0008, snapT,        0.001, 0.02, gate);
clickEnv = en.adsre(0.0004, 0.028,        0.001, 0.01, gate);

i1 = (0.80 + 3.20*brightness) * velIdx * kscale * idx1Env;
i2 = (1.60 + 5.00*bite)       * velIdx * kscale * idx2Env;

// 7.01 rather than 7: a hair of inharmonicity makes the bite ring instead of
// fusing into the harmonic series.
car = sn(ph(kf) + i1*sn(ph(kf)) + i2*sn(ph(kf*7.01))) * ampEnv;

clkF  = min(4500.0, max(800.0, kf*13.0));
click = no.noise : fi.resonbp(clkF, 1.1, 1.0) : *(clickEnv * 0.55 * bite * velAmp);

// Tone control tracks the note so the top octave does not go dull.
cut = min(15000.0, (700.0 + 7500.0*brightness) * pow(kf/65.0, 0.30));

voice = (car*0.92 + click)
      : fi.lowpass(2, cut)
      : fi.highpass(2, 38)   // no sub-40 rumble; the fundamental is the low end
      : *(velAmp);

sat(x) = ma.tanh(x*k) / ma.tanh(k) with { k = 1.0 + 7.0*drive; };

// Mid/side by construction: L+R is exactly the input, so the spread cannot
// thin the bass on a mono system. Only the band above 600 Hz is decorrelated.
spread(x) = (x + s, x - s)
with {
    hb = x : fi.highpass(2, 600);
    s  = 0.5 * width * (hb - (hb : @(26)));
};

process = voice : sat : *(0.55) : spread;
