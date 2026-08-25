// breathy-texture: an airy, barely-pitched noise pad.
//
// The sound is filtered noise, not filtered oscillators. Pitch is present only as a
// set of narrow resonances riding on the same noise the body is made of, so a note
// reads as a colour rather than a tone. The "wind" is a bandpass pair sweeping on a
// modulator built from three sine LFOs at mutually irrational rates plus a slow random
// walk: the sum has no common period, which is what stops the movement from looping.
// Left and right take decorrelated noise and independent modulators, so the width is
// real rather than a delay effect.

import("stdfaust.lib");

//---------------------------------------------------------------- poly interface
freq = hslider("freq", 440, 20, 8000, 0.001);
gain = hslider("gain", 0.5, 0, 1, 0.001);
gate = button("gate");

//---------------------------------------------------------------- macros
// where the wind sits: dark rushing air up to a bright hiss
brightness = hslider("brightness", 0.52, 0, 1, 0.001) : si.smoo;
// how far and how fast the filter opens and closes
movement   = hslider("movement", 0.62, 0, 1, 0.001) : si.smoo;
// breath against pitch. at 1 it is almost pure air, at 0 a hollow whistle tone
air        = hslider("air", 0.78, 0, 1, 0.001) : si.smoo;
// how slowly a note swells in
bloom      = hslider("bloom", 0.45, 0, 1, 0.001);
// how long it takes to disappear after the key is up
fade       = hslider("fade", 0.55, 0, 1, 0.001);

//---------------------------------------------------------------- envelope
att  = 0.05 + 2.4 * bloom;
rel  = 0.25 + 4.5 * fade;
env  = en.asr(att, 1.0, rel, gate);
// the filter lags the amplitude, so the note is heard to open after it appears
benv = env : si.smooth(ba.tau2pole(0.45));

// velocity: louder, and a little brighter, the way harder breath actually behaves
vel = 0.30 + 0.70 * gain;

//---------------------------------------------------------------- modulation
// rates are pulled slightly by pitch so two held notes never breathe in step
vscale = 0.75 + 0.55 * (freq / 900.0 : min(1.0) : max(0.05));
mrate  = (0.28 + 1.45 * movement) * vscale;

lfo(r) = os.osc(r * mrate);

driftA = 0.46 * lfo(0.0730) + 0.31 * lfo(0.0311) + 0.23 * lfo(0.1270);
driftB = 0.46 * lfo(0.0611) + 0.31 * lfo(0.0367) + 0.23 * lfo(0.1103);

modA = 0.68 * driftA + 0.32 * no.lfnoise(0.33 * mrate);
modB = 0.68 * driftB + 0.32 * no.lfnoise(0.27 * mrate);

//---------------------------------------------------------------- the wind
// partial keyboard tracking: full tracking makes high notes shriek, none makes the
// texture the same colour everywhere
ftrack = pow(freq / 261.626, 0.55);
fcbase = 900.0 * (0.35 + 3.1 * brightness) * ftrack * (0.62 + 0.55 * gain);

cut(m) = fcbase * pow(2.0, 2.4 * movement * m) * (0.42 + 0.62 * benv)
       : max(70.0) : min(0.42 * ma.SR);

// a wide band for the body and a narrow one on the same centre for the whistle
wind(n, m) = 0.75 * (n : fi.resonbp(fq, 1.1, 1.0))
           + 0.45 * (n : fi.resonbp(fq, 5.0 + 3.0 * brightness, 1.0))
with {
    fq = cut(m);
};

//---------------------------------------------------------------- the pitch, barely
p1 = freq            : min(0.42 * ma.SR);
p2 = 2.0 * freq      : min(0.42 * ma.SR);
p3 = 3.0 * freq      : min(0.42 * ma.SR);

// a narrow resonator passes energy proportional to its bandwidth, which scales with
// centre frequency, so low notes need the compensation to stay as present as high ones
pcomp = sqrt(440.0 / freq) : min(2.2) : max(0.45);

tone(n) = pcomp * ( 0.60 * (n : fi.resonbp(p1, 30.0, 1.0))
                  + 0.34 * (n : fi.resonbp(p2, 26.0, 1.0))
                  + 0.20 * (n : fi.resonbp(p3, 20.0, 1.0)) );

// just enough steady tone to tell the listener which note it is
sines = 0.050 * os.osc(p1) + 0.026 * os.osc(p2);

windlvl = 0.40 + 0.60 * air;
tonelvl = 1.00 - 0.75 * air;

chan(n, m) = windlvl * wind(n, m) + tonelvl * (5.5 * tone(n) + 0.85 * sines);

//---------------------------------------------------------------- voice
src = no.noise;
nL  = src;
// white noise delayed by more than one sample is uncorrelated with itself, which is
// the cheapest honest way to get two independent breaths
nR  = src : @(211);

trim = 0.14;
ampL = trim * env * vel * (0.76 + 0.24 * modA);
ampR = trim * env * vel * (0.76 + 0.24 * modB);

process = (chan(nL, modA), chan(nR, modB)) : (*(ampL), *(ampR));

//---------------------------------------------------------------- shared chain
width = hslider("width", 0.45, 0, 1, 0.001) : si.smoo;
space = hslider("space", 0.55, 0, 1, 0.001) : si.smoo;

wetamt = 0.20 + 0.65 * space;

rev = re.zita_rev1_stereo(0.0, 200.0, 5500.0, 4.5, 2.8, 48000);

widen(l, r) = m + s, m - s
with {
    m = (l + r) * 0.5;
    s = (l - r) * 0.5 * (0.30 + 1.70 * width);
};

hp = fi.highpass(2, 42.0);

// bounded above by 0.80 (-1.9 dBFS) and transparent well below it: a safety ceiling,
// not a compressor
ceiling = *(1.25) : ma.tanh : *(0.80);

effect = _,_
       <: ( *(1.0 - 0.5 * wetamt),
            *(1.0 - 0.5 * wetamt),
            ( (*(wetamt), *(wetamt)) : rev ) )
       :> widen
        : (hp : ceiling), (hp : ceiling);
