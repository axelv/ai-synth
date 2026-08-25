declare name "juno-106";
declare description "Roland Juno-106 style polysynth. One DCO per voice with saw, PWM pulse, square sub and noise, a 1-pole HPF into a 4-pole resonant VCF with key follow, a single ADSR serving both filter and amp, and the BBD stereo chorus.";

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
brightness = hslider("brightness[panel:VCF][idx:1]", 0.44, 0, 1, 0.001) : si.smoo;
resonance  = hslider("resonance[panel:VCF][idx:2]", 0.20, 0, 1, 0.001) : si.smoo;
sweep      = hslider("sweep[panel:VCF][idx:3]", 0.42, 0, 1, 0.001) : si.smoo;
tone       = hslider("tone[panel:DCO][idx:1]", 0.62, 0, 1, 0.001) : si.smoo;
swell      = hslider("swell[panel:ENV][idx:1]", 0.26, 0, 1, 0.001);

//======================================================================
// Voice
//======================================================================

// The DCOs are deliberately NOT decorrelated per voice. A Juno's oscillators are
// digitally reset and stay phase-locked, which is why its chords sit so still and
// why the machine needs a chorus at all. Only the noise differs per voice, as it
// does in hardware, where every voice board carries its own source.
vseed  = ma.frac(freq * 0.0177);
vnoise = no.noise : de.delay(4096, int(vseed * 4000));

// A harder player should get brighter more than louder, so velocity is compressed
// on level and spent on the filter.
vel = gain ^ 0.6;

// ONE envelope generator, as on the panel. The 106 has a single ADSR wired to both
// the VCF and the VCA, so reusing this signal is the architecture rather than a
// shortcut: any attack slow enough to swell the filter also swells the level.
att = 0.004 + swell * 2.40;
dec = 0.55  + swell * 1.30;
rel = 0.10  + swell * 3.20;
env = en.adsre(att, dec, 0.74, rel, gate);

// One LFO per voice with the panel's DELAY, so it ramps in after note-on instead of
// being at full depth from the first sample. The rate is a voicing choice, not a
// measured one: PWM this slow reads as the pad drifting rather than as vibrato, and
// it is where a Juno pad player leaves the knob. Measured, the rate barely touches
// level at all, 0.53 dB of sustain swing at 4.2 Hz against 0.47 dB at 1.35 Hz.
lfo = os.osc(1.35) * en.adsre(1.15, 0.05, 1.0, 0.15, gate);

// PWM sweeps from square DOWN toward a narrow hollow pulse, which is the direction
// the Juno's own control moves. Never reaches zero width, where the pulse vanishes,
// and shallow enough that narrowing the pulse does not read as a level change.
pwDepth = 0.10 + 0.18 * tone;
pw      = 0.5 - pwDepth * (0.5 + 0.5 * lfo);

// A few cents of LFO to pitch. The panel default for VCO mod is zero and this is
// less than that; it is here so the delayed LFO is audible as movement on a held
// note rather than only as a slow change of pulse width.
f0 = freq * (1 + 0.0018 * lfo);

saw = os.sawtooth(f0);
// Normalised by the fundamental of a pulse of this width, (4/pi)*sin(pi*pw).
// Narrowing the pulse moves energy up into harmonics the VCF then removes, so without
// this the width sweep is partly a level sweep: measured on a held note it costs
// 1.21 dB of sustain swing at tone=1 against 0.47 dB with it, and 0.28 dB is the floor
// this measurement has with the LFO off entirely. Floored so the divisor cannot
// approach zero.
pul = os.pulsetrain(f0, pw) / max(0.34, sin(ma.PI * pw));
sub = os.square(f0 * 0.5);

// The DCO mixer's four faders, driven as one intent: bright and reedy at 0, hollow
// and fat at 1. Normalised to constant energy so `tone` moves timbre and not volume.
sawL = 0.95 - 0.55 * tone;
pulL = 0.08 + 0.80 * tone;
subL = 0.10 + 0.52 * tone;
nseL = 0.012 + 0.040 * tone;
mixG = 1.0 / sqrt(sawL*sawL + pulL*pulL + subL*subL + nseL*nseL);
dco  = (saw*sawL + pul*pulL + sub*subL + vnoise*nseL) * mixG;

// The panel's 4-position HPF, made continuous and folded into the same intent: the
// thin setting wants the bottom out of the way, the fat setting wants it back. One
// pole, because that is what the hardware is, and because a steeper filter here
// would eat the fundamental at the bottom of the keyboard.
hpF = 34 + 300 * (1 - tone) * (1 - tone);

// Half-power key tracking: an octave up moves the cutoff by a fifth, so the top of
// the keyboard stays soft rather than turning glassy.
ktrack = (freq / 261.6256) ^ 0.5;

// `sweep` opens the envelope's depth AND pulls the standing cutoff down underneath
// it, so turning it up buys movement instead of just brightness.
cutBase = 120 * pow(35, brightness) * (1 - 0.45 * sweep);

// The ceiling is a property of ve.moog_vcf, not a taste decision. Measured on this
// material it returns non-finite samples above roughly 7 kHz at low resonance and
// above 5 kHz at high resonance, so the safe cutoff depends on both. Written against
// ma.SR because the instability scales with sample rate and the page may run at 48k.
fcMax = ma.SR * (0.142 - 0.036 * resonance);
fc = max(60, min(fcMax, cutBase * ktrack * (0.55 + 0.45*vel) * (1 + sweep * 6.5 * env)));

// Resonance also drives the filter harder and makes up the gain again, because a
// pushed filter is half of what a resonant Juno actually sounds like, and because
// without the makeup the control would be a loudness knob.
res    = 0.05 + resonance * 0.87;
push   = 1.0 + 1.10 * resonance;
// Makeup BOOSTS. A ladder loses passband level as it resonates, so the obvious
// compensation is upside down: written as a divider it turned `resonance` into a
// 13 dB fader, which is the volume-control failure with the sign flipped.
makeup = 1.0 + 0.90 * resonance;

// Small-signal unity, compresses above it. Per voice, never on the shared bus, so
// the timbre does not change with how many notes are held.
sat(x) = x / sqrt(1 + 0.8 * x * x);

// The DC blocker is not decoration. `sat` is an odd function, so it cannot make DC
// from a symmetric wave, but a saw plus a narrow pulse is not symmetric about zero
// and an odd curve applied to it has a nonzero mean: measured +0.039 without this,
// four times the threshold, and gone entirely with the saturator bypassed.
voice = dco : fi.highpass(1, hpF) : *(push) : sat
            : ve.moog_vcf(res, fc) : fi.dcblocker : *(makeup);
amp   = env * (0.32 + 0.68 * vel) * 0.30;

// The voice is mono. Every bit of width in this instrument comes from the chorus,
// which is historically exactly the case.
process = voice * amp <: _,_;

//======================================================================
// Shared effect chain: the BBD chorus, and nothing else
//======================================================================
// Off / I / II, as three positions rather than a continuous control, because that
// is what the machine has: two buttons, and both up is off.
chorus = hslider("chorus[panel:CHORUS][positions:OFF|I|II]", 0.5, 0, 1, 0.5);

chOn = chorus > 0.25;
fast = chorus > 0.75;

bbd(l, r) = l*dry + wl*wet, r*dry + wr*wet
with {
    rate  = select2(fast, 0.513, 0.863);      // mode I and mode II, measured rates
    depth = select2(fast, 2.55, 3.30);        // ms of deviation either side
    base  = 4.60;                             // ms
    clfo  = os.osc(rate);
    // Antiphase on the two lines is what makes the width. The hardware instead
    // inverts one wet output, which is why a Juno through a mono desk loses its
    // chorus completely; this keeps the mono sum intact and the width comparable.
    dl = ma.SR * 0.001 * (base + depth * clfo);
    dr = ma.SR * 0.001 * (base - depth * clfo);
    // A bucket brigade is dark and noisy at the top. The lowpass is the part of
    // that worth keeping.
    wl = de.fdelay(2048, dl, l) : fi.lowpass(2, 7200);
    wr = de.fdelay(2048, dr, r) : fi.lowpass(2, 7200);
    dry = 1 - 0.28 * chOn;
    wet = 0.62 * chOn;
};

// No reverb. The 106 has none, and it is the one machine whose owners all agree
// belongs in front of whatever reverb they already have.
effect = bbd : par(i, 2, *(0.95));
