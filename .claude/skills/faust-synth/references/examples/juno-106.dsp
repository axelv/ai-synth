declare name "juno-106";
declare description "Roland Juno-106 style polysynth, laid out as the machine's own panel. One DCO per voice with saw, PWM pulse, square sub and noise, a 1-pole HPF into a 4-pole resonant VCF with key follow, one ADSR serving both filter and amp, and the BBD stereo chorus.";

import("stdfaust.lib");

//======================================================================
// Poly interface. Fixed by the Faust polyphonic convention.
//======================================================================
freq = hslider("freq", 440, 20, 8000, 0.001);
gain = hslider("gain", 0.5, 0, 1, 0.001);
gate = button("gate");

//======================================================================
// The panel.
//
// This file models a named machine, so the controls are the machine's faders
// rather than a handful of intents. That is a deliberate departure from
// patch-design.md, which asks for 4 to 7 macros each driving several
// destinations: here the point IS the one-to-one correspondence with the
// hardware, and a player who knows a Juno should find every fader where they
// expect it. The cost is recorded in measured.md and is real: LEVEL is a
// volume control and measures as one.
//
// `panel` places a control in a section, `idx` orders it within that section
// (Faust emits controls alphabetically, not in source order), and `cap` gives
// the short name printed on the hardware. Two sections both have a fader
// called LFO and two have one called ENV, which is why the Faust labels have
// to differ from the printed names.
//======================================================================

// ---- LFO ----
lfoRate  = hslider("rate[panel:LFO][idx:1][cap:RATE]", 1.35, 0.1, 30, 0.01) : si.smoo;
lfoDelay = hslider("delay[panel:LFO][idx:2][cap:DELAY]", 1.15, 0, 3, 0.01);

// ---- DCO ----
// RANGE is the footage switch: 16', 8', 4'.
dcoRange = hslider("range[panel:DCO][idx:1][cap:RANGE][positions:16|8|4]", 1, 0, 2, 1);
dcoLfo   = hslider("dcoLfo[panel:DCO][idx:2][cap:LFO]", 0.04, 0, 1, 0.001) : si.smoo;
dcoPwm   = hslider("pwm[panel:DCO][idx:3][cap:PWM]", 0.47, 0, 1, 0.001) : si.smoo;
// The panel's PWM source switch. MAN holds the width the fader sets; LFO sweeps it.
pwmMode  = hslider("pwmMode[panel:DCO][idx:4][cap:MODE][positions:LFO|MAN]", 0, 0, 1, 1);
dcoSaw   = hslider("saw[panel:DCO][idx:5][cap:SAW][positions:OFF|ON]", 1, 0, 1, 1) : si.smoo;
dcoPulse = hslider("pulse[panel:DCO][idx:6][cap:PULSE][positions:OFF|ON]", 1, 0, 1, 1) : si.smoo;
dcoSub   = hslider("sub[panel:DCO][idx:7][cap:SUB]", 0.45, 0, 1, 0.001) : si.smoo;
dcoNoise = hslider("noise[panel:DCO][idx:8][cap:NOISE]", 0.04, 0, 1, 0.001) : si.smoo;

// ---- HPF ----
// Four positions, not a continuous fader: the panel board encodes the slider onto two
// lines through a diode matrix, and the panel prints 3 2 1 0 beside it where every
// continuous fader prints 10 5 0. What each position DOES is off the jack board in the
// service notes, and it is not what a name like "HPF" suggests. See hpfStage below.
hpfPos = hslider("hpf[panel:HPF][idx:1][cap:FREQ]", 1, 0, 3, 1);

// ---- VCF ----
vcfFreq = hslider("cutoff[panel:VCF][idx:1][cap:FREQ]", 0.38, 0, 1, 0.001) : si.smoo;
vcfRes  = hslider("res[panel:VCF][idx:2][cap:RES]", 0.20, 0, 1, 0.001) : si.smoo;
vcfPol  = hslider("envPol[panel:VCF][idx:3][cap:POL][positions:NORM|INV]", 0, 0, 1, 1);
vcfEnv  = hslider("vcfEnv[panel:VCF][idx:4][cap:ENV]", 0.42, 0, 1, 0.001) : si.smoo;
vcfLfo  = hslider("vcfLfo[panel:VCF][idx:5][cap:LFO]", 0, 0, 1, 0.001) : si.smoo;
vcfKybd = hslider("kybd[panel:VCF][idx:6][cap:KYBD]", 0.5, 0, 1, 0.001) : si.smoo;

// ---- VCA ----
vcaMode  = hslider("vcaMode[panel:VCA][idx:1][cap:MODE][positions:ENV|GATE]", 0, 0, 1, 1);
vcaLevel = hslider("level[panel:VCA][idx:2][cap:LEVEL]", 0.80, 0, 1, 0.001) : si.smoo;

// ---- ENV ----
envA = hslider("attack[panel:ENV][idx:1][cap:A]", 0.60, 0, 1, 0.001);
envD = hslider("decay[panel:ENV][idx:2][cap:D]", 0.40, 0, 1, 0.001);
envS = hslider("sustain[panel:ENV][idx:3][cap:S]", 0.74, 0, 1, 0.001);
envR = hslider("release[panel:ENV][idx:4][cap:R]", 0.42, 0, 1, 0.001);

//======================================================================
// Voice
//======================================================================

// The DCOs are deliberately NOT decorrelated per voice. A Juno's oscillators are
// digitally reset and stay phase-locked, which is why its chords sit so still and
// why the machine needs a chorus at all.
//
// The noise IS decorrelated here, and that is a departure from the hardware rather
// than a model of it. The service notes show one noise generator for the whole
// instrument, a selected 2SC945 into a BA662, shared by all six voices; an earlier
// comment here claimed each voice board carried its own, and that was wrong. Kept
// because Faust poly voices are otherwise bit-identical, so a shared source would
// make noise sum coherently across a chord instead of forming a texture.
vseed  = ma.frac(freq * 0.0177);
vnoise = no.noise : de.delay(4096, int(vseed * 4000));

// A harder player should get brighter more than louder, so velocity is compressed
// on level and spent on the filter.
vel = gain ^ 0.6;

// ONE envelope generator, as on the panel. The 106 has a single ADSR wired to both
// the VCF and the VCA, so reusing this signal is the architecture rather than a
// shortcut: any attack slow enough to swell the filter also swells the level.
// Cubic on the time faders, so the short end stays controllable and the long end
// still reaches the machine's several seconds.
att = 0.001 + 3.0  * envA * envA * envA;
dec = 0.005 + 12.0 * envD * envD * envD;
rel = 0.005 + 12.0 * envR * envR * envR;
env = en.adsre(att, dec, envS, rel, gate);

// One LFO per voice with the panel's DELAY, so it ramps in after note-on instead of
// being at full depth from the first sample. Both the rate and the delay are now
// the player's, where they used to be baked in at 1.35 Hz and 1.15 s; those remain
// the defaults because that is where a Juno pad player leaves them.
lfo = os.osc(lfoRate) * en.adsre(max(0.005, lfoDelay), 0.05, 1.0, 0.15, gate);

// PWM sweeps from square DOWN toward a narrow hollow pulse, which is the direction
// the Juno's own control moves. The depth stops at 0.36, so the narrowest width is
// 0.14 and sin(pi*w) is 0.43, comfortably clear of the 0.34 floor in the normaliser
// below. Measured: at a depth of 0.45 the width reached 0.05, the floor engaged and
// the compensation stopped working, so the last sixth of the PWM fader was a 4.3 dB
// level drop. At 0.36 the RMS holds inside 1 dB across the whole fader.
pwDepth = 0.36 * dcoPwm;
// MAN parks the width where the fader puts it; LFO sweeps down from square.
pwLfo = 0.5 - pwDepth * (0.5 + 0.5 * lfo);
pwMan = 0.5 - pwDepth;
pw    = select2(pwmMode, pwLfo, pwMan);

// The footage switch. 16' 8' 4'.
oct = select3(int(dcoRange), 0.5, 1.0, 2.0);

// A few cents of LFO to pitch. The panel default for VCO mod is near zero, and so
// is this; it is here so the delayed LFO is audible as movement on a held note.
f0 = freq * oct * (1 + 0.05 * dcoLfo * lfo);

saw = os.sawtooth(f0);
// Normalised by the fundamental of a pulse of this width, (4/pi)*sin(pi*pw).
// Narrowing the pulse moves energy up into harmonics the VCF then removes, so
// without this the width sweep is partly a level sweep: measured on a held note it
// costs 1.21 dB of sustain swing against 0.47 dB with it. Floored so the divisor
// cannot approach zero.
//
// The leading 0.5 puts the pulse and the saw at the same fundamental, and it is
// arithmetic rather than taste: normalised this way the pulse's fundamental is 4/pi
// where a sawtooth's is 2/pi, so without it the pulse arrives 6.02 dB hot. Measured
// 5.71 dB, and with both waveforms switched on it drowned the saw: turning SAW off
// moved the spectrum by 0.9 dB, which is a switch that does nothing. It mattered
// less when one `tone` macro crossfaded the two across a 15 dB range. As two
// independent on/off switches, each has to carry its own weight.
pul = os.pulsetrain(f0, pw) * 0.5 / max(0.34, sin(ma.PI * pw));
sub = os.square(f0 * 0.5);

// The DCO mixer, as four independent faders. There is deliberately no constant-energy
// normalisation here, unlike the single `tone` macro this replaces: on the hardware
// these faders ARE level controls per source, and normalising them would mean turning
// SUB up quietly turned SAW down. The fixed scale below is headroom, not normalisation,
// and `sat` catches the rest when everything is at maximum.
dco = (saw * dcoSaw + pul * dcoPulse + sub * dcoSub + vnoise * dcoNoise) * 0.52;

// The panel's HPF, read off the jack board: a 4052 selects one of four paths into a
// 47K virtual earth with 47K feedback, so each is unity gain and exactly one pole.
//
//   3   4700 pF into 47K   high-pass at 720 Hz
//   2   0.015 uF into 47K  high-pass at 226 Hz
//   1   direct             FLAT. This is the bypass, not position 0
//   0   shelf network      bass BOOST, about +10 dB below 72 Hz
//
// The two surprises are worth stating plainly, because both were modelled wrong here
// from the photograph alone. The bypass is position 1, and the lowest position adds
// bass rather than removing it, which is why a Juno with the HPF "off" sounds fatter
// than one with it at 1. Do not carry the Juno-60's numbers over: it has no boost
// position and different corners.
hpfStage(x) = select2(hpfPos > 0.5, boost(x),
                select2(hpfPos > 1.5, x,
                  select2(hpfPos > 2.5, hp(226.0, x), hp(720.0, x))))
with {
    hp(f, sig) = sig : fi.highpass(1, f);
    boost(sig) = sig : fi.lowshelf(1, 10.0, 72.0);
};

// Key tracking, now with the panel's KYBD amount as the exponent: 0 is no tracking,
// 0.5 is half-power, where an octave up moves the cutoff by a fifth.
ktrack = (freq / 261.6256) ^ vcfKybd;

cutBase = 120 * pow(35, vcfFreq);

// The ceiling is a property of ve.moog_vcf, not a taste decision. Measured on this
// material it returns non-finite samples above roughly 7 kHz at low resonance and
// above 5 kHz at high resonance, so the safe cutoff depends on both. Written against
// ma.SR because the instability scales with sample rate and the page may run at 48k.
fcMax = ma.SR * (0.142 - 0.036 * vcfRes);

// ENV amount with the panel's polarity switch, plus the VCF's own LFO amount.
// Both are multiplicative, so they move the cutoff in octaves and cannot drive it
// negative. Written as a sum instead, INV subtracted more than the standing cutoff
// and pinned it to the 60 Hz floor: measured 33.3 dB of level swing and a peak of
// 0.012 against 0.598, which is the voice disappearing rather than a filter sweep.
// A ratio also makes INV the mirror of NORM, which is what the switch claims.
pol   = select2(vcfPol, 1.0, -1.0);
fcMod = pow(1 + vcfEnv * 6.5 * env, pol) * pow(1 + 3.0 * vcfLfo, lfo);
fc = max(60, min(fcMax, cutBase * ktrack * (0.55 + 0.45 * vel) * fcMod));

// Resonance also drives the filter harder and makes up the gain again, because a
// pushed filter is half of what a resonant Juno actually sounds like, and because
// without the makeup the control would be a loudness knob.
res    = 0.05 + vcfRes * 0.87;
push   = 1.0 + 1.10 * vcfRes;
// Makeup BOOSTS. A ladder loses passband level as it resonates, so the obvious
// compensation is upside down: written as a divider it turned resonance into a
// 13 dB fader, which is the volume-control failure with the sign flipped.
makeup = 1.0 + 0.90 * vcfRes;

// Small-signal unity, compresses above it. Per voice, never on the shared bus, so
// the timbre does not change with how many notes are held.
sat(x) = x / sqrt(1 + 0.8 * x * x);

// The DC blocker is not decoration. `sat` is an odd function, so it cannot make DC
// from a symmetric wave, but a saw plus a narrow pulse is not symmetric about zero
// and an odd curve applied to it has a nonzero mean: measured +0.039 without this,
// four times the threshold, and gone entirely with the saturator bypassed.
voice = dco : hpfStage : *(push) : sat
            : ve.moog_vcf(res, fc) : fi.dcblocker : *(makeup);

// The VCA's ENV/GATE switch. GATE is the organ setting: full level for as long as
// the key is down. Smoothed, or it clicks on every note.
gateAmp = gate : si.smooth(ba.tau2pole(0.004));
vca     = select2(vcaMode, env, gateAmp);
amp     = vca * (0.32 + 0.68 * vel) * vcaLevel * 0.38;

// The voice is mono. Every bit of width in this instrument comes from the chorus,
// which is historically exactly the case.
process = voice * amp <: _,_;

//======================================================================
// Shared effect chain: the BBD chorus, and nothing else
//======================================================================
// Off / I / II, as three positions rather than a continuous control, because that
// is what the machine has: two buttons, and both up is off.
chorus = hslider("chorus[panel:CHORUS][idx:1][positions:OFF|I|II]", 0.5, 0, 1, 0.5);

chOn = chorus > 0.25;
fast = chorus > 0.75;

bbd(l, r) = l*dry + wl*wet, r*dry + wr*wet
with {
    // Solved from the 106's own triangle oscillator: a 2SK30A shunts R3 2.2M across
    // R4 680K to change the integrator drive. The 0.513 and 0.863 that stood here
    // are measurements of a Juno-60, which is a different machine.
    rate  = select2(fast, 0.553, 0.898);
    depth = select2(fast, 2.55, 3.30);        // ms of deviation either side
    base  = 4.60;                             // ms
    clfo  = os.osc(rate);
    // Antiphase on the two lines is what makes the width, and that IS the hardware:
    // one LFO feeds the two MN3101 clock generators with opposite polarity, at TP3
    // and TP4. An earlier comment here claimed the hardware inverts one wet output
    // instead. It does not: both output mixers on the jack board are identical
    // inverting summers, same ratio, same polarity.
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
