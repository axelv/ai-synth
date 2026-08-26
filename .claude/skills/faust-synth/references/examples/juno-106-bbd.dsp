declare name "juno-106-bbd";
declare description "Roland Juno-106 style polysynth, laid out as the machine's own panel. One DCO per voice with saw, PWM pulse, square sub and noise, a 1-pole HPF into a 4-pole resonant VCF with key follow, one ADSR serving both filter and amp, and the BBD stereo chorus reproduced from the board schematic rather than sketched in the delay domain.";

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
// NORM multiplies the cutoff by the envelope's depth and INV divides by the same
// amount, which is the mirror the switch claims. Written as a select rather than
// pow(depth, +-1): the exponent would come from a slider, so Faust cannot fold it
// and would emit a runtime log/exp pair every sample in every voice to choose
// between a number and its reciprocal. depth is never below 1, so the divide is
// safe. The LFO term keeps its pow, whose exponent genuinely varies.
envDepth = 1 + vcfEnv * 6.5 * env;
fcMod = select2(vcfPol, envDepth, 1.0 / envDepth) * pow(1 + 3.0 * vcfLfo, lfo);
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
// Shared effect chain: the chorus BOARD, and nothing else
//======================================================================
// This is the only thing that differs from juno-106.dsp, and it is the whole
// point of this variant. The chorus there is a delay-domain sketch: a triangle
// written straight onto delay time, and one 2-pole lowpass standing in for the
// filtering. Here the board is reproduced stage by stage, traced from a Juno-60
// CHORUS BOARD sheet of 10 April 1983, on the grounds that the two machines
// run the same design: two MN3009 bucket brigades on their own MN3101 clocks,
// handed one triangle LFO in opposite polarity.
//
// What is 106 and what is 60 is kept straight deliberately, because this repo
// has already been burned once for blurring it:
//
//   106, from its own service notes:  the topology, and the RATES below.
//   60, from the chorus sheet:        every COMPONENT VALUE below.
//
// The component values are NOT verified for a 106. The 106's jack board would
// settle them and nobody has read it. Nothing here was tuned by ear.

chorus = hslider("chorus[panel:CHORUS][idx:1][positions:OFF|I|II]", 0.5, 0, 1, 0.5);

chOn = chorus > 0.25;
fast = chorus > 0.75;

//----------------------------------------------------------------------
// The board's filter idioms, written as the component values they are
//----------------------------------------------------------------------
// One corner on this board (clock rejection, 45.4 kHz) sits above Nyquist at
// ordinary sample rates, where fi.lowpass designs an unstable section and the
// output leaves for NaN. Clamping keeps the values in the source, which is the
// reason for writing them as values at all.
nyq(f) = min(f, 0.45 * ma.SR);

rc(r, c) = fi.lowpass(1, nyq(1.0 / (2.0 * ma.PI * r * c)));   // series R, shunt C
cr(c, r) = fi.highpass(1, nyq(1.0 / (2.0 * ma.PI * r * c)));  // coupling C, shunt R

// Sallen-Key 2-pole, built the way the board builds them: two equal series
// resistors into a unity-gain emitter follower, a feedback cap from their
// junction to the emitter, a shunt cap to ground. f0 and Q are solved from the
// four values rather than chosen.
sk(r, cfb, csh) = fi.resonlp(f0, q, 1.0)
with {
    rt = sqrt(r * r * cfb * csh);
    f0 = nyq(1.0 / (2.0 * ma.PI * rt));
    q  = rt / (csh * 2.0 * r);
};

// The board builds this same 4-pole three times: once shared ahead of both
// bucket brigades as the anti-alias filter, and once after each of them as the
// reconstruction filter. Solved, the sections are 9.69 kHz at Q 0.549 and
// 10.38 kHz at Q 1.291. A 4th-order Butterworth wants Q 0.541 and 1.307 at one
// frequency, so this is a Butterworth and Roland designed it as one.
butter4 = sk(22e3, 820e-12, 680e-12) : sk(22e3, 1800e-12, 270e-12);

//----------------------------------------------------------------------
// MN3009, 256 stages
//----------------------------------------------------------------------
// Delay is 128 over the clock frequency, and the LFO sweeps the CLOCK, not the
// delay. That is not a detail. Delay goes as 1/f, so sensitivity is highest
// where the clock is lowest: the line whips out to its long delays and comes
// straight back, sitting near its short ones the rest of the time. Measured off
// this model in mode I, the delay spends 22.3% of the LFO cycle in the longer
// half of its 2.05 to 7.14 ms range, mean 3.59 ms against an arithmetic
// midpoint of 4.60 ms. juno-106.dsp writes a triangle onto delay time instead,
// which spends 50% and departs from this by up to 1.54 ms, about 30% of the
// whole sweep.
STAGES = 256.0;

// The clock centre and the LFO amplitude are set on the panel board, which is
// not on the chorus sheet. These land the sweep on the ranges quoted for a 60.
FCLK  = 40171.0;                          // Hz
sweep = select2(fast, 0.554, 0.597);      // fraction of centre

// 106, solved from its own triangle oscillator: a 2SK30A shunts R3 2.2M across
// R4 680K to change the integrator drive. The 60's own rates are 0.513 and
// 0.863, and they are deliberately NOT used here.
rate = select2(fast, 0.553, 0.898);

// The board carries no compander anywhere, which is why a real one hisses.
// Held at zero so an A/B against juno-106.dsp isolates the circuit change
// rather than measuring a noise floor. Raise it to hear the part honestly.
HISS = 0.0;

bbdLine(tri) = cr(0.1e-6, 100e3)      // C6 into R14 100K, 15.9 Hz
             : rc(10e3, 2.2e-9)       // R15 10K into C7 0.0022, 7.23 kHz
             : delayStage(tri)
             : rc(1.594e3, 2.2e-9)    // R18/R19 3.3K pair into C10, 45.4 kHz
             : butter4                // R22/R21/C9/C8 then R24/R26/C11/C3
             : cr(1e-6, 22e3)         // C12 into R27 22K, 7.2 Hz
with {
    delayStage(t) = de.fdelay(4096, dsamp) : +(no.noise * HISS)
    with {
        fclk  = max(4000.0, FCLK * (1.0 + sweep * t));
        dsamp = ma.SR * STAGES / (2.0 * fclk);
    };
};

//----------------------------------------------------------------------
// The board
//----------------------------------------------------------------------
// At the node after R78 the signal splits, and only ONE branch is filtered.
// DIRECT SIG runs straight down the page to both mixers, unfiltered. The other
// branch runs the 4-pole Butterworth and feeds BOTH bucket brigades. So the wet
// carries nine poles to the dry's none, and that asymmetry is the character of
// the effect. Modelling the dry as filtered, or the wet as one gentle lowpass,
// both miss it.
board(x) = mix(w1), mix(w2)
with {
    dry = x;                            // DIRECT SIG, unfiltered
    fed = x : butter4;                  // TR19/TR18, shared anti-alias

    tri = os.lf_triangle(rate);         // panel board, pin 9
    w1  = fed : bbdLine(tri);
    w2  = fed : bbdLine(-tri);          // pin 10, drawn as the inverse

    // CHORUS OFF (pin 38, 1 = off) drives TR21 into the wet mute FETs TR8 and
    // TR16. It mutes the WET ONLY. The dry is never touched, which is why the
    // machine gets LOUDER when the chorus comes on, and why juno-106.dsp's
    // `dry = 1 - 0.28 * chOn` is a compensation the board does not do. C44
    // 4.7uF through 47K makes it a 0.22 s fade rather than a switch.
    mute = chOn : si.smooth(ba.tau2pole(0.221));

    // Both mixers are inverting summers: 100K feedback, dry through 47K, wet
    // through 39K. Normalised so dry alone is unity, the wet sits at 47/39,
    // which is 1.62 dB ABOVE the dry.
    mix(w) = (dry + w * (47.0 / 39.0) * mute)
           : cr(10e-6, 1.5e3);          // C40 10uF NP into R90 1.5K, 10.6 Hz
};

// No reverb. The 106 has none, and it is the one machine whose owners all agree
// belongs in front of whatever reverb they already have.
//
// The board has one input, at pin 19. The voice above is mono duplicated, so
// summing to mono here loses nothing and is what the hardware does.
//
// The 0.68 is headroom, and it is needed because the mixer above is faithful.
// juno-106.dsp keeps its peaks near 0.6 by ducking the dry 2.85 dB whenever the
// chorus is on; this board does not duck anything, so at the old 0.95 the patch
// peaked at 1.18 and the harness failed seven macros for clipping at the top of
// their range. Taking it out here instead leaves the 47K/39K ratio inside the
// mixer untouched, which is the part that is sourced. On the machine this is
// the VOLUME control, and turning it down is what a player does when the chorus
// makes the instrument louder.
effect(l, r) = board((l + r) * 0.5) : par(i, 2, *(0.68));
