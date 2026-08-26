declare name "juno60chorus";
declare version "1.0";
declare author "traced from the Roland Juno-60 CHORUS BOARD sheet, 10 April 1983";
declare description "The Juno-60 chorus board reproduced stage by stage from its schematic: one shared 4-pole Butterworth anti-alias filter, two MN3009 bucket brigades on clocks swept by anti-phase triangles, a matched 4-pole reconstruction filter on each, and two inverting summers that mix an unfiltered dry against one wet each.";

import("stdfaust.lib");

//======================================================================
// What is and is not on the sheet
//======================================================================
// Every resistor and capacitor named in a comment below is read off the
// schematic, and every filter here is solved from those values rather than
// tuned by ear. Three things are NOT on the sheet and are marked where they
// appear: the LFO rate and depth, which are generated on the panel board;
// the BBD clock centre frequency, which the panel board's VCO sets; and the
// BBD noise floor, which is a property of the part rather than of the wiring.
//
// The board also carries the synth's switched HPF (IC5, an HD14051B picking
// one of four paths) and its VCA (IC6). Those are the instrument's, not the
// chorus's, and they sit upstream of everything modelled here.

//======================================================================
// Panel
//======================================================================
// Two buttons, and both up is off, so three positions rather than a fader.
chorus = hslider("chorus[panel:CHORUS][positions:OFF|I|II]", 0.5, 0, 1, 0.5);

chOn = chorus > 0.25;
modeII = chorus > 0.75;

// NOT on this sheet. The panel board generates the LFO and this page only
// receives it, at pins 9 and 10. These two rates are Juno-60 measurements,
// and they are used here for the reason they were removed from the 106 patch
// in this repo: they belong to a 60, and this is a 60.
rate = select2(modeII, 0.513, 0.863);

// NOT on this sheet either. The board fixes the delay range through the
// clock VCO's centre frequency and the LFO's amplitude, neither of which is
// drawn here. These two land the sweep on the ranges quoted for a 60:
// 2.05 to 7.15 ms in mode I, 2.00 to 7.90 ms in mode II.
FCLK = 40171.0;                              // Hz, clock centre
sweep = select2(modeII, 0.554, 0.597);       // fraction of centre

// NOT on this sheet. There is no compander anywhere on this board, which is
// the sheet's contribution to the question: an MN3009 run open like this
// hisses, and the hiss is a large part of why a Juno chorus sounds the way it
// does. The level is a choice; zero it to hear the circuit alone.
hiss = hslider("hiss[unit:dB]", -74, -120, -50, 1) : ba.db2linear;

// NOT on the board. The board has no output compensation at all, and engaging
// the chorus therefore makes the machine louder, by about 3.9 dB on an
// incoherent sum. Left at 0 dB this model does the same thing.
trim = hslider("trim[unit:dB]", 0, -12, 12, 0.1) : ba.db2linear : si.smoo;

//======================================================================
// The board's two filter idioms, written as component values
//======================================================================
// Corners are solved from component values, and one of them (the board's
// clock-rejection pole, 45.4 kHz) lands above Nyquist at any sane sample
// rate. Clamping keeps the filter designs stable without editing the values
// out of the source, which is the point of writing them down.
nyq(f) = min(f, 0.45 * ma.SR);

// A one-pole RC, series R into a shunt C.
rc(r, c) = fi.lowpass(1, nyq(1.0 / (2.0 * ma.PI * r * c)));

// A one-pole coupling capacitor into a shunt R.
cr(c, r) = fi.highpass(1, nyq(1.0 / (2.0 * ma.PI * r * c)));

// A Sallen-Key 2-pole low-pass built the way this board builds them: two
// equal series resistors into a unity-gain emitter follower, a feedback cap
// from the resistor junction to the emitter, and a shunt cap to ground. f0
// and Q are solved from the four values, not chosen.
sk(r, cfb, csh) = fi.resonlp(f0, q, 1.0)
with {
    rt = sqrt(r * r * cfb * csh);
    f0 = nyq(1.0 / (2.0 * ma.PI * rt));
    q  = rt / (csh * 2.0 * r);
};

// The board builds the same 4-pole twice, once before the bucket brigades and
// once after each of them. Solved, the two sections are 9.69 kHz at Q 0.549
// and 10.38 kHz at Q 1.291. A 4th-order Butterworth wants Q 0.541 and 1.307
// at one frequency, so this is a Butterworth and Roland designed it as one;
// the split in f0 is component-value rounding.
butter4 = sk(22e3, 820e-12, 680e-12) : sk(22e3, 1800e-12, 270e-12);

//======================================================================
// MN3009 bucket brigade
//======================================================================
// 256 stages, so the delay is 128 over the clock frequency. The LFO sweeps
// the CLOCK, not the delay, and that is not a detail. Delay goes as 1/f, so
// sensitivity is highest where the clock is lowest: the line whips out to its
// long delays and comes straight back, and sits near its short ones the rest
// of the time. Measured off this model in mode I, the delay spends just 22.3%
// of the LFO cycle in the longer half of its 2.05 to 7.14 ms range, and its
// mean is 3.59 ms against an arithmetic midpoint of 4.60 ms. A triangle
// written straight onto delay time, which is what most chorus models do,
// spends 50% and departs from this by up to 1.54 ms, or about 30% of the
// whole sweep.
STAGES = 256.0;

// tri is the panel board's triangle, -1 to 1. Channel 2 is handed its
// negation, which is the whole of the stereo mechanism on this board.
bbd(tri, x) = de.fdelay(4096, dsamp, x)
with {
    // Guarded so the clock cannot reach zero; sweep < 1 already ensures it.
    fclk  = max(4000.0, FCLK * (1.0 + sweep * tri));
    dsamp = ma.SR * STAGES / (2.0 * fclk);
};

//======================================================================
// One delay line, C6 through C12
//======================================================================
// Designators are channel 1's; channel 2 is the same circuit with C18, R44,
// R43, C19, IC4, R48/R47, R49, C22, R51/R50, C20/C23, R53/R55, C21/C2, C24.
line(tri) = cr(0.1e-6, 100e3)          // C6 into R14 100K, 15.9 Hz
          : rc(10e3, 2.2e-9)           // R15 10K into C7 0.0022, 7.23 kHz
          : bbdStage(tri)
          : rc(1.594e3, 2.2e-9)        // R18/R19 3.3K pair into C10, 45.4 kHz
          : butter4                    // R22/R21/C9/C8 then R24/R26/C11/C3
          : cr(1e-6, 22e3)             // C12 into R27 22K, 7.2 Hz
with {
    // The BBD itself, plus the noise it makes. Two independent sources, one
    // per channel, which is correct here: this is an effect and runs once,
    // so unlike a poly voice it does not duplicate bit-identically.
    bbdStage(t) = bbd(t) : +(no.noise * hiss);
};

//======================================================================
// The board
//======================================================================
// SIG IN arrives at pin 19 already filtered and enveloped by the HPF and VCA
// upstream. At the node after R78 33K it splits two ways, and the split is
// the thing most chorus models get wrong:
//
//   DIRECT SIG  ->  down the page, unfiltered, to both mixers through 47K
//   R83/R84 ... ->  4-pole Butterworth, TR19 and TR18, feeding BOTH bucket
//                   brigades
//
// So the dry is genuinely unfiltered and the wet carries nine poles: the
// shared Butterworth, the 7.23 kHz pole in each BBD input, the clock-
// rejection pole, and a second Butterworth on the way out. The wet is a great
// deal darker than the dry, and that asymmetry is the sound.
board(x) = out1, out2
with {
    dry = x;                            // DIRECT SIG, no filtering at all
    fed = x : butter4;                  // TR19/TR18, shared anti-alias

    tri = os.lf_triangle(rate);         // panel board, pin 9
    w1  = fed : line(tri);
    w2  = fed : line(-tri);             // pin 10, drawn as the inverse

    // CHORUS OFF at pin 38 (1 = off) drives TR21 into the wet mute FETs TR8
    // and TR16. It mutes the WET ONLY; the dry path is never touched, which
    // is why the board gets louder when the chorus comes on. C44 4.7uF
    // through 47K makes it a 0.22 s fade rather than a switch.
    mute = chOn : si.smooth(ba.tau2pole(0.221));

    // Both mixers are inverting summers: 100K feedback, dry through 47K,
    // wet through 39K. Normalised so dry alone is unity, the wet sits at
    // 47/39, which is 1.62 dB ABOVE the dry.
    DRYG = 1.0;
    WETG = 47.0 / 39.0;

    mix(w) = (dry * DRYG + w * WETG * mute)
           : cr(10e-6, 1.5e3)           // C40 10uF NP into R90 1.5K, 10.6 Hz
           : *(trim);

    out1 = mix(w1);                     // IC8a, R86/R89/R95, to OUT1 pin 17
    out2 = mix(w2);                     // IC8b, R87/R88/R91, to OUT2 pin 15
};

//======================================================================
// What is deliberately not modelled
//======================================================================
// - The BBD's actual sampling. A real MN3009 samples at the clock and folds
//   everything above half of it back down, which is why the board spends four
//   poles on an anti-alias filter and one more per channel. At the bottom of
//   the mode II sweep the clock reaches 16.2 kHz, whose Nyquist of 8.1 kHz is
//   barely below the filter's 10 kHz corner, so a real 60 genuinely does
//   alias there. This model interpolates instead and stays clean.
// - VR1 and VR2, the per-channel BBD bias trims, set for minimum distortion.
//   They are why no two units match and why the wet path distorts
//   asymmetrically. Modelling them without a unit in front of you would be
//   inventing a nonlinearity, so there is none here.
//
// The board has one input, at pin 19. This takes two and sums them, so it
// drops in as a Faust `effect` block; feed it the same signal twice for the
// board's own behaviour.
process(l, r) = board((l + r) * 0.5);
