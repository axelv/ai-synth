declare name "dx7-fm";
declare description "Yamaha DX7 style 6-operator phase-modulation synth, laid out as the machine's own operator grid. Four of the 32 algorithms, per-operator coarse ratios and detune, one envelope contour biased across the operator stack, the pitch/amplitude LFO, keyboard level and rate scaling, and the operator feedback loop written the way the hardware actually computes it.";

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
// Same departure from patch-design.md that juno-106.dsp makes, for the same
// reason: this models a named machine, so the controls are the machine's own
// parameters rather than a handful of intents. A DX7 has no fader panel at all
// (one data slider and a membrane keypad), so what is modelled here is its
// PARAMETER grid, which is the thing a DX7 programmer actually holds in their
// head: six operators in a column each with a coarse ratio, an algorithm, a
// feedback level, and the global LFO and scaling pages.
//
// Deliberately NOT modelled, and each one is a real simplification:
//   - Six independent 4-rate/4-level operator EGs, which is 48 controls. One
//     ADSR biased across the stack by CONTOUR stands in for them.
//   - 28 of the 32 algorithms. The four here are chosen to span the shapes:
//     a two-carrier stack, three 2-op pairs, one carrier under three
//     modulators, and six carriers in parallel.
//   - Keyboard level scaling breakpoints and curves; LV SCALE is a single
//     exponent instead.
//======================================================================

// ---- ALGO ----
// Four of the 32. All four carry their feedback on operator 6, which is where
// the hardware algorithm ROM puts it for most of the set.
algo   = hslider("algo[panel:ALGO][idx:1][cap:ALG][positions:1|5|16|32]", 1, 0, 3, 1);
index  = hslider("index[panel:ALGO][idx:2][cap:INDEX]", 0.46, 0, 1, 0.001) : si.smoo;
fbAmt  = hslider("feedback[panel:ALGO][idx:3][cap:F.BACK]", 0.30, 0, 1, 0.001) : si.smoo;
// The one control here that exists to be A/B'd rather than played. See fb6.
fbMode = hslider("fbMode[panel:ALGO][idx:4][cap:F.LOOP][positions:AVG|1TAP]", 0, 0, 1, 1);

// ---- OSC ----
// Coarse ratio per operator, as a detented fader over a DX7-ish ratio set
// rather than a continuous sweep: FM wants rational ratios, and a fader that
// glides between them mostly produces inharmonic mush on the way.
ratio1 = hslider("op1[panel:OSC][idx:1][cap:OP1][positions:.5|1|1.41|2|3|4|5|7|9|11|14|20]", 1, 0, 11, 1);
ratio2 = hslider("op2[panel:OSC][idx:2][cap:OP2][positions:.5|1|1.41|2|3|4|5|7|9|11|14|20]", 1, 0, 11, 1);
ratio3 = hslider("op3[panel:OSC][idx:3][cap:OP3][positions:.5|1|1.41|2|3|4|5|7|9|11|14|20]", 1, 0, 11, 1);
ratio4 = hslider("op4[panel:OSC][idx:4][cap:OP4][positions:.5|1|1.41|2|3|4|5|7|9|11|14|20]", 5, 0, 11, 1);
ratio5 = hslider("op5[panel:OSC][idx:5][cap:OP5][positions:.5|1|1.41|2|3|4|5|7|9|11|14|20]", 3, 0, 11, 1);
ratio6 = hslider("op6[panel:OSC][idx:6][cap:OP6][positions:.5|1|1.41|2|3|4|5|7|9|11|14|20]", 7, 0, 11, 1);
detune = hslider("detune[panel:OSC][idx:7][cap:DETUNE]", 0.34, 0, 1, 0.001) : si.smoo;

// ---- EG ----
attack  = hslider("attack[panel:EG][idx:1][cap:A]", 0.10, 0, 1, 0.001);
decay   = hslider("decay[panel:EG][idx:2][cap:D]", 0.46, 0, 1, 0.001);
sustain = hslider("sustain[panel:EG][idx:3][cap:S]", 0.55, 0, 1, 0.001);
release = hslider("release[panel:EG][idx:4][cap:R]", 0.38, 0, 1, 0.001);
// How much faster the top of the operator stack runs than the bottom. This is
// what makes FM sound struck rather than blown: on a DX7 it is six separate
// envelopes, and a programmer spends most of their time setting exactly this.
contour = hslider("contour[panel:EG][idx:5][cap:CONTOUR]", 0.62, 0, 1, 0.001) : si.smoo;

// ---- LFO ----
lfoRate = hslider("lfoRate[panel:LFO][idx:1][cap:SPEED]", 4.6, 0.1, 24, 0.01) : si.smoo;
lfoWave = hslider("lfoWave[panel:LFO][idx:2][cap:WAVE][positions:TRI|SAW|SQR|SIN]", 0, 0, 3, 1);
pmd     = hslider("pmd[panel:LFO][idx:3][cap:P.MOD]", 0.20, 0, 1, 0.001) : si.smoo;
// AMD drives the MODULATORS' levels, not the carriers' amplitude, which is
// what AMS does on the hardware and the reason it reads as timbre and not as
// tremolo. Writing it on the carriers instead turns it into a volume control.
amd     = hslider("amd[panel:LFO][idx:4][cap:A.MOD]", 0.30, 0, 1, 0.001) : si.smoo;

// ---- SCALE ----
// The two scaling pages, both of which exist because a fixed FM patch does not
// survive being played five octaves up.
lvScale = hslider("lvScale[panel:SCALE][idx:1][cap:LV.SCALE]", 0.62, 0, 1, 0.001) : si.smoo;
rtScale = hslider("rtScale[panel:SCALE][idx:2][cap:RT.SCALE]", 0.45, 0, 1, 0.001) : si.smoo;
velSens = hslider("velSens[panel:SCALE][idx:3][cap:VEL.SENS]", 0.70, 0, 1, 0.001) : si.smoo;

// ---- OUT ----
// The DX7's converter, as a control. 16 bits down to 10.
grain   = hslider("grain[panel:OUT][idx:1][cap:GRAIN]", 0.30, 0, 1, 0.001);
spread  = hslider("spread[panel:OUT][idx:2][cap:SPREAD]", 0.40, 0, 1, 0.001) : si.smoo;

//======================================================================
// Voice
//======================================================================

// The sine table, at the hardware's resolution or coarser.
//
// The OPS chip truncates phase to 12 bits before the log-sine ROM, so the DX7's
// oscillators run on a 4096-point table however wide the accumulator is. That
// truncation is audible: it lands inharmonic hash around every operator, worst
// on high ratios and worst of all inside the feedback loop, and it is a large
// part of why the machine sounds like a DX7 and not like six clean sines.
//
// GRAIN at 0 is 12 bits, which is the hardware. Above that it coarsens to 6,
// which is not a model of anything and is there because it is a good control.
//
// The first thing tried here was the output word depth instead, 15 bits down to
// 7 at the DAC. It measured 0.34 dB of level-normalised shape and failed as
// inert, and the reason is worth keeping: quantisation noise at the output sits
// under a bright FM spectrum that already has energy in every band, so there is
// no band for it to show up in. Phase truncation moves the partials themselves.
tq    = pow(2.0, 12.0 - 6.0 * grain);
sn(p) = sin(2.0 * ma.PI * floor(ma.frac(p) * tq) / tq);
ph(f) = os.phasor(1.0, f);

// The coarse ratio set. Not the DX7's own 0.5 plus 1..31, which is 32 detents
// nobody can hit on a fader: these are the ones patches actually use, plus
// 1.41 for the inharmonic metal that a whole family of DX7 patches lives on.
rat(i) = i : rdtable(waveform{0.5, 1.0, 1.41, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 14.0, 20.0});

ratSel(0) = ratio1;
ratSel(1) = ratio2;
ratSel(2) = ratio3;
ratSel(3) = ratio4;
ratSel(4) = ratio5;
ratSel(5) = ratio6;

// LFO. One per voice and bit-identical across voices, which is the same
// exception juno-106.dsp records: the hardware has ONE LFO for the whole
// instrument, so coherent is the machine rather than a defect. See
// references/faust-poly.md.
lfoSig = (os.lf_triangle(lfoRate),
          os.lf_sawpos(lfoRate) * 2.0 - 1.0,
          os.lf_squarewave(lfoRate),
          os.osc(lfoRate)) : ba.selectn(4, int(lfoWave));

// Pitch mod, in cents rather than as a ratio, so the depth means the same
// thing at both ends of the keyboard.
kf = max(20.0, freq * pow(2.0, pmd * 0.90 * lfoSig / 12.0));

// Detune spread across the stack. The DX7's DETUNE is +-7 units of about half
// a cent; this reaches +-8 cents at full, which is past the hardware and is
// what makes a stack of ratios beat instead of fusing. Spread by operator so
// the six do not all move together.
dt(i) = pow(2.0, detune * 0.08 * (i - 2.5) / 2.5 / 12.0);

fop(i) = kf * rat(ratSel(i)) * dt(i);

// Rate scaling: envelopes run faster as pitch rises, because a struck string
// does and because a fixed decay sounds sluggish two octaves up.
rs = pow(max(0.25, kf / 261.6256), 0.55 * rtScale);

att = (0.0015 + 1.10 * attack * attack * attack) / rs;
dec = (0.0100 + 8.00 * decay  * decay  * decay ) / rs;
rel = (0.0100 + 9.00 * release * release * release) / rs;

// CONTOUR, the stand-in for six independent EGs. Operator 6 ends up with a
// decay 0.18x operator 1's and a sustain 0.42x, so the top of the stack is an
// attack transient and the bottom is the held tone. Both are geometric in the
// operator index, which is what the hardware's rate curves approximately do.
tdk(i) = pow(pow(0.18, contour), i / 5.0);
tsu(i) = pow(pow(0.42, contour), i / 5.0);

eg(i) = en.adsre(att, dec * tdk(i), sustain * tsu(i), rel * tdk(i), gate);

// Keyboard level scaling. Without it the top octave is fizz: a fixed phase-mod
// index puts its sidebands a fixed number of HARMONICS up, which is a fixed
// number of octaves up, so the spectrum runs off the top of the band. The
// exponent is the control. Clamped because at the bottom of the keyboard the
// same expression would multiply the index up without limit.
kscale = max(0.30, min(2.20, pow(110.0 / kf, 0.45 * lvScale)));

// Harder is brighter more than louder, and VEL SENS says how much. At 0 the
// patch ignores velocity for index, which is a real DX7 setting.
velIdx = 1.0 - velSens * (1.0 - gain);
velAmp = 0.32 + 0.68 * (gain ^ 0.6);

// AMD on the modulator levels. Unity at lfo = -1 rather than a centred
// tremolo, so turning A.MOD up cannot make the patch louder than it was.
amGate = 1.0 - 0.55 * amd * (0.5 + 0.5 * lfoSig);

// A modulator's output, in CYCLES of phase deviation on whatever it feeds.
mlev(i) = (0.15 + 2.70 * index) * velIdx * kscale * amGate * eg(i);

// One operator: a sine at its own frequency, phase-modulated by m.
op(i, m) = sn(ph(fop(i)) + m);

//----------------------------------------------------------------------
// Operator 6, the one with the feedback loop, written the way the OPS chip
// actually computes it.
//
// The hardware does NOT feed back one sample. It holds the previous TWO
// outputs of the operator in per-voice shift registers, sums and halves them,
// and modulates with that. Yamaha's patent (Tomisawa, US 4,249,447) gives the
// reason: a single-sample loop makes a large modulation produce a small output
// and vice versa, so the signal alternates high and low every sample, which is
// an oscillation at Nyquist. The mean of two consecutive samples is a 2-tap FIR
// with a zero exactly there, so averaging suppresses the hunting.
//
//   1TAP   y[n] = sin(p[n] + k * y[n-1])
//   AVG    y[n] = sin(p[n] + k * (y[n-1] + y[n-2]) / 2)
//
// These are different systems and they diverge worst at high feedback, which is
// exactly where a DX7 patch reaches for feedback. faustlibraries' own dx7.lib
// writes the single tap, and its file header carries the matching open TODO:
// "There are artifacts that sound like aliasing for high feedback values."
// F.LOOP is here so the difference can be heard and measured rather than
// asserted.
//
// Simplification worth stating: the hardware takes feedback from the operator's
// ENVELOPED output, so its feedback dies with the note. This takes it from the
// raw sine and applies the envelope outside the loop.
//----------------------------------------------------------------------
// Quadratic, because the useful part of a feedback control is all in its
// bottom third. At 1.0 this is 0.62 cycles, about 3.9 radians, which is well
// into the noisy end where the two loop forms disagree most.
fbk = 0.62 * fbAmt * fbAmt;

fb6 = loop ~ _
with {
    // y is the output one sample back; y' reaches the one before that.
    loop(y) = sn(ph(fop(5)) + fbk * select2(fbMode, 0.5 * (y + y'), y));
};

//----------------------------------------------------------------------
// The four algorithms.
//
// Written out rather than table-driven. The hardware holds them in a 96-entry
// ROM and a Faust port could do the same with a static table, but at four
// algorithms the explicit form is the one a reader can check against the
// machine's own diagram.
//----------------------------------------------------------------------
carr(i, m) = op(i, m) * eg(i);

m6   = fb6 * mlev(5);              // op6, with feedback, as a modulator
o5   = op(4, m6) * mlev(4);        // op5 <- op6
o4   = op(3, o5) * mlev(3);        // op4 <- op5
o4f  = op(3, 0.0) * mlev(3);       // op4, unmodulated
o3f  = op(2, o4f) * mlev(2);       // op3 <- op4
o2   = op(1, 0.0) * mlev(1);       // op2, unmodulated

// 1:  (op2 -> op1) + (op6fb -> op5 -> op4 -> op3)     two carriers
alg1  = carr(0, o2) + carr(2, o4);
// 5:  (op2 -> op1) + (op4 -> op3) + (op6fb -> op5)    three carriers, the
//     E.PIANO / tubular-bell shape
alg5  = carr(0, o2) + carr(2, o4f) + carr(4, m6);
// 16: op2, (op4 -> op3), (op6fb -> op5) all into op1  one carrier
alg16 = carr(0, o2 + o3f + o5);
// 32: six carriers in parallel, op6 fed back                 the organ shape
alg32 = carr(0, 0.0) + carr(1, 0.0) + carr(2, 0.0)
      + carr(3, 0.0) + carr(4, 0.0) + fb6 * eg(5);

// COM, the algorithm ROM's output-count compensation. Six carriers at unity is
// 15.6 dB hotter than one, so without this the ALG switch is mostly a volume
// control. The hardware stores this per operator as a power-of-two shift and
// applies it; Dexed computes the same carrier count in n_out() under #ifdef
// VERBOSE and never applies it at render time. These are trimmed from measured
// peaks rather than being exactly 1/n: the carriers are phase-locked sines at
// different ratios, so they sum somewhere between coherently and not.
com   = (0.62, 0.50, 1.00, 0.30) : ba.selectn(4, int(algo));

fmSum = (alg1, alg5, alg16, alg32) : ba.selectn(4, int(algo));

// Small-signal unity, compresses above it. Fixed rather than a macro, so it
// cannot become a drive control that is really a gain control. Per voice, never
// on the shared bus: on the bus the timbre would change with how many notes are
// held.
sat(x) = ma.tanh(x * 1.6) / ma.tanh(1.6);

// The voice is mono, which is what the machine is: one output jack.
process = fmSum * com * velAmp * 0.62 : sat : *(0.18) <: _, _;

//======================================================================
// Shared effect chain
//======================================================================
// A DX7 has no effects at all, and this is an explicit departure rather than a
// model of anything: a mono sine-carrier FM voice in a browser is a thin thing
// to hand someone. Only the band above 700 Hz is decorrelated, so the low end
// stays exactly centred, and the matrix is energy-normalised so SPREAD cannot
// become a loudness control the way fm-bass.dsp's widener does.
wid(l, r) = (l + s) * g, (r - s) * g
with {
    w  = 0.30 * spread;
    hb = (l + r) * 0.5 : fi.highpass(2, 700);
    s  = w * (hb - hb @ 29);
    g  = 1.0 / sqrt(1.0 + 2.0 * w * w);
};

effect = wid : par(i, 2, fi.dcblocker : *(0.98));
