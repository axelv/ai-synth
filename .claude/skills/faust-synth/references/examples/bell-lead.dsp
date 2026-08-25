declare name "bell-lead";
declare description "Glassy struck-bar bell lead: inharmonic additive ring plus an FM/noise strike.";

import("stdfaust.lib");

//----------------------------------------------------------------- poly interface
freq = hslider("freq", 440, 20, 8000, 0.001);
gain = hslider("gain", 0.5, 0, 1, 0.001);
gate = button("gate");

//----------------------------------------------------------------- macro controls
// Named for the gesture, not the primitive: "brightness" tilts the whole partial
// stack, it is not one filter cutoff.
brightness = hslider("brightness", 0.62, 0, 1, 0.001) : si.smoo;
sparkle    = hslider("sparkle", 0.55, 0, 1, 0.001) : si.smoo;
decayT     = hslider("decay", 4.5, 0.3, 14, 0.01);
attackMs   = hslider("attack", 3, 0.5, 60, 0.01);
shimmer    = hslider("shimmer", 0.4, 0, 1, 0.001) : si.smoo;
width      = hslider("width", 0.7, 0, 1, 0.001) : si.smoo;
space      = hslider("space", 0.35, 0, 1, 0.001) : si.smoo;

//----------------------------------------------------------------- envelopes
// One sample impulse at note-on. A bell is struck, so the envelope is set by the
// strike and then ignores the gate; holding the key does not sustain it.
trig = gate > gate';

// Release does not cut the ring, it damps it: letting go is a hand on the bar,
// which is what keeps repeated notes from smearing while the tail stays long.
strikeEnv(t60) = (\(y).(select2(trig, y * coef, 1.0))) ~ _
with {
    tEff = max(0.02, t60) * select2(gate > 0.5, 0.45, 1.0);
    coef = pow(0.001, 1.0 / max(1.0, tEff * ma.SR));
};

// Smoothing the jump is the attack: a mallet-hardness control in disguise.
soften = si.smooth(ba.tau2pole(attackMs * 0.001));

//----------------------------------------------------------------- partial stack
NP = 8;
// Struck-bar modes (1, 2.76, 5.40, 8.93) interleaved with near-harmonic glass
// partials. Pure bar modes read as marimba; pure harmonics read as an organ.
ratios  = (1.0, 2.0, 2.76, 3.0, 4.16, 5.40, 6.79, 8.93);
amps    = (1.0, 0.55, 0.42, 0.30, 0.22, 0.18, 0.12, 0.09);
// Upper modes lose energy first, which is the whole reason a bell sounds struck.
decays  = (1.0, 0.82, 0.60, 0.68, 0.44, 0.34, 0.25, 0.18);
// Fundamental stays centred so the melodic line has a fixed place; only the
// inharmonic upper modes get thrown wide.
pans    = (0.0, -0.55, 0.70, -0.85, 0.60, -0.75, 0.90, -0.65);
lfos    = (0.11, 0.17, 0.23, 0.31, 0.41, 0.53, 0.67, 0.83);

// Harder strikes are brighter, and high notes are pulled back so the top of the
// keyboard stays bright rather than piercing.
keyDamp = 1.0 - 0.13 * max(0.0, min(3.0, log(max(20.0, freq) / 261.626) / log(2.0)));
bright  = min(1.0, brightness * (0.55 + 0.65 * gain) * keyDamp);
tiltExp = 2.2 - 2.0 * bright;

partial(i) = osc * amp * env * alive <: (_ * gl, _ * gr)
with {
    r    = ba.take(i + 1, ratios);
    a0   = ba.take(i + 1, amps);
    dm   = ba.take(i + 1, decays);
    p    = ba.take(i + 1, pans) * width;
    lf   = ba.take(i + 1, lfos);
    // Slow independent drift per mode: the beating that reads as "glass", not chorus.
    det  = 1.0 + shimmer * 0.0022 * (i > 0) * os.osc(lf);
    f    = freq * r * det;
    alive = f < 17000.0;
    osc  = os.osc(f);
    amp  = a0 * pow(r, -tiltExp) * (1.0 + shimmer * 0.10 * os.osc(lf * 0.7));
    env  = strikeEnv(decayT * dm) : soften;
    ang  = (p + 1.0) * ma.PI / 4.0;
    gl   = cos(ang);
    gr   = sin(ang);
};

bell = par(i, NP, partial(i)) :> _, _;

//----------------------------------------------------------------- strike transient
// Fast inharmonic FM burst plus a filtered tick. This is the "sparkle in the
// attack": it is gone in under a fifth of a second and never touches the tail.
strikeFM = sin(2.0 * ma.PI * (ph + md)) * env * lvl
with {
    ph  = os.phasor(1.0, freq * 2.03);
    ie  = strikeEnv(0.07) : si.smooth(ba.tau2pole(0.0015));
    md  = 0.55 * sparkle * (0.4 + 0.9 * gain) * ie * os.osc(freq * 3.51);
    env = strikeEnv(0.20) : soften;
    lvl = 0.22 * sparkle * (0.3 + 0.7 * gain) * (freq * 2.03 < 17000.0);
};

strikeTick = no.noise * env * lvl : fi.resonbp(fc, 1.4, 1.0)
with {
    env = strikeEnv(0.035) : si.smooth(ba.tau2pole(0.0012));
    lvl = 0.30 * sparkle * (0.2 + 0.8 * gain);
    fc  = min(9000.0, max(600.0, freq * 6.0));
};

// The transient is the widest thing in the patch, so it is hard-split L/R with a
// short offset; that is what makes the onset feel like it happens in a room.
transient = (strikeFM + strikeTick) <: (_ * tl, (_ : @(37)) * tr)
with {
    tl = cos((0.35 * width + 1.0) * ma.PI / 4.0);
    tr = sin((0.35 * width + 1.0) * ma.PI / 4.0);
};

//----------------------------------------------------------------- voice
level = 0.76 * (0.28 + 0.72 * gain);

voice = (bell, transient) :> (_ * level, _ * level)
      : (fi.highpass(2, 55), fi.highpass(2, 55))
      : (fi.lowpass(2, 15000), fi.lowpass(2, 15000));

process = voice;

//----------------------------------------------------------------- shared effect
wet = space * 0.85;
dry = 1.0 - 0.45 * space;

effect = _, _ <: (_ * dry, _ * dry),
                 (re.zita_rev1_stereo(0, 220, 6500, 3.2, 2.4, 48000) : _ * wet, _ * wet)
              :> _, _;
