import("stdfaust.lib");
freq = hslider("freq", 440, 20, 8000, 0.001);
gain = hslider("gain", 0.5, 0, 1, 0.001);
gate = button("gate");
env = en.adsr(0.01, 0.2, 0.7, 0.4, gate);
process = os.sawtooth(freq) * env * gain * 0.2 <: _,_;
effect = _,_;
