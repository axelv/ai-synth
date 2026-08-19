"""Differentiable port of the shared `effect` chain in synth.py.

Three of the four stages are exact ports, verified against isolated Faust probes
(see verify_fx.py): the chorus is a linear-interpolating gather, the ping-pong
delay is a truncated Neumann series of the feedback comb, and the tilt EQ is the
Butterworth shelf pair evaluated as a static frequency response.

The chorus needs one piece of Faust's arithmetic, not just its algebra. os.osc
runs its phase as a float32 accumulator whose rounding is biased, so the LFO lags
an exact ramp by about 3e-4 of the elapsed time, which is several samples of
chorus delay by the end of the clip. faust_phasor replays that accumulation bit
for bit, cheaply, and it enters as a detached correction: the value is Faust's,
the gradient is the ideal sine's. Without it the isolated chorus agrees only to
5e-2 rather than 1e-5.

The reverb is NOT a port. re.zita_rev1_stereo is an 8x8 feedback delay network
with allpass combs; reimplementing it differentiably would cost more than the
gradient is worth. Instead it is a decorrelated-noise impulse response whose
per-band decay is derived from zita's own damping filters (rev_t60), which
reproduces the decay, the per-octave energy and the stereo decorrelation but not
the modal fine structure. That is the main reason the fitted patch has to be
re-rendered through PadRenderer before it is believed.

Quirk kept on purpose: chDepth multiplies the chorus twice, once inside the
chorus chain and once as the wetdry mix, so the chorus contributes chDepth
squared. That is what the Faust source does and what patch.json was fitted to.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

SR = 44100

# --- chorus (de.fdelay lengths are in SAMPLES, not seconds)
CH_BASE = 220.0
CH_SWING = 200.0
CH_WIDTH = 0.8          # ef.stereo_width(0.8)
CH_RATE_SPREAD = 0.13   # os.osc(chRate + 0.13*i)
OSC_TABLE = 1 << 16     # os.oscsin reads a pl.tablesize sine table with int() truncation

# --- ping-pong delay
DLY_TAP_TOL = 1e-5      # stop the Neumann series once dlyFb**k drops below this
DLY_MAX_TAPS = 64

# --- tilt EQ
TILT_HI_FREQ = 1200.0
TILT_LO_FREQ = 300.0
TILT_HI_DB = 12.0    # fi.highshelf(2, tilt*12, 1200)
TILT_LO_DB = -6.0    # fi.lowshelf(2, -tilt*6, 300)
BUTTER2_A1 = math.sqrt(2.0)

# --- reverb surrogate
LN1000 = 3.0 * math.log(10.0)
REV_F1 = 200.0              # dc/mid crossover, hardwired in synth.py
REV_BANDS = 20              # log-spaced decay bands in the noise IR
REV_F_LO = 25.0
REV_F_HI = 16000.0
REV_SEED = 20240817         # fixed, so a render is reproducible
REV_MAX_T60 = 8.0           # revSize <= 0.98 in PARAMS, so t60dc <= 7.84 s
REV_PREDELAY = 843          # measured: first nonzero sample of the zita IR at any setting
# zita_rev_fdn's eight loop lengths in seconds (its `tdelays`), which set how often
# each loop applies its damping filter and therefore the decay above f2.
REV_TDELAYS = (0.153129, 0.210389, 0.127837, 0.256891,
               0.174713, 0.192303, 0.125000, 0.219991)
# zita's impulse response is not decaying noise alone: the input reaches the output
# through the allpass combs at roughly 0.14 gain, so a loud early transient carries
# most of the energy and makes the total nearly independent of revSize. Two envelope
# terms, diffuse plus early, fitted to zita's per-octave band energies over a
# revSize x revDamp grid (see verify_fx.py).
REV_GAIN = 0.003115
REV_EARLY_GAIN = 0.006951
REV_EARLY_T60 = 0.1636

_BAND_NOISE: dict[tuple[str, int, int, int], torch.Tensor] = {}


# ---------------------------------------------------------------- helpers


def _frac(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)


def faust_phasor(freq: float, n: int, sr: int = SR) -> np.ndarray:
    """os.phasor's float32 accumulator, replayed bit for bit without a sample loop.

    Faust runs `(+(freq/SR) : ma.decimal) ~ _` in float32, and the rounding of that
    accumulation is biased, so the phase lags an exact ramp by roughly 3e-4 of the
    elapsed time. That is small but it is several samples of chorus delay by the end
    of the clip, and it is the whole difference between agreeing with the Faust
    chorus to 1e-5 and to 5e-2.

    ma.decimal subtracts 1.0 from a value in [1, 1+incr), which is exact, so the
    recursion restarts cleanly at every wrap. Between wraps it is a running sum of
    one constant, and np.cumsum on float32 accumulates sequentially in float32 just
    like the generated C++, so one cumsum per wrap reproduces the drift exactly.
    """
    incr = np.float32(np.float32(freq) / np.float32(sr))
    if incr <= np.float32(0.0):
        return np.zeros(n, dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    span = int(1.0 / float(incr)) + 64
    one = np.float32(1.0)
    i, start = 0, np.float32(0.0)
    while i < n:
        m = min(span, n - i)
        step = np.full(m, incr, dtype=np.float32)
        step[0] = start
        run = np.cumsum(step, dtype=np.float32)
        wrap = np.nonzero(run >= one)[0]
        if wrap.size:
            w = int(wrap[0])
            out[i:i + w] = run[:w]
            i += w
            start = np.float32(run[w] - one)
        else:
            out[i:i + m] = run
            i += m
            start = np.float32(run[m - 1] + incr)
            if start >= one:
                start = np.float32(start - one)
    return out


def _osc(freq: torch.Tensor, n: int, sr: int, device: torch.device) -> torch.Tensor:
    """os.osc: quantised sine table read by Faust's own drifting float32 phasor.

    The drift and the table quantisation are added as a detached correction, so the
    value is what Faust produces while the gradient is the one of the ideal sine.
    Differentiating the real phasor would mean differentiating float32 rounding,
    which is chaotic in chRate and useless as a search direction.
    """
    t = torch.arange(n, device=device, dtype=torch.float32)
    ph = _frac(t * (freq / sr))
    ref = faust_phasor(float(freq.detach()), n, sr).astype(np.float64)
    quantised = torch.from_numpy(np.floor(ref * OSC_TABLE) / OSC_TABLE)
    ph = ph + (quantised.to(device=device, dtype=torch.float32) - ph).detach()
    return torch.sin(2.0 * math.pi * ph)


def _fdelay_static(x: torch.Tensor, d0: int, frac: torch.Tensor) -> torch.Tensor:
    """de.fdelay with a delay that does not change over the render."""
    n = x.shape[-1]
    a = F.pad(x, (d0, 0))[..., :n] * (1.0 - frac)
    b = F.pad(x, (d0 + 1, 0))[..., :n] * frac
    return a + b


def _fdelay_tv(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """de.fdelay with a per-sample delay. Gradient flows through the interpolation."""
    n = x.shape[-1]
    d0 = torch.floor(d)
    frac = d - d0
    i0 = torch.arange(n, device=x.device)[None, :] - d0.long()
    i1 = i0 - 1
    g0 = torch.gather(x, 1, i0.clamp(min=0)) * (i0 >= 0)
    g1 = torch.gather(x, 1, i1.clamp(min=0)) * (i1 >= 0)
    return g0 * (1.0 - frac) + g1 * frac


def _fft_conv(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal convolution truncated to x's length; the channel axes broadcast."""
    n = x.shape[-1]
    nfft = 1 << (n + h.shape[-1] - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, n=nfft) * torch.fft.rfft(h, n=nfft), n=nfft)
    return y[..., :n]


def _biquad_response(b: tuple[float, float, float], a: tuple[float, float, float],
                     w: torch.Tensor) -> torch.Tensor:
    z = torch.exp(-1j * w)
    z2 = z * z
    return (b[0] + b[1] * z + b[2] * z2) / (a[0] + a[1] * z + a[2] * z2)


def _butter2(fc: float, sr: int, highpass: bool) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """fi.lowpass(2,fc) / fi.highpass(2,fc), i.e. tf2s bilinear-transformed."""
    c = 1.0 / math.tan(math.pi * fc / sr)
    csq = c * c
    d = 1.0 + BUTTER2_A1 * c + csq
    a = (1.0, 2.0 * (1.0 - csq) / d, (1.0 - BUTTER2_A1 * c + csq) / d)
    if highpass:
        return (csq / d, -2.0 * csq / d, csq / d), a
    return (1.0 / d, 2.0 / d, 1.0 / d), a


# ---------------------------------------------------------------- stages


def stereo_width(x: torch.Tensor, w: float = CH_WIDTH) -> torch.Tensor:
    """ef.stereo_width: Blumlein shuffle, so out_L = L + (1-w)*R."""
    lo, hi = x[0], x[1]
    return torch.stack([lo + (1.0 - w) * hi, (1.0 - w) * lo + hi])


def chorus(x: torch.Tensor, ch_rate: torch.Tensor, ch_depth: torch.Tensor,
           sr: int = SR) -> torch.Tensor:
    n = x.shape[-1]
    wide = stereo_width(x)
    lfo = torch.stack([_osc(ch_rate + CH_RATE_SPREAD * i, n, sr, x.device) for i in (0, 1)])
    d = CH_BASE + CH_SWING * ch_depth * lfo
    return _fdelay_tv(wide, d) * ch_depth


def pingpong(x: torch.Tensor, dly_time: torch.Tensor, dly_fb: torch.Tensor,
             sr: int = SR) -> torch.Tensor:
    """par(i,2,(+ : de.fdelay(65536, dlyTime*SR)) ~ *(dlyFb)).

    Per channel, so despite the name there is no L/R crossing. The loop delay is
    the fdelay plus the one sample Faust inserts in a `~` recursion, so the k-th
    echo sits at D + k*(D+1) samples. Summing those taps replaces the recursion.
    """
    n = x.shape[-1]
    delay = dly_time * float(sr)
    d0 = int(math.floor(float(delay.detach())))
    frac = delay - float(d0)
    fb = float(dly_fb.detach())
    taps = 0
    if fb > 1e-6:
        taps = min(int(math.ceil(math.log(DLY_TAP_TOL) / math.log(min(fb, 0.999)))),
                   n // (d0 + 1) + 1, DLY_MAX_TAPS)
    acc = _fdelay_static(x, d0, frac)
    out = acc
    for _ in range(taps):
        acc = dly_fb * F.pad(_fdelay_static(acc, d0, frac), (1, 0))[..., :n]
        out = out + acc
    return out


def rev_t60(freq: torch.Tensor, rev_size: torch.Tensor, rev_damp: torch.Tensor,
            sr: int = SR) -> torch.Tensor:
    """T60 per frequency of zita_rev_fdn, derived from its own damping filters.

    Every loop i applies gM*low_shelf1_l(g0/gM, f1) : special_lowpass(gM, f2) once
    per tdelay(i) seconds, with g0 and gM chosen per loop so the loop decays at t60dc
    and t60m exactly. Above f2 the extra one-pole loss no longer scales with the loop
    length, so the loops decay at different rates, and the late field a Schroeder fit
    measures is the slowest of them. Hence the min over loops, which is an exact tie
    at dc and mid (so t60dc and t60m come out right by construction) and measured
    better than averaging the loops, per octave, across a revSize x revDamp grid.
    No fitted constant, and differentiable in revSize (g0, gM) and revDamp (f2).
    """
    td = torch.tensor([math.floor(0.5 + sr * t) / sr for t in REV_TDELAYS],
                      device=freq.device)[:, None]
    g0 = torch.exp(-LN1000 * td / (rev_size * 8.0))
    g_m = torch.exp(-LN1000 * td / (rev_size * 4.0))
    f2 = 1200.0 + (1.0 - rev_damp) * 8000.0
    gs = g_m * g_m
    c = torch.cos(2.0 * math.pi * f2 / sr)
    mbo2 = (1.0 - gs * c) / (1.0 - gs)
    pole = mbo2 - torch.sqrt(torch.clamp(mbo2 * mbo2 - 1.0, min=0.0))
    w = 2.0 * math.pi * freq[None, :] / sr
    lp = (1.0 - pole) / torch.sqrt(1.0 - 2.0 * pole * torch.cos(w) + pole * pole)
    # |1 + (G0-1)*fi.lowpass(1,f1)|, the bilinear one-pole written out real-valued
    cf = 1.0 / math.tan(math.pi * REV_F1 / sr)
    b1, a1 = 1.0 / (1.0 + cf), (1.0 - cf) / (1.0 + cf)
    gain = (g0 / g_m - 1.0) * b1
    n0, n1 = 1.0 + gain, a1 + gain
    cw = torch.cos(w)
    shelf = torch.sqrt((n0 * n0 + n1 * n1 + 2.0 * n0 * n1 * cw)
                       / (1.0 + a1 * a1 + 2.0 * a1 * cw))
    rate = -torch.log(g_m * shelf * lp) / td
    return LN1000 / rate.amin(dim=0)


def _band_noise(n_ir: int, sr: int, device: torch.device) -> torch.Tensor:
    """(REV_BANDS, 2, n_ir) decorrelated noise, split into power-complementary bands."""
    key = (str(device), n_ir, sr, REV_SEED)
    hit = _BAND_NOISE.get(key)
    if hit is not None:
        return hit
    gen = torch.Generator().manual_seed(REV_SEED)
    noise = torch.fft.rfft(torch.randn(2, n_ir, generator=gen), n=n_ir)
    freq = torch.fft.rfftfreq(n_ir, 1.0 / sr).clamp(REV_F_LO, REV_F_HI)
    step = math.log2(REV_F_HI / REV_F_LO) / (REV_BANDS - 1)
    s = torch.log2(freq / REV_F_LO) / step
    k = torch.arange(REV_BANDS, dtype=torch.float32)[:, None]
    mask = torch.cos(0.5 * math.pi * (s[None, :] - k)) * ((s[None, :] - k).abs() < 1.0)
    bands = torch.fft.irfft(noise[None] * mask[:, None, :], n=n_ir)
    out = bands.to(device=device, dtype=torch.float32)
    _BAND_NOISE[key] = out
    return out


def reverb(x: torch.Tensor, rev_size: torch.Tensor, rev_damp: torch.Tensor,
           sr: int = SR) -> torch.Tensor:
    """Surrogate for re.zita_rev1_stereo(0, 200, f2, revSize*8, revSize*4, SR).

    Independent noise per output channel, so the wet signal is decorrelated the
    way zita's is; the input is mono-summed, which zita does not do exactly but
    which costs nothing while the pre-reverb signal is near mono.
    """
    n = x.shape[-1]
    t60dc = float(rev_size.detach()) * 8.0
    n_ir = min(int(math.ceil(min(t60dc, REV_MAX_T60) * sr)), n)
    bands = _band_noise(int(math.ceil(REV_MAX_T60 * sr)), sr, x.device)[:, :, :n_ir]
    # built on cpu: torch.logspace has no mps kernel
    centres = torch.logspace(math.log10(REV_F_LO), math.log10(REV_F_HI), REV_BANDS).to(x.device)
    t60 = rev_t60(centres, rev_size, rev_damp, sr)
    t = torch.arange(n_ir, device=x.device, dtype=torch.float32) / sr
    early = REV_EARLY_GAIN * torch.exp(-LN1000 * t / REV_EARLY_T60)
    ir = torch.zeros(2, n_ir, device=x.device)
    for k in range(REV_BANDS):
        ir = ir + bands[k] * (REV_GAIN * torch.exp(-LN1000 * t / t60[k]) + early)
    ir = F.pad(ir, (REV_PREDELAY, 0))
    return _fft_conv(x.mean(dim=0, keepdim=True), ir)


def tilt_eq(x: torch.Tensor, tilt: torch.Tensor, sr: int = SR) -> torch.Tensor:
    """fi.highshelf(2, tilt*12, 1200) then fi.lowshelf(2, -tilt*6, 300).

    fi.filterbank(2,(fx)) splits into (highband, lowband); the shelf scales one
    of them, so each shelf is exactly one pair of Butterworth biquads. Static,
    hence exact as a single frequency-domain multiply.
    """
    n = x.shape[-1]
    nfft = 1 << (n + 8192 - 1).bit_length()
    w = 2.0 * math.pi * torch.fft.rfftfreq(nfft, device=x.device)
    g_hi = torch.exp(math.log(10.0) * tilt * TILT_HI_DB / 20.0)
    g_lo = torch.exp(math.log(10.0) * tilt * TILT_LO_DB / 20.0)
    b, a = _butter2(TILT_HI_FREQ, sr, highpass=True)
    resp = g_hi * _biquad_response(b, a, w)
    b, a = _butter2(TILT_HI_FREQ, sr, highpass=False)
    resp = resp + _biquad_response(b, a, w)
    b, a = _butter2(TILT_LO_FREQ, sr, highpass=True)
    lo = _biquad_response(b, a, w)
    b, a = _butter2(TILT_LO_FREQ, sr, highpass=False)
    resp = resp * (lo + g_lo * _biquad_response(b, a, w))
    return torch.fft.irfft(torch.fft.rfft(x, n=nfft) * resp, n=nfft)[..., :n]


# ---------------------------------------------------------------- chain


def _wetdry(dry: torch.Tensor, wet: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return dry * (1.0 - w) + wet * w


def effects(stereo: torch.Tensor, p: dict[str, torch.Tensor], sr: int = SR) -> torch.Tensor:
    """The Faust `effect` chain: chorus, delay, reverb, tilt, outGain. (2,N) -> (2,N)."""
    x = stereo
    x = _wetdry(x, chorus(x, p["chRate"], p["chDepth"], sr), p["chDepth"])
    x = _wetdry(x, pingpong(x, p["dlyTime"], p["dlyFb"], sr), p["dlyWet"])
    x = _wetdry(x, reverb(x, p["revSize"], p["revDamp"], sr), p["revWet"])
    return tilt_eq(x, p["tilt"], sr) * p["outGain"]
