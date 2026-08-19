"""Differentiable oscillator bank: everything in the Faust voice before the filter.

Reproduces this fragment of synth.DSP, per MIDI note, as a (V, N) tensor:

    ratio(i) = 2 ** ((i - (NVOICE-1)/2) * detune / 1200)
    vib      = 2 ** (os.osc(lfoRate) * lfoAmt / 1200) * bend
    oscmix(f)= os.sawtooth(f)*(1-sqrMix) + os.square(f)*sqrMix
    osc      = centre*(1-uniMix) + spread*uniMix + os.osc(freq*0.5*vib)*subLvl

Why a transcription of Faust's own algorithms rather than an idealised
band-limited oscillator: the surrogate exists to predict what Faust will do, so
any deviation is a lie the gradient will happily optimise. Faust's oscillators
are cheap closed forms, so copying them costs nothing and removes a whole class
of disagreement:

  * os.sawtooth is saw2 is saw2ptr: a trivial ramp whose single post-wrap sample
    is replaced by the order-2 Polynomial Transition Region value
    p = 1 + d*(2 - p0), with d the wrapped phase and p0 the period in samples.
    That is a PolyBLEP-class correction (O(1) per sample, one polynomial residual
    at each wrap) and it is exactly what Faust computes, which is better than
    exactly what an ideal saw would be.
  * os.square is NOT two offset PTR saws. It is pulsetrainN(2): the *DPW* saw
    (differentiated square of the naive ramp) minus a linearly interpolated
    half-period-delayed copy of itself, with the frequency clipped at 23.449 Hz.
    The two-offset-saws identity holds for the naive ramp, and that is precisely
    how Faust builds it, delay line and all.
  * os.osc is a 65536-point sine table read with a truncated index. We use exact
    sin; the truncation is worth at most 2*pi/65536 = 9.6e-5 in amplitude.

Phase accumulates as a cumulative sum of the instantaneous phase increment, so
detune, lfoRate, lfoAmt and the frozen bend curve all get gradients through it.
A plain float32 cumsum does not lose the sum, it loses the *fraction*: 789566
samples at 523 Hz reach 9213 cycles, where a float32 step is 9.8e-4, so the only
part we use is quantised to 4.9e-4 cycles. The sum is therefore blocked, an
error-compensated within-block float32 cumsum plus a per-block offset reduced
mod 1 in float64, which holds the phase to 1.4e-6 cycles on either device.
All three numbers are measured in verify_osc.test_precision.

That fixes the summation, not the summands. The increment (hz/sr)*vib is itself a
float32 number, wrong by ~6e-8 of itself, and 9200 cycles of accumulating it
carries about 1e-3 cycles of phase whichever way the sum is arranged; MPS and the
CPU round exp2 and the products differently, so the two diverge over the clip.
Measured at full scale against a float64 run of this same module: 1.8e-3 relative
L2 on the CPU and 4.6e-3 on MPS, which is drift concentrated in the one sample
either side of each saw wrap, and 1.3e-3 / 2.5e-3 once projected onto the
magnitude STFT the fit actually sees. Faust's own float32 phasor drifts further
than that (verify_osc.test_drift). Both numbers are in verify_osc.test_scale.

Two dawdreamer facts this module depends on, both measured, not assumed:
  * A voice's phasors (oscillators AND the vibrato LFO) start from zero at each
    note-on. dsp_voice::keyOn does not clear voice state, but a free voice is
    never computed, so a voice that has not been reused starts at phase 0 and its
    phase is a function of time since its note-on. Two renders of the same note at
    different onsets agree bit-exactly after shifting, and the 29-note score
    renders bit-identically at 24 and at 64 voices, so no voice in it is reused.
  * Parameter automation is applied once per 512-sample block, so Faust sees a
    block-held bend curve. We hold it the same way. Integrating the curve at
    sample rate instead would put the phase 0.13 cycles out at the lowest note of
    the score and 1.9 cycles out at the highest.

Voice gain (velocity/127, measured) belongs to the amplitude stage, not here:
Faust applies it as `filtered * aenv * gain`.
"""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

from synth import BLOCK, NVOICE
from torch_common import SR, NoteEvents

SQUARE_FMIN = 96000.0 / (2.0 * 2047.0)  # os.pulsetrainN clips freq at 23.4489 Hz
SAWN_FMIN = 20.0                        # os.sawN clips again, always below SQUARE_FMIN
DELMAX = 2047.0                         # os.pulsetrainN delay ceiling, in samples
PHASE_BLOCK = 512                       # inner block of the mod-1 phase accumulator
VOICE_CHUNK = 4                         # voices per autograd checkpoint segment


def _frac(x: torch.Tensor) -> torch.Tensor:
    """Positive fractional part. torch.frac truncates toward zero, which is wrong here."""
    return x - torch.floor(x)


def _shift1(x: torch.Tensor) -> torch.Tensor:
    """x delayed by one sample along the last axis, zero-filled (Faust's x')."""
    return torch.cat([torch.zeros_like(x[..., :1]), x[..., :-1]], dim=-1)


def _block_offsets(total: torch.Tensor) -> torch.Tensor:
    """Exclusive cumulative sum of per-block totals, reduced mod 1, in float64.

    Small tensor (one value per block), so float64 is free. MPS has no float64,
    hence the detour through the CPU; autograd tracks it, the gradient of frac is 1.
    """
    d = total.double() if total.device.type == "cpu" else total.cpu().double()
    c = torch.cumsum(d, dim=-1)
    ex = torch.cat([torch.zeros_like(c[..., :1]), c[..., :-1]], dim=-1)
    # cast before moving: the backward of a combined .to() would try to make a
    # float64 tensor on the accelerator, which MPS refuses
    return (ex - torch.floor(ex)).to(dtype=total.dtype).to(device=total.device)


def _cum_blocks(inc: torch.Tensor, block: int = PHASE_BLOCK) -> torch.Tensor:
    """Within-block inclusive cumulative sum, shaped (..., n_blocks, block).

    Kept in this shape so several oscillators can share it: the phase of a
    detuned copy is the same sum times a constant, and reducing mod 1 only needs
    the per-block totals, which are small enough to add up in float64.

    The sum is error-compensated because MPS's float32 cumsum rounds with a
    systematic bias: its per-block totals are then wrong in the same direction
    every block and the mod-1 accumulator integrates that into 4.6e-4 cycles of
    phase over the clip at 523 Hz, against 1.4e-6 on the CPU. `e` is the exact
    rounding error of each cumsum step, both subtractions being exact by
    Sterbenz, and removing its running sum brings MPS to the CPU number. It is
    gradient-neutral: d e / d inc cancels to exactly zero, so autograd still
    sees a plain cumulative sum.
    """
    n = inc.shape[-1]
    pad = (-n) % block
    if pad:
        inc = torch.cat([inc, torch.zeros_like(inc[..., :pad])], dim=-1)
    x = inc.reshape(*inc.shape[:-1], (n + pad) // block, block)
    s = x.cumsum(-1)
    e = (s - _shift1(s)) - x
    return s - e.cumsum(-1)


def _wrap(w: torch.Tensor, n: int) -> torch.Tensor:
    """frac of the full inclusive cumulative sum whose within-block part is `w`.

    Inclusive because Faust's saw2ptr accumulator already holds t0 at sample 0.
    """
    off = _block_offsets(w[..., -1]).unsqueeze(-1)
    return _frac(off + w).flatten(-2)[..., :n]


def _wrapped_phase(inc: torch.Tensor, block: int = PHASE_BLOCK) -> torch.Tensor:
    """frac of the inclusive cumulative sum of `inc` along the last axis."""
    return _wrap(_cum_blocks(inc, block), inc.shape[-1])


def _ptr_saw(phase: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
    """os.sawtooth: trivial ramp with the order-2 PTR value at the wrap sample."""
    wrap = phase < _shift1(phase)
    p = torch.where(wrap, torch.addcmul(phase.new_ones(()), phase, 2.0 - p0), phase)
    return p.mul_(2.0).sub_(1.0)


def _dpw_saw(phase: torch.Tensor, p0: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """os.sawN(2): (s^2 - (s^2)') * p0/4 on the naive ramp, first sample blanked."""
    zero = phase.new_zeros(())
    s = phase.mul(2.0).sub_(1.0)
    u = torch.where(live, s * s, zero)
    return torch.where(_shift1(live), (u - _shift1(u)) * (p0 * 0.25), zero)


def _interp_delay(x: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
    """x delayed by a time-varying fractional number of samples (de.fdelay).

    Both taps read at the same index, one of them from x delayed a sample, so one
    int64 index buffer is enough. Reads from before the start need no masking:
    x[..., 0] is always exactly 0 here (sawN blanks its own first sample), which
    is what Faust's zero-filled delay line returns.
    """
    i0 = torch.floor(delay)
    fr = delay - i0
    j = (torch.arange(x.shape[-1], device=x.device) - i0.long()).clamp_(min=0)
    return torch.lerp(torch.gather(x, -1, j), torch.gather(_shift1(x), -1, j), fr)


def _at_onset(x: torch.Tensor, onset: torch.Tensor) -> torch.Tensor:
    """Sample of x (..., N) at each voice's own onset index, keepdim."""
    idx = onset.clamp(0, x.shape[-1] - 1)
    view = idx.reshape(-1, *([1] * (x.dim() - 1)))
    return torch.gather(x, -1, view.expand(*x.shape[:-1], 1))


def _voices(
    freq: torch.Tensor,
    onset: torch.Tensor,
    bend: torch.Tensor,
    detune: torch.Tensor,
    lfo_rate: torch.Tensor,
    lfo_amt: torch.Tensor,
    sqr_mix: torch.Tensor,
    uni_mix: torch.Tensor,
    sub_lvl: torch.Tensor,
    sr: float,
) -> torch.Tensor:
    """One chunk of voices. freq/onset are (C,), bend is (N,), returns (C, N).

    Everything here is one pass over a (C, NVOICE, N) tensor, so the ordering is
    chosen to keep as few of those alive at once as possible: the phase
    accumulator is shared between the unison copies, the sub and (when its
    frequency clip does not bind) the square.
    """
    n = int(bend.shape[-1])
    dt = freq.dtype
    zero = freq.new_zeros(())
    since = torch.arange(n, device=freq.device)[None, :] - onset[:, None]  # (C, N)
    live = since >= 0
    lfo = torch.sin(2.0 * math.pi * _frac(lfo_rate * since.to(dt) / sr))
    vib = torch.exp2(lfo * lfo_amt / 1200.0) * bend[None, :]               # (C, N)

    i = torch.arange(NVOICE, device=freq.device, dtype=dt)
    ratio = torch.exp2((i - (NVOICE - 1) / 2.0) * detune / 1200.0)         # (7,)
    hz1 = freq[:, None] * ratio[None, :]                                   # (C, 7) at vib=1
    live7 = live[:, None, :]
    # phase increment, zero before the note-on so the phase starts there;
    # p0 is the period in samples and must stay finite everywhere
    inc = (hz1 / sr)[:, :, None] * torch.where(live, vib, zero)[:, None, :]
    p0 = (sr / hz1)[:, :, None] * torch.reciprocal(vib)[:, None, :]
    w = _cum_blocks(inc)
    phase = _wrap(w, n)
    saw = _ptr_saw(phase, p0)

    # os.square runs off its own frequency, clipped at 23.449 Hz. When the clip
    # does not bind the two phases are identical, so reuse the accumulator; the
    # branch is an optimisation, both sides compute the same thing.
    fmin = max(SQUARE_FMIN, SAWN_FMIN)
    if bool((hz1.min() * vib.min() < fmin).item()):
        hz_sq = torch.clamp(hz1[:, :, None] * vib[:, None, :], min=fmin)
        p0_sq = sr / hz_sq
        inc_sq = torch.where(live7, hz_sq / sr, zero)
        ph_sq = _wrapped_phase(inc_sq)
    else:
        inc_sq, p0_sq, ph_sq = inc, p0, phase
    # the naive phasor is zero at the voice's first sample, one increment behind saw2ptr
    dpw = _dpw_saw(_frac(ph_sq - _at_onset(inc_sq, onset)), p0_sq, live7)
    # no clamp on the delay: Faust's own 23.449 Hz floor is chosen so that half a
    # period is at most 940 samples here, well inside pulsetrainN's 2047 ceiling
    oscmix = torch.lerp(saw, dpw - _interp_delay(dpw, 0.5 * p0_sq), sqr_mix)
    centre = oscmix[:, NVOICE // 2, :]  # ratio(3) is exactly 1, so Faust's centre
    voice = torch.lerp(centre, oscmix.mean(dim=1), uni_mix)

    w_sub = 0.5 * w[:, NVOICE // 2]  # the ratio-1 accumulator, one octave down
    ph_sub = _frac(_wrap(w_sub, n) - 0.5 * _at_onset(inc[:, NVOICE // 2], onset))
    return torch.where(live, torch.addcmul(voice, torch.sin(2.0 * math.pi * ph_sub), sub_lvl),
                       zero)


def osc_bank(
    ev: NoteEvents,
    p: dict[str, torch.Tensor],
    bend: torch.Tensor,
    sr: float = SR,
    chunk: int = VOICE_CHUNK,
    bend_block: int = BLOCK,
) -> torch.Tensor:
    """Pre-filter oscillator sum, one row per MIDI note. Returns (V, N).

    `chunk` trades memory for autograd checkpoint recomputation: the working set
    is chunk * NVOICE * N, so 29 voices at once would need several GB.
    `bend_block` mirrors dawdreamer's per-block parameter automation; pass 1 to
    integrate the bend curve at sample rate instead.

    Cost is memory traffic, not arithmetic: 29 voices times 7 unison copies times
    789566 samples is 160M oscillator samples, and every torch op walks all of it.
    Measured on this machine at the full length, forward plus backward, in
    verify_osc.test_scale.
    """
    n = ev.n_samples
    dtype = ev.freq.dtype
    b = bend[:n].to(device=ev.device, dtype=dtype)
    if b.shape[-1] < n:
        b = torch.cat([b, b[-1:].expand(n - b.shape[-1])])
    if bend_block > 1:
        keep = (n // bend_block) * bend_block
        held = b[:keep].reshape(-1, bend_block)[:, :1].expand(-1, bend_block).reshape(-1)
        b = torch.cat([held, b[keep:keep + 1].expand(n - keep)]) if n > keep else held

    args = (b, p["detune"], p["lfoRate"], p["lfoAmt"], p["sqrMix"], p["uniMix"], p["subLvl"])
    grad = torch.is_grad_enabled() and any(
        isinstance(a, torch.Tensor) and a.requires_grad for a in args
    )
    rows = []
    for s in range(0, ev.n_voices, chunk):
        seg = (ev.freq[s:s + chunk], ev.onset[s:s + chunk]) + args
        if grad:
            rows.append(checkpoint(_voices, *seg, sr, use_reentrant=False))
        else:
            rows.append(_voices(*seg, sr))
    return torch.cat(rows, dim=0)
