"""Full differentiable surrogate: the four ported stages wired like synth.DSP.

The signal flow is a transcription of the Faust source, not a reinterpretation of
it, because the whole value of the surrogate is that its gradient points where the
real renderer's loss actually falls:

    osc      = centre*(1-uniMix) + uni*uniMix + sub              -> torch_osc
    shaped   = osc + drive*(tanh(osc*dgain)/tanh(dgain) - osc)    -> saturate
    fenv     = en.adsr(fA, fD, fS, aR, gate)                      -> torch_env
    aenv     = en.adsr(aA, aD, aS, aR, gate)                      -> torch_env
    fc       = max(30, min(16000, cutoff*(freq/C4)^kbdTrk + envAmt*fenv))
    filtered = shaped : fi.resonlp(fc, reso, 1) : fi.lowpass(2, fc) -> torch_filter
    process  = filtered * aenv * gain <: *(panL), *(panR)         -> pan_gains
    effect   = chorus : pingpong : zita : tiltEQ : outGain         -> torch_fx

One consequence of `spread` worth stating: torch_fx.reverb mono-sums its input before
convolving with two decorrelated IRs, which was free while the pre-reverb signal was
mono and is now an approximation. It costs the surrogate the part of the wet field
that follows the dry image, so at large spread the surrogate under-reports width.
Faust decides anyway, and the measured L/R correlation quoted anywhere is a
PadRenderer render, never a surrogate one.

Two things about that flow are easy to get wrong and are spelled out here rather
than left implicit:

  * `freq` in trackedCut is the note's own MIDI frequency, with neither the
    vibrato nor the frozen bend applied. Only the oscillators see vib.
  * `gain` is velocity/127 (measured, not assumed) and Faust applies it after the
    filter, so it scales the amplitude stage and nothing that feeds the cutoff.

What the assembled surrogate is worth, all measured in verify_torch_synth.py against
real PadRenderer renders. The patch these numbers were taken at is the 27-parameter one
that was delivered before drive and spread existed; it is kept as
out/patch_baseline27.json, since out/patch.json has since moved on to 29 parameters:

  * Per voice, with the fx chain matched, the surrogate tracks Faust to a median
    relative L2 of 0.11 and a median multi-resolution STFT distance of 0.15.
  * Summed over 29 voices and 17.9 s it does not, and cannot: Faust's saw2ptr
    accumulates phase in float32, so two legitimate Faust renderings of the same
    patch (the polyphonic render, and the sum of 29 one-note renders) already
    differ by mrstft 1.18 and relative L2 1.84. The surrogate's distance to Faust,
    mrstft 0.99 and relative L2 1.80, sits at that floor. Judge this port by the
    loss it produces and by its gradient, never by waveform distance.
  * Loss: Faust 1.5564, surrogate 1.6178, a bias of +0.06. At the 29-parameter patch
    now delivered the bias is the same size, Faust 1.5450 against surrogate 1.6244.
  * The surrogate gradient is a descent direction for the TRUE Faust loss away
    from the optimum (at stage2.seeded_start, cosine +0.272 against a 27-parameter
    Faust finite difference, and one backward pass plus a line search takes the true
    Faust loss from 3.6052 to 1.8808) but not at a converged patch itself, where CMA-ES
    has already converged: there the cosine is -0.063 and all eight step sizes tried
    made the true loss worse, monotonically past 0.03. Use it to get near, not to
    polish.
  * dlyTime's gradient is meaningless by construction: torch_fx.pingpong takes the
    integer part of the delay with a detached floor, so only the interpolation
    fraction carries a derivative, and the measured value is -9.0 against Faust's
    +0.01. It dominates the raw gradient norm and should be frozen with
    Patch.freeze(["dlyTime", "chRate", "lfoRate"]) before any step is taken.

dawdreamer's polyphonic wrapper adds one behaviour that is not in the Faust source:
a voice stops being computed partway through its release, so the tail of a release
that has gone quiet is cut off instead of ramped to zero. Measured at that same patch
over the 23 notes whose release fits inside the render, the cut lands between 11 and
95 percent of aR (median 55 percent), it moves with the output level (scaling the
voice by 100 removes it, by 0.01 makes it immediate) and it is not a monotone
function of pitch. VoiceLife applies those measured instants as a detached per-voice
gate. It is OFF by default because every measurement got worse with it: the loss
went 1.6178 -> 1.6993, the render gap 0.99 -> 1.08, and the gradient cosine at the
seed +0.272 -> +0.007. Per isolated voice it does help (rel L2 0.113 -> 0.092),
so the effect is real; what fails is transplanting cut instants measured from
one-note renders into the polyphonic sum. Left in, off, and documented.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import torch_env
import torch_filter
import torch_fx
import torch_osc
from stage2 import load_notes
from synth import BLOCK
from torch_common import KBD_REF, SR, NoteEvents, Patch, bend_tensor, default_n_samples, schedule

FC_MIN = 30.0
FC_MAX = 16000.0
DRIVE_GAIN = 12.0        # synth.DSP: dgain = 1 + drive*12
PAN_TURNS = 0.6180339887  # synth.DSP: turns of pan phase per semitone above C4


def cutoff_curve(ev: NoteEvents, p: dict[str, torch.Tensor], fenv: torch.Tensor) -> torch.Tensor:
    """(V, N) per-voice, per-sample filter cutoff, exactly synth.DSP's `fc`."""
    tracked = p["cutoff"] * (ev.freq[:, None] / KBD_REF) ** p["kbdTrk"]
    return torch.clamp(tracked + p["envAmt"] * fenv, FC_MIN, FC_MAX)


def saturate(x: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    """synth.DSP's `shaped`: peak-normalised tanh, mixed in by drive itself.

    Written the same way round as the Faust source, so drive=0 leaves x untouched
    instead of leaving tanh's own curvature behind, and the gradient at drive=0 is the
    finite harmonic content the shaper would add rather than zero.
    """
    g = 1.0 + drive * DRIVE_GAIN
    return x + drive * (torch.tanh(x * g) / torch.tanh(g) - x)


def pan_gains(ev: NoteEvents, spread: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(V,), (V,) constant-power L/R gains, exactly synth.DSP's panL/panR.

    Deterministic in the note's own pitch because the Faust voice DSP has no voice
    index to hash. Both gains are 1 at spread=0, which is the old mono split.
    """
    semis = 12.0 * torch.log2(ev.freq / KBD_REF)
    pos = 0.5 + 0.5 * spread * torch.sin(2.0 * math.pi * PAN_TURNS * semis)
    return torch.sqrt(2.0 * (1.0 - pos)), torch.sqrt(2.0 * pos)


@dataclass
class VoiceLife:
    """Per-voice sample index where dawdreamer stops computing the voice.

    Measured from Faust rather than derived: the cut instant depends on which sample
    lands on a block boundary and is not a differentiable function of the parameters.
    See verify_torch_synth.step_life, and the module docstring for why this is off by
    default despite being a real effect.
    """

    end: torch.Tensor
    block: int = BLOCK

    def gate(self, ev: NoteEvents) -> torch.Tensor:
        t = torch.arange(ev.n_samples, device=ev.device)
        return (t[None, :] < self.end.to(ev.device)[:, None]).to(torch.float32)


class TorchPad(nn.Module):
    """Differentiable render of the whole patch. Stage 1 is frozen input, not state.

    The note schedule, the bend curve and the device are fixed at construction:
    they are what stage 1 decided and nothing here may move them. Only the
    synth.PARAMS carried by a Patch flow through forward().
    """

    def __init__(
        self,
        notes: list[tuple[int, int, float, float]],
        n_samples: int,
        device: torch.device,
        sr: int = SR,
        filter_block: int = torch_filter.BLOCK,
        osc_chunk: int = torch_osc.VOICE_CHUNK,
        voice_life: VoiceLife | None = None,
    ) -> None:
        super().__init__()
        self.ev = schedule(notes, n_samples, device, sr)
        self.bend = bend_tensor(n_samples, device)
        self.sr = sr
        self.filter_block = filter_block
        self.osc_chunk = osc_chunk
        self.voice_life = voice_life

    @property
    def n_samples(self) -> int:
        return self.ev.n_samples

    def voice_output(self, p: dict[str, torch.Tensor]) -> torch.Tensor:
        """(V, N) post-filter, post-envelope, post-gain voice signals."""
        ev, sr = self.ev, self.sr
        osc = saturate(torch_osc.osc_bank(ev, p, self.bend, sr, chunk=self.osc_chunk), p["drive"])
        fenv = torch_env.adsr(p["fA"], p["fD"], p["fS"], p["aR"], ev, sr)
        fc = cutoff_curve(ev, p, fenv)
        y = torch_filter.tv_lowpass(osc, fc, p["reso"], sr, block=self.filter_block)
        aenv = torch_env.adsr(p["aA"], p["aD"], p["aS"], p["aR"], ev, sr)
        out = y * aenv * ev.gain[:, None]
        if self.voice_life is not None:
            out = out * self.voice_life.gate(ev)
        return out

    def render(self, p: dict[str, torch.Tensor]) -> torch.Tensor:
        """(2, N) stereo output from a dict of real parameter values."""
        voices = self.voice_output(p)
        gl, gr = pan_gains(self.ev, p["spread"])
        pre = torch.stack([(voices * gl[:, None]).sum(dim=0), (voices * gr[:, None]).sum(dim=0)])
        return torch_fx.effects(pre, p, self.sr)

    def forward(self, patch: Patch) -> torch.Tensor:
        return self.render(patch.values())


def pad_from_patch_file(
    path: str,
    device: torch.device,
    n_samples: int | None = None,
    midi: str = "out/transcription.mid",
    **kwargs,
) -> tuple[TorchPad, Patch, np.ndarray]:
    """TorchPad plus a Patch initialised from a stage-2 patch json.

    Returns the normalized vector as well, so the same numbers can be handed to
    synth.PadRenderer for the authoritative render.
    """
    z = np.asarray(json.load(open(path))["normalized"], dtype=float)
    n = default_n_samples() if n_samples is None else n_samples
    pad = TorchPad(load_notes(midi), n, device, **kwargs)
    return pad, Patch(z).to(device), z
