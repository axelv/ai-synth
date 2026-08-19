"""Shared contract for the differentiable PyTorch surrogate of synth.py.

Stage 1 stays frozen: the MIDI and the measured bend curve are inputs here, never
parameters. What changes is how stage 2 searches. CMA-ES needed roughly 7500 Faust
renders because it can only see the loss value; a differentiable synth gets the
gradient of the same loss with respect to every parameter from one render.

The surrogate is not the deliverable. Faust remains authoritative: anything the
gradient finds is re-rendered through PadRenderer and only kept if the true loss
improves. That is what keeps the delivered patch portable to a real synth.

Module contract, so the pieces can be built and tested independently:

    torch_osc.osc_bank(ev, p, bend, sr)        -> (V, N)  pre-filter oscillator
    torch_env.adsr(a, d, s, r, ev, sr)         -> (V, N)  envelope per voice
    torch_filter.tv_lowpass(x, fc, q)          -> (V, N)  resonlp then lowpass(2)
    torch_fx.effects(stereo, p, sr)            -> (2, N)  chorus/delay/reverb/tilt

V is one entry per MIDI note (no voice allocation needed: notes are independent
and summed, which is what group_voices=True does in dawdreamer).
"""

from __future__ import annotations

from dataclasses import dataclass

import auraloss
import librosa
import numpy as np
import torch
import torch.nn as nn

from synth import PARAMS, PARAM_INDEX

SR = 44100
DUR = 17.904
HOP = 512
KBD_REF = 261.6255  # C4, the reference pitch for keyboard tracking in the Faust DSP


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- parameters


class Patch(nn.Module):
    """synth.PARAMS as unconstrained logits mapped through sigmoid to [0,1].

    Sigmoid rather than clamping so a parameter that wants to sit at a box edge
    does not kill its own gradient. The [0,1] space and the log-scaling are
    identical to synth.denorm, so a Patch and a CMA-ES vector are interchangeable.
    """

    def __init__(self, normalized: np.ndarray | None = None) -> None:
        super().__init__()
        z = np.full(len(PARAMS), 0.5) if normalized is None else np.asarray(normalized, float)
        z = np.clip(z, 1e-4, 1 - 1e-4)
        self.logits = nn.Parameter(torch.from_numpy(np.log(z / (1 - z))).float())

    def normalized(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def values(self) -> dict[str, torch.Tensor]:
        return denorm_torch(self.normalized())

    def to_numpy(self) -> np.ndarray:
        return self.normalized().detach().cpu().numpy().astype(float)

    def freeze(self, names: list[str]) -> None:
        """Exclude parameters from the gradient step (they stay with CMA-ES)."""
        mask = torch.ones_like(self.logits)
        for n in names:
            mask[PARAM_INDEX[n]] = 0.0
        self.logits.register_hook(lambda g: g * mask)


def denorm_torch(z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Differentiable, log-aware [0,1] -> real. Mirrors synth.denorm exactly."""
    out: dict[str, torch.Tensor] = {}
    for i, p in enumerate(PARAMS):
        v = z[i]
        if p.log:
            lo, hi = float(np.log(p.lo)), float(np.log(p.hi))
            out[p.name] = torch.exp(lo + v * (hi - lo))
        else:
            out[p.name] = p.lo + v * (p.hi - p.lo)
    return out


# ---------------------------------------------------------------- scheduling


@dataclass
class NoteEvents:
    """One entry per MIDI note. All tensors live on the same device."""

    freq: torch.Tensor    # (V,) base frequency in Hz, from the MIDI pitch
    gain: torch.Tensor    # (V,) velocity / 127
    onset: torch.Tensor   # (V,) note-on sample index, int64
    offset: torch.Tensor  # (V,) note-off sample index, int64
    n_samples: int
    device: torch.device

    @property
    def n_voices(self) -> int:
        return int(self.freq.shape[0])

    def time_since_onset(self) -> torch.Tensor:
        """(V, N) seconds since note-on, negative before it. Basis for envelopes."""
        t = torch.arange(self.n_samples, device=self.device, dtype=torch.float32)
        return (t[None, :] - self.onset[:, None].float()) / SR

    def held(self) -> torch.Tensor:
        """(V, N) float gate: 1 while the note is held, 0 outside."""
        t = torch.arange(self.n_samples, device=self.device)
        return ((t[None, :] >= self.onset[:, None]) & (t[None, :] < self.offset[:, None])).float()


def schedule(notes, n_samples: int, device: torch.device, sr: int = SR) -> NoteEvents:
    """notes = [(pitch, velocity, start_sec, dur_sec)], as returned by stage2.load_notes."""
    pitch = np.array([n[0] for n in notes], dtype=np.float64)
    vel = np.array([n[1] for n in notes], dtype=np.float64)
    start = np.array([n[2] for n in notes], dtype=np.float64)
    dur = np.array([n[3] for n in notes], dtype=np.float64)
    return NoteEvents(
        freq=torch.tensor(librosa.midi_to_hz(pitch), dtype=torch.float32, device=device),
        gain=torch.tensor(vel / 127.0, dtype=torch.float32, device=device),
        onset=torch.tensor(np.round(start * sr), dtype=torch.long, device=device),
        offset=torch.tensor(np.round((start + dur) * sr), dtype=torch.long, device=device),
        n_samples=n_samples,
        device=device,
    )


# ---------------------------------------------------------------- objective


def _env(x: torch.Tensor, hop: int = HOP) -> torch.Tensor:
    """Scale-invariant RMS envelope. Identical to stage2.Objective._env."""
    f = x.reshape(1, 1, -1).unfold(-1, hop * 2, hop)
    e = f.pow(2).mean(-1).sqrt()
    return e / (e.mean() + 1e-9)


class SpectralLoss(nn.Module):
    """The stage-2 objective, differentiable.

    Deliberately the same numbers as stage2.Objective (same FFT sizes, same
    w_env) so a loss from the surrogate is comparable to the 1.5564 that CMA-ES
    reached. It runs on CPU: auraloss STFTs on MPS were not worth the transfer.
    """

    def __init__(self, target_path: str = "data/original.wav", w_env: float = 0.35) -> None:
        super().__init__()
        y, _ = librosa.load(target_path, sr=SR, mono=True)
        self.n = len(y)
        self.w_env = w_env
        self.register_buffer("target", torch.from_numpy(y).float().view(1, 1, -1))
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048, 4096],
            hop_sizes=[128, 256, 512, 1024],
            win_lengths=[512, 1024, 2048, 4096],
            w_sc=1.0,
            w_log_mag=1.0,
            w_lin_mag=0.0,
        )

    def forward(self, stereo: torch.Tensor) -> torch.Tensor:
        """stereo is (2, N) as produced by the surrogate. Returns a scalar."""
        mono = stereo.mean(dim=0).to(self.target.device)
        n = min(mono.shape[-1], self.n)
        pred = mono[:n].reshape(1, 1, -1)
        tgt = self.target[..., :n]
        return self.mrstft(pred, tgt) + self.w_env * (_env(pred) - _env(tgt)).abs().mean()


# ---------------------------------------------------------------- bend input


def bend_tensor(n_samples: int, device: torch.device) -> torch.Tensor:
    """The measured intro glide + late +3 semitone bend, as a frozen (N,) multiplier."""
    from bend2 import bend_curve

    curve = np.asarray(bend_curve(n_samples), dtype=np.float32)[:n_samples]
    if len(curve) < n_samples:
        curve = np.pad(curve, (0, n_samples - len(curve)), constant_values=curve[-1])
    return torch.from_numpy(curve).to(device)


def default_n_samples(dur: float = DUR, sr: int = SR) -> int:
    return int(dur * sr)
