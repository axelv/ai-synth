"""Differentiable port of Faust's en.adsr, closed form so it has no recursion.

en.adsr in envelopes.lib is written as two one-pole counters plus min/max clipping:

    an, dn, rn = max(1, at*SR), max(1, dt*SR), max(1, rt*SR)
    atime      = +(gate) ~ *(gate' >= gate)        // counts held samples, freezes at note-off
    rtime      = (+(1) : *(gate == 0)) ~ _         // counts samples since note-off
    A          = atime / an
    D          = 1 + (an - atime) * (1 - sl) / dn
    out        = max(0, min(A, max(D, sl)) * (1 - rtime / rn))

Both counters are pure functions of the gate, not of the parameters, so they can be
written down directly from the note schedule instead of being integrated sample by
sample. That is what makes this vectorised over all voices and differentiable in
(a, d, s, r) at the same time: the only param-dependent ops are three divisions and
two clips, all of them elementwise on precomputed integer ramps.

Two details that the closed form has to reproduce exactly, because they are where a
naive ADSR diverges:

  * atime is 1, not 0, on the first gate sample. The attack therefore reaches 1.0
    after an samples, not an+1, and the value at half the attack time is 0.5002 for
    a=0.1 rather than 0.5.
  * atime FREEZES at note-off rather than resetting, and the release is a multiply by
    (1 - rtime/rn) applied to that frozen ADS value. So when a note is released during
    its attack or its decay (which happens here: notes are 2-6 s but aA/aD reach 3-4 s)
    the release ramps down from whatever level was reached, never from the sustain
    level. Clamping atime to the hold length is what encodes that.

Deliberately NOT modelled here: voices in the full synth go silent well before aR
elapses (at out/patch.json, per-note tails run 858 to 26407 samples after note-off
against aR = 34063). verify_env.check_poly_wrapper shows that is not the envelope:
rendered through the same polyphonic PadRenderer with the oscillator and filter
removed, en.adsr still runs its whole release and still matches this port. The early
death is in the oscillator path, so it belongs to torch_osc, not here.
"""

from __future__ import annotations

import torch

from torch_common import SR, NoteEvents


def adsr(
    a: torch.Tensor,
    d: torch.Tensor,
    s: torch.Tensor,
    r: torch.Tensor,
    ev: NoteEvents,
    sr: int = SR,
) -> torch.Tensor:
    """(V, N) linear-segment ADSR, one row per note, matching en.adsr sample for sample."""
    hold = (ev.offset - ev.onset).to(torch.float32)[:, None]
    since = torch.round(ev.time_since_onset() * sr)

    atime = torch.clamp(since + 1.0, min=0.0)
    atime = torch.minimum(atime, hold)
    rtime = torch.clamp(since - hold + 1.0, min=0.0)

    an = torch.clamp(a * sr, min=1.0)
    dn = torch.clamp(d * sr, min=1.0)
    rn = torch.clamp(r * sr, min=1.0)

    attack = atime / an
    decay = 1.0 + (an - atime) * (1.0 - s) / dn
    ads = torch.minimum(attack, torch.maximum(decay, s))
    return torch.clamp(ads * (1.0 - rtime / rn), min=0.0)


def gate_events(
    hold_samples: list[int] | torch.Tensor,
    n_samples: int,
    device: torch.device,
) -> NoteEvents:
    """NoteEvents for probe gates that all open at sample 0, for verification only."""
    hold = torch.as_tensor(hold_samples, dtype=torch.long, device=device)
    v = int(hold.shape[0])
    return NoteEvents(
        freq=torch.full((v,), 440.0, device=device),
        gain=torch.ones(v, device=device),
        onset=torch.zeros(v, dtype=torch.long, device=device),
        offset=hold,
        n_samples=n_samples,
        device=device,
    )
