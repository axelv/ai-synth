"""The target is two instruments, so the synth is two instruments.

Measured, not assumed: stereo width (side over mid, dB) in the recording rises from
-23.80 at 30-45 Hz to +0.40 at 500-700 Hz, an 18.6 dB climb from a dead-centre low end
to a fully decorrelated midrange. Our single-voice render is flat to within 0.4 dB
across the same span, because one voice path through one stereo chain has exactly one
width. That is a mono-centred bass under a wide reverberant pad, and no setting of a
single patch can be both.

The frozen transcription already contains the split and nobody noticed: four notes at
MIDI 25/27/29, all at velocity 100, one landing on each chord boundary, against
twenty-five notes at MIDI 37-72 at velocity 75. Eight semitones of clear air between
them. Stage 1 is not re-opened here; its note list is only partitioned.

Two PadRenderers summed rather than one dawdreamer graph with two processors: the
layers are independent, so summing their audio is exact, and it keeps each layer a
plain patch that a person could load into a real synth. Both layers get the same
measured bend automation, because the +3 semitone bend at 13.35 s falls inside the bass
note that starts at 10.40 s.
"""

from __future__ import annotations

import json

import numpy as np

import eq_stage
import synth
from bend2 import bend_curve
from stage2 import DUR, SR, load_notes

SPLIT_MIDI = 33          # the gap runs 29 -> 37; 33 sits in the middle of it
BASS, PAD = "bass", "pad"

# The bass occupies the bottom of the spectrum and has four notes to justify its
# parameters, so it does not get to move all 26 bands. Bands above 500 Hz stay at 0 dB.
# This is regularisation by restriction rather than by penalty, and it is cheaper and
# more honest than letting the fit put gain where the layer has no energy.
BASS_EQ = tuple(i for i, f in enumerate(eq_stage.band_freqs()) if f < 500.0)
PAD_EQ = tuple(range(eq_stage.N_BANDS))


def split_notes(notes=None, at: int = SPLIT_MIDI):
    """Partition the frozen note list into (bass, pad).

    Asserts the register gap still exists, so that a future change to stage 1 fails
    loudly here instead of silently assigning a pad note to the bass layer.
    """
    notes = load_notes() if notes is None else notes
    bass = [n for n in notes if n[0] < at]
    pad = [n for n in notes if n[0] >= at]
    if not bass or not pad:
        raise ValueError(f"split at MIDI {at} left a layer empty")
    gap = min(n[0] for n in pad) - max(n[0] for n in bass)
    if gap < 4:
        raise ValueError(
            f"register gap is only {gap} semitones (bass top "
            f"{max(n[0] for n in bass)}, pad bottom {min(n[0] for n in pad)}); "
            "the layer split is no longer unambiguous"
        )
    return bass, pad


# ---------------------------------------------------------------- parameters


def flat(params: dict[str, dict[str, float]]) -> dict[str, float]:
    """{'bass': {...}, 'pad': {...}} -> {'bass.cutoff': ..., 'pad.cutoff': ...}"""
    return {f"{k}.{n}": v for k, layer in params.items() for n, v in layer.items()}


def unflat(d: dict[str, float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {BASS: {}, PAD: {}}
    for k, v in d.items():
        layer, _, name = k.partition(".")
        if layer not in out:
            raise KeyError(f"unknown layer in {k!r}")
        out[layer][name] = v
    return out


def default_params(base: str = "out/patch.json") -> dict[str, dict[str, float]]:
    """Both layers start from the current single-layer patch, then diverge.

    The bass starts mono and nearly dry, which is where the recording says it sits; the
    pad keeps the fitted EQ and is free to become wide. Starting the bass from the wet
    patch would waste the first fit undoing reverb the layer should never have had.
    """
    p = dict(json.load(open(base))["params"])
    bass, pad = dict(p), dict(p)
    bass.update(spread=0.0, revWet=0.02, dlyWet=0.0, chDepth=0.0)
    bass.update({f"eq{i}": 0.0 for i in range(eq_stage.N_BANDS)})
    return {BASS: bass, PAD: pad}


def eq_of(params: dict[str, float]) -> np.ndarray:
    return np.array([params[f"eq{i}"] for i in range(eq_stage.N_BANDS)], dtype=float)


def with_eq(params: dict[str, float], gains: np.ndarray) -> dict[str, float]:
    out = dict(params)
    out.update(eq_stage.gain_dict(gains))
    return out


# ---------------------------------------------------------------- rendering


class LayerRenderer:
    """Two independent synths, summed. Faust remains the authoritative renderer."""

    def __init__(self, notes=None, n_voices: int = 24, dsp: str = synth.DSP) -> None:
        bass_notes, pad_notes = split_notes(notes)
        self.notes = {BASS: bass_notes, PAD: pad_notes}
        self.r: dict[str, synth.PadRenderer] = {}
        bend = bend_curve(int(DUR * SR) + SR)
        for k in (BASS, PAD):
            r = synth.PadRenderer(n_voices=n_voices, dsp=dsp)
            r.set_notes(self.notes[k])
            r.set_bend(bend)
            self.r[k] = r

    def render_layer(self, layer: str, params: dict[str, float],
                     dur: float = DUR) -> np.ndarray:
        self.r[layer].set_params(params)
        return self.r[layer].render(dur)

    def render(self, params: dict[str, dict[str, float]],
               dur: float = DUR) -> np.ndarray:
        a = self.render_layer(BASS, params[BASS], dur)
        b = self.render_layer(PAD, params[PAD], dur)
        n = min(a.shape[1], b.shape[1])
        return a[:, :n] + b[:, :n]


def describe() -> str:
    bass, pad = split_notes()
    return (
        f"bass {len(bass)} notes, MIDI {sorted({n[0] for n in bass})}, "
        f"vel {sorted({n[1] for n in bass})}; "
        f"pad {len(pad)} notes, MIDI {min(n[0] for n in pad)}-{max(n[0] for n in pad)}, "
        f"vel {sorted({n[1] for n in pad})}; "
        f"bass EQ bands {len(BASS_EQ)} of {eq_stage.N_BANDS} (below 500 Hz)"
    )


if __name__ == "__main__":
    from stage2 import Objective

    print(describe())
    lr = LayerRenderer()
    p = default_params()
    obj = Objective(load_notes())

    a = lr.render(p)
    d = obj.loss_parts(a)
    print(f"\ntwo layers at the starting parameters: "
          f"mono {d['mono']:.4f}  side {d['side']:.4f}  total {d['total']:.4f}")
    for k in (BASS, PAD):
        x = lr.render_layer(k, p[k])
        rms = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)
        print(f"  {k:<5} alone: rms {rms:6.2f} dB")
