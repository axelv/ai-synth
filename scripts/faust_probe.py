"""Render isolated Faust sub-chains, so the PyTorch port can be fitted to ground truth.

The point of the port is to be a fast differentiable *surrogate* of the Faust
synth in synth.py. A surrogate is only worth anything if it agrees with what it
stands in for, so every module gets fitted against real Faust output rather than
against my recollection of what Faust's library functions do.

Two entry points:
  * render_gen  - a DSP with no audio inputs (envelopes, oscillators)
  * render_fx   - a DSP fed by a playback processor (filters, effects, reverb)

Gates are synthesised inside the probe DSP from ba.time rather than driven over
MIDI, which keeps the probes free of dawdreamer's polyphony layer.
"""

from __future__ import annotations

import os

import dawdreamer as daw
import numpy as np

SR = 44100
BLOCK = 512
FAUST_LIBS = os.path.join(os.path.dirname(daw.__file__), "faustlibraries")

# Drop this in a probe DSP to get a gate that opens at 0 and closes at `hold`.
GATE_SNIPPET = """
hold = hslider("hold", 1.0, 0.001, 20, 0.000001);
gate = ba.time < int(hold * ma.SR);
"""


def _make(engine: daw.RenderEngine, dsp: str, params: dict[str, float]):
    proc = engine.make_faust_processor("probe")
    proc.faust_libraries_path = FAUST_LIBS
    if not proc.set_dsp_string(dsp):
        raise RuntimeError("faust compile failed:\n" + dsp)
    pidx = {d["label"]: d["index"] for d in proc.get_parameters_description()}
    unknown = [k for k in params if k not in pidx]
    if unknown:
        raise KeyError(f"probe DSP does not expose {unknown}; it has {sorted(pidx)}")
    for name, val in params.items():
        proc.set_parameter(pidx[name], float(val))
    return proc


def render_gen(dsp: str, params: dict[str, float], dur: float, sr: int = SR) -> np.ndarray:
    """Render a generator DSP (no audio inputs). Returns (n_channels, n_samples)."""
    engine = daw.RenderEngine(sr, BLOCK)
    proc = _make(engine, dsp, params)
    engine.load_graph([(proc, [])])
    engine.render(dur)
    return engine.get_audio()


def render_fx(dsp: str, params: dict[str, float], x: np.ndarray, sr: int = SR,
              tail: float = 0.0) -> np.ndarray:
    """Render an effect DSP fed by `x`. x is (n_samples,) or (n_channels, n_samples)."""
    sig = np.asarray(x, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[None, :]
    if tail > 0.0:
        sig = np.concatenate([sig, np.zeros((sig.shape[0], int(tail * sr)), np.float32)], axis=1)
    engine = daw.RenderEngine(sr, BLOCK)
    play = engine.make_playback_processor("src", sig)
    proc = _make(engine, dsp, params)
    engine.load_graph([(play, []), (proc, ["src"])])
    engine.render(sig.shape[1] / sr)
    return engine.get_audio()


def impulse(n: int, sr: int = SR, channels: int = 1) -> np.ndarray:
    x = np.zeros((channels, n), dtype=np.float32)
    x[:, 0] = 1.0
    return x


def sweep(n: int, f0: float = 20.0, f1: float = 20000.0, sr: int = SR) -> np.ndarray:
    """Exponential sweep, for measuring a filter's magnitude response."""
    t = np.arange(n) / sr
    T = n / sr
    k = np.log(f1 / f0)
    phase = 2 * np.pi * f0 * T / k * (np.exp(t * k / T) - 1.0)
    return (0.5 * np.sin(phase)).astype(np.float32)


def save_fixture(path: str, **arrays: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"wrote {path} ({', '.join(f'{k}{tuple(np.shape(v))}' for k, v in arrays.items())})")
