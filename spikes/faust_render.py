"""Render an arbitrary polyphonic Faust instrument to a wav, via dawdreamer.

Spike 2 harness. `scripts/synth.py` can only render its own DSP: PadRenderer asserts
that every name in PARAMS survived the compiler, and it owns the note list. This takes
any source that follows the Faust poly convention (`process` is one voice reading
freq/gain/gate, optional `effect` is the shared chain) and plays a fixed audition
pattern through it, so five candidate patches are heard under identical conditions.

Patterns are per-instrument-type on purpose. A bass patch auditioned on a four-note
held pad chord is being judged on material it was never written for, which would
measure the pattern rather than the patch.
"""

from __future__ import annotations

import os
import sys

import dawdreamer as daw
import numpy as np
import soundfile as sf

SR = 44100
BLOCK = 512
FAUST_LIBS = os.path.join(os.path.dirname(daw.__file__), "faustlibraries")

# (pitch, velocity, start_sec, dur_sec)
Note = tuple[int, int, float, float]


def _chord(pitches: list[int], vel: int, start: float, dur: float) -> list[Note]:
    return [(p, vel, start, dur) for p in pitches]


# Every pattern ends well before its render duration so the release tail is audible;
# a patch judged on a truncated tail is judged on the wrong thing.
PATTERNS: dict[str, tuple[list[Note], float]] = {
    # slow harmonic movement, long holds: what a pad is for
    "pad": (
        _chord([53, 60, 64, 69], 90, 0.0, 3.4)
        + _chord([48, 60, 63, 67], 90, 3.6, 3.4)
        + _chord([55, 62, 67, 71], 80, 7.2, 4.6),
        14.0,
    ),
    # single line, short notes, wide register: exposes attack and note-off clicks
    "bass": (
        [(36, 110, 0.0, 0.42), (36, 100, 0.5, 0.22), (43, 105, 0.8, 0.38),
         (36, 110, 1.3, 0.42), (39, 100, 1.8, 0.22), (41, 105, 2.1, 0.55),
         (36, 110, 2.8, 0.42), (48, 100, 3.3, 0.22), (43, 110, 3.6, 0.9),
         (31, 110, 4.8, 1.6)],
        8.0,
    ),
    # sustained melodic phrase plus one held tail: exposes vibrato and legato behaviour
    "lead": (
        [(72, 100, 0.0, 0.9), (74, 100, 0.95, 0.45), (76, 100, 1.45, 0.9),
         (79, 105, 2.4, 1.4), (76, 95, 3.9, 0.45), (74, 95, 4.4, 0.45),
         (72, 100, 4.9, 2.6)],
        10.0,
    ),
    # fast repeats and a chord: exposes voice stealing and decay length
    "pluck": (
        [(60, 110, 0.0, 0.18), (64, 105, 0.25, 0.18), (67, 105, 0.5, 0.18),
         (72, 110, 0.75, 0.18), (67, 100, 1.0, 0.18), (64, 100, 1.25, 0.18)]
        + _chord([48, 55, 60, 64, 67], 110, 1.6, 0.3)
        + [(60, 110, 3.0, 0.18), (63, 105, 3.2, 0.18), (65, 105, 3.4, 0.18),
           (70, 110, 3.6, 1.0)],
        9.0,
    ),
}


def render_dsp(dsp: str, notes: list[Note], dur: float, n_voices: int = 16,
               sr: int = SR, release: float = 4.0) -> np.ndarray:
    """Compile `dsp` and play `notes` through it. Returns (2, n).

    Raises RuntimeError with faust's own message on a compile failure, which is the
    signal spike 2 counts as a wasted round.
    """
    engine = daw.RenderEngine(sr, BLOCK)
    proc = engine.make_faust_processor("inst")
    proc.faust_libraries_path = FAUST_LIBS
    proc.num_voices = n_voices
    proc.release_length = release
    proc.group_voices = True
    if not proc.set_dsp_string(dsp):
        raise RuntimeError("faust compile failed")
    engine.load_graph([(proc, [])])
    proc.clear_midi()
    for p, v, s, d in notes:
        proc.add_midi_note(int(p), int(v), float(s), float(d))
    engine.render(dur)
    return engine.get_audio()


class Instrument:
    """A compiled Faust instrument that can be re-rendered at different macro settings.

    A macro sweep is dozens of renders of the same DSP, and `render_dsp` recompiles on
    every call, so the sweep would spend most of its time in the Faust compiler. This
    compiles once and swaps parameters, the way `synth.PadRenderer` already does for the
    stage-2 fit. Parameters are addressed by LABEL, never by the Faust path, because the
    path shape changes with whether the DSP happens to declare `effect` and `name`.
    """

    def __init__(self, dsp: str, n_voices: int = 16, sr: int = SR,
                 release: float = 4.0) -> None:
        self.sr = sr
        self.engine = daw.RenderEngine(sr, BLOCK)
        self.proc = self.engine.make_faust_processor("inst")
        self.proc.faust_libraries_path = FAUST_LIBS
        self.proc.num_voices = n_voices
        self.proc.release_length = release
        self.proc.group_voices = True
        if not self.proc.set_dsp_string(dsp):
            raise RuntimeError("faust compile failed")
        desc = self.proc.get_parameters_description()
        self.index = {d["label"]: d["index"] for d in desc}
        # freq/gain/gate belong to the voice and are driven by MIDI, so whatever is left
        # is the patch's own control surface
        self.macros = [
            {"label": d["label"], "lo": float(d["min"]), "hi": float(d["max"]),
             "default": float(d["value"])}
            for d in desc if d["label"] not in ("freq", "gain", "gate")
        ]
        self.defaults = {m["label"]: m["default"] for m in self.macros}
        self.engine.load_graph([(self.proc, [])])

    def render(self, notes: list[Note], dur: float,
               params: dict[str, float] | None = None) -> np.ndarray:
        """Render `notes`, with `params` overriding defaults. Unset macros go to default."""
        values = dict(self.defaults)
        values.update(params or {})
        for label, val in values.items():
            if label in self.index:
                self.proc.set_parameter(self.index[label], float(val))
        self.proc.clear_midi()
        for p, v, s, d in notes:
            self.proc.add_midi_note(int(p), int(v), float(s), float(d))
        self.engine.render(dur)
        return self.engine.get_audio()


def render_file(dsp_path: str, out_path: str, pattern: str) -> dict[str, float]:
    with open(dsp_path) as fh:
        dsp = fh.read()
    notes, dur = PATTERNS[pattern]
    audio = render_dsp(dsp, notes, dur)
    # PCM_24, matching scripts/synth.write_render: 16-bit costs real measurable error
    # in the top bands, and these files exist to be listened to critically.
    sf.write(out_path, np.asarray(audio).T, SR, subtype="PCM_24")
    return audio_stats(audio)


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    """Enough to catch a patch that is silent, clipped, or mono, without listening."""
    a = np.asarray(audio, dtype=float)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    rms = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
    mid = (a[0] + a[1]) / 2
    side = (a[0] - a[1]) / 2
    side_db = 20 * np.log10(max(float(np.sqrt(np.mean(side ** 2))), 1e-12)
                            / max(float(np.sqrt(np.mean(mid ** 2))), 1e-12))
    return {
        "peak": round(peak, 4),
        "rms_db": round(20 * np.log10(max(rms, 1e-12)), 2),
        "side_db": round(float(side_db), 2),
        "clipped": float(np.mean(np.abs(a) > 0.999)),
    }


if __name__ == "__main__":
    dsp_path, out_path, pattern = sys.argv[1], sys.argv[2], sys.argv[3]
    print(render_file(dsp_path, out_path, pattern))
