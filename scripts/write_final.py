"""Write the final MIDI + render, guaranteeing the two artifacts agree.

Fixes a real defect: my measured region boundaries overlapped by ~50 ms
(e.g. region 2 ended at 10.45 while region 3 began at 10.40), and F3 occurs in
both chords. Writing that to a MIDI file puts a note-off for the earlier F3 in
the middle of the later one, truncating it to 50 ms — so the delivered
transcription.mid did not contain what was rendered.

Two changes:
  * region ends are snapped to the next region's start, so no same-pitch overlap
    can occur (chords abut, as they should);
  * the render is produced from notes RELOADED FROM THE FILE, so render.wav is
    exactly what transcription.mid encodes.

Both are then asserted rather than assumed.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

import bend2
from metrics import report
from stage2 import Objective, load_notes
from synth import denorm

SR = 44100
DUR = 17.904
RESOLUTION = 1920  # ticks per beat; keeps round-trip error well under 1 ms


def to_pm(notes, path: str, tempo: float = 82.0) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=RESOLUTION)
    inst = pretty_midi.Instrument(program=90, name="pad")
    for p, v, s, d in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p),
                                           start=float(s), end=float(s + d)))
    pm.instruments.append(inst)
    pm.write(path)


def check_no_pitch_overlap(notes) -> None:
    by_pitch: dict[int, list[tuple[float, float]]] = {}
    for p, _, s, d in notes:
        by_pitch.setdefault(p, []).append((s, s + d))
    bad = []
    for p, spans in by_pitch.items():
        spans.sort()
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if s2 < e1 - 1e-9:
                bad.append((p, s1, e1, s2, e2))
    if bad:
        for p, s1, e1, s2, e2 in bad:
            print(f"  OVERLAP pitch {p} ({librosa.midi_to_note(p)}): "
                  f"[{s1:.3f},{e1:.3f}] vs [{s2:.3f},{e2:.3f}]")
        raise SystemExit("same-pitch overlaps would be truncated on MIDI write")
    print("  no same-pitch overlaps")


def main() -> None:
    choice = json.load(open("out/stage1_choice.json"))
    patch = json.load(open("out/patch.json"))
    px = np.array(patch["normalized"], dtype=float)
    vel = choice["velocities"]
    regions = choice["regions"]

    # snap each region's end to the next region's start
    for a, b in zip(regions, regions[1:]):
        if abs(a["t1"] - b["t0"]) > 1e-9:
            print(f"  snapping region end {a['t1']:.2f} -> {b['t0']:.2f}")
            a["t1"] = b["t0"]

    notes = [(p, vel["intro"], 0.02, min(3.53, regions[0]["t0"] - 0.02)) for p in choice["intro"]]
    for r in regions:
        d = r["t1"] - r["t0"]
        notes.append((r["bass"], 100, r["t0"], d))
        for s in r["voicing"]:
            notes.append((r["bass"] + s, vel["upper"], r["t0"], d))
    notes = sorted(notes, key=lambda z: (z[2], z[0]))

    print(f"building {len(notes)} notes")
    check_no_pitch_overlap(notes)

    to_pm(notes, "out/transcription.mid")
    reloaded = load_notes()
    print(f"  wrote + reloaded {len(reloaded)} notes")
    assert len(reloaded) == len(notes), f"note count changed: {len(notes)} -> {len(reloaded)}"
    worst = max(max(abs(a[2] - b[2]), abs(a[3] - b[3]))
                for a, b in zip(sorted(notes, key=lambda z: (z[2], z[0])),
                                sorted(reloaded, key=lambda z: (z[2], z[0]))))
    print(f"  worst round-trip timing error: {worst * 1000:.3f} ms")
    assert worst < 0.002, f"round-trip error too large: {worst * 1000:.2f} ms"

    # render FROM THE FILE so the artifacts cannot disagree
    obj = Objective(reloaded)
    obj.renderer.set_bend(bend2.bend_curve(int(DUR * SR) + SR))
    loss = obj(px)
    aud = obj.renderer.render(DUR)
    orig, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    m = report(orig, aud)
    print(f"\nloss={loss:.4f}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    sf.write("out/render.wav", aud.T, SR)
    patch["metrics"] = m
    patch["loss"] = loss
    patch["rendered_from"] = "out/transcription.mid (reloaded)"
    json.dump(patch, open("out/patch.json", "w"), indent=2)
    json.dump(choice, open("out/stage1_choice.json", "w"), indent=2)
    print("wrote out/transcription.mid + out/render.wav")

    for r in regions:
        v = " ".join(librosa.midi_to_note(r["bass"] + x) for x in r["voicing"])
        print(f"  [{r['t0']:5.2f}-{r['t1']:5.2f}] {r.get('label', '?'):12s} "
              f"{librosa.midi_to_note(r['bass']):>4s} + {v}")


if __name__ == "__main__":
    main()
