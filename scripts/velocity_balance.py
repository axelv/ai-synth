"""Infer note velocities from measured energy balance.

The long-term average spectrum showed the render 10-20 dB light between 250 and
900 Hz — the register the upper voicing occupies. Velocities were set by hand
(bass 100 / upper 66 / intro 60); here they are fitted against the same loss
stage 2 uses, then frozen before the patch is re-polished.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pretty_midi

from build_midi import to_pm
from stage2 import Objective, load_notes
from synth import denorm

BASS_V = 100


def rebuild(choice: dict, upper_v: int, intro_v: int):
    from build_midi import INTRO_END
    from polish_midi import REGIONS

    notes = [(p, intro_v, 0.02, INTRO_END - 0.02) for p in choice["intro"]]
    for (t0, t1, bass), r in zip(REGIONS, choice["regions"]):
        d = t1 - t0
        notes.append((bass, BASS_V, t0, d))
        for s in r["voicing"]:
            notes.append((bass + s, upper_v, t0, d))
    return notes


def main() -> None:
    choice = json.load(open("out/stage1_choice.json"))
    patch = json.load(open("out/patch.json"))
    x = np.array(patch["normalized"], dtype=float)

    obj = Objective(load_notes())
    obj.renderer.set_params(denorm(x))

    results = []
    for upper_v, intro_v in itertools.product((60, 75, 90, 105, 120), (50, 70, 90, 110)):
        notes = rebuild(choice, upper_v, intro_v)
        obj.renderer.set_notes(notes)
        loss = obj(x)
        results.append((loss, upper_v, intro_v))
        print(f"  upper={upper_v:3d} intro={intro_v:3d}  loss={loss:.4f}")

    results.sort()
    loss, upper_v, intro_v = results[0]
    print(f"\nbest: upper={upper_v} intro={intro_v}  loss={loss:.4f} "
          f"(was upper=66 intro=60 -> {patch['loss']:.4f})")

    notes = rebuild(choice, upper_v, intro_v)
    to_pm(notes, "out/transcription.mid")
    choice["velocities"] = {"bass": BASS_V, "upper": upper_v, "intro": intro_v}
    with open("out/stage1_choice.json", "w") as fh:
        json.dump(choice, fh, indent=2)
    print(f"wrote out/transcription.mid ({len(notes)} notes)")


if __name__ == "__main__":
    main()
