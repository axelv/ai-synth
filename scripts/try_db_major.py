"""Swap the Db region to Db major and re-render.

The adopted transcription carried a written E3 (from the author's "Dbm" label),
but the clean spectral test found no E natural in that region (~34 dB below the
root). Db major replaces that E3 with F3. The no-third voicing is measured too,
since the major third itself is confounded by harmonic 5 of the Db bass and so
cannot be confirmed either way.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import soundfile as sf

import bend2
from build_midi import to_pm
from metrics import report
from stage2 import Objective
from synth import denorm

SR = 44100
DUR = 17.904

VARIANTS = {
    "Db minor (current)": [12, 19, 24, 27, 31],   # C#2 G#2 C#3 E3  G#3
    "Db major": [12, 19, 24, 28, 31],             # C#2 G#2 C#3 F3  G#3
    "Db no third": [12, 19, 24, 31],              # C#2 G#2 C#3     G#3
}
ADOPT = "Db major"
DB_REGION_INDEX = 2  # 7.45-10.45


def main() -> None:
    choice = json.load(open("out/stage1_choice.json"))
    patch = json.load(open("out/patch.json"))
    px = np.array(patch["normalized"], dtype=float)
    vel = choice["velocities"]
    regions = choice["regions"]
    assert abs(regions[DB_REGION_INDEX]["t0"] - 7.45) < 1e-6, "unexpected region layout"

    orig_st, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    obj = Objective([(60, 100, 0.0, 1.0)])
    obj.renderer.set_bend(bend2.bend_curve(int(DUR * SR) + SR))

    def build(db_voicing):
        notes = [(p, vel["intro"], 0.02, 3.53) for p in choice["intro"]]
        for i, r in enumerate(regions):
            vo = db_voicing if i == DB_REGION_INDEX else r["voicing"]
            d = r["t1"] - r["t0"]
            notes.append((r["bass"], 100, r["t0"], d))
            for s in vo:
                notes.append((r["bass"] + s, vel["upper"], r["t0"], d))
        return notes

    results = {}
    for name, vo in VARIANTS.items():
        notes = build(vo)
        obj.renderer.set_notes(notes)
        loss = obj(px)
        aud = obj.renderer.render(DUR)
        m = report(orig_st, aud)
        results[name] = (loss, m, aud, notes, vo)
        tag = " ".join(librosa.midi_to_note(25 + x) for x in vo)
        print(f"  {name:20s} loss={loss:.4f}  chroma={m['chroma_agree']:.4f}  "
              f"mel={m['mel_dist']:.2f}  env={m['env_l1']:.4f}   {tag}")

    loss, m, aud, notes, vo = results[ADOPT]
    print(f"\nadopting {ADOPT} as requested")
    to_pm(notes, "out/transcription.mid")
    sf.write("out/render.wav", aud.T, SR)
    regions[DB_REGION_INDEX]["voicing"] = vo
    regions[DB_REGION_INDEX]["label"] = ADOPT
    choice["db_region_test"] = {k: {"loss": v[0], "chroma": v[1]["chroma_agree"],
                                    "mel": v[1]["mel_dist"]} for k, v in results.items()}
    json.dump(choice, open("out/stage1_choice.json", "w"), indent=2)
    patch["metrics"] = m
    patch["loss"] = loss
    json.dump(patch, open("out/patch.json", "w"), indent=2)
    print(f"wrote out/transcription.mid ({len(notes)} notes) + out/render.wav")


if __name__ == "__main__":
    main()
