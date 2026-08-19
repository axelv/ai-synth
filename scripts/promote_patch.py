"""Promote a fitted candidate to the delivered artifacts, refusing to do it silently.

The rule this fit has run under is that Faust decides: a candidate replaces
out/patch.json only if its re-rendered true loss is lower. The loss and metrics written
here are the ones defect_check.py measured from the re-render it also saved, so
report.md, patch.json and render.wav cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import shutil

from synth import denorm, pad_normalized


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="patch json to promote")
    ap.add_argument("--label", required=True, help="its key in out/defect_check.json")
    ap.add_argument("--against", default="delivered_before",
                    help="key of the currently delivered render in out/defect_check.json")
    args = ap.parse_args()

    dc = json.load(open("out/defect_check.json"))
    new, old = dc[args.label], dc[args.against]
    print(f"{args.label} loss {new['loss']:.6f} vs {args.against} {old['loss']:.6f}")
    if new["loss"] >= old["loss"]:
        print("candidate does not beat what is delivered; nothing written")
        return

    z = pad_normalized(json.load(open(args.candidate))["normalized"])
    with open("out/patch.json", "w") as fh:
        json.dump({"loss": new["loss"], "metrics": new["metrics"], "params": denorm(z),
                   "pinned": {}, "normalized": z.tolist()}, fh, indent=2)
    shutil.copyfile(new["render"], "out/render.wav")
    print(f"wrote out/patch.json and out/render.wav from {args.candidate} / {new['render']}")


if __name__ == "__main__":
    main()
