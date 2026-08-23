"""Assemble one self-contained HTML page from a Faust DSP.

Spike 3. The delivery target is an Artifact, whose CSP blocks every external host, so
nothing may be fetched at runtime: the faustwasm runtime, both wasm modules and their
metadata all have to be inside the file. faust2wasm's own generated glue fetches its
assets by relative URL, which is exactly what cannot work here, so this bypasses it and
hands `FaustPolyDspGenerator.createNode` already-compiled modules instead.

Usage:
    uv run python spikes/build_page.py spikes/dsp/warm-pad.dsp spikes/web/warm-pad.html
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "page_template.html")
# faustwasm ships libfaust compiled to wasm, so the build needs node but NOT the native
# faust CLI. Installed by `npm install` at the repo root; see package.json.
REPO = os.path.dirname(HERE)
FAUST2WASM = os.environ.get(
    "FAUST2WASM",
    os.path.join(REPO, "node_modules", "@grame", "faustwasm", "scripts", "faust2wasm.js"),
)

BLURB = ("Faust compiled to WebAssembly, running in an AudioWorklet. "
         "Everything is inside this one file: no network, no server. "
         "Press start, then play the keys.")


def compile_dsp(dsp_path: str, out_dir: str) -> None:
    if not os.path.exists(FAUST2WASM):
        raise RuntimeError(
            f"faust2wasm not found at {FAUST2WASM}. Run `npm install` at the repo root.")
    r = subprocess.run(
        ["node", FAUST2WASM, dsp_path, out_dir, "-poly", "-standalone"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"faust2wasm failed:\n{r.stdout}\n{r.stderr}")


def build(dsp_path: str, out_path: str, voices: int = 16,
          title: str | None = None) -> dict[str, int]:
    name = os.path.splitext(os.path.basename(dsp_path))[0]
    # The Faust name has to stay the slug (it keys the worklet processor registration);
    # only the heading and <title> get the readable form.
    display = title or name.replace("-", " ").title()
    tmp = tempfile.mkdtemp(prefix="faustbuild-")
    try:
        compile_dsp(dsp_path, tmp)

        def read(fn: str) -> bytes:
            with open(os.path.join(tmp, fn), "rb") as fh:
                return fh.read()

        def b64(fn: str) -> str:
            return base64.b64encode(read(fn)).decode("ascii")

        has_effect = os.path.exists(os.path.join(tmp, "effect-module.wasm"))
        payload = {
            "name": name,
            "dsp": b64("dsp-module.wasm"),
            "mixer": b64("mixer-module.wasm"),
            # the generator wants the metadata as a JSON *string*, not an object
            "dspMeta": read("dsp-meta.json").decode(),
            "effect": b64("effect-module.wasm") if has_effect else None,
            "effectMeta": read("effect-meta.json").decode() if has_effect else None,
        }
        runtime = read(os.path.join("faustwasm", "index.js")).decode()
        # Pasting the bundle into an inline module leaves its trailing `export { X as Y }`
        # list behind. Dropping it is not cosmetic: the names it re-exports stay in scope
        # for the code below, and renaming the block instead of deleting it is a syntax
        # error on the `as` clauses.
        i = runtime.rindex("\nexport {")
        runtime = runtime[:i] + runtime[runtime.index("};", i) + 2:]

        with open(TEMPLATE) as fh:
            html = fh.read()
        html = (html
                .replace("__FAUSTWASM__", runtime)
                .replace("__PAYLOAD__", json.dumps(payload))
                .replace("__VOICES__", str(voices))
                .replace("__TITLE__", display)
                .replace("__EYEBROW__", "ai-synth")
                .replace("__BLURB__", BLURB))

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(html)

        return {
            "dsp_wasm": len(read("dsp-module.wasm")),
            "effect_wasm": len(read("effect-module.wasm")) if has_effect else 0,
            "mixer_wasm": len(read("mixer-module.wasm")),
            "runtime_js": len(runtime),
            "page": len(html),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    stats = build(sys.argv[1], sys.argv[2],
                  title=sys.argv[3] if len(sys.argv) > 3 else None)
    total = stats["page"]
    print(f"{sys.argv[2]}")
    for k, v in stats.items():
        print(f"  {k:12} {v/1024:9.1f} KiB")
    print(f"  {'cap':12} {16*1024:9.1f} KiB  ({100*total/(16*1024*1024):.2f}% used)")
