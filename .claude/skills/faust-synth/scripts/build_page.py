"""Assemble one self-contained HTML page from a Faust DSP.

Spike 3. The delivery target is an Artifact, whose CSP blocks every external host, so
nothing may be fetched at runtime: the faustwasm runtime, both wasm modules and their
metadata all have to be inside the file. faust2wasm's own generated glue fetches its
assets by relative URL, which is exactly what cannot work here, so this bypasses it and
hands `FaustPolyDspGenerator.createNode` already-compiled modules instead.

Usage:
    uv run python <skill>/scripts/build_page.py patch.dsp out/patch.html "Display Name"
    uv run python <skill>/scripts/build_page.py juno.dsp out/juno.html "Juno-106" --skin juno
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
TEMPLATE = os.path.join(ASSETS, "page_template.html")
SKINS = os.path.join(ASSETS, "skins")
REL = os.path.join("node_modules", "@grame", "faustwasm", "scripts", "faust2wasm.js")


def find_faust2wasm() -> str:
    """Locate faust2wasm.js, which ships libfaust compiled to wasm.

    The build therefore needs node but NOT the native faust CLI. Searched from the
    working directory upward so the skill works in any project that has run
    `npm install @grame/faustwasm`, rather than only in the one it was written in.
    """
    if os.environ.get("FAUST2WASM"):
        return os.environ["FAUST2WASM"]
    d = os.path.abspath(os.getcwd())
    while True:
        cand = os.path.join(d, REL)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.join(os.getcwd(), REL)  # reported in the error below
        d = parent

BLURB = ("Faust compiled to WebAssembly, running in an AudioWorklet. "
         "Everything is inside this one file: no network, no server. "
         "Press start, then play the keys.")


def load_skin(name: str) -> tuple[str, str]:
    """Return a skin's (css, js).

    A skin is one HTML fragment holding a <style> and a <script>. It is spliced into
    the shared template rather than replacing it, so the keyboard, the MIDI handling
    and the boot path exist in exactly one place however many skins there are. The JS
    lands inside the page's module, which is what lets it assign to SKIN.
    """
    if name == "plain":
        return "", ""
    path = os.path.join(SKINS, name + ".html")
    if not os.path.exists(path):
        have = sorted(f[:-5] for f in os.listdir(SKINS) if f.endswith(".html"))
        raise RuntimeError(f"no skin {name!r}. Available: plain, {', '.join(have)}")
    with open(path) as fh:
        frag = fh.read()

    def block(tag: str) -> str:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", frag, re.S)
        if not m:
            raise RuntimeError(f"skin {name!r} has no <{tag}> block")
        return m.group(1)

    return block("style"), block("script")


def compile_dsp(dsp_path: str, out_dir: str) -> None:
    faust2wasm = find_faust2wasm()
    if not os.path.exists(faust2wasm):
        raise RuntimeError(
            "faust2wasm not found. Run `npm install @grame/faustwasm` in the project "
            f"root, or set FAUST2WASM. Looked for {faust2wasm}")
    r = subprocess.run(
        ["node", faust2wasm, dsp_path, out_dir, "-poly", "-standalone"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"faust2wasm failed:\n{r.stdout}\n{r.stderr}")


def build(dsp_path: str, out_path: str, voices: int = 16,
          title: str | None = None, skin: str = "plain") -> dict[str, int]:
    name = os.path.splitext(os.path.basename(dsp_path))[0]
    # The Faust name has to stay the slug (it keys the worklet processor registration);
    # only the heading and <title> get the readable form.
    display = title or name.replace("-", " ").title()
    skin_css, skin_js = load_skin(skin)
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
        # The skin goes in first: its CSS and JS are allowed to contain none of the
        # other markers, and substituting it last would let a marker inside it expand.
        html = (html
                .replace("__SKIN_CSS__", skin_css)
                .replace("__SKIN_JS__", skin_js)
                .replace("__SKIN__", skin)
                .replace("__FAUSTWASM__", runtime)
                .replace("__PAYLOAD__", json.dumps(payload))
                .replace("__VOICES__", str(voices))
                .replace("__TITLE__", display)
                .replace("__EYEBROW__", "ai-synth")
                .replace("__TAGLINE__", f"{voices}-voice polyphonic synthesizer")
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dsp")
    ap.add_argument("out")
    ap.add_argument("title", nargs="?")
    ap.add_argument("--voices", type=int, default=16)
    ap.add_argument("--skin", default="plain",
                    help="panel look: plain, or a name under assets/skins/")
    a = ap.parse_args()

    stats = build(a.dsp, a.out, voices=a.voices, title=a.title, skin=a.skin)
    total = stats["page"]
    print(f"{a.out}  [skin: {a.skin}]")
    for k, v in stats.items():
        print(f"  {k:12} {v/1024:9.1f} KiB")
    print(f"  {'cap':12} {16*1024:9.1f} KiB  ({100*total/(16*1024*1024):.2f}% used)")
