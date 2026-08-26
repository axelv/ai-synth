"""Assemble one self-contained HTML page from a Faust DSP.

Spike 3. The delivery target is an Artifact, whose CSP blocks every external host, so
nothing may be fetched at runtime: the faustwasm runtime, both wasm modules and their
metadata all have to be inside the file. faust2wasm's own generated glue fetches its
assets by relative URL, which is exactly what cannot work here, so this bypasses it and
hands `FaustPolyDspGenerator.createNode` already-compiled modules instead.

The page template is a fragment starting at <title>, because the Artifact wrapper
supplies the document shell at publish time. Serving that fragment over HTTP instead
parses it in quirks mode, so by default the fragment is wrapped in a shell here; see
wrap_document. `--fragment` skips the wrapping, for the Artifact path.

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
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
FONTS = os.path.join(ASSETS, "fonts")
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


def wrap_document(fragment: str) -> str:
    """Wrap the template fragment in a document shell, for serving over HTTP.

    Without a doctype the browser falls into quirks mode, and a phone then lays the
    page out against a 980 px virtual viewport, so a skin's width media query never
    fires and the panel comes out too small to play. The viewport meta is the part
    that actually fixes that; the doctype keeps the CSS out of quirks mode as well.
    Only the <title> is hoisted into the head. Everything else the fragment opens
    with stays exactly where it sits: a <style> applies from the body just as well,
    and hoisting more would mean this file having to know the template's shape.
    """
    m = re.match(r"\s*(<title>.*?</title>)\s*", fragment, re.S)
    head_title, body = (m.group(1), fragment[m.end():]) if m else ("", fragment)
    return ('<!doctype html>\n<html lang="en">\n<head>\n'
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'{head_title}\n'
            '</head>\n<body>\n'
            f'{body}'
            '</body>\n</html>\n')


# The latin subsets of the two families the template asks for, as Google Fonts serves
# them. Archivo is a variable font, so one file answers every weight; IBM Plex Mono is
# static on Google Fonts, so each weight is its own file. Only weights the template's
# CSS actually asks for are here, and only latin: anything outside the range falls
# through to the next family in the stack instead of shipping glyphs nobody renders.
FACES = [
    ("Archivo", "100 900", "archivo-latin.woff2"),
    ("IBM Plex Mono", "400", "ibm-plex-mono-400-latin.woff2"),
    ("IBM Plex Mono", "500", "ibm-plex-mono-500-latin.woff2"),
    ("IBM Plex Mono", "600", "ibm-plex-mono-600-latin.woff2"),
]

# Faces a single skin asks for, kept off every other page. The juno panel's wordmark
# is the only case so far.
SKIN_FACES = {
    "juno": [("Michroma", "400", "michroma-latin.woff2")],
}

# Visible attribution for a skin that models a real machine's panel. Per-skin rather
# than a build flag, because a flag is a thing a call site forgets and the whole point
# is that the page states the homage where a reader can see it. A skin that models
# nothing needs no notice, and gets no empty element.
SKIN_NOTICES = {
    "juno": ("Panel layout is a visual homage to the Roland Juno-106, drawn from a "
             "public-domain photograph. An independent Faust patch: not affiliated "
             "with or endorsed by Roland Corporation. Roland and Juno are trademarks "
             "of their respective owners."),
}


def notice_html(skin: str) -> str:
    """The attribution paragraph, or nothing at all for a skin that claims nothing."""
    text = SKIN_NOTICES.get(skin)
    return f'<p class="notice">{text}</p>' if text else ""


LATIN = ("U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, "
         "U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, "
         "U+2212, U+2215, U+FEFF, U+FFFD")


def fonts_css(skin: str = "plain") -> str:
    """Every face the page uses, as data: URIs, so it fetches nothing at all.

    Two external requests used to hide here: a fonts.googleapis.com stylesheet in the
    template, and an @import for Michroma at the top of the juno skin's own <style>.
    Both broke the page offline and over file://, which are delivery modes the skill
    documents, and both put a third-party request on a file that otherwise touches no
    network. Costs about 86 KiB of base64, plus 15 KiB on the juno skin.

    The OFL requires the copyright notice and the licence to travel with every copy of
    the font software, so they are emitted here rather than kept only beside the woff2.
    """
    with open(os.path.join(FONTS, "OFL.txt")) as fh:
        licence = fh.read().strip()
    out = ["/*", licence, "*/"]
    for family, weight, fn in FACES + SKIN_FACES.get(skin, []):
        with open(os.path.join(FONTS, fn), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        out.append(
            "@font-face {"
            f" font-family: '{family}';"
            " font-style: normal;"
            f" font-weight: {weight};"
            " font-display: swap;"
            f" src: url(data:font/woff2;base64,{b64}) format('woff2');"
            f" unicode-range: {LATIN};"
            " }")
    return "\n".join(out)


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
          title: str | None = None, skin: str = "plain",
          fragment: bool = False) -> dict[str, int]:
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
        # The skin goes in LAST, so that nothing rescans it. Every replace runs over
        # the whole document, including whatever the previous ones inserted, so a
        # marker appearing inside a skin's CSS or JS would be expanded if the skin
        # were substituted first. A skin is a separate file that a person edits
        # without thinking about this file's markers, which is exactly the case worth
        # protecting. The same reasoning does not save the runtime bundle, which has
        # to go in before the markers it might contain are searched for; it has never
        # contained one.
        html = (html
                .replace("__FAUSTWASM__", runtime)
                .replace("__FONTS__", fonts_css(skin))
                .replace("__NOTICE__", notice_html(skin))
                .replace("__PAYLOAD__", json.dumps(payload))
                .replace("__VOICES__", str(voices))
                .replace("__TITLE__", display)
                .replace("__EYEBROW__", "ai-synth")
                .replace("__TAGLINE__", f"{voices}-voice polyphonic synthesizer")
                .replace("__BLURB__", BLURB)
                .replace("__SKIN_CSS__", skin_css)
                .replace("__SKIN_JS__", skin_js)
                .replace("__SKIN__", skin))
        # After the markers, never before: the shell must not be rescanned for them.
        if not fragment:
            html = wrap_document(html)

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
    ap.add_argument("--fragment", action="store_true",
                    help="emit the bare fragment, for publishing as an Artifact whose "
                         "wrapper supplies its own document shell")
    a = ap.parse_args()

    stats = build(a.dsp, a.out, voices=a.voices, title=a.title, skin=a.skin,
                  fragment=a.fragment)
    total = stats["page"]
    mode = "fragment" if a.fragment else "document"
    print(f"{a.out}  [skin: {a.skin}, {mode}]")
    for k, v in stats.items():
        print(f"  {k:12} {v/1024:9.1f} KiB")
    print(f"  {'cap':12} {16*1024:9.1f} KiB  ({100*total/(16*1024*1024):.2f}% used)")
