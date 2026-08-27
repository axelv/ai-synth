"""Build the public site: one landing page, one page per instrument.

Stdlib plus node, deliberately. `build_page.py` needs no Python packages, so the whole
site builds from a clean checkout with `npm install` and nothing else; pulling in the
project's own dependencies would put torch on the critical path to publishing a
landing page.

    python site/build.py [--out site/dist] [--patch SLUG ...]

The landing page's job is narrow: get sound out of the first click, then make the one
next action obvious. Everything it says is either in `catalogue.json` (which
instruments) or `config.json` (where a request goes). Neither is code.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKILL = os.path.join(ROOT, ".claude", "skills", "faust-synth")
sys.path.insert(0, os.path.join(SKILL, "scripts"))

import build_page  # noqa: E402  the path above is what makes this importable

TEMPLATE = os.path.join(HERE, "templates", "index.html")
PAGE_TEMPLATE = os.path.join(SKILL, "assets", "page_template.html")


def read_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def tokens() -> str:
    """The palette, taken from the instrument template rather than copied beside it.

    A landing page whose colours drift from the pages it links to reads as two
    different products, and a copy drifts the first time only one of them is edited.
    """
    with open(PAGE_TEMPLATE) as fh:
        src = fh.read()
    m = re.search(r"/\* --- tokens:.*?\*/\n(.*?)/\* --- end tokens --- \*/", src, re.S)
    if not m:
        raise RuntimeError(f"token markers not found in {PAGE_TEMPLATE}")
    return m.group(1).rstrip()


# Inline, because a page that fetches nothing at runtime should not make its one
# exception a request the browser issues on its own. Without it every page logs a 404
# for /favicon.ico, which is noise in exactly the console someone debugs audio in.
FAVICON = (
    '<link rel="icon" href="data:image/svg+xml,'
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='6' fill='%2311191b'/%3E"
    "%3Cpath d='M4 16c3-9 6-9 9 0s6 9 9 0 4-5 6-2' fill='none' stroke='%2345b9c1' "
    "stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E\">"
)


def head(cfg: dict, title: str, description: str, path: str) -> str:
    """The head content shared by every page: what a shared link shows, plus counting.

    No og:image yet. Every platform that renders one wants a raster, and this build is
    stdlib-only by design, so there is nothing here that can draw one. A link therefore
    previews as text, which costs reach and is worth fixing with a committed PNG.
    """
    url = cfg["site_url"].rstrip("/") + path
    esc = lambda s: html.escape(s, quote=True)
    tags = [
        f'<meta name="description" content="{esc(description)}">',
        f'<link rel="canonical" href="{esc(url)}">',
        '<meta property="og:type" content="website">',
        f'<meta property="og:title" content="{esc(title)}">',
        f'<meta property="og:description" content="{esc(description)}">',
        f'<meta property="og:url" content="{esc(url)}">',
        '<meta name="twitter:card" content="summary">',
        FAVICON,
    ]
    if cfg.get("analytics"):
        tags.append(cfg["analytics"])
    return "\n".join(tags)


def inject_head(page: str, extra: str) -> str:
    """Put the site's head content into a page the skill built without knowing about it.

    `build_page.py` emits pages that stand alone anywhere, including offline and over
    file://, so social metadata and a counter are not its business. They are this
    build's, and this is the seam.
    """
    if "</head>" not in page:
        raise RuntimeError("page has no </head>: was it built with --fragment?")
    return page.replace("</head>", extra + "\n</head>", 1)


def card(p: dict) -> str:
    esc = lambda s: html.escape(s, quote=False)
    caveat = (f'\n      <span class="caveat">{esc(p["caveat"])}</span>'
              if p.get("caveat") else "")
    return (f'\n    <a class="card" href="p/{p["slug"]}.html">'
            f'\n      <span class="fam">{esc(p["family"])}</span>'
            f'\n      <h3>{esc(p["name"])}</h3>'
            f'\n      <p>{esc(p["blurb"])}</p>{caveat}'
            f'\n      <span class="cta">Play it &rarr;</span>'
            f'\n    </a>')


def request_block(cfg: dict) -> str:
    """The form, and the GitHub fallback under it.

    Both, because they fail on opposite people: a form endpoint needs an account this
    project's author has to create, and an issue needs one the visitor has to have.
    """
    issue = html.escape(cfg["request_issue_url"], quote=True)
    form_url = cfg.get("request_form_url", "").strip()
    if not form_url:
        print("  note: request_form_url is empty, so the GitHub issue is the only "
              "path. Anyone without a GitHub account cannot ask for a sound.",
              file=sys.stderr)
        return (f'<a class="btn" href="{issue}">Request a sound on GitHub</a>\n'
                '    <p class="note">A GitHub account is needed for this one. A form '
                'that needs nothing is coming.</p>')
    return (
        f'<form method="POST" action="{html.escape(form_url, quote=True)}">\n'
        '      <div>\n'
        '        <label for="sound">The sound you want</label>\n'
        '        <textarea id="sound" name="sound" rows="4" required\n'
        '          placeholder="A warm 80s pad, the kind that sits under a chorus and '
        'swells. Or a link to a clip."></textarea>\n'
        '      </div>\n'
        '      <div>\n'
        '        <label for="email">Where to send the link</label>\n'
        '        <input id="email" name="email" type="email" required\n'
        '          placeholder="you@example.com">\n'
        '      </div>\n'
        '      <button class="btn" type="submit">Send it</button>\n'
        '    </form>\n'
        f'    <p class="note">Rather use GitHub? <a href="{issue}">Open it as an issue</a> '
        'instead, where the request and what came back are both public.</p>')


def dev_block(cfg: dict) -> str:
    repo = cfg["repo"]
    if cfg.get("plugin_ready"):
        cmd = (f"/plugin marketplace add {repo}\n"
               "/plugin install faust-synth\n\n"
               '&gt; build me a warm analog pad')
    else:
        cmd = (f"git clone https://github.com/{repo}\n"
               "cd ai-synth &amp;&amp; npm install\n"
               "claude\n\n"
               '&gt; build me a warm analog pad')
    return (f'<pre>{cmd}</pre>\n'
            f'    <p class="note">The skill is <a href="https://github.com/{repo}/blob/'
            'main/.claude/skills/faust-synth/SKILL.md">.claude/skills/faust-synth</a>. '
            'It also writes down what it measured, which is the half that took the '
            'work.</p>')


LANDING_TITLE = "ai-synth: describe a sound, get a synth you can play"
LANDING_DESC = ("Playable synth patches built from a plain-language description. Real "
                "Faust DSP with knobs on it, running in your browser: no install, no "
                "account, and a MIDI keyboard works.")


def build_landing(cfg: dict, cat: dict, out: str) -> int:
    with open(TEMPLATE) as fh:
        tpl = fh.read()
    patches = cat["patches"]
    featured = next(p for p in patches if p["slug"] == cat["featured"])
    body = (tpl
            .replace("__FONTS__", build_page.fonts_css("plain"))
            .replace("__TOKENS__", tokens())
            .replace("__HERO_SLUG__", featured["slug"])
            .replace("__HERO_NAME__", html.escape(featured["name"], quote=True))
            .replace("__COUNT__", str(len(patches)))
            .replace("__CARDS__", "".join(card(p) for p in patches))
            .replace("__REQUEST__", request_block(cfg))
            .replace("__DEV__", dev_block(cfg))
            .replace("__REPO__", cfg["repo"]))
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{html.escape(LANDING_TITLE)}</title>\n'
        f'{head(cfg, LANDING_TITLE, LANDING_DESC, "/")}\n'
        f'</head>\n<body>\n{body}</body>\n</html>\n')
    path = os.path.join(out, "index.html")
    with open(path, "w") as fh:
        fh.write(page)
    return len(page)


def build_patch(cfg: dict, p: dict, out: str) -> int:
    dst = os.path.join(out, "p", p["slug"] + ".html")
    build_page.build(
        os.path.join(ROOT, p["dsp"]), dst,
        voices=p.get("voices", 16), title=p["name"],
        skin=p.get("skin", "plain"), demo=p.get("demo", "chord"),
    )
    with open(dst) as fh:
        page = fh.read()
    desc = f'{p["name"]}: {p["blurb"]}'
    title = f'{p["name"]} — a playable synth patch'
    page = inject_head(page, head(cfg, title, desc, f'/p/{p["slug"]}.html'))
    with open(dst, "w") as fh:
        fh.write(page)
    return len(page)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(HERE, "dist"))
    ap.add_argument("--patch", action="append", default=None,
                    help="build only these slugs, for a quick loop on one instrument")
    a = ap.parse_args()

    cfg = read_json(os.path.join(HERE, "config.json"))
    cat = read_json(os.path.join(HERE, "catalogue.json"))
    patches = cat["patches"]
    if a.patch:
        known = {p["slug"] for p in patches}
        unknown = sorted(set(a.patch) - known)
        if unknown:
            raise SystemExit(f"no such patch: {', '.join(unknown)}")
        patches = [p for p in patches if p["slug"] in a.patch]

    out = os.path.abspath(a.out)
    if not a.patch and os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "p"), exist_ok=True)
    # Pages runs Jekyll over anything without it, which eats directories starting with
    # an underscore and rewrites nothing here for the better.
    open(os.path.join(out, ".nojekyll"), "w").close()

    total = 0
    for p in patches:
        size = build_patch(cfg, p, out)
        total += size
        print(f"  p/{p['slug']}.html {size/1024:9.1f} KiB")
    if not a.patch:
        size = build_landing(cfg, cat, out)
        total += size
        print(f"  index.html      {size/1024:9.1f} KiB")
    print(f"{out}  {len(patches)} instruments, {total/1024:.1f} KiB total")


if __name__ == "__main__":
    main()
