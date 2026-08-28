# Song reconstruction, design exploration

Three directions for the page that shows what a reconstruction found: which synths a
track is built from, and how each one is configured. Worked against Take On Me as the
example, broken down as two Juno-60 layers and three DX7 layers.

Published canvas: https://claude.ai/code/artifact/dfcced5f-ea65-401a-a274-0a44957460d3

| artboard | direction |
|---|---|
| `Main.dc.html` | Option A, patch rack. Layer rail plus the selected layer's panel, then a placement map. A Juno layer prints as faders; a DX7 layer prints its algorithm, because that is where its timbre lives |
| `Exploded.dc.html` | Option B, exploded stack. Five sheets low to high, each carrying its spectral fingerprint. Shows overlap, hides configuration |
| `Placement.dc.html` | Option C, placement chart. Band, level, stereo width and register played. No hardware at all |
| `DX7.dc.html` | Drill-down. One FM patch in full: operator graph, envelopes, per-operator parameters |
| `canvas.json` | layout, sticky notes, launch view |

## Two decisions worth not re-litigating

**The look is not invented.** Every hex, and the section-band-plus-hairline grammar, is
lifted from `.claude/skills/faust-synth/assets/skins/juno.html`, so a page built from
this design and a page built by `build_page.py` read as one product. Fonts are Michroma,
Archivo and IBM Plex Mono, the same three the skin embeds.

**Colour carries the instrument, never the layer.** Juno-60 `#4a86bd` (the skin's own
blue) and DX7 `#c98500`. Two hues rather than five: it passes the categorical checks in
the `dataviz` skill on all pairs against the dark panel surface, and it puts "three DX7
and two Juno" in front of the reader first. Layer identity is always the printed lane
label, so nothing rests on colour alone. Adding a third instrument means re-running
`dataviz/scripts/validate_palette.js` rather than picking a hue by eye.

## Traps found by measuring, not by reading

- **A `data:` URI inside a `style` attribute is stripped by the canvas runtime.** It
  survives as far as `url("data:image/svg+xml")` and loses the payload, so the element
  renders empty with no error. Every graphic here is therefore real DOM: the operator
  graphs are positioned boxes and hairline divs, the envelopes are `clip-path` polygons.
- **`left: X%` on an absolutely positioned child resolves against its ancestor's
  padding box.** A `padding-left` gutter does not move it, and the octave ruler under
  the keyboard silently disagreed with the keyboard until the offset moved into `left`.
- **The layer data is a reconstruction candidate, not fact.** Layer attribution carries
  a confidence badge and the panel values are illustrative. If the real attribution is
  known it belongs here.

## Rebuilding

The tracked `.dc.html` files are the source; the seeded page is generated and ignored.
Reseed with the `design` skill's helper, from this directory:

```
node "<design skill base>/seed-canvas.mjs" \
  --template "<design skill base>/payload.template.html" \
  --out take-on-me-breakdown.html --title "Take On Me Breakdown" \
  --artboard Main.dc.html --artboard Exploded.dc.html \
  --artboard Placement.dc.html --artboard DX7.dc.html \
  --canvas canvas.json
```
