# Assemble the spec

Type: task
Status: resolved
Blocked by: 03, 04, 05, 06, 07, 08, 09, 10, 12, 13, 14

## Question

The destination. Not a decision: the write-up of every decision this map made, as one
document a person or an agent can build from.

It has to contain:

- Every screen and state of the flow, from arrival to playing with macros, with the
  transitions between them.
- The contract the fitting side must satisfy: what the app needs back, in what order,
  within what time.
- The gaps found along the way, named as gaps rather than quietly resolved. This was the
  stated reason for the whole exercise.
- The exits, stubbed: export, returning to an old patch, everything on the map's Out of
  scope list, with a line each on where they attach.

Link the prototypes from the tickets that produced them rather than redrawing them.

## Answer

Written to **`SPEC.md`** at the repo root, beside `VISION.md` and `CLAUDE.md`, because that
is where someone building from it will look and where agents working in this repo will find
it. 606 lines, eleven sections.

Structure, and why it is not one screen-by-screen walkthrough: the flow has two surfaces and
a rail that spans both, so a linear walkthrough would describe the rail three times and the
play surface once for each of its six states. Instead the surfaces are specified once each
with their state tables, and the transitions are named at the point they fire.

- §1 the seven steps and the cost disclosure
- §2 the setup surface, S1 to S3, with the Start transition enumerated
- §3 the rail: per-step display, eight states, and the tab-title channel
- §4 the play surface: comparison frame, ear check, keyboard, evidenced span, device line,
  six states, the swap
- §5 the macros
- §6 when it goes wrong, split into a fit that came back wrong and machinery that failed
- §7 persistence, the URL and the exits
- §8 nine gaps
- §9 the contract for the fitting side, as a sequence-and-timing table plus requirements
- §10 out of scope, with where each item would attach
- §11 which ticket argued what

Prototypes are linked rather than redrawn.

**The gaps came out at nine**, up from the four the map was carrying. The five that had not
been collected anywhere: the stand-in being playable but not adjustable, the rail saying "Play
it" while you are playing, the cost disclosure resting on a grey subtitle, the visual language,
and how many architectures exist. Four of those were surfaced by the last three tickets rather
than by the assembly, which is the map working as intended.

**One thing the assembly itself found.** Two decisions read as contradictory when placed a page
apart: "there is no refit" in §6.1 and "a dead fit retries once" in §6.2. The distinction is
real, no-refit applies to a fit that completed and has therefore already explored its reachable
set, but it only became visible when both were written down in one document. The spec states it
explicitly at the point of collision.
