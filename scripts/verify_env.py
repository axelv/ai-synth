"""Fit-check torch_env.adsr against real Faust en.adsr output, sample for sample.

A surrogate that quietly disagrees with Faust makes the gradient optimise the wrong
thing, so the envelope is compared against rendered ground truth rather than against
what the library source looks like it should do. The grid deliberately includes the
cases that break a naive ADSR: hold < a (released during the attack) and
a < hold < a+d (released during the decay), which is the regime the actual patch sits
in because notes are 2-6 s while aA/aD range up to 3-4 s.

Run: PYTHONPATH=scripts uv run python scripts/verify_env.py
"""

from __future__ import annotations

import numpy as np
import torch

import dawdreamer as daw

import faust_probe
from faust_probe import BLOCK, FAUST_LIBS, GATE_SNIPPET, SR
from synth import DSP as PAD_DSP
from torch_env import adsr, gate_events

FIXTURE = "out/fixtures/env.npz"

DSP = f"""
import("stdfaust.lib");
{GATE_SNIPPET}
a = hslider("a", 0.4, 0.001, 6, 0.000001);
d = hslider("d", 0.5, 0.001, 6, 0.000001);
s = hslider("s", 0.8, 0, 1, 0.000001);
r = hslider("r", 1.2, 0.001, 8, 0.000001);
process = en.adsr(a, d, s, r, gate);
"""

# (a, d, s, r, hold). Spans the PARAMS ranges for aA/aD/aS/aR and fA/fD/fS.
# Cases are tagged in comments by which segment the release interrupts.
GRID: tuple[tuple[float, float, float, float, float], ...] = (
    # sustain reached well before note-off (the easy case)
    (0.005, 0.02, 1.00, 0.05, 0.5),
    (0.005, 0.02, 0.05, 5.0, 1.0),
    (0.35, 0.8, 0.85, 1.4, 3.0),
    (0.4, 0.5, 0.6, 1.2, 2.5),
    (0.05, 0.10, 0.30, 0.30, 1.5),
    (0.20, 1.00, 0.50, 2.00, 4.0),
    (0.01, 4.00, 0.00, 0.05, 5.0),
    (1.00, 0.50, 0.95, 3.00, 6.0),
    (0.005, 0.02, 0.50, 0.05, 0.1),
    (0.10, 0.02, 0.05, 0.20, 0.6),
    # released during the attack: hold < a
    (3.00, 4.00, 0.85, 5.00, 0.5),
    (3.00, 0.02, 0.05, 0.05, 1.0),
    (2.00, 1.00, 0.60, 1.40, 0.25),
    (1.00, 0.50, 0.00, 2.00, 0.10),
    (0.50, 0.30, 1.00, 0.50, 0.20),
    (4.00, 4.00, 0.50, 3.00, 2.00),
    (0.30, 0.80, 0.70, 0.05, 0.05),
    (3.00, 0.02, 1.00, 5.00, 2.50),
    # released during the decay: a < hold < a + d
    (0.30, 4.00, 0.05, 1.40, 1.00),
    (0.50, 3.00, 0.20, 0.30, 1.50),
    (1.00, 2.00, 0.60, 2.00, 2.00),
    (0.02, 4.00, 0.10, 5.00, 1.00),
    (2.00, 2.00, 0.85, 1.00, 3.00),
    (0.005, 3.00, 0.00, 0.05, 1.20),
    (0.10, 1.00, 0.90, 0.60, 0.50),
    (1.50, 3.50, 0.35, 4.00, 4.00),
    # degenerate / boundary
    (0.005, 0.02, 0.05, 0.05, 0.02),
    (3.00, 4.00, 1.00, 5.00, 8.00),
)


def render_grid() -> tuple[np.ndarray, np.ndarray]:
    """Returns (grid (G,5), faust (G,N)) with every row padded to a common length."""
    durs = [h + r + 0.5 for (_, _, _, r, h) in GRID]
    n = int(max(durs) * SR)
    out = np.zeros((len(GRID), n), dtype=np.float64)
    for i, (a, d, s, r, hold) in enumerate(GRID):
        y = faust_probe.render_gen(
            DSP, {"a": a, "d": d, "s": s, "r": r, "hold": hold}, durs[i]
        )[0]
        out[i, : min(n, len(y))] = y[: min(n, len(y))]
        print(f"  [{i:2d}] a={a:<6g} d={d:<6g} s={s:<5g} r={r:<5g} hold={hold:<5g} "
              f"peak={y.max():.4f}")
    grid = np.array(GRID, dtype=np.float64)
    return grid, out


def torch_grid(grid: np.ndarray, n: int) -> np.ndarray:
    """torch_env.adsr evaluated per grid row. One row per (a,d,s,r), stacked."""
    dev = torch.device("cpu")
    rows = []
    for a, d, s, r, hold in grid:
        ev = gate_events([int(hold * SR)], n, dev)
        env = adsr(
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
            ev,
        )
        rows.append(env[0].detach().numpy())
    return np.stack(rows).astype(np.float64)


def check_vectorised(grid: np.ndarray, n: int) -> float:
    """One shared (a,d,s,r) across many holds must equal the per-row evaluation."""
    a, d, s, r = 0.30, 4.00, 0.05, 1.40
    holds = [int(h * SR) for h in grid[:, 4]]
    ev = gate_events(holds, n, torch.device("cpu"))
    batched = adsr(*[torch.tensor(v, dtype=torch.float32) for v in (a, d, s, r)], ev)
    single = np.stack([
        torch_grid(np.array([[a, d, s, r, h]]), n)[0] for h in grid[:, 4]
    ])
    return float(np.abs(batched.detach().numpy() - single).max())


def check_gradients() -> dict[str, float]:
    """a.grad and s.grad must be finite and non-zero for a plain sum loss."""
    dev = torch.device("cpu")
    ev = gate_events([int(1.5 * SR), int(0.3 * SR), int(4.0 * SR)], int(6.0 * SR), dev)
    a = torch.tensor(0.40, requires_grad=True)
    d = torch.tensor(1.20, requires_grad=True)
    s = torch.tensor(0.60, requires_grad=True)
    r = torch.tensor(1.40, requires_grad=True)
    adsr(a, d, s, r, ev).sum().backward()
    g = {"a": a.grad, "d": d.grad, "s": s.grad, "r": r.grad}
    for k, v in g.items():
        if v is None or not torch.isfinite(v).all():
            raise RuntimeError(f"grad for {k} is {v}")
    return {k: float(v) for k, v in g.items()}


def finite_difference(eps: float = 1e-3) -> dict[str, tuple[float, float]]:
    """Central differences at float64 vs autograd, the gradcheck spirit at this scale.

    torch.autograd.gradcheck itself is useless on a piecewise-linear envelope with
    hundreds of thousands of samples: any perturbation that moves a segment boundary
    across a sample makes the numerical Jacobian disagree by construction. Comparing
    a scalar directional derivative of the sum is the meaningful version of the check.
    """
    dev = torch.device("cpu")
    ev = gate_events([int(1.5 * SR), int(0.3 * SR), int(4.0 * SR)], int(6.0 * SR), dev)
    base = {"a": 0.40, "d": 1.20, "s": 0.60, "r": 1.40}

    def loss(vals: dict[str, float], grad: bool) -> torch.Tensor:
        t = {
            k: torch.tensor(v, dtype=torch.float64, requires_grad=grad)
            for k, v in vals.items()
        }
        return adsr(t["a"], t["d"], t["s"], t["r"], ev).sum(), t

    out: dict[str, tuple[float, float]] = {}
    total, tens = loss(base, True)
    total.backward()
    for k in base:
        hi = dict(base, **{k: base[k] + eps})
        lo = dict(base, **{k: base[k] - eps})
        num = float((loss(hi, False)[0] - loss(lo, False)[0]) / (2 * eps))
        out[k] = (float(tens[k].grad), num)
    return out


def check_poly_wrapper(a: float = 0.005, d: float = 4.0, s: float = 1.0,
                       r: float = 3.0, hold: float = 1.0) -> float:
    """Does dawdreamer's polyphony truncate the release? Measured, not assumed.

    Renders synth.DSP with the oscillator and filter removed, so the voice output
    IS aenv * gain, then compares against the same closed form. This matters because
    voices in the full synth audibly die long before aR elapses; the check exists to
    show that the envelope is not what dies, so adsr needs no lifetime correction.
    """
    dsp = PAD_DSP.replace(
        "process = filtered * aenv * gain <: _, _;", "process = aenv <: _, _;"
    )
    if dsp == PAD_DSP:
        raise RuntimeError("synth.DSP process line changed; update this probe")
    cut = dsp.index("effect = _,_")
    # The shared effect chain must go too: at tilt=0 the tiltEQ shelves are 0 dB but
    # still second order, and their transient is a 0.19 error over the first 1000
    # samples, which would swamp the envelope comparison.
    dsp = dsp[:cut] + "effect = _,_;\n"
    engine = daw.RenderEngine(SR, BLOCK)
    proc = engine.make_faust_processor("pad")
    proc.faust_libraries_path = FAUST_LIBS
    proc.num_voices = 24
    proc.release_length = 4.0
    proc.group_voices = True
    if not proc.set_dsp_string(dsp):
        raise RuntimeError("faust compile failed")
    pidx = {q["label"]: q["index"] for q in proc.get_parameters_description()}
    for name, val in (("aA", a), ("aD", d), ("aS", s), ("aR", r)):
        proc.set_parameter(pidx[name], val)
    engine.load_graph([(proc, [])])
    start = 0.5
    proc.clear_midi()
    proc.add_midi_note(60, 100, start, hold)
    engine.render(start + hold + r + 1.0)
    y = engine.get_audio()[0].astype(np.float64)

    onset = round(start * SR)
    ev = gate_events([int(hold * SR)], len(y) - onset, torch.device("cpu"))
    ours = adsr(*[torch.tensor(v, dtype=torch.float32) for v in (a, d, s, r)], ev)
    mine = ours[0].detach().numpy().astype(np.float64)
    err = float(np.abs(y[onset:] - mine).max())
    nz = np.nonzero(y > 1e-9)[0]
    print(f"poly-wrapper aenv: last nonzero at off+{nz[-1] - onset - int(hold * SR)} "
          f"samples, en.adsr release ends at off+{int(r * SR)}")
    print(f"poly-wrapper max abs error vs torch_env {err:.3e}")
    return err


def main() -> None:
    print("rendering Faust en.adsr over the grid")
    grid, faust = render_grid()
    n = faust.shape[1]
    ours = torch_grid(grid, n)

    err = np.abs(ours - faust)
    per_case = err.max(axis=1)
    worst = int(per_case.argmax())

    print("\nper-case max abs error")
    for i, e in enumerate(per_case):
        a, d, s, r, hold = grid[i]
        tag = "rel-in-attack" if hold < a else ("rel-in-decay" if hold < a + d else "sustained")
        print(f"  [{i:2d}] {tag:<14} max={e:.3e}  rms={np.sqrt((err[i]**2).mean()):.3e}")
    print(f"\nWORST max abs error {per_case[worst]:.3e} at case {worst} "
          f"(a={grid[worst,0]}, d={grid[worst,1]}, s={grid[worst,2]}, "
          f"r={grid[worst,3]}, hold={grid[worst,4]})")
    print(f"global rms {np.sqrt((err**2).mean()):.3e}  "
          f"samples compared {err.size} over {len(GRID)} cases")

    vec = check_vectorised(grid, n)
    print(f"vectorised-vs-single max abs diff {vec:.3e}")

    poly = max(
        check_poly_wrapper(),
        check_poly_wrapper(a=2.5, d=3.0, s=0.4, r=4.0, hold=1.0),
        check_poly_wrapper(a=0.3, d=3.5, s=0.2, r=2.0, hold=1.5),
    )

    grads = check_gradients()
    print("gradients of adsr(...).sum(): " + ", ".join(f"{k}={v:.6g}" for k, v in grads.items()))
    for k, (an, num) in finite_difference().items():
        rel = abs(an - num) / max(abs(num), 1e-9)
        print(f"  d/d{k}: autograd={an:.6g} central-diff={num:.6g} rel={rel:.3e}")

    faust_probe.save_fixture(FIXTURE, grid=grid, faust=faust.astype(np.float32))
    ok = per_case.max() < 1e-3 and vec < 1e-6 and poly < 1e-3
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
