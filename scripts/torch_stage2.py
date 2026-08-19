"""Hybrid stage 2: the surrogate proposes, Faust decides.

CMA-ES reached loss 1.5564 with roughly 7500 PadRenderer renders because a
black-box search only ever sees a loss value. The surrogate in torch_synth gives
the gradient of the same loss for all 27 parameters from one forward and one
backward pass, so the question this driver answers is narrow and empirical: how
much of that search can the gradient replace, measured in true Faust renders.

Nothing the gradient produces is trusted. Every candidate goes back through
stage2.Objective (a real PadRenderer render against data/original.wav) and is
kept only if the TRUE loss improves on the best so far. That is the only reason
the delivered patch stays portable, and it is also the only honest way to score a
surrogate whose own loss is biased by about +0.06.

The freeze list is not a guess, it is what the two auditors measured:

  * dlyTime, chRate, lfoRate, detune, lfoAmt: the autograd value disagrees in sign
    with a finite difference of the SURROGATE'S OWN loss over the same interval, so
    these are either detached graphs (torch_fx.pingpong floors the delay length) or
    derivatives of an 18 s phase accumulation whose secant collapses by h = 3e-3.
    Together they carry 95 percent of the raw gradient norm and none of it transfers.
  * aD, dlyFb, revSize, revDamp: numerically dead, |dL/dlogit| between 2e-07 and
    3e-03. aD is dead for a correct reason, aS sits exactly at the box ceiling which
    makes en.adsr's decay segment flat.
  * kbdTrk, aR, dlyWet: the surrogate's loss is anti-correlated or uncorrelated with
    Faust's along these axes over a 5-point sweep (-0.06, +0.05, -0.46), and aR is
    where dawdreamer's release truncation diverges worst from en.adsr.

Those 12 stay with CMA-ES. The 15 that remain are the axes where the surrogate's
loss tracks Faust's with correlation 0.95 to 1.00.

Two gradient phases, because the auditors found the answer depends entirely on
where you stand:

  polish  from out/patch.json, where CMA-ES has already converged. The audit
          measured the step there as worse than chance (1.5695 against 1.5639 for a
          random direction of the same length), so this phase is expected to be
          rejected and is run to confirm it with our own renders, not to be tuned
          until it passes.
  coarse  from stage2.seeded_start, loss 3.6052, where cos(autograd, Faust FD) is
          +0.994 over the 23 non-phase parameters. This is where the port earns its
          keep, and --race quantifies it: CMA-ES is then run from the same seed and
          we count how many Faust renders it needs to reach the loss the gradient
          reached.

  uv run python scripts/torch_stage2.py --budget 600 --steps 18
"""

from __future__ import annotations

import argparse
import json
import time

import librosa
import numpy as np
import torch

from metrics import report
from stage2 import Objective, load_notes, run_cma, seeded_start
from synth import PARAM_INDEX, PARAMS, denorm, pad_normalized, write_render
from torch_common import Patch, SpectralLoss, default_n_samples
from torch_synth import TorchPad

SR = 44100
DUR = 17.904
POPSIZE = 16  # run_cma's popsize, needed to convert a render budget into generations

# measured untrustworthy, see the module docstring; handed to CMA-ES instead
FROZEN = (
    "detune", "kbdTrk", "aD", "aR", "lfoRate", "lfoAmt",
    "chRate", "dlyTime", "dlyFb", "dlyWet", "revSize", "revDamp",
)
FREE = tuple(p.name for p in PARAMS if p.name not in FROZEN)

# appended after the first fit; --pin-new holds them at 0 to reproduce the 27-parameter
# architecture, which is the only number comparable to the 1.5564 baseline
NEW = ("drive", "spread")


class Trust:
    """Faust as the only authority: it holds the best patch and counts the renders."""

    def __init__(self, obj: Objective) -> None:
        self.obj = obj
        self.trace: list[float] = []
        self.best_x: np.ndarray | None = None
        self.best_l = float("inf")

    @property
    def renders(self) -> int:
        return self.obj.calls

    def offer(self, x: np.ndarray, label: str) -> tuple[bool, float]:
        z = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        loss = float(self.obj(z))
        self.trace.append(loss)
        keep = loss < self.best_l
        if keep:
            self.best_x, self.best_l = z.copy(), loss
        print(f"  faust[{self.renders:4d}] {label:28s} {loss:.4f} "
              f"{'ACCEPT' if keep else 'reject'} (best {self.best_l:.4f})")
        return keep, loss


def gradient_phase(
    pad: TorchPad,
    lossfn: SpectralLoss,
    x0: np.ndarray,
    device: torch.device,
    lr: float,
    steps: int,
    check_every: int,
    trust: Trust,
    label: str,
    frozen: list[str],
    gamma: float = 0.9,
    clip: float = 5.0,
) -> tuple[np.ndarray, list[float], int]:
    """Adam on the Patch logits against the surrogate, trust-rendering checkpoints.

    Adam in logit space rather than plain SGD in [0,1]: the loss surface scales
    differ by four orders of magnitude across the 27 parameters (envAmt in Hz
    against revWet in a unit interval) and the sigmoid already keeps every step
    inside the box without a projection that would kill edge gradients.
    """
    patch = Patch(x0).to(device)
    patch.freeze(frozen)
    opt = torch.optim.Adam([patch.logits], lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    surrogate: list[float] = []
    gsteps = 0
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = lossfn(pad(patch))
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_([patch.logits], clip))
        opt.step()
        sched.step()
        gsteps += 1
        surrogate.append(float(loss.detach()))
        print(f"  {label} step {step:2d}/{steps}  surrogate {surrogate[-1]:.4f}  "
              f"|g| {gnorm:.3f}  lr {opt.param_groups[0]['lr']:.4f}")
        if step % check_every == 0 or step == steps:
            trust.offer(patch.to_numpy(), f"{label} step {step}")
    return patch.to_numpy(), surrogate, gsteps


def cma_phase(
    obj: Objective, trust: Trust, x0: np.ndarray, free: list[str],
    sigma: float, budget: int, seed: int, label: str,
) -> None:
    """A short CMA-ES pass over the parameters the gradient is not allowed to touch."""
    gens = budget // POPSIZE
    if gens < 2:
        print(f"  [{label}] skipped, only {budget} renders left in the budget")
        return
    x, _ = run_cma(obj, x0, free, gens, sigma, seed, label)
    trust.offer(x, f"{label} best")


def race_cma(x_seed: np.ndarray, target: float, notes, max_renders: int) -> tuple[int, float]:
    """How many Faust renders CMA-ES needs from the same seed to reach `target`.

    Its own Objective so the render count is separate from the hybrid budget. This
    is the comparison that decides whether the gradient bought anything: the
    gradient's cost is its trust renders plus its wall clock, CMA-ES's cost is this.
    """
    import cma

    obj = Objective(notes)
    es = cma.CMAEvolutionStrategy(
        x_seed.tolist(), 0.22,
        {"bounds": [0, 1], "popsize": POPSIZE, "seed": 100, "verbose": -9},
    )
    best = float("inf")
    while obj.calls < max_renders and not es.stop():
        sols = es.ask()
        vals = []
        for s in sols:
            vals.append(obj(np.asarray(s)))
            best = min(best, vals[-1])
            if best <= target:
                break
        if best <= target:
            break
        es.tell(sols, vals)
    return obj.calls, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="out/patch.json", help="patch json to start and to beat")
    ap.add_argument("--steps", type=int, default=18, help="Adam steps in the coarse phase")
    ap.add_argument("--lr", type=float, default=0.06, help="coarse Adam lr; polish uses lr/20..lr/2")
    ap.add_argument("--out", default="out/patch_torch.json")
    ap.add_argument("--budget", type=int, default=600, help="max true Faust renders")
    # CPU by default: the integrator measured forward+backward at 16.2 s on CPU
    # against 22.5 s on MPS, so MPS is a loss for this graph.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--race", type=int, default=320,
                    help="max renders for the CMA-ES-from-seed control; 0 disables it")
    ap.add_argument("--skip-polish", action="store_true")
    ap.add_argument("--skip-coarse", action="store_true")
    ap.add_argument("--pin-new", action="store_true",
                    help="hold drive and spread at 0, i.e. refit the original 27 parameters")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    notes = load_notes()
    obj = Objective(notes)
    trust = Trust(obj)
    pinned = list(NEW) if args.pin_new else []
    frozen = list(FROZEN) + pinned
    searchable = [p.name for p in PARAMS if p.name not in pinned]
    free = [n for n in searchable if n not in frozen]
    print(f"{len(notes)} notes, device {device}, {len(free)} free / {len(FROZEN)} frozen "
          f"params, pinned at 0: {pinned or 'none'}")

    # 1. honest baseline, computed here rather than read from patch.json
    x_init = pad_normalized(json.load(open(args.init))["normalized"])
    for name in pinned:
        x_init[PARAM_INDEX[name]] = 0.0
    _, baseline = trust.offer(x_init, "baseline " + args.init)
    print(f"baseline true faust loss {baseline:.4f}")

    n = default_n_samples()
    pad = TorchPad(notes, n, device)
    lossfn = SpectralLoss()
    log: dict[str, object] = {"baseline": baseline, "device": str(device), "init": args.init,
                              "frozen": frozen, "free": free, "pinned_at_zero": pinned}
    gsteps = 0

    # 2. polish attempt at the converged patch, three learning rates
    if not args.skip_polish:
        per = max(2, args.steps // 3)
        for lr in (args.lr / 20.0, args.lr / 6.0, args.lr / 2.0):
            if trust.renders + 4 > args.budget:
                break
            print(f"\n--- gradient polish from {args.init}, lr {lr:.4f} ---")
            _, surr, gs = gradient_phase(pad, lossfn, x_init, device, lr, per,
                                         max(1, per // 2), trust, f"polish lr{lr:.4f}", frozen)
            gsteps += gs
            log[f"polish_lr{lr:.4f}_surrogate"] = surr
        log["polish_best"] = trust.best_l
        if trust.best_l < baseline:
            print(f"gradient polish improved the converged patch: {baseline:.4f} -> {trust.best_l:.4f}")
        else:
            print(f"gradient polish did NOT beat the converged patch (best {trust.best_l:.4f})")

    # 3. coarse phase from the seeded start, where the audit measured cos +0.994
    if not args.skip_coarse and trust.renders + 6 <= args.budget:
        x_seed = seeded_start()
        _, seed_loss = trust.offer(x_seed, "seeded_start")
        mark = len(trust.trace)
        print(f"\n--- gradient coarse from seeded_start, lr {args.lr:.4f} ---")
        _, surr, gs = gradient_phase(pad, lossfn, x_seed, device, args.lr, args.steps,
                                     max(1, args.steps // 3), trust, "coarse", frozen)
        gsteps += gs
        log["coarse_surrogate"] = surr
        log["seed_faust_loss"] = seed_loss
        coarse_best = min(trust.trace[mark:])
        log["coarse_best_faust"] = coarse_best
        print(f"coarse: seed {seed_loss:.4f} -> {coarse_best:.4f} in {gs} gradient steps")

        # 3b. the control: what CMA-ES costs to reach the same loss from the same seed
        if args.race > 0:
            print(f"\n--- control: CMA-ES from the same seed, target {coarse_best:.4f} ---")
            r_calls, r_best = race_cma(x_seed, coarse_best, notes, args.race)
            log["race_renders"] = r_calls
            log["race_best"] = r_best
            hit = r_best <= coarse_best
            print(f"CMA-ES reached {r_best:.4f} after {r_calls} renders "
                  f"({'matched' if hit else 'did NOT match'} the gradient result)")

    # 4. hand the frozen and untrusted parameters to a short CMA-ES pass, warm started
    left = args.budget - trust.renders
    if trust.best_x is not None and left > 2 * POPSIZE:
        gradient_frozen = [n for n in FROZEN if n not in pinned]
        print(f"\n--- CMA-ES on the {len(gradient_frozen)} frozen params, {left} renders left ---")
        cma_phase(obj, trust, trust.best_x, gradient_frozen, 0.06, left // 2, 300, "frozen")
        left = args.budget - trust.renders
        if left > 2 * POPSIZE:
            print(f"\n--- CMA-ES on all {len(searchable)}, sigma 0.04, {left} renders left ---")
            cma_phase(obj, trust, trust.best_x, searchable, 0.04, left, 301, "full")

    # anything the objective saw is a true Faust loss, so its own best is admissible
    if obj.best[1] is not None and obj.best[0] < trust.best_l:
        trust.best_x, trust.best_l = np.clip(obj.best[1], 0.0, 1.0), obj.best[0]
        print(f"objective's own best is lower: {trust.best_l:.4f}")

    wall = time.time() - t0
    print(f"\nbest true faust loss {trust.best_l:.4f} (baseline {baseline:.4f}) "
          f"after {trust.renders} renders, {gsteps} gradient steps, {wall:.0f}s")

    np.save("out/torch_loss_history.npy", np.array(trust.trace, dtype=float))
    log["best_faust_loss"] = trust.best_l
    log["renders"] = trust.renders
    log["gradient_steps"] = gsteps
    log["wall_seconds"] = wall

    improved = trust.best_l < baseline
    if improved:
        aud = obj.render(trust.best_x)
        write_render("out/patch_torch_render.wav", aud)
        orig, _ = librosa.load("data/original.wav", sr=SR, mono=False)
        m = report(orig, aud)
        print("metrics:", {k: round(v, 4) for k, v in m.items()})
        with open(args.out, "w") as fh:
            json.dump({"loss": trust.best_l, "metrics": m, "params": denorm(trust.best_x),
                       "pinned": {}, "normalized": trust.best_x.tolist()}, fh, indent=2)
        log["metrics"] = m
        print(f"wrote {args.out} + out/patch_torch_render.wav")
    else:
        print("no candidate beat the baseline, out/patch.json stands and nothing was written")
    with open("out/torch_stage2_log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print("log: out/torch_stage2_log.json, trace: out/torch_loss_history.npy")


if __name__ == "__main__":
    main()
