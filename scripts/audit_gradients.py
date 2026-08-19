"""Audit whether the surrogate's GRADIENT is usable, not whether it sounds close.

A surrogate that renders something similar but whose derivative points the wrong way
is worse than no surrogate: a gradient step walks confidently downhill in the wrong
direction and every render spent on it is wasted. So the question here is narrow,
and it is answered against the only authority there is, a real PadRenderer render.

Three gradient vectors are measured at the same point, because two of them are
needed to tell a broken graph apart from a genuinely bad loss surface:

    g_auto   autograd through the surrogate, chain-ruled into normalized [0,1] space
    g_torch  central finite difference of the SURROGATE loss, same intervals
    g_faust  central finite difference of the TRUE stage2.Objective loss

    cos(g_auto, g_torch)  is the graph healthy? a detached or straight-through op
                          shows up here and nowhere else, because both sides are
                          the same function
    cos(g_torch, g_faust) is the PORT faithful? immune to autograd pathologies
    cos(g_auto, g_faust)  the headline: is the thing an optimiser would actually
                          use a descent direction for the real renderer

Finite differences of the Faust loss are taken at three step sizes (0.01, 0.03,
0.08 in normalized space) rather than one. dawdreamer renders are bit-deterministic
for identical parameters, so the spread across h is not measurement noise, it is
real roughness of the loss surface at the scale a gradient step would use, and it
is the only way to say whether a disagreeing parameter has a broken graph (the
Faust reference is stable and the surrogate still differs) or a bad surface (the
Faust reference itself changes sign with h).

Cosines are also reported restricted to the interior of the box. At out/patch.json
eight of the 27 parameters sit within 0.04 of a bound, and CMA-ES clips to [0,1],
so that point is a CONSTRAINED optimum: components that push out through a face
cannot be taken by any feasible step and inflating the cosine with them, in either
direction, measures nothing an optimiser can use.

Every claim about a step is checked against a random control of the same length.
At a converged point every direction makes the loss worse, so "the loss went up"
is not by itself evidence of a bad gradient; what matters is whether -g does
better or worse than chance.

    uv run python scripts/audit_gradients.py noise     # render determinism, FD floor
    uv run python scripts/audit_gradients.py faust     # 27 params x 3 step sizes
    uv run python scripts/audit_gradients.py torchfd   # surrogate FD, graph check
    uv run python scripts/audit_gradients.py auto      # autograd at patch and seed
    uv run python scripts/audit_gradients.py seed      # faust FD at the seeded start
    uv run python scripts/audit_gradients.py micro     # phase params at tiny h
    uv run python scripts/audit_gradients.py lines     # steps vs random controls
    uv run python scripts/audit_gradients.py report    # assemble the table

Scalars go to stdout, the per-parameter table to out/gradient_audit.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from stage2 import Objective, load_notes, seeded_start
from synth import PARAMS, denorm
from torch_common import Patch, SpectralLoss, default_n_samples, denorm_torch, get_device
from torch_synth import TorchPad

OUT = "out/gradient_audit.json"
PATCH = "out/patch.json"
NAMES = [p.name for p in PARAMS]
H_GRID = (0.01, 0.03, 0.08)
EDGE = 0.04  # a parameter this close to a bound is treated as on the box face
FREEZE_CANDIDATES = ("dlyTime", "chRate", "lfoRate")
# the four whose autograd value disagrees in sign with a finite difference of the
# surrogate's OWN loss, measured in step_torchfd: everything that enters as an
# oscillator or delay-line phase over 18 s of accumulation
PHASE_PARAMS = ("detune", "lfoRate", "chRate", "dlyTime")


def record(**vals) -> None:
    have = json.load(open(OUT)) if os.path.exists(OUT) else {}
    have.update(vals)
    with open(OUT, "w") as fh:
        json.dump(have, fh, indent=2, sort_keys=True)


def stored(key: str):
    return json.load(open(OUT))[key]


def point(name: str) -> np.ndarray:
    if name == "patch":
        return np.asarray(json.load(open(PATCH))["normalized"], dtype=float)
    if name == "seed":
        return seeded_start()
    return np.asarray(json.load(open(name))["normalized"], dtype=float)


def fd_interval(z: np.ndarray, i: int, h: float) -> tuple[np.ndarray, np.ndarray, float]:
    """The same clipped interval on both sides, so torch and Faust differences match.

    synth.denorm clips to [0,1], so at a parameter sitting on a box face the central
    difference degenerates to a one-sided one. Returning the true width keeps the
    quotient a valid divided difference instead of silently halving it.
    """
    lo, hi = z.copy(), z.copy()
    hi[i] = min(1.0, z[i] + h)
    lo[i] = max(0.0, z[i] - h)
    return lo, hi, float(hi[i] - lo[i])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


def faust_objective() -> Objective:
    return Objective(load_notes())


# ---------------------------------------------------------------- noise floor


def step_noise() -> None:
    """Is the Faust loss deterministic, and how small can h be before it is noise?"""
    obj = faust_objective()
    z = point("patch")
    a = obj.render(z)
    b = obj.render(z)
    losses = [obj(z) for _ in range(3)]
    tiny = {}
    for name in ("cutoff", "dlyTime", "aR", "outGain"):
        i = NAMES.index(name)
        lo, hi, w = fd_interval(z, i, 0.002)
        tiny[name] = (obj(hi) - obj(lo)) / w
    record(noise={
        "render_max_abs_diff_same_params": float(np.abs(a - b).max()),
        "loss_repeats": losses,
        "loss_spread": float(max(losses) - min(losses)),
        "fd_at_h_0.002": tiny,
    })
    print(f"same-params render max|diff| {np.abs(a - b).max():.3g}, "
          f"loss spread over 3 calls {max(losses) - min(losses):.3g}")
    print("h=0.002 FD: " + "  ".join(f"{k} {v:+.3f}" for k, v in tiny.items()))


# ---------------------------------------------------------------- faust FD


def _faust_fd(obj: Objective, z: np.ndarray, hs: tuple[float, ...]) -> dict:
    base = obj(z)
    out: dict[str, dict[str, float]] = {}
    for h in hs:
        g = {}
        for i, nm in enumerate(NAMES):
            lo, hi, w = fd_interval(z, i, h)
            g[nm] = (obj(hi) - obj(lo)) / w if w > 0 else 0.0
        out[str(h)] = g
        print(f"  h={h}: |g| = {np.linalg.norm([g[n] for n in NAMES]):.4f}")
    return {"base_loss": base, "fd": out}


def step_faust() -> None:
    obj = faust_objective()
    z = point("patch")
    print(f"faust FD at patch.json, {len(H_GRID) * 2 * len(NAMES)} renders")
    res = _faust_fd(obj, z, H_GRID)
    record(faust_fd_patch=res)
    print(f"base faust loss {res['base_loss']:.4f} after {obj.calls} renders")


def step_seed() -> None:
    obj = faust_objective()
    z = point("seed")
    res = _faust_fd(obj, z, (0.03,))
    record(faust_fd_seed=res)
    print(f"seed base faust loss {res['base_loss']:.4f}")


# ---------------------------------------------------------------- surrogate


def _pad_and_loss(device: torch.device) -> tuple[TorchPad, SpectralLoss]:
    return TorchPad(load_notes(), default_n_samples(), device), SpectralLoss()


def torch_loss_at(pad: TorchPad, loss_fn: SpectralLoss, z: np.ndarray,
                  device: torch.device) -> float:
    with torch.no_grad():
        zt = torch.tensor(np.clip(z, 0.0, 1.0), dtype=torch.float32, device=device)
        return float(loss_fn(pad.render(denorm_torch(zt))))


def step_torchfd(device_name: str, h: float = 0.03) -> None:
    """Central FD of the surrogate itself. Disagreement with autograd is a graph bug."""
    device = torch.device(device_name)
    pad, loss_fn = _pad_and_loss(device)
    z = point("patch")
    base = torch_loss_at(pad, loss_fn, z, device)
    g = {}
    for i, nm in enumerate(NAMES):
        lo, hi, w = fd_interval(z, i, h)
        g[nm] = (torch_loss_at(pad, loss_fn, hi, device)
                 - torch_loss_at(pad, loss_fn, lo, device)) / w if w > 0 else 0.0
    record(torch_fd_patch={"h": h, "base_loss": base, "device": device_name, "fd": g})
    print(f"surrogate base loss {base:.4f}, FD |g| {np.linalg.norm([g[n] for n in NAMES]):.4f}")


def autograd_normalized(z: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """(dL/dz, dL/dlogit, loss). Both spaces, because they are different directions.

    An optimiser stepping the Patch moves logits, and the diagonal z(1-z) between
    the two spaces is not a scalar, so a direction that descends in one space is not
    the same direction in the other.
    """
    pad, loss_fn = _pad_and_loss(device)
    patch = Patch(z).to(device)
    loss = loss_fn(pad(patch))
    loss.backward()
    g_logit = patch.logits.grad.detach().cpu().numpy().astype(float)
    zc = np.clip(z, 1e-4, 1 - 1e-4)
    return g_logit / (zc * (1.0 - zc)), g_logit, float(loss)


def step_auto(device_name: str) -> None:
    device = torch.device(device_name)
    for tag in ("patch", "seed"):
        z = point(tag)
        gz, gl, loss = autograd_normalized(z, device)
        record(**{f"autograd_{tag}": {
            "device": device_name,
            "loss": loss,
            "grad_normalized": dict(zip(NAMES, gz.tolist())),
            "grad_logit": dict(zip(NAMES, gl.tolist())),
            "norm_normalized": float(np.linalg.norm(gz)),
        }})
        print(f"[{tag}] surrogate loss {loss:.4f}, |dL/dz| {np.linalg.norm(gz):.4f}, "
              f"|dL/dlogit| {np.linalg.norm(gl):.4f}")


# ---------------------------------------------------------------- line search


def _direction(g: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d = -g * mask
    return d / (np.linalg.norm(d) + 1e-20)


def step_micro(device_name: str, hs: tuple[float, ...] = (0.0003, 0.003)) -> None:
    """Is the phase-parameter gradient right at a step size nobody would ever take?

    A derivative that only describes the loss over an interval of 3e-4 is useless to
    an optimiser but tells us which of two failures we have: an analytic derivative
    that is correct and the surface that is pathological, or a derivative of
    something other than the rendered signal.
    """
    device = torch.device(device_name)
    pad, loss_fn = _pad_and_loss(device)
    obj = faust_objective()
    z = point("patch")
    out: dict[str, dict[str, dict[str, float]]] = {}
    for nm in PHASE_PARAMS:
        i = NAMES.index(nm)
        out[nm] = {}
        for h in hs:
            lo, hi, w = fd_interval(z, i, h)
            out[nm][str(h)] = {
                "torch": (torch_loss_at(pad, loss_fn, hi, device)
                          - torch_loss_at(pad, loss_fn, lo, device)) / w,
                "faust": (obj(hi) - obj(lo)) / w,
            }
        print(f"{nm:8s} " + "  ".join(
            f"h={h}: torch {out[nm][str(h)]['torch']:+9.3f} faust {out[nm][str(h)]['faust']:+9.3f}"
            for h in hs))
    record(micro_fd_patch=out)


def step_lines(seed: int = 7) -> None:
    """Step along -g and along random directions of the same length, in Faust.

    Without the random control, "the loss got worse" says nothing: at a converged
    point it gets worse along every direction. The control turns the test into a
    comparison against chance, which is what usable means.
    """
    obj = faust_objective()
    rng = np.random.default_rng(seed)
    def keep(names: tuple[str, ...]) -> np.ndarray:
        m = np.ones(len(NAMES))
        for nm in names:
            m[NAMES.index(nm)] = 0.0
        return m

    mask, mask4 = keep(FREEZE_CANDIDATES), keep(PHASE_PARAMS)
    out = {}
    for tag, etas in (("patch", (0.03, 0.1, 0.3)), ("seed", (0.5, 1.0, 1.5))):
        z = point(tag)
        g = np.array([stored(f"autograd_{tag}")["grad_normalized"][n] for n in NAMES])
        base = obj(z)
        runs: dict[str, dict[str, float]] = {}
        for label, d in (("grad_all", _direction(g, np.ones_like(mask))),
                         ("grad_frozen", _direction(g, mask)),
                         ("grad_frozen4", _direction(g, mask4))):
            runs[label] = {}
            for e in etas:
                runs[label][str(e)] = obj(np.clip(z + e * d, 0.0, 1.0))
        for k in range(3):
            r = rng.normal(size=len(NAMES)) * mask4
            r /= np.linalg.norm(r)
            runs[f"random_{k}"] = {str(e): obj(np.clip(z + e * r, 0.0, 1.0)) for e in etas}
        out[tag] = {"base_loss": base, "etas": list(etas), "runs": runs}
        best_r = min(min(runs[f"random_{k}"].values()) for k in range(3))
        print(f"[{tag}] base {base:.4f}  "
              + "  ".join(f"best -grad({lb}) {min(runs[lb].values()):.4f}"
                          for lb in ("grad_all", "grad_frozen", "grad_frozen4"))
              + f"  best of 3 random {best_r:.4f}")
    record(lines=out)


# ---------------------------------------------------------------- report


def classify(row: dict) -> str:
    """Name the failure mode, because the two failures need opposite responses.

    A broken graph is the surrogate lying about its own loss and can only be fixed
    in the port. A rough surface is the objective itself having no direction at the
    scale a step would use, and no surrogate can repair that: those parameters
    belong to CMA-ES whatever the port does.
    """
    a, t = row["g_auto_normalized"], row["g_torch_fd"]
    f = row["g_faust_fd"]["0.03"]
    ratio = abs(a / t) if abs(t) > 1e-9 else float("inf")
    # dead first: below this level neither gradient can move the parameter, so a
    # sign disagreement between two numbers near zero is not worth naming a bug
    if abs(a) < 1e-2 and abs(f) < 0.05:
        return "dead"
    if np.sign(a) != np.sign(t) or not 0.3 < ratio < 3.0:
        return "broken_graph"
    if not row["faust_fd_sign_stable_over_h"]:
        return "rough_surface"
    if np.sign(a) != np.sign(f):
        return "port_sign_error"
    if abs(f) > 1e-9 and not 0.1 < abs(a / f) < 10.0:
        return "trust_sign_only"
    return "trust"


def step_report() -> None:
    d = json.load(open(OUT))
    z = point("patch")
    gz = np.array([d["autograd_patch"]["grad_normalized"][n] for n in NAMES])
    gl = np.array([d["autograd_patch"]["grad_logit"][n] for n in NAMES])
    ffd = {h: np.array([v[n] for n in NAMES]) for h, v in d["faust_fd_patch"]["fd"].items()}
    ref = ffd["0.03"]
    tfd = np.array([d["torch_fd_patch"]["fd"][n] for n in NAMES])
    zc = np.clip(z, 1e-4, 1 - 1e-4)
    chain = zc * (1.0 - zc)

    interior = np.array([EDGE < v < 1.0 - EDGE for v in z])
    free = np.array([n not in FREEZE_CANDIDATES for n in NAMES])
    free4 = np.array([n not in PHASE_PARAMS for n in NAMES])
    hs = sorted(ffd, key=float)
    stack = np.stack([ffd[h] for h in hs])
    stable = (np.sign(stack) == np.sign(stack[0])).all(axis=0)

    rows = {}
    for i, nm in enumerate(NAMES):
        vals = stack[:, i]
        span = float(np.abs(vals).max())
        rows[nm] = {
            "z": float(z[i]),
            "on_box_face": bool(not interior[i]),
            "g_auto_normalized": float(gz[i]),
            "g_auto_logit": float(gl[i]),
            "g_torch_fd": float(tfd[i]),
            "g_faust_fd": {h: float(ffd[h][i]) for h in hs},
            "faust_fd_sign_stable_over_h": bool(stable[i]),
            "faust_fd_spread_over_h": float(vals.max() - vals.min()),
            "faust_fd_rel_spread": float((vals.max() - vals.min()) / (span + 1e-12)),
            "sign_agree_auto_vs_faust": bool(np.sign(gz[i]) == np.sign(ref[i])),
            "sign_agree_autofd_vs_torchfd": bool(np.sign(gz[i]) == np.sign(tfd[i])),
            "ratio_auto_over_faust": float(gz[i] / ref[i]) if ref[i] != 0 else None,
            "ratio_auto_over_torchfd": float(gz[i] / tfd[i]) if tfd[i] != 0 else None,
        }

    for i, nm in enumerate(NAMES):
        rows[nm]["verdict"] = classify(rows[nm])

    def subset(m: np.ndarray) -> dict:
        return {
            "n": int(m.sum()),
            "cos_auto_vs_faust": cosine(gz[m], ref[m]),
            "cos_torchfd_vs_faust": cosine(tfd[m], ref[m]),
            "cos_auto_vs_torchfd": cosine(gz[m], tfd[m]),
            "cos_auto_vs_faust_logit_space": cosine(gz[m] * chain[m], ref[m] * chain[m]),
            "sign_agree_auto_vs_faust": int((np.sign(gz[m]) == np.sign(ref[m])).sum()),
        }

    summary = {
        "all": subset(np.ones(len(NAMES), bool)),
        "interior": subset(interior),
        "free": subset(free),
        "free_no_phase4": subset(free4),
        "interior_and_free": subset(interior & free),
        "interior_and_free_no_phase4": subset(interior & free4),
        "cos_faust_h001_vs_h003": cosine(ffd["0.01"], ffd["0.03"]),
        "cos_faust_h003_vs_h008": cosine(ffd["0.03"], ffd["0.08"]),
        "faust_fd_sign_unstable": [n for n, s in zip(NAMES, stable) if not s],
        "norms": {
            "g_auto": float(np.linalg.norm(gz)),
            "g_torch_fd": float(np.linalg.norm(tfd)),
            "g_faust_fd_h003": float(np.linalg.norm(ref)),
        },
        "phase4_share_of_grad_norm_sq": float((gz[~free4] ** 2).sum() / (gz @ gz)),
        "top_share_of_auto_norm": sorted(
            ((n, float(v ** 2 / (gz @ gz))) for n, v in zip(NAMES, gz)),
            key=lambda kv: -kv[1])[:5],
        "directional_derivative_frozen_dir": {
            "predicted_by_surrogate": float(-np.linalg.norm(gz[free])),
            "true_faust": float(ref[free] @ _direction(gz, free.astype(float))[free]),
        },
    }
    if "autograd_seed" in d and "faust_fd_seed" in d:
        gzs = np.array([d["autograd_seed"]["grad_normalized"][n] for n in NAMES])
        fs = np.array([d["faust_fd_seed"]["fd"]["0.03"][n] for n in NAMES])
        summary["seed"] = {
            "cos_auto_vs_faust": cosine(gzs, fs),
            "cos_auto_vs_faust_free": cosine(gzs[free], fs[free]),
            "cos_auto_vs_faust_no_phase4": cosine(gzs[free4], fs[free4]),
            "sign_agree": int((np.sign(gzs) == np.sign(fs)).sum()),
            "sign_agree_no_phase4": int((np.sign(gzs[free4]) == np.sign(fs[free4])).sum()),
            "phase4_share_of_grad_norm_sq": float((gzs[~free4] ** 2).sum() / (gzs @ gzs)),
        }
    verdicts: dict[str, list[str]] = {}
    for nm, row in rows.items():
        verdicts.setdefault(row["verdict"], []).append(nm)
    summary["verdicts"] = verdicts
    # Two different reasons to freeze, and they are not interchangeable. A broken
    # graph is a property of the port and holds everywhere, so those parameters come
    # out of the gradient step for good. A sign disagreement seen only at the
    # converged patch is a property of that point, where the true gradient has fallen
    # below the surrogate's accuracy: the same parameters agree at the seed, so
    # freezing them permanently would throw away the gradient's best coordinates.
    always = [nm for nm, row in rows.items() if row["verdict"] == "broken_graph"]
    if "autograd_seed" in d and "faust_fd_seed" in d:
        seed_bad = [nm for nm in NAMES
                    if np.sign(d["autograd_seed"]["grad_normalized"][nm])
                    != np.sign(d["faust_fd_seed"]["fd"]["0.03"][nm])]
        summary["sign_disagree_at_seed"] = seed_bad
        summary["sign_disagree_at_both_points"] = sorted(
            set(seed_bad) & {nm for nm, row in rows.items()
                             if not row["sign_agree_auto_vs_faust"]})
    summary["recommended_freeze"] = {
        "always": sorted(always),
        "dead_leave_to_cma": sorted(nm for nm, row in rows.items()
                                    if row["verdict"] == "dead"),
        "near_optimum_only": sorted(nm for nm, row in rows.items()
                                    if row["verdict"] == "port_sign_error"),
    }
    record(per_parameter=rows, summary=summary)
    for k in sorted(verdicts):
        print(f"{k:16s} {len(verdicts[k]):2d}  {' '.join(verdicts[k])}")
    s = summary
    print(f"cos(auto, faust) all {s['all']['cos_auto_vs_faust']:+.4f}  "
          f"interior {s['interior']['cos_auto_vs_faust']:+.4f}  "
          f"free {s['free']['cos_auto_vs_faust']:+.4f}  "
          f"interior+free {s['interior_and_free']['cos_auto_vs_faust']:+.4f}")
    print(f"cos(auto, torchFD) all {s['all']['cos_auto_vs_torchfd']:+.4f}  "
          f"cos(torchFD, faust) all {s['all']['cos_torchfd_vs_faust']:+.4f}")
    print(f"faust FD self-consistency: cos(h.01,h.03) {s['cos_faust_h001_vs_h003']:+.4f}  "
          f"cos(h.03,h.08) {s['cos_faust_h003_vs_h008']:+.4f}  "
          f"sign-unstable {s['faust_fd_sign_unstable']}")
    print(f"without the 4 phase params: cos(auto,faust) "
          f"{s['free_no_phase4']['cos_auto_vs_faust']:+.4f} at patch, "
          f"{s.get('seed', {}).get('cos_auto_vs_faust_no_phase4', float('nan')):+.4f} at seed; "
          f"cos(auto,torchFD) {s['free_no_phase4']['cos_auto_vs_torchfd']:+.4f}")
    print(f"sign agreement {s['all']['sign_agree_auto_vs_faust']}/27 "
          f"(interior {s['interior']['sign_agree_auto_vs_faust']}/{s['interior']['n']})")
    print(f"wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["noise", "faust", "torchfd", "auto", "seed",
                                     "micro", "lines", "report"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("mps" if get_device().type == "mps" else "cpu")
    if args.step == "noise":
        step_noise()
    elif args.step == "faust":
        step_faust()
    elif args.step == "torchfd":
        step_torchfd(dev)
    elif args.step == "auto":
        step_auto(dev)
    elif args.step == "seed":
        step_seed()
    elif args.step == "micro":
        step_micro(dev)
    elif args.step == "lines":
        step_lines()
    else:
        step_report()


if __name__ == "__main__":
    main()
