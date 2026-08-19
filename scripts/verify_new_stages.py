"""Ground-truth checks for the two stages added after the first fit: drive and spread.

Four questions, all answered from rendered audio rather than from reading the source,
because a surrogate that disagrees with Faust makes the gradient optimise the wrong
thing and an appended parameter that is not the identity at its default silently
invalidates the patch that was fitted before it existed:

  1. does the torch waveshaper equal the Faust one, sample for sample
  2. do the per-note pan gains a render actually produces equal pan_gains()
  3. does the 27-parameter patch that was delivered before, padded with the new
     defaults, still render to its recorded loss
  4. what do drive and spread buy on a coarse Faust grid at the current best patch,
     measured on the two defects they were added for and not only on the loss

Everything numeric lands in out/new_stages.json; stdout is scalars only.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import torch

import faust_probe
from metrics import band_db_error, lta_band_error, stereo_decorrelation
from stage2 import Objective, load_notes
from synth import DSP, PARAM_INDEX, PARAMS, PadRenderer, denorm, norm_defaults, pad_normalized
from torch_common import Patch, default_n_samples, get_device, schedule
from torch_synth import TorchPad, pan_gains, saturate

SR = 44100
OLD_LOSS = 1.5563665382564067  # out/patch.json, measured by stage2 before PARAMS grew
BASELINE27 = "out/patch_baseline27.json"  # the same patch, kept once patch.json moved on

DRIVE_DSP = """
import("stdfaust.lib");
drive = hslider("drive", 0.0, 0, 1, 0.001);
dgain = 1.0 + drive * 12.0;
shape(x) = x + drive * (ma.tanh(x * dgain) / ma.tanh(dgain) - x);
process = shape;
"""

# effects neutralised, so the L/R ratio a render shows is the pan law and nothing else
NEUTRAL_FX = {"chDepth": 0.0, "dlyWet": 0.0, "revWet": 0.0, "tilt": 0.0, "outGain": 1.0}

DRIVES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SPREADS = (0.0, 0.3, 0.7, 1.0)
PAN_PITCHES = (29, 53, 68)

GRID_DRIVE = (0.0, 0.08, 0.16, 0.3, 0.5, 0.75)
GRID_SPREAD = (0.0, 0.25, 0.5, 0.75, 1.0)


def test_drive() -> dict[str, float]:
    """Faust ma.tanh against torch.tanh on the same signal, at six drive settings."""
    n = 1 << 15
    t = np.arange(n) / SR
    x = (0.9 * np.sin(2 * np.pi * 110 * t) + 0.5 * np.sin(2 * np.pi * 437 * t)
         + 0.3 * np.sin(2 * np.pi * 1911 * t)).astype(np.float32)
    xt = torch.from_numpy(x)
    worst = {"max_abs": 0.0, "rel_l2": 0.0, "at_drive": 0.0}
    for d in DRIVES:
        ref = faust_probe.render_fx(DRIVE_DSP, {"drive": d}, x)[0]
        got = saturate(xt, torch.tensor(float(d))).numpy()
        err = float(np.abs(got - ref).max())
        rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))
        if err > worst["max_abs"]:
            worst = {"max_abs": err, "rel_l2": rel, "at_drive": float(d)}
        if d == 0.0 and err != 0.0:
            raise AssertionError(f"drive=0 is not a bypass in torch: max abs {err:.3e}")
    return worst


def test_pan() -> dict[str, float]:
    """Per-channel RMS of a real one-note render against the predicted pan gains."""
    r = PadRenderer(n_voices=4)
    r.set_bend(None)
    base = denorm(norm_defaults())
    base.update(NEUTRAL_FX)
    worst = {"max_ratio_err": 0.0, "at_spread": 0.0, "at_pitch": 0}
    for pitch in PAN_PITCHES:
        r.set_notes([(pitch, 100, 0.05, 1.0)])
        ev = schedule([(pitch, 100, 0.05, 1.0)], int(1.6 * SR), torch.device("cpu"))
        for s in SPREADS:
            r.set_params({**base, "spread": s})
            a = r.render(1.6)
            rms = np.sqrt((a.astype(np.float64) ** 2).mean(axis=1))
            gl, gr = pan_gains(ev, torch.tensor(float(s)))
            want = float(gr[0] / gl[0])
            got = float(rms[1] / max(rms[0], 1e-12))
            err = abs(got - want)
            if err > worst["max_ratio_err"]:
                worst = {"max_ratio_err": err, "at_spread": float(s), "at_pitch": int(pitch),
                         "predicted": want, "measured": got}
    return worst


def _replace_once(s: str, old: str, new: str) -> str:
    if s.count(old) != 1:
        raise AssertionError(f"expected exactly one {old!r} in the DSP, found {s.count(old)}")
    return s.replace(old, new)


def legacy_dsp() -> str:
    """synth.DSP with the two new sliders replaced by their identity constants.

    Faust's own simplifier then folds `osc + 0*(...)` back to `osc` and both pan gains
    to 1, so this compiles to the pre-drive, pre-spread DSP. That makes the parity
    check a real A/B of two renders rather than a claim about float arithmetic.
    """
    s = _replace_once(DSP, 'drive    = hslider("drive", 0.0, 0, 1, 0.001);', "drive = 0.0;")
    return _replace_once(s, 'spread   = hslider("spread", 0.0, 0, 1, 0.001);', "spread = 0.0;")


def test_compat(obj: Objective, notes, path: str = BASELINE27) -> dict[str, float]:
    """The pre-drive, pre-spread patch padded with the new defaults must still render
    to the loss that was recorded for it."""
    x = pad_normalized(json.load(open(path))["normalized"])
    if len(x) != len(PARAMS):
        raise AssertionError(f"padded vector is {len(x)}, PARAMS is {len(PARAMS)}")
    if x[PARAM_INDEX["drive"]] != 0.0 or x[PARAM_INDEX["spread"]] != 0.0:
        raise AssertionError(f"{path} does not have the new stages at their defaults")
    new_audio = obj.render(x)
    old_obj = Objective(notes, dsp=legacy_dsp())
    old_audio = old_obj.render(x)
    n = min(new_audio.shape[1], old_audio.shape[1])
    d = new_audio[:, :n] - old_audio[:, :n]
    return {
        "old_loss_recorded": OLD_LOSS,
        "legacy_dsp_loss": old_obj.loss_of(old_audio),
        "padded_loss": obj.loss_of(new_audio),
        "render_max_abs": float(np.abs(d).max()),
        "render_rel_l2": float(np.linalg.norm(d) / (np.linalg.norm(old_audio[:, :n]) + 1e-12)),
    }


def test_surrogate(obj: Objective, x: np.ndarray) -> dict[str, float]:
    """Surrogate loss and gradient against the true Faust loss, both new stages on.

    The gradient entries for the two new parameters are the point of the port: they are
    exact ports, unlike the reverb, so their derivative is the one Faust would give if
    Faust could be differentiated.
    """
    from torch_common import SpectralLoss

    device = get_device(prefer_mps=False)
    pad = TorchPad(load_notes(), default_n_samples(), device)
    lossfn = SpectralLoss()
    patch = Patch(x).to(device)
    surr = lossfn(pad(patch))
    surr.backward()
    g = patch.logits.grad.detach().cpu().numpy()
    faust = float(obj(x))
    return {
        "surrogate": float(surr.detach()),
        "faust": faust,
        "bias": float(surr.detach()) - faust,
        "grad_finite": bool(np.isfinite(g).all()),
        "grad_drive": float(g[PARAM_INDEX["drive"]]),
        "grad_spread": float(g[PARAM_INDEX["spread"]]),
        "grad_max_abs": float(np.abs(g).max()),
    }


def best_gain_loss(obj: Objective, audio: np.ndarray) -> tuple[float, float]:
    """Lowest loss reachable by rescaling the render, and the factor that reaches it.

    outGain is the last operation in the Faust effect chain, so scaling the rendered
    audio is exactly equivalent to re-rendering with outGain scaled the same way. A
    stage that changes the output level would otherwise be judged on the level change
    rather than on the spectrum, and the loss's spectral-convergence term is not
    scale-invariant.
    """
    best = (float("inf"), 1.0)
    for g in np.geomspace(0.5, 2.0, 13):
        best = min(best, (obj.loss_of(audio * g), float(g)))
    return best


def grid(obj: Objective, x0: np.ndarray, orig: np.ndarray) -> dict[str, object]:
    """Coarse Faust scan over drive x spread with the other parameters held."""
    di, si = PARAM_INDEX["drive"], PARAM_INDEX["spread"]
    rows = []
    best = None
    for d in GRID_DRIVE:
        for s in GRID_SPREAD:
            x = x0.copy()
            x[di], x[si] = d, s
            aud = obj.render(x)
            gain_loss, gain = best_gain_loss(obj, aud)
            row = {"drive": d, "spread": s, "loss": obj.loss_of(aud),
                   "gain_matched_loss": gain_loss, "gain": gain,
                   "decorr": stereo_decorrelation(aud),
                   "mid_signed_db": band_db_error(orig, aud)["mean_signed_db"],
                   "bands_db": lta_band_error(orig, aud)}
            rows.append(row)
            if best is None or row["gain_matched_loss"] < best["gain_matched_loss"]:
                best = row | {"normalized": x.tolist()}
    return {"rows": rows, "best": best, "renders": len(rows)}


def main() -> None:
    out: dict[str, object] = {}
    out["drive_parity"] = test_drive()
    print("drive parity  worst max_abs {max_abs:.3e} rel_l2 {rel_l2:.3e} "
          "at drive {at_drive}".format(**out["drive_parity"]))

    out["pan_parity"] = test_pan()
    print("pan parity    worst |R/L ratio err| {max_ratio_err:.3e} "
          "at spread {at_spread} pitch {at_pitch}".format(**out["pan_parity"]))

    notes = load_notes()
    obj = Objective(notes)
    out["compat"] = test_compat(obj, notes)
    print("compat        recorded {old_loss_recorded:.10f} legacy-dsp {legacy_dsp_loss:.10f} "
          "padded {padded_loss:.10f}".format(**out["compat"]))
    print("compat        padded-vs-legacy render max_abs {render_max_abs:.2e} "
          "rel_l2 {render_rel_l2:.2e}".format(**out["compat"]))

    orig, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    x0 = pad_normalized(np.asarray(json.load(open("out/patch_torch.json"))["normalized"], float))
    g = grid(obj, x0, orig)
    out["grid"] = g
    b = g["best"]
    zero = next(r for r in g["rows"] if r["drive"] == 0.0 and r["spread"] == 0.0)
    for tag, r in (("identity", zero), ("best", b)):
        print(f"grid {tag:8s} drive {r['drive']:.2f} spread {r['spread']:.2f} "
              f"loss {r['loss']:.4f} gain-matched {r['gain_matched_loss']:.4f} "
              f"(x{r['gain']:.2f}) decorr {r['decorr']:.4f} mid {r['mid_signed_db']:+.2f} dB")

    out["surrogate_at_grid_best"] = test_surrogate(obj, np.asarray(b["normalized"], float))
    print("surrogate     {surrogate:.4f} vs faust {faust:.4f} (bias {bias:+.4f})"
          .format(**out["surrogate_at_grid_best"]))
    print("gradient      finite {grad_finite}  dL/dlogit drive {grad_drive:+.2e} "
          "spread {grad_spread:+.2e}  max |g| {grad_max_abs:.2e}"
          .format(**out["surrogate_at_grid_best"]))

    total = obj.calls + int(g["renders"])
    out["renders_used"] = total
    with open("out/new_stages.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote out/new_stages.json ({total} faust renders of the full clip)")


if __name__ == "__main__":
    main()
