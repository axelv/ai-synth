"""Fit the bass and pad layers against the stereo objective, one at a time.

Why alternating rather than joint: the layers are summed, so with one layer's render
held fixed the other's fit is an ordinary problem, and each pass costs a handful of real
renders instead of a search over twice as many coupled parameters. Two passes each has
been enough in every alternation this project has run.

Why the EQ is not searched by CMA-ES: the band cascade sits after the layer's voice sum
and before the layer is added to the other, and both operations are linear, so a layer
rendered once with flat bands and then pushed through the same Faust cascade IS the
audio its sliders would have produced. One real render per macro candidate covers an
entire gain fit at about 35 ms per candidate. The gains still come out of Faust, so
nothing here is a model of the synth.

The curvature penalty is not optional. Fitting 26 unpenalised bands to one chord earlier
in this project produced a comb tuned to that chord's partials: mean neighbour jump
10.2 dB, three bands on the rail, 0.153 WORSE on a different chord, and env_l1 regressed
from 0.104 to 0.189. Penalising the second difference of the gain curve is what keeps a
fitted curve a spectral envelope instead of a filter bank.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.optimize import minimize

import eq_stage
import layers
import synth
from layers import BASS, PAD
from stage2 import DUR, SR, Objective, W_SIDE, load_notes

LAM = 3e-5          # curvature weight; chosen by sweep in fit_eq_full, 10x smoother for 0.017
GAIN_LIMIT = eq_stage.GAIN_LIMIT


class CachedMRSTFT:
    """auraloss MultiResolutionSTFTLoss with the target's spectra computed once.

    The target never changes across a fit, but auraloss recomputes its STFT on every
    call and also computes a phase spectrum that is discarded when w_phs is 0. Together
    that is most of the cost: 0.166 s per call, and the objective calls it twice, which
    put a single scored candidate at 0.49 s and a full fit at several hours.

    This is a reimplementation, so it is checked rather than trusted: `verify` asserts
    agreement with the real auraloss module on actual renders. If that assertion ever
    fires, the fit is scoring something other than stage 2's objective and the numbers
    are not comparable.
    """

    def __init__(self, target: torch.Tensor, fft_sizes=(512, 1024, 2048, 4096),
                 hop_sizes=(128, 256, 512, 1024), win_lengths=(512, 1024, 2048, 4096),
                 eps: float = 1e-8) -> None:
        self.cfg = list(zip(fft_sizes, hop_sizes, win_lengths))
        self.eps = eps
        self.windows = [torch.hann_window(w) for _, _, w in self.cfg]
        t = target.reshape(-1)
        self.y_mag = [self._mag(t, i) for i in range(len(self.cfg))]
        self.y_log = [m.log() for m in self.y_mag]
        self.y_norm = [m.norm(p="fro") for m in self.y_mag]

    def _mag(self, x: torch.Tensor, i: int) -> torch.Tensor:
        n, h, w = self.cfg[i]
        s = torch.stft(x, n, h, w, self.windows[i], return_complex=True)
        return torch.sqrt(torch.clamp(s.real ** 2 + s.imag ** 2, min=self.eps))

    def __call__(self, x: torch.Tensor) -> float:
        total = 0.0
        with torch.no_grad():
            for i in range(len(self.cfg)):
                m = self._mag(x.reshape(-1), i)
                sc = (self.y_mag[i] - m).norm(p="fro") / self.y_norm[i]
                lg = (m.log() - self.y_log[i]).abs().mean()
                total += float(sc + lg)
        return total / len(self.cfg)


def verify_cached(obj: Objective, clips: list[np.ndarray], tol: float = 2e-5) -> float:
    """Assert CachedMRSTFT reproduces auraloss on real audio before any fit uses it."""
    worst = 0.0
    fast = CachedMRSTFT(obj.target)
    for a in clips:
        mono = np.asarray(a).mean(axis=0)
        n = min(len(mono), obj.n)
        p = torch.from_numpy(mono[:n].copy()).float().view(1, 1, -1)
        with torch.no_grad():
            ref = float(obj.mrstft(p, obj.target[..., :n]))
        worst = max(worst, abs(fast(p) - ref))
    if worst > tol:
        raise AssertionError(f"cached MRSTFT differs from auraloss by {worst:.2e}")
    return worst


class FastObjective:
    """stage2's objective with both target spectra cached. Same formulas, same weights."""

    def __init__(self, obj: Objective, w_side: float) -> None:
        self.obj = obj
        self.w_side = w_side
        self.mono = CachedMRSTFT(obj.target)
        self.side = CachedMRSTFT(torch.from_numpy(obj.target_side).float().view(1, 1, -1))
        self.n = obj.n
        self.ns = obj.target_side.shape[0]

    def __call__(self, audio: np.ndarray, w_env: float = 0.35) -> dict[str, float]:
        a = np.asarray(audio)
        if not np.isfinite(a).all():
            return {"mono": 1e6, "side": 1e6, "total": 1e6}
        m = a.mean(axis=0)
        n = min(len(m), self.n)
        p = torch.from_numpy(m[:n].copy()).float().view(1, 1, -1)
        with torch.no_grad():
            env = float((self.obj._env(p) - self.obj._env(self.obj.target[..., :n])).abs().mean())
        mono = self.mono(p) + w_env * env
        ns = min(a.shape[1], self.ns)
        s = torch.from_numpy(self.obj._side(a[:, :ns]).copy()).float().view(1, 1, -1)
        side = self.side(s)
        return {"mono": mono, "side": side, "total": mono + self.w_side * side}


def cascade(stereo: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """A stereo clip through the real Faust band cascade.

    Applied per channel because the DSP applies `par(i, 2, eqCurve)`, i.e. the same
    cascade independently to L and R, so doing it channel by channel is the same
    operation and not an approximation.
    """
    return np.stack([eq_stage.eq_window(stereo[0], gains),
                     eq_stage.eq_window(stereo[1], gains)])


def curvature(g: np.ndarray) -> float:
    return float((np.diff(g, 2) ** 2).sum())


class LayerFit:
    """Fit one layer's EQ gains and level with the other layer's audio held fixed."""

    def __init__(self, obj, other: np.ndarray, free_bands, w_side: float) -> None:
        self.obj = obj
        self.other = other
        self.free = np.asarray(free_bands, dtype=int)
        self.w_side = w_side
        self.calls = 0

    def mix(self, flat_render: np.ndarray, gains: np.ndarray, level: float) -> np.ndarray:
        y = cascade(flat_render, gains) * level
        n = min(y.shape[1], self.other.shape[1])
        return y[:, :n] + self.other[:, :n]

    def score(self, flat_render: np.ndarray, gains: np.ndarray, level: float) -> dict:
        self.calls += 1
        return self.obj(self.mix(flat_render, gains, level))

    def fit(self, flat_render: np.ndarray, g0: np.ndarray, level0: float = 1.0,
            maxfev: int = 1500) -> tuple[np.ndarray, float, dict]:
        """Powell over the free bands plus log level. Bands outside `free` stay at 0 dB."""
        x0 = np.concatenate([g0[self.free], [np.log(level0)]])

        def unpack(x):
            g = np.zeros(eq_stage.N_BANDS)
            g[self.free] = np.clip(x[:-1], -GAIN_LIMIT, GAIN_LIMIT)
            return g, float(np.exp(np.clip(x[-1], -3.0, 3.0)))

        def f(x):
            g, lv = unpack(x)
            return self.score(flat_render, g, lv)["total"] + LAM * curvature(g)

        res = minimize(f, x0, method="Powell",
                       bounds=[(-GAIN_LIMIT, GAIN_LIMIT)] * len(self.free) + [(-3.0, 3.0)],
                       options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-5})
        g, lv = unpack(res.x)
        return g, lv, self.score(flat_render, g, lv)


def fold_level(params: dict[str, float], level: float) -> dict[str, float]:
    """Fold a fitted level into outGain, which is the chain's last operation so this is
    exact rather than an approximation."""
    lo, hi = next((q.lo, q.hi) for q in synth.PARAMS if q.name == "outGain")
    out = dict(params)
    out["outGain"] = float(np.clip(params["outGain"] * level, lo, hi))
    return out


def report(tag: str, d: dict, extra: str = "") -> None:
    print(f"  {tag:<34} mono {d['mono']:.4f}  side {d['side']:.4f}  "
          f"total {d['total']:.4f}  {extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--w-side", type=float, default=W_SIDE)
    ap.add_argument("--out", default="out/patch_layers.json")
    ap.add_argument("--wav", default="out/render_layers.wav")
    a = ap.parse_args()

    print(layers.describe())
    obj = Objective(load_notes())
    lr = layers.LayerRenderer()
    P = layers.default_params()

    incumbent = obj.render(np.array(json.load(open("out/patch.json"))["normalized"]))
    start_audio = lr.render(P)
    # The fast objective is a reimplementation, so prove it matches auraloss on the very
    # clips this run will be scoring before any of them are used to make a decision.
    worst = verify_cached(obj, [incumbent, start_audio])
    fast = FastObjective(obj, a.w_side)
    print(f"cached objective agrees with auraloss to {worst:.1e} on this run's own audio")

    single = fast(incumbent)
    print()
    report("single-layer incumbent", single)
    report("two layers, starting params", fast(start_audio))
    print()

    # Macro candidates per layer. Deliberately small: the recording says where these
    # belong, and every extra macro is another real render per round.
    BASS_MACROS = [
        {"spread": 0.0, "revWet": 0.02, "chDepth": 0.0},
        {"spread": 0.0, "revWet": 0.08, "chDepth": 0.0},
        {"spread": 0.0, "revWet": 0.20, "chDepth": 0.0},
    ]
    PAD_MACROS = [
        {"spread": 0.9, "revWet": 0.35},
        {"spread": 0.9, "revWet": 0.55},
        {"spread": 1.0, "revWet": 0.70},
        {"spread": 1.0, "revWet": 0.85},
    ]

    for rnd in range(1, a.rounds + 1):
        for layer, macros, free in ((BASS, BASS_MACROS, layers.BASS_EQ),
                                    (PAD, PAD_MACROS, layers.PAD_EQ)):
            other = lr.render_layer(PAD if layer == BASS else BASS,
                                    P[PAD if layer == BASS else BASS])
            fitter = LayerFit(fast, other, free, a.w_side)
            # Coarse pass to rank the macros, then one full fit on the winner. Running a
            # full gain fit per macro quadruples the cost to decide something a short
            # fit already orders correctly.
            ranked = []
            for m in macros:
                cand = layers.with_eq({**P[layer], **m}, np.zeros(eq_stage.N_BANDS))
                flat_render = lr.render_layer(layer, cand)
                g, lv, d = fitter.fit(flat_render, layers.eq_of(P[layer]), maxfev=180)
                tag = " ".join(f"{k}={v}" for k, v in m.items())
                report(f"r{rnd} {layer} coarse: {tag}", d, f"lvl {20*np.log10(lv):+.1f} dB")
                ranked.append((d["total"], cand, flat_render, g, lv))
            ranked.sort(key=lambda z: z[0])
            _, cand, flat_render, g0, lv0 = ranked[0]
            g, lv, d = fitter.fit(flat_render, g0, lv0, maxfev=2500)
            P[layer] = fold_level(layers.with_eq(cand, g), lv)
            report(f"r{rnd} {layer}: KEPT", d,
                   f"max|g| {np.abs(g).max():.1f} dB, jump {np.abs(np.diff(g)).mean():.1f}")
        print()

    audio = lr.render(P)
    final = fast(audio)
    # Confirm the fitted result through the real auraloss objective, not the cached one.
    check = obj.loss_parts(audio, w_side=a.w_side)
    report("final, rendered through both synths", final)
    report("  same, scored by auraloss itself", check)

    synth.write_render(a.wav, audio)
    json.dump({"params": layers.flat(P),
               "normalized": {k: synth.normalize(v).tolist() for k, v in P.items()},
               "loss": final, "w_side": a.w_side, "lam": LAM,
               "single_layer_reference": single,
               "split_midi": layers.SPLIT_MIDI,
               "note": "two independent synths, bass (MIDI<33) and pad, summed. "
                       "Fitted against the stereo objective; mono loss reported "
                       "alongside so it stays comparable to earlier single-layer work."},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out} and {a.wav}")


if __name__ == "__main__":
    main()
