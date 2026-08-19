"""Fit the 26-band EQ on the WHOLE clip with a smoothness penalty.

Why this exists. Fitting the cascade on one chord (fit_chord.py) reached 1.2565 on
that chord but produced a comb: mean neighbour jump 10.2 dB, nine sign alternations,
three bands on the +-18 dB rail. That is not a spectral envelope, it is a filter
tuned to F major's partial frequencies, and it behaves like one: its broadband gain
swings 8.5 dB as the harmony moves, the bend-free window at 10.40-13.35 s got 0.153
WORSE, and full-clip env_l1 regressed 0.1041 -> 0.1885.

Two changes fix the cause rather than the symptom.

1. Fit on all 18 s, not one chord. A curve fitted to every chord cannot specialise to
   one chord's partials. This is affordable only because of the commutation below.
2. Penalise the second difference of the gain curve. The 26-band bank at Q 2.8 is
   well conditioned but not orthogonal, so the loss has a near-flat alternating
   direction; without a penalty an optimiser walks down it into the rails. The
   penalty is justified by measurement, not taste: the oracle band curve fitted
   independently on each of the four bend-free windows agrees to 1.28 dB while the
   shape itself is 13.5 dB deep, so the true curve is smooth and chord-independent.

The trick that makes a full-clip fit cheap. The cascade sits after the voice sum and
before outGain, both of which are linear, so it commutes with them: pushing an
already-rendered flat-EQ clip through the same Faust cascade gives exactly the audio
the sliders would have produced. So one 18 s poly render is enough, and every
candidate costs a cascade render instead. Audio still comes out of Faust either way.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import minimize

import eq_stage
import synth
from bend2 import bend_curve
from metrics import env_l1
from stage2 import DUR, SR, Objective, load_notes

FLAT = np.zeros(eq_stage.N_BANDS)


def flat_render(patch_path: str = "out/patch.json") -> np.ndarray:
    """The 18 s clip at `patch_path` with every EQ band at 0 dB, mono.

    DSP_SAW rather than DSP: the 160-partial bank costs 23x render time and the
    workflow measured its contribution at exactly zero, so it is not in the loop.
    """
    r = synth.PadRenderer(n_voices=24, dsp=synth.DSP_SAW)
    r.set_notes(load_notes())
    p = dict(json.load(open(patch_path))["params"])
    p.update(eq_stage.gain_dict(FLAT))
    r.set_params(p)
    r.set_bend(bend_curve(int(DUR * SR) + SR))
    return r.render(DUR).mean(0)


class FullScore:
    """stage2's objective on a mono clip, plus the level the objective actually wants.

    Level is not searched. The loss is a MAGNITUDE spectral loss, and its spectral
    convergence term is least squares on magnitudes, so its optimal gain is
    <|O|,|R|> / <|R|,|R|>, which factorises as (level ratio) x cos(theta). Note this
    must be computed on magnitude spectra: the time-domain inner product of two
    different signals is essentially random in sign and gives a meaningless answer.
    Solving it per candidate stops the optimiser spending gain bands on broadband level.
    """

    NFFT = 2048

    def __init__(self, dsp: str = synth.DSP_SAW) -> None:
        self.obj = Objective(load_notes(), dsp=dsp)
        self.target = self.obj.target.numpy().ravel()
        self._tmag: dict[int, np.ndarray] = {}

    def _mag(self, x: np.ndarray) -> np.ndarray:
        import librosa
        return np.abs(librosa.stft(np.ascontiguousarray(x, dtype=np.float32),
                                   n_fft=self.NFFT,
                                   hop_length=self.NFFT // 4)).astype(np.float64)

    def level(self, x: np.ndarray) -> tuple[float, float]:
        """(optimal gain, cos theta) for this clip against the target."""
        n = min(len(x), len(self.target))
        if n not in self._tmag:
            self._tmag[n] = self._mag(self.target[:n])
        O, R = self._tmag[n], self._mag(x[:n])
        m = min(O.shape[1], R.shape[1])
        O, R = O[:, :m], R[:, :m]
        g = float((O * R).sum() / ((R * R).sum() + 1e-30))
        cos = float((O * R).sum() / (np.linalg.norm(O) * np.linalg.norm(R) + 1e-30))
        return g, cos

    def __call__(self, mono: np.ndarray) -> tuple[float, float, float]:
        n = min(len(mono), len(self.target))
        x = mono[:n]
        g, cos = self.level(x)
        return self.obj.loss_of((x * g)[None, :]), g, cos


def penalty(g: np.ndarray) -> float:
    """Second difference of the curve, i.e. curvature. Flat and tilted are both free."""
    return float((np.diff(g, 2) ** 2).sum())


def fit(flat: np.ndarray, score: FullScore, lam: float, x0: np.ndarray,
        maxfev: int = 4000) -> tuple[np.ndarray, dict]:
    hist: list[float] = []

    def obj(g: np.ndarray) -> float:
        g = np.clip(g, -eq_stage.GAIN_LIMIT, eq_stage.GAIN_LIMIT)
        loss, _, _ = score(eq_stage.eq_window(flat, g))
        v = loss + lam * penalty(g)
        hist.append(v)
        return v

    res = minimize(obj, x0, method="Powell",
                   bounds=[(-eq_stage.GAIN_LIMIT, eq_stage.GAIN_LIMIT)] * len(x0),
                   options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-5})
    g = np.clip(res.x, -eq_stage.GAIN_LIMIT, eq_stage.GAIN_LIMIT)
    loss, lvl, corr = score(eq_stage.eq_window(flat, g))
    return g, {"loss": loss, "level": lvl, "cos": corr, "penalty": penalty(g),
               "evals": len(hist)}


def oracle_start(flat: np.ndarray, score: FullScore) -> np.ndarray:
    """Warm start: the band gains that best match the flat render's spectrum to the
    target's, read off band by band. Cheap, and it puts Powell in the right basin."""
    import librosa
    n = min(len(flat), len(score.target))
    nfft = 8192
    O = np.abs(librosa.stft(score.target[:n], n_fft=nfft, hop_length=nfft // 4)).astype(float)
    R = np.abs(librosa.stft(flat[:n], n_fft=nfft, hop_length=nfft // 4)).astype(float)
    f = librosa.fft_frequencies(sr=SR, n_fft=nfft)
    fc = eq_stage.band_freqs()
    edges = np.sqrt(fc[:-1] * fc[1:])
    edges = np.concatenate([[fc[0] / 1.1], edges, [fc[-1] * 1.1]])
    want = np.empty(len(fc))
    for i in range(len(fc)):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if not m.any():
            want[i] = 0.0
            continue
        want[i] = 20 * np.log10((O[m] * R[m]).sum() / ((R[m] ** 2).sum() + 1e-30) + 1e-30)
    want -= want.mean()                      # the level is solved in closed form, not here
    g, _ = eq_stage.fit_gains(fc, want, ridge=0.05)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="out/patch.json")
    ap.add_argument("--lam", type=float, nargs="*",
                    default=[0.0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--maxfev", type=int, default=4000)
    ap.add_argument("--out", default="out/patch_eq_full.json")
    a = ap.parse_args()

    score = FullScore()
    flat = flat_render(a.patch)
    base_loss, base_lvl, base_corr = score(flat)
    print(f"flat EQ, level-optimal: loss {base_loss:.6f}  level {20*np.log10(base_lvl):+.2f} dB")
    print(f"incumbent out/render.wav reference: 1.544635\n")

    x0 = oracle_start(flat, score)
    l0, _, _ = score(eq_stage.eq_window(flat, x0))
    print(f"oracle warm start: loss {l0:.6f}  curvature {penalty(x0):.1f}  "
          f"max|g| {np.abs(x0).max():.1f} dB\n")

    tgt = score.target
    print(f"{'lambda':>8} {'loss':>9} {'level':>8} {'curv':>8} {'max|g|':>7} "
          f"{'jump':>6} {'rail':>5} {'env_l1':>7}")
    print("-" * 68)
    best = None
    for lam in a.lam:
        g, info = fit(flat, score, lam, x0, a.maxfev)
        y = eq_stage.eq_window(flat, g)
        _, lvl, _ = score(y)
        e = env_l1(tgt[:len(y)], (y * lvl)[:len(tgt)])
        jump = float(np.abs(np.diff(g)).mean())
        rail = int((np.abs(g) >= eq_stage.GAIN_LIMIT - 0.5).sum())
        print(f"{lam:>8.1e} {info['loss']:>9.6f} {20*np.log10(lvl):>+7.2f} "
              f"{info['penalty']:>8.1f} {np.abs(g).max():>7.1f} {jump:>6.1f} "
              f"{rail:>5} {e:>7.4f}")
        rec = {"lam": lam, "gains": g.tolist(), "loss": info["loss"],
               "level": lvl, "env_l1": e, "curvature": info["penalty"],
               "mean_jump": jump, "railed": rail, "evals": info["evals"]}
        if best is None or info["loss"] < best["loss"]:
            best = rec
        json.dump(rec, open(f"out/eq_full_lam{lam:.0e}.json", "w"), indent=1)

    print(f"\nbest: lambda {best['lam']:.1e}, loss {best['loss']:.6f}, "
          f"env_l1 {best['env_l1']:.4f}, mean jump {best['mean_jump']:.1f} dB")
    print("gains dB:", " ".join(f"{x:+.1f}" for x in best["gains"]))
    json.dump(best, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
