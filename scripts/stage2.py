"""Stage 2: match the patch with CMA-ES against a multi-resolution STFT loss.

The MIDI from stage 1 is frozen. Only synth parameters move. Optimisation is
hierarchical: first the oscillator/filter/amp core with the effects held at a
modest setting, then everything together.
"""

from __future__ import annotations

import argparse
import json
import time

import auraloss
import cma
import librosa
import numpy as np
import pretty_midi
import torch

from bend2 import bend_curve
from metrics import report
from synth import PAD, Architecture, PadRenderer, write_render

SR = 44100
DUR = 17.904

CORE = [
    "detune", "uniMix", "subLvl", "sqrMix", "cutoff", "reso", "envAmt", "kbdTrk",
    "fA", "fD", "fS", "aA", "aD", "aS", "aR", "lfoRate", "lfoAmt", "tilt", "outGain",
    "drive", "spread",
]
FX = ["chRate", "chDepth", "dlyTime", "dlyFb", "dlyWet", "revSize", "revDamp", "revWet"]

# The 26 EQ band gains. Kept out of CORE deliberately: they are the largest single
# improvement this project found (0.162 of loss, against 0.012 for all the CMA-ES and
# gradient work before them), but they are a LINEAR problem in the log-magnitude domain
# and CMA-ES is the wrong tool for them. fit_eq_full.py solves them by coordinate search
# with a curvature penalty at roughly one poly render for the whole fit, because the
# cascade commutes with the voice sum and outGain. A CMA-ES pass that includes them
# would spend thousands of renders rediscovering that, and without the penalty it would
# land in the near-singular alternating direction that fitting one chord already found:
# a comb tuned to that chord's partials, which cost 0.15 on a different chord and
# regressed env_l1 from 0.104 to 0.189. Name them explicitly if you really want them
# searched.
EQ = [f"eq{i}" for i in range(26)]


def load_notes(path: str = "data/transcription.mid"):
    pm = pretty_midi.PrettyMIDI(path)
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((n.pitch, n.velocity, float(n.start), float(n.end - n.start)))
    return sorted(notes, key=lambda z: z[2])


# Weight of the side-channel term in the stereo objective. The mono objective could not
# see stereo at all, which is not a subtlety here: the recording is a mono-centred bass
# under a wide pad, its width rising 18.6 dB from 30 Hz to 3 kHz while our render is flat
# to 0.4 dB. Scored in mono, the correct reverb depth reads as a 0.21 regression, roughly
# the size of the largest genuine improvement the project has found, so the fit drove
# revWet to 0.12 when the image wants about 0.58.
W_SIDE = 0.35


class Objective:
    """The stage-2 objective. Mono by default, stereo when asked.

    loss_of keeps its mono meaning and its historical numbers (out/patch.json reproduces
    to 1.2e-07), because a silent change of yardstick would invalidate every figure in
    the repo at once. Stereo scoring is opt-in through loss_parts, which reports the mono
    and side terms separately so the two are always comparable.
    """

    def __init__(self, notes, target_path: str = "data/original.wav",
                 arch: Architecture = PAD, loss: str | None = None) -> None:
        self.arch = arch
        y, _ = librosa.load(target_path, sr=SR, mono=True)
        self.n = len(y)
        self.target_mono = y
        self.target = torch.from_numpy(y).float().view(1, 1, -1)
        self.target_env = self._env(self.target)
        # Stereo reference. Loaded separately and NOT with mono=True: original.wav is
        # 48 kHz, so this has to resample, and reading it any other way compares audio
        # that is 8.8% wrong in time and pitch.
        ys, _ = librosa.load(target_path, sr=SR, mono=False)
        ys = np.atleast_2d(ys)
        self.target_side = self._side(ys if ys.shape[0] > 1 else np.vstack([ys, ys]))
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048, 4096],
            hop_sizes=[128, 256, 512, 1024],
            win_lengths=[512, 1024, 2048, 4096],
            w_sc=1.0,
            w_log_mag=1.0,
            w_lin_mag=0.0,
        )
        self.renderer = PadRenderer(arch, n_voices=24)
        self.renderer.set_notes(notes)
        # intro glide: sample-accurate bend, fixed (measured), never optimised
        self.renderer.set_bend(bend_curve(int(DUR * SR) + SR))
        # An alternative objective, for the bake-off. `loss_of` is deliberately left
        # alone: it keeps its mono MRSTFT meaning and its historical numbers, so a run
        # under a candidate loss still reports the incumbent figure alongside and the
        # two stay comparable. Only __call__, what the optimiser actually descends,
        # switches.
        self.loss_name = loss
        if loss is None:
            self.alt = None
        else:
            from losses import LOSSES, load_candidates
            load_candidates()
            if loss not in LOSSES:
                raise SystemExit(f"unknown loss {loss!r}; have {sorted(LOSSES)}")
            self.alt = LOSSES[loss](y.astype(np.float32), SR)
        self.calls = 0
        self.best = (1e9, None)
        self.history: list[float] = []

    @staticmethod
    def _env(x: torch.Tensor, hop: int = 512) -> torch.Tensor:
        f = x.view(1, 1, -1).unfold(-1, hop * 2, hop)
        e = f.pow(2).mean(-1).sqrt()
        return e / (e.mean() + 1e-9)

    @staticmethod
    def _side(a: np.ndarray) -> np.ndarray:
        """(L-R)/2, normalised to unit RMS.

        Normalised because the side channel is 3 to 20 dB below the mid depending on the
        band, so without it w_side would be setting a level rather than a weight, and the
        term would be swamped by however loud the render happens to be. Normalising also
        makes the term measure the SHAPE of the stereo image, which is what differs: our
        render has too much side energy in the bass and too little in the pad.
        """
        s = (a[0] - a[1]) / 2.0
        return s / (float(np.sqrt((s ** 2).mean())) + 1e-12)

    def loss_parts(self, audio: np.ndarray, w_env: float = 0.35,
                   w_side: float = W_SIDE) -> dict[str, float]:
        """Mono loss, side loss and their weighted total, in one pass.

        Returned as parts rather than a scalar so a caller can always report the mono
        figure next to the stereo one. Every number in the repo predates the side term.
        """
        mono = self.loss_of(audio, w_env)
        a = np.asarray(audio)
        if a.ndim < 2 or a.shape[0] < 2 or not np.isfinite(a).all():
            return {"mono": mono, "side": 1e6, "total": 1e6}
        n = min(a.shape[1], self.target_side.shape[0])
        p = torch.from_numpy(self._side(a[:, :n]).copy()).float().view(1, 1, -1)
        t = torch.from_numpy(self.target_side[:n].copy()).float().view(1, 1, -1)
        with torch.no_grad():
            side = float(self.mrstft(p, t))
        if not np.isfinite(side):
            side = 1e6
        return {"mono": mono, "side": side, "total": mono + w_side * side}

    def stereo_loss(self, audio: np.ndarray, w_env: float = 0.35,
                    w_side: float = W_SIDE) -> float:
        return self.loss_parts(audio, w_env, w_side)["total"]

    def render(self, x: np.ndarray) -> np.ndarray:
        self.renderer.set_params(self.arch.denorm(x))
        return self.renderer.render(DUR)

    def loss_of(self, audio: np.ndarray, w_env: float = 0.35) -> float:
        """Loss of an already rendered clip, so a caller that needs the audio too
        does not have to pay for a second render."""
        mono = audio.mean(axis=0)
        if not np.isfinite(mono).all():
            return 1e6
        n = min(len(mono), self.n)
        pred = torch.from_numpy(mono[:n].copy()).float().view(1, 1, -1)
        tgt = self.target[..., :n]
        with torch.no_grad():
            spec = float(self.mrstft(pred, tgt))
            env = float((self._env(pred) - self._env(tgt)).abs().mean())
        loss = spec + w_env * env
        return float(loss) if np.isfinite(loss) else 1e6

    def alt_of(self, audio: np.ndarray) -> float:
        """The candidate objective on an already rendered clip."""
        mono = audio.mean(axis=0)
        if not np.isfinite(mono).all():
            return 1e6
        v = self.alt(mono[: self.n].astype(np.float32))
        return float(v) if np.isfinite(v) else 1e6

    def __call__(self, x: np.ndarray, w_env: float = 0.35) -> float:
        self.calls += 1
        audio = self.render(np.asarray(x))
        loss = self.alt_of(audio) if self.alt is not None else self.loss_of(audio, w_env)
        if loss < self.best[0]:
            self.best = (loss, np.array(x, dtype=float))
        return loss


def seeded_start() -> np.ndarray:
    """Seed from the stage-0 analysis rather than from the middle of the box."""
    p = PAD.denorm(PAD.norm_defaults())
    p.update(
        detune=65.0,     # ~70 cent spread measured between unison partials
        uniMix=0.85,     # partials appear as clusters, little dry centre
        subLvl=0.40,     # strong energy at the fundamental
        sqrMix=0.0,      # complete harmonic series -> saw, not square
        cutoff=850.0,    # spectral centroid ~1050 Hz, rolloff95 ~3750 Hz
        reso=1.0,
        envAmt=500.0,
        kbdTrk=0.25,
        fA=0.9, fD=1.6, fS=0.8,
        aA=0.55, aD=1.2, aS=0.9, aR=1.8,   # slow swell, long tail
        lfoRate=4.0, lfoAmt=0.0,
        chRate=0.4, chDepth=0.25,
        dlyTime=0.35, dlyFb=0.25, dlyWet=0.05,
        revSize=0.9, revDamp=0.5, revWet=0.45,
        tilt=-0.2, outGain=0.5,
    )
    return np.clip(PAD.normalize(p), 0.001, 0.999)


def run_cma(obj: Objective, x0: np.ndarray, free: list[str], gens: int, sigma: float,
            seed: int, label: str, plateau: float | None = 0.99):
    """CMA-ES over the named subset of the vector.

    `plateau` is the fraction of the 20-generation-ago best that counts as progress;
    None runs the full `gens`. It is a knob rather than a constant because telling a
    stalled optimiser from a stopped one needs a run without it, and the self-recovery
    bench is exactly that question.
    """
    idx = [obj.arch.index[n] for n in free]
    base = x0.copy()

    def sub(z: np.ndarray) -> float:
        x = base.copy()
        x[idx] = np.clip(z, 0.0, 1.0)
        return obj(x)

    es = cma.CMAEvolutionStrategy(
        base[idx].tolist(), sigma,
        {"bounds": [0, 1], "popsize": 16, "seed": seed, "verbose": -9, "maxiter": gens},
    )
    t0 = time.time()
    g = 0
    milestone = None  # best loss 20 generations ago
    while not es.stop() and g < gens:
        sols = es.ask()
        vals = [sub(np.asarray(s)) for s in sols]
        es.tell(sols, vals)
        g += 1
        obj.history.append(float(min(vals)))
        if g % 10 == 0 or g == 1:
            print(f"  [{label}] gen {g:3d}  best {es.result.fbest:.4f}  "
                  f"({obj.calls} renders, {time.time() - t0:.0f}s)")
        # stop when the best loss has improved < 1% over 20 generations
        if plateau is not None and g % 20 == 0:
            fb = float(es.result.fbest)
            if milestone is not None and fb > milestone * plateau:
                print(f"  [{label}] plateau at gen {g} ({milestone:.4f} -> {fb:.4f}), stopping")
                break
            milestone = fb
    out = base.copy()
    out[idx] = np.clip(np.asarray(es.result.xbest), 0.0, 1.0)
    return out, float(es.result.fbest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-gens", type=int, default=90)
    ap.add_argument("--full-gens", type=int, default=140)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--pin", action="append", default=[],
                    help="name=real_value : hold a parameter fixed and exclude it from the search")
    ap.add_argument("--out", default="out/patch.json")
    ap.add_argument("--init", default=None, help="warm-start from an existing patch json")
    ap.add_argument("--sigma", type=float, default=None,
                    help="initial CMA step; use a small value (~0.05) for a local polish")
    args = ap.parse_args()

    notes = load_notes()
    print(f"loaded {len(notes)} notes from data/transcription.mid")
    obj = Objective(notes)

    if args.init:
        x0 = obj.arch.padded(json.load(open(args.init))["normalized"])
        print(f"warm start from {args.init}")
    else:
        x0 = seeded_start()

    pinned: dict[str, float] = {}
    for spec in args.pin:
        name, _, val = spec.partition("=")
        v = float(val)
        i = obj.arch.index[name]
        nx = obj.arch.params[i].normalize(v)
        x0[i] = float(np.clip(nx, 0.0, 1.0))
        pinned[name] = v
    if pinned:
        print("pinned:", pinned)
        for lst in (CORE, FX):
            for n in list(lst):
                if n in pinned:
                    lst.remove(n)
    print(f"seed loss: {obj(x0):.4f}")

    best_x, best_l = x0, obj(x0)
    free_all = [p.name for p in obj.arch.params if p.name not in pinned]
    for rs in range(args.restarts):
        print(f"\n--- restart {rs} : core ---")
        sig_c = args.sigma if args.sigma else 0.22 + 0.08 * rs
        xc, lc = run_cma(obj, best_x if rs == 0 else x0, CORE, args.core_gens, sig_c, 100 + rs, f"core{rs}")
        print(f"--- restart {rs} : full ---")
        sig_f = args.sigma * 0.7 if args.sigma else 0.14 + 0.06 * rs
        xf, lf = run_cma(obj, xc, free_all, args.full_gens, sig_f, 200 + rs, f"full{rs}")
        print(f"restart {rs}: core {lc:.4f} -> full {lf:.4f}")
        if lf < best_l:
            best_x, best_l = xf, lf

    if obj.best[1] is not None and obj.best[0] < best_l:
        best_x, best_l = obj.best[1], obj.best[0]

    print(f"\nfinal loss {best_l:.4f} after {obj.calls} renders")
    vals = obj.arch.denorm(best_x)
    aud = obj.render(best_x)
    # default run owns out/render.wav + out/loss_history.npy; variants get a suffix
    stem = args.out.removesuffix(".json")
    is_default = args.out == "out/patch.json"
    render_path = "out/render.wav" if is_default else f"{stem}_render.wav"
    hist_path = "out/loss_history.npy" if is_default else f"{stem}_loss_history.npy"

    write_render(render_path, aud)
    orig, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    m = report(orig, aud)
    print("metrics:", {k: round(v, 4) for k, v in m.items()})

    with open(args.out, "w") as fh:
        json.dump({"loss": best_l, "metrics": m, "params": vals,
                   "pinned": pinned, "normalized": best_x.tolist()}, fh, indent=2)
    np.save(hist_path, np.array(obj.history))
    print(f"wrote {args.out} + {render_path}")


if __name__ == "__main__":
    main()
