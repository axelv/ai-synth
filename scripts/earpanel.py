"""Day 1 of the retrieval prototype: can any fingerprint see a near neighbour at all?

Retrieval replaces the search with a lookup. Fingerprint the target, fingerprint every
patch in a bank, rank by descriptor distance. That only works if a patch genuinely close
in parameter space lands closer than the whole crowd of unrelated ones, by a margin wide
enough to survive the bank's own size: the closest of N unrelated patches sits about
sqrt(2 ln N) standard deviations below the crowd mean purely by chance, so a neighbour
buried under that is unretrievable however good the descriptor looks in isolation.

Costs zero renders. Everything here is measured on the cached corpus the loss bake-off
already built, which is why this runs before any preset bank is assembled: it is the one
question whose answer can kill the whole approach for a day of arithmetic.

Four blocks, deliberately not more:

  band26    the synth's own EQ grid, as the mean of PER-FRAME dB. Not dB of the mean
            power and not power-0.3 compression: three separate proposals reached for
            those and both destroy the quiet-frame detail the grid exists to carry.
            Bottom edge at 80 Hz rather than the EQ bank's 40 Hz, because the two lowest
            bands hold almost no energy on this material and only add phase noise.
  mel_mean  average spectrum on a perceptual grid. Overlaps band26 by design; the pair
            says whether the synth's own geometry buys anything over a generic one.
  mel_std   how much each band moves over time. The only block here with any hope of
            separating the frame-shuffled control, which every time-averaged block reads
            as identical to the target.
  env       loudness over time. Included on suspicion, kept only if it earns its place.

Weighting is the one design choice that could be fiddled, so it is fixed by measurement
instead: each coordinate is scaled by ITS OWN phase noise, the amount that coordinate
moves when the same patch is rendered with the oscillators at different phases. Nothing
is scaled by its spread across the bank, which would be tuning on the answer. A distance
of 1.0 therefore means "on average, as different as one patch is from itself".

Run:  PYTHONPATH=scripts uv run python scripts/earpanel.py
"""

from __future__ import annotations

import argparse
import glob
import os

import librosa
import numpy as np
from scipy.stats import norm

from losscorpus import CORPUS

SR = 44100
N_FFT = 2048
HOP = 512
FLOOR_DB = -80.0

N_BAND = 26
BAND_LO, BAND_HI = 80.0, 16000.0
BAND_HALFWIDTH = 0.254          # octaves; the measured support of the EQ bank's Q

N_MELS = 64
ENV_POINTS = 192

# A coordinate whose phase noise is a fluke fraction of the rest would otherwise dominate
# the weighted distance. Floored at a quarter of the block's median.
WEIGHT_FLOOR = 0.25

NEAR_RADIUS = 0.05              # the neighbour a bank is realistically expected to hold
CROWD_RADIUS = 0.33             # unrelated patches; chance separation here is rms 0.376
KEEP_NUISANCE = 0.2             # phase twin must cost at most a fifth of a real error
KEEP_MARGIN = 3.0               # neighbour must sit this many crowd sd below the crowd


def _band_filters(sr: int = SR, n_fft: int = N_FFT) -> np.ndarray:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    fb = np.zeros((N_BAND, len(freqs)))
    for i, fc in enumerate(np.geomspace(BAND_LO, BAND_HI, N_BAND)):
        lo, hi = fc * 2 ** -BAND_HALFWIDTH, fc * 2 ** BAND_HALFWIDTH
        m = (freqs >= lo) & (freqs <= hi)
        if not m.any():                        # bands narrower than the bin spacing
            fb[i, np.argmin(np.abs(freqs - fc))] = 1.0
        else:
            fb[i] = m.astype(float)
        fb[i] /= fb[i].sum()
    return fb


_FB = _band_filters()
_MEL = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                           fmin=BAND_LO, fmax=BAND_HI)


# Concatenations, because a fingerprint is allowed to be more than one block and the
# blocks answer different questions: mel_std is the only one that sees time, band26 and
# mel_mean the only ones that see the spectrum. Weighted per coordinate as always, so a
# concatenation cannot win by simply having more numbers in it.
COMBO_ALL = ("band26", "mel_mean", "mel_std", "env")
COMBOS = {
    "mel_both": ("mel_mean", "mel_std"),
    "band+std": ("band26", "mel_std"),
    "all4": COMBO_ALL,
}


def blocks(audio: np.ndarray) -> dict[str, np.ndarray]:
    """The four candidate fingerprints, each a 1-D vector in dB."""
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))
    band = np.maximum(20.0 * np.log10(_FB @ S + 1e-12), FLOOR_DB)
    mel = np.maximum(10.0 * np.log10(_MEL @ (S ** 2) + 1e-12), FLOOR_DB)

    e = librosa.feature.rms(y=audio, hop_length=HOP)[0]
    e = e / (e.mean() + 1e-9)                  # scale invariant, as env_l1 already is
    e = np.interp(np.linspace(0, 1, ENV_POINTS), np.linspace(0, 1, len(e)), e)

    return {
        "band26": band.mean(axis=1),
        "mel_mean": mel.mean(axis=1),
        "mel_std": mel.std(axis=1),
        "env": np.maximum(20.0 * np.log10(e + 1e-6), FLOOR_DB),
    }


def delevel(v: np.ndarray) -> np.ndarray:
    """Remove overall level, which a real recording sets by its gain knob, not its patch."""
    return v - v.mean()


def describe(path: str, level_invariant: bool) -> dict[str, dict[str, np.ndarray]]:
    """label -> block -> descriptor, for one corpus file."""
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as npz:
        for label in npz.files:
            if label in ("target", "labels", "dists"):
                continue
            b = blocks(npz[label].astype(np.float64))
            b = {k: (delevel(v) if level_invariant else v) for k, v in b.items()}
            for name, parts in COMBOS.items():
                b[name] = np.concatenate([b[q] for q in parts])
            out[label] = b
    return out


def weights(desc: dict[str, dict[str, dict[str, np.ndarray]]], block: str) -> np.ndarray:
    """Per-coordinate phase noise, pooled over targets. The only scaling applied."""
    noise = np.mean([np.abs(d["nuisance"][block] - d["truth"][block])
                     for d in desc.values()], axis=0)
    return 1.0 / np.maximum(noise, WEIGHT_FLOOR * np.median(noise))


def dist(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Mean weighted absolute difference, in units of one patch's distance from itself."""
    return float((w * np.abs(a - b)).mean())


def radius_labels(labels, r: float) -> list[str]:
    return [x for x in labels if x.startswith(f"radius:{r}:")]


def crowd_of(desc, tids: list[str], tid: str, block: str, w: np.ndarray
             ) -> tuple[np.ndarray, float]:
    """Distances from this truth to unrelated patches, plus a within-family sd.

    A bank is a mixture of families, so the pooled spread is the honest model of one and
    it is what the margin is quoted against. The within-family sd drops the between-group
    part, which is the most generous reading available; it is reported alongside so the
    verdict cannot be an artefact of how the crowd was assembled.
    """
    truth = desc[tid]["truth"][block]
    groups = [[dist(truth, desc[tid][l][block], w)
               for l in radius_labels(desc[tid], CROWD_RADIUS)],
              [dist(truth, desc[o][l][block], w)
               for o in tids if o != tid for l in radius_labels(desc[o], CROWD_RADIUS)],
              [dist(truth, desc[o]["truth"][block], w) for o in tids if o != tid]]
    pooled = np.concatenate([np.array(g) for g in groups])
    within = np.concatenate([np.array(g) - np.mean(g) for g in groups]).std(ddof=1)
    return pooled, within


def needed_margin(n: float) -> float:
    """How far below the crowd mean the nearest of n unrelated patches is expected to sit.

    Blom's approximation to the expected extreme of n standard normals, not the textbook
    sqrt(2 ln n). The asymptotic form is the one everybody reaches for and it is badly
    wrong at the sizes this bench actually measures: at n = 11 it claims 2.19 where the
    true expectation is 1.59, which flatters the verdict by inventing a bar the crowd
    could never clear. Gaussian either way, and a heavier-tailed crowd is worse than both.
    """
    return float(norm.ppf((n - 0.375) / (n + 0.25)))


def supported_n(margin: float) -> float:
    """Largest bank in which a neighbour this far below the crowd still expects to win."""
    p = float(norm.cdf(max(margin, 0.0)))
    return (0.375 + 0.25 * p) / max(1.0 - p, 1e-12)


def panel(desc, block: str) -> dict:
    w = weights(desc, block)
    tids = sorted(desc)
    rows = []
    for tid in tids:
        d = desc[tid]
        truth = d["truth"][block]
        here = lambda lab: dist(truth, d[lab][block], w)
        crowd, within = crowd_of(desc, tids, tid, block, w)
        mu, sd = crowd.mean(), crowd.std(ddof=1)

        near = float(np.mean([here(lab) for lab in radius_labels(d, NEAR_RADIUS)]))
        r002 = float(np.mean([here(lab) for lab in radius_labels(d, 0.02)]))
        r0002 = float(np.mean([here(lab) for lab in radius_labels(d, 0.002)]))
        nui = here("nuisance")

        rows.append({
            "tid": tid, "nui": nui, "r002": r002, "crowd_mu": mu, "near": near,
            "nui_r02": nui / r002, "nui_r002": nui / r0002, "cov": sd / mu,
            "margin": (mu - near) / sd, "margin_within": (mu - near) / within,
            "shuffled_z": (here("shuffled") - mu) / sd,
        })
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("nui_r02", "nui_r002", "cov", "margin", "margin_within",
                     "shuffled_z")}
    agg["margin_min"] = float(min(r["margin"] for r in rows))
    agg["n_supported"] = supported_n(agg["margin_min"])
    agg["rows"] = rows
    return agg


def report(desc, names: list[str], title: str) -> dict[str, dict]:
    print(f"\n=== {title} ===")
    print(f"{'block':10s} {'nui/r.02':>9s} {'nui/r.002':>10s} {'CoV':>6s} "
          f"{'margin':>8s} {'worst':>7s} {'generous':>9s} {'bank N':>7s} "
          f"{'shuffled':>9s}  keep")
    res = {}
    for name in names:
        a = panel(desc, name)
        res[name] = a
        keep = a["nui_r02"] <= KEEP_NUISANCE and a["margin_min"] >= KEEP_MARGIN
        print(f"{name:10s} {a['nui_r02']:9.3f} {a['nui_r002']:10.3f} {a['cov']:6.2f} "
              f"{a['margin']:5.2f} sd {a['margin_min']:4.2f} sd {a['margin_within']:6.2f} sd "
              f"{a['n_supported']:7.0f} {a['shuffled_z']:6.2f} sd  {'yes' if keep else 'NO'}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--per-target", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.corpus, "t[0-9][0-9].npz")))
    if not paths:
        raise SystemExit(f"no corpus in {args.corpus}; run losscorpus.py first")

    names = ["band26", "mel_mean", "mel_std", "env"] + list(COMBOS)
    for invariant in (False, True):
        desc = {os.path.basename(p)[:-4]: describe(p, invariant) for p in paths}
        title = ("level invariant, each descriptor de-meaned"
                 if invariant else "absolute level, as rendered")
        res = report(desc, names, title)
        if args.per_target:
            for name in names:
                for r in res[name]["rows"]:
                    print(f"  {name:10s} {r['tid']} nui {r['nui']:7.3f} r0.02 "
                          f"{r['r002']:7.3f} crowd {r['crowd_mu']:7.3f} near "
                          f"{r['near']:7.3f} margin {r['margin']:6.2f} sd")

    print("\nmargin a bank of N demands, as the expected nearest of N strangers:")
    for n in (11, 100, 1000, 3000, 10000, 200000):
        print(f"  N = {n:>7d}   {needed_margin(n):.2f} sd"
              f"   (sqrt(2 ln N) would claim {np.sqrt(2 * np.log(n)):.2f})")
    print(f"\nkeep rule: nui/r0.02 <= {KEEP_NUISANCE} and worst-target r0.05 margin "
          f">= {KEEP_MARGIN} sd")


if __name__ == "__main__":
    main()
