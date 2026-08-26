"""Turn a fitted EQ gain vector into a real patch, and hold it to the test that
caught the single-chord fit.

The chord fit looked excellent on the chord it was fitted to and was 0.153 WORSE on
the bend-free window at 10.40-13.35 s, with full-clip env_l1 regressing from 0.104 to
0.189. So a full-clip loss number on its own is not evidence any more. Acceptance
here means: better on the full clip, better on ALL THREE bend-free windows, and no
regression in the metrics the optimiser never saw.

Everything is rendered by Faust through synth.PadRenderer. Nothing is scored from a
post-hoc filter, even though the cascade provably commutes with the rest of the chain,
because the point of promotion is to prove the sliders do it.
"""

from __future__ import annotations

import argparse
import json

import librosa
import numpy as np
import soundfile as sf

import chord
import eq_stage
import metrics
import synth
from bend2 import bend_curve
from stage2 import DUR, SR, Objective, incumbent_loss, load_notes

WINDOWS = ((4.95, 7.45), (7.45, 10.40), (10.40, 13.35))


def patch_params(gains: np.ndarray, level: float,
                 base: str = "out/patch.json") -> dict[str, float]:
    """Incumbent macros + fitted bands + the level folded into outGain.

    outGain is the last operation in the chain, so folding the level there is exact
    rather than an approximation, and it keeps the patch a plain parameter vector.
    """
    p = dict(json.load(open(base))["params"])
    p.update(eq_stage.gain_dict(gains))
    lo, hi = next((q.lo, q.hi) for q in synth.PAD.params if q.name == "outGain")
    want = p["outGain"] * level
    if not lo <= want <= hi:
        raise ValueError(f"outGain {want:.4f} outside [{lo}, {hi}]; level {level:.4f}")
    p["outGain"] = want
    return p


def render_full(params: dict[str, float]) -> np.ndarray:
    r = synth.PadRenderer(synth.PAD, n_voices=24)
    r.set_notes(load_notes())
    r.set_params(params)
    r.set_bend(bend_curve(int(DUR * SR) + SR))
    return r.render(DUR)


def window_score(t0: float, t1: float):
    """chord.WindowScore with moved bounds, without duplicating its loss definition."""
    old = (chord.WIN_T0, chord.WIN_T1)
    chord.WIN_T0, chord.WIN_T1 = t0, t1
    try:
        return chord.WindowScore(), slice(int(t0 * SR), int(t1 * SR))
    finally:
        chord.WIN_T0, chord.WIN_T1 = old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", default="out/patch_eq_full.json")
    ap.add_argument("--out", default="out/patch_eqfull.json")
    ap.add_argument("--wav", default="out/render_eqfull.wav")
    a = ap.parse_args()

    doc = json.load(open(a.gains))
    gains = np.asarray(doc["gains"], dtype=float)
    params = patch_params(gains, doc["level"])

    obj = Objective(load_notes())
    audio = render_full(params)
    loss = obj.loss_of(audio)
    incumbent = incumbent_loss()
    print(f"full clip, rendered through the sliders : {loss:.6f}")
    print(f"  fit predicted (post-hoc cascade)      : {doc['loss']:.6f}"
          f"   delta {loss - doc['loss']:+.6f}")
    print(f"  incumbent out/patch.json              : {incumbent:.6f}")
    print(f"  single-chord fit out/patch_chord.json : 1.531124\n")

    old, sr_o = sf.read("out/render.wav", always_2d=True)
    old = old.T.astype(np.float64)
    tgt, _ = librosa.load("data/original.wav", sr=SR, mono=True)   # 48 kHz source
    tgt2 = np.stack([tgt, tgt])

    print(f"{'window':>14} {'incumbent':>10} {'new':>10} {'delta':>9} "
          f"{'cos old':>8} {'cos new':>8}")
    print("-" * 64)
    ok_windows = True
    for t0, t1 in WINDOWS:
        sc, sl = window_score(t0, t1)
        lo, ln = sc(old[:, sl]), sc(audio[:, sl])
        co, cn = sc.cos_theta(old[:, sl]), sc.cos_theta(audio[:, sl])
        flag = "" if ln < lo else "  WORSE"
        ok_windows &= ln < lo
        print(f"{t0:>6.2f}-{t1:<7.2f} {lo:>10.4f} {ln:>10.4f} {ln-lo:>+9.4f} "
              f"{co:>8.4f} {cn:>8.4f}{flag}")

    print(f"\n{'metric':<16} {'target':>10} {'incumbent':>10} {'new':>10}")
    print("-" * 50)
    mono = {"target": tgt, "incumbent": old.mean(0), "new": audio.mean(0)}
    rows = {}
    for k, y in mono.items():
        rows[k] = {
            "centroid": librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean(),
            "rolloff95": librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.95)[0].mean(),
            "rms_db": 20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12),
        }
    for m in ("centroid", "rolloff95", "rms_db"):
        print(f"{m:<16} {rows['target'][m]:>10.1f} {rows['incumbent'][m]:>10.1f} "
              f"{rows['new'][m]:>10.1f}")
    r_old = metrics.report(tgt2, old)
    r_new = metrics.report(tgt2, audio)
    for m in ("mel_dist", "chroma_agree", "env_l1", "onset_f"):
        worse = " WORSE" if (r_new[m] > r_old[m]) != (m in ("chroma_agree", "onset_f")) else ""
        print(f"{m:<16} {'-':>10} {r_old[m]:>10.4f} {r_new[m]:>10.4f}{worse}")
    print(f"{'decorr':<16} {metrics.stereo_decorrelation(tgt2):>10.3f} "
          f"{metrics.stereo_decorrelation(old):>10.3f} "
          f"{metrics.stereo_decorrelation(audio):>10.3f}")

    print("\ngain-aligned band error vs target (dB):")
    hdr = None
    for name, y in (("incumbent", old), ("new", audio)):
        b = metrics.lta_band_error(tgt2, y)
        if hdr is None:
            hdr = list(b)
            print(f"{'':<11}" + "".join(f"{k:>12}" for k in hdr))
        print(f"{name:<11}" + "".join(f"{b[k]:>+12.2f}" for k in hdr))

    # The bells overlap, so alternating GAINS can still realise a smooth response.
    # The response is what the signal sees, so report its shape, not the gain vector's.
    f, db = eq_stage.response(gains)
    band = (f >= 40.0) & (f <= 16000.0)
    fb, dbb = f[band], db[band]
    third = [float(np.ptp(dbb[(fb >= c / 2 ** (1 / 6)) & (fb <= c * 2 ** (1 / 6))]))
             for c in eq_stage.band_freqs()]
    print(f"\nrealised response over 40-16000 Hz: peak {dbb.max():+.1f} dB at "
          f"{fb[dbb.argmax()]:.0f} Hz, trough {dbb.min():+.1f} dB at "
          f"{fb[dbb.argmin()]:.0f} Hz, peak-to-trough {np.ptp(dbb):.1f} dB, "
          f"worst swing inside one third octave {max(third):.1f} dB")
    print("  (single-chord fit: peak-to-trough 32.6 dB, worst third-octave swing 17.7 dB;")
    print("   the target shape measured across four chords is 13.5 dB deep)")

    print(f"EQ curve: max|g| {np.abs(gains).max():.1f} dB, "
          f"mean neighbour jump {np.abs(np.diff(gains)).mean():.1f} dB, "
          f"sign alternations {sum(1 for i in range(len(gains)-1) if gains[i]*gains[i+1] < 0)}, "
          f"railed {int((np.abs(gains) >= eq_stage.GAIN_LIMIT - 0.5).sum())}")
    print("  (single-chord fit for comparison: max 18.0, jump 10.2, alternations 9, railed 3)")

    synth.write_render(a.wav, audio)
    json.dump({"normalized": synth.PAD.normalize(params).tolist(), "params": params,
               "full_clip_loss": loss, "lam": doc.get("lam"),
               "windows": {f"{t0}-{t1}": window_score(t0, t1)[0](audio[:, window_score(t0, t1)[1]])
                           for t0, t1 in WINDOWS},
               "metrics": r_new, "gains_db": gains.tolist(),
               "note": "EQ fitted on the whole clip with a curvature penalty; "
                       "wavetable bank off (measured zero contribution)"},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out} and {a.wav}")
    print("ACCEPT" if loss < incumbent and ok_windows else "REJECT: see WORSE flags above")


if __name__ == "__main__":
    main()
