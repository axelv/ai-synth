"""Acceptance battery for the two-layer patch.

A lower loss is not sufficient here, and that is not a general principle but a lesson
this project paid for: a fit that improved the aggregate loss by 0.0135 was 0.153 WORSE
on one chord and regressed env_l1 from 0.104 to 0.189. So every check below has to pass,
and each one is something the fit did not directly optimise.

Everything is scored from the WAV on disk rather than from the array in memory, because
writing at soundfile's default PCM_16 once cost 0.089 of loss in this project without
changing a single number that was being reported.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import chord
import diagnose
import layers
import metrics
import synth
from stage2 import SR, Objective, W_SIDE, load_notes

WINDOWS = ((4.95, 7.45), (7.45, 10.40), (10.40, 13.35))
# The single-layer incumbent, measured. Every comparison below is against these.
REF = {"mono": 1.382293, "side": 2.6094, "total": 2.2956,
       "cos_mid": 0.7940, "cos_side": 0.5132, "width_rise": -1.8}


def window_score(t0: float, t1: float):
    """chord.WindowScore with moved bounds, without duplicating its loss definition."""
    old = (chord.WIN_T0, chord.WIN_T1)
    chord.WIN_T0, chord.WIN_T1 = t0, t1
    try:
        return chord.WindowScore(), slice(int(t0 * SR), int(t1 * SR))
    finally:
        chord.WIN_T0, chord.WIN_T1 = old


def side_band_error(a: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """Side-channel energy, render minus target, per band in dB."""
    import librosa
    n = min(a.shape[1], t.shape[1])
    out = {}
    for lo, hi in ((30, 90), (90, 300), (300, 1000), (1000, 4000), (4000, 16000)):
        def e(x):
            S = np.abs(librosa.stft(np.ascontiguousarray((x[0] - x[1])[:n]) / 2,
                                    n_fft=2048, hop_length=512))
            f = librosa.fft_frequencies(sr=SR, n_fft=2048)
            return float((S[(f >= lo) & (f < hi)] ** 2).sum())
        out[f"{lo}_{hi}"] = 10 * np.log10((e(a) + 1e-30) / (e(t) + 1e-30))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="out/render_layers.wav")
    ap.add_argument("--patch", default="out/patch_layers.json")
    a = ap.parse_args()

    tgt = diagnose.load("data/original.wav")
    new = diagnose.load(a.wav)
    old = diagnose.load("out/render.wav")
    n = min(tgt.shape[1], new.shape[1], old.shape[1])
    tgt, new, old = tgt[:, :n], new[:, :n], old[:, :n]

    obj = Objective(load_notes())
    d = obj.loss_parts(new, w_side=W_SIDE)
    ok: list[tuple[str, bool]] = []

    print("1. LOSS (scored from the file on disk)")
    print(f"   mono  {d['mono']:.4f}   incumbent {REF['mono']:.4f}")
    print(f"   side  {d['side']:.4f}   incumbent {REF['side']:.4f}")
    print(f"   total {d['total']:.4f}   incumbent {REF['total']:.4f}")
    ok.append(("total loss improves", d["total"] < REF["total"]))

    print("\n2. STEREO STRUCTURE (the point of the rebuild)")
    wt, wn, wo = (diagnose.width_rise(x) for x in (tgt, new, old))
    print(f"   width rise: target {wt:+.1f} dB   new {wn:+.1f} dB   incumbent {wo:+.1f} dB")
    ct = diagnose.channel_cos(new, tgt)
    print(f"   cos mid  {ct['mid']:.4f}   incumbent {REF['cos_mid']:.4f}")
    print(f"   cos side {ct['side']:.4f}   incumbent {REF['cos_side']:.4f}")
    ok.append(("width rise moves toward target", abs(wn - wt) < abs(wo - wt)))
    ok.append(("side cos improves", ct["side"] > REF["cos_side"]))
    ok.append(("mid cos does not regress", ct["mid"] >= REF["cos_mid"] - 0.005))

    print("\n   width by band (dB), target / incumbent / new:")
    wsT, wsO, wsN = (diagnose.width_spectrum(x) for x in (tgt, old, new))
    for k in list(wsT)[:9]:
        print(f"     {k:>10}  {wsT[k]:>+7.2f}  {wsO[k]:>+7.2f}  {wsN[k]:>+7.2f}")

    print("\n3. SIDE-CHANNEL ENERGY ERROR vs target (dB)")
    eo, en = side_band_error(old, tgt), side_band_error(new, tgt)
    worst_o = max(abs(v) for v in eo.values())
    worst_n = max(abs(v) for v in en.values())
    for k in eo:
        print(f"   {k:>10}  incumbent {eo[k]:>+7.2f}   new {en[k]:>+7.2f}")
    print(f"   worst |error|: incumbent {worst_o:.2f} dB, new {worst_n:.2f} dB")
    ok.append(("side energy error shrinks", worst_n < worst_o))

    print("\n4. PER-WINDOW (the gate that caught the single-chord comb)")
    allw = True
    for t0, t1 in WINDOWS:
        sc, sl = window_score(t0, t1)
        lo_, ln = sc(old[:, sl]), sc(new[:, sl])
        flag = "" if ln < lo_ else "   WORSE"
        allw &= ln < lo_
        print(f"   {t0:5.2f}-{t1:<6.2f} incumbent {lo_:.4f}  new {ln:.4f}  "
              f"{ln - lo_:+.4f}{flag}")
    ok.append(("all bend-free windows improve", allw))

    print("\n5. METRICS THE FIT NEVER SAW")
    ro, rn = metrics.report(tgt, old), metrics.report(tgt, new)
    for m, better_high in (("mel_dist", False), ("chroma_agree", True),
                           ("env_l1", False), ("onset_f", True)):
        good = (rn[m] > ro[m]) if better_high else (rn[m] < ro[m])
        print(f"   {m:<14} incumbent {ro[m]:.4f}   new {rn[m]:.4f}"
              f"   {'better' if good else 'WORSE'}")
        ok.append((f"{m} holds", good or abs(rn[m] - ro[m]) < 1e-3))

    print("\n" + "=" * 62)
    for name, passed in ok:
        print(f"   [{'PASS' if passed else 'FAIL'}] {name}")
    print("=" * 62)
    print("ACCEPT" if all(p for _, p in ok) else "REJECT")

    json.dump({"loss": d, "cos": ct, "width_rise": {"target": wt, "new": wn, "old": wo},
               "side_band_error": en, "metrics": rn,
               "checks": {k: bool(v) for k, v in ok}},
              open("out/verify_layers.json", "w"), indent=1)
    print("wrote out/verify_layers.json")


if __name__ == "__main__":
    main()
