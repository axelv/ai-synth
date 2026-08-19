"""What is still audibly wrong, measured with things the fitting loss cannot see.

Why this exists. The stage-2 loss has stopped being informative. Calibrated against
deliberately-wrong controls, cos theta on this material runs 0.68 for the original with
its frames shuffled, 0.79 for the original delayed a full second, and 0.84 for the best
possible shaping oracle. The render now sits at 0.79, i.e. inside the band where the
metric cannot separate right from badly wrong, so a further tenth of a percent of loss
means nothing. Every measure here is therefore chosen to be one the fit never optimised.

Each block reports the render against the target AND against a control whose answer is
known, so a number can be read as good or bad rather than just recorded.
"""

from __future__ import annotations

import json

import librosa
import numpy as np
import soundfile as sf

SR = 44100
WINDOWS = ((4.95, 7.45), (7.45, 10.40), (10.40, 13.35), (16.25, 17.90))


def load(path: str) -> np.ndarray:
    a, sr = sf.read(path, always_2d=True)
    a = a.T.astype(np.float64)
    if sr != SR:                      # data/original.wav is 48 kHz
        a = np.stack([librosa.resample(ch, orig_sr=sr, target_sr=SR) for ch in a])
    return a


def align_level(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match broadband level, so nothing below is a level difference in disguise."""
    return x * np.sqrt((ref ** 2).mean() / ((x ** 2).mean() + 1e-30))


# ---------------------------------------------------------------- measures

def attack_time(y: np.ndarray, sr: int = SR) -> float:
    """Median seconds from a MIDI note-on to 90% of the following peak.

    Anchored to the frozen transcription, NOT to onset detection. Detected onsets with
    backtrack land part-way up a pad's swell, where the RMS is already near its peak, so
    the measured rise collapses to zero for every signal and the measure says nothing.
    That is not hypothetical: it read exactly 0.000 for two renders whose true 90% rise
    times were 0.31 s and 0.26 s.
    """
    from stage2 import load_notes
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    out = []
    for _, _, start, dur in load_notes():
        if dur < 0.8:
            continue
        f0 = int(start * sr / 256)
        seg = rms[f0:f0 + int(1.5 * sr / 256)]
        if len(seg) < 20 or seg.max() <= 0:
            continue
        out.append(float(np.argmax(seg >= 0.9 * seg.max())) * 256 / sr)
    return float(np.median(out)) if out else float("nan")


def flux(y: np.ndarray, sr: int = SR) -> float:
    """Mean spectral flux: how much the timbre moves per frame. A static patch and a
    moving one can share a long-term average spectrum and sound nothing alike."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S = S / (S.sum(0, keepdims=True) + 1e-12)
    return float(np.abs(np.diff(S, axis=1)).sum(0).mean())


def roughness(y: np.ndarray, sr: int = SR) -> float:
    """Energy of the 20-200 Hz amplitude modulation of the 200-2000 Hz band.

    This is the beating between detuned partials, i.e. what makes a supersaw sound thick
    rather than clean. The magnitude STFT loss is blind to it at these hop sizes.
    """
    b = librosa.stft(y, n_fft=2048, hop_length=64)
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    m = (f >= 200) & (f <= 2000)
    env = np.abs(b[m]).sum(0)
    env = env - env.mean()
    P = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    fr = np.fft.rfftfreq(len(env), 512 / sr)      # hop 64 -> frame rate sr/64
    band = (fr >= 20) & (fr <= 200)
    return float(10 * np.log10(P[band].sum() / (P.sum() + 1e-30) + 1e-30))


def stereo_width(a: np.ndarray) -> float:
    """Side over mid energy in dB. The loss is mono, so this is never optimised."""
    if a.shape[0] < 2:
        return float("nan")
    mid, side = (a[0] + a[1]) / 2, (a[0] - a[1]) / 2
    return float(10 * np.log10(((side ** 2).mean() + 1e-30) / ((mid ** 2).mean() + 1e-30)))


def hf_decay(y: np.ndarray, sr: int = SR) -> float:
    """Slope of 4-12 kHz energy over the clip, dB per second: does the air fade like
    the reference's, or does it sit flat because a static EQ put it there."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    e = 20 * np.log10(S[(f >= 4000) & (f <= 12000)].mean(0) + 1e-12)
    t = np.arange(len(e)) * 512 / sr
    keep = e > e.max() - 40
    return float(np.polyfit(t[keep], e[keep], 1)[0])


MEASURES = {
    "attack_s": lambda a: attack_time(a.mean(0)),
    "flux": lambda a: flux(a.mean(0)),
    "roughness_db": lambda a: roughness(a.mean(0)),
    "width_db": stereo_width,
    "hf_decay_db_s": lambda a: hf_decay(a.mean(0)),
}


def main() -> None:
    tgt = load("data/original.wav")
    new = load("out/render.wav")
    old = load("out/render_pretimbre.wav")
    n = min(tgt.shape[1], new.shape[1], old.shape[1])
    tgt, new, old = tgt[:, :n], new[:, :n], old[:, :n]
    new, old = align_level(new, tgt), align_level(old, tgt)

    rows = {"target": tgt, "pre-EQ": old, "delivered": new}
    print("whole clip, level-aligned:\n")
    print(f"{'measure':<16}" + "".join(f"{k:>12}" for k in rows) + f"{'verdict':>26}")
    print("-" * (16 + 12 * len(rows) + 26))
    report: dict[str, dict[str, float]] = {}
    for mname, fn in MEASURES.items():
        vals = {k: float(fn(v)) for k, v in rows.items()}
        report[mname] = vals
        d_new = vals["delivered"] - vals["target"]
        d_old = vals["pre-EQ"] - vals["target"]
        better = abs(d_new) < abs(d_old)
        verdict = (f"{d_new:+.3f} vs target "
                   f"({'closer' if better else 'further'} than pre-EQ)")
        print(f"{mname:<16}" + "".join(f"{vals[k]:>12.3f}" for k in rows) + f"{verdict:>26}")

    print("\nper bend-free window, delivered minus target:\n")
    print(f"{'window':>14}" + "".join(f"{m:>15}" for m in MEASURES))
    print("-" * (14 + 15 * len(MEASURES)))
    for t0, t1 in WINDOWS:
        sl = slice(int(t0 * SR), int(t1 * SR))
        tw = tgt[:, sl]
        nw = align_level(new[:, sl], tw)
        line = "".join(f"{float(fn(nw)) - float(fn(tw)):>15.3f}" for fn in MEASURES.values())
        print(f"{t0:>6.2f}-{t1:<7.2f}" + line)

    json.dump(report, open("out/diagnose.json", "w"), indent=1)
    print("\nwrote out/diagnose.json")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- stereo structure

WIDTH_EDGES = (30, 45, 65, 90, 125, 175, 250, 350, 500, 700, 1000,
               1400, 2000, 3000, 4500, 7000, 11000, 16000)


def width_spectrum(a: np.ndarray, nfft: int = 8192,
                   edges: tuple = WIDTH_EDGES) -> dict[str, float]:
    """Side over mid energy in dB, per band. The single most diagnostic measure here.

    The recording climbs 18.6 dB from -23.80 at 30-45 Hz to +0.40 at 500-700 Hz, i.e.
    a dead-centre bass under a fully decorrelated pad. A single voice path through one
    stereo chain is flat across the same span whatever its parameters, so this is the
    measure that says whether the two-layer structure is actually there.
    """
    if a.shape[0] < 2:
        return {}
    L = librosa.stft(np.ascontiguousarray(a[0]), n_fft=nfft, hop_length=nfft // 4)
    R = librosa.stft(np.ascontiguousarray(a[1]), n_fft=nfft, hop_length=nfft // 4)
    mid, side = (L + R) / 2, (L - R) / 2
    f = librosa.fft_frequencies(sr=SR, n_fft=nfft)
    out: dict[str, float] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        num = float((np.abs(side[m]) ** 2).mean())
        den = float((np.abs(mid[m]) ** 2).mean())
        out[f"{lo}_{hi}"] = 10 * np.log10((num + 1e-30) / (den + 1e-30))
    return out


def width_rise(a: np.ndarray) -> float:
    """How much wider the pad region is than the bass region, in dB. Target 18.6."""
    w = width_spectrum(a)
    lowk = [k for k in w if int(k.split("_")[1]) <= 90]
    hik = [k for k in w if 700 <= int(k.split("_")[0]) < 3000]
    return float(np.mean([w[k] for k in hik]) - np.mean([w[k] for k in lowk]))


def channel_cos(a: np.ndarray, target: np.ndarray, nfft: int = 2048) -> dict[str, float]:
    """Scale-invariant spectral agreement on mid and side separately.

    Reported apart because they diverge sharply: the delivered single-layer render sits
    at 0.79 on mid but only 0.51 on side, which is what identified the stereo image as
    where the remaining error lives.
    """
    n = min(a.shape[1], target.shape[1])

    def mag(x):
        return np.abs(librosa.stft(np.ascontiguousarray(x[:n]), n_fft=nfft,
                                   hop_length=nfft // 4)).astype(np.float64)

    def cos(A, B):
        m = min(A.shape[1], B.shape[1])
        A, B = A[:, :m], B[:, :m]
        return float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-30))

    out = {}
    for name, idx in (("mid", 1), ("side", -1)):
        out[name] = cos(mag((target[0] + idx * target[1]) / 2),
                        mag((a[0] + idx * a[1]) / 2))
    return out
