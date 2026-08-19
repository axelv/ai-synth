"""Measure torch_fx against isolated Faust probes, stage by stage and end to end.

Every probe DSP is built from synth.DSP's own effect section, so the reference can
never drift from the DSP being ported. Nothing here is asserted: the numbers are
printed, good or bad, and the arrays go to out/fixtures/fx.npz so a later change
to torch_fx can be diffed against the same ground truth.
"""

from __future__ import annotations

import json
import math
import time

import auraloss
import librosa
import numpy as np
import torch
from scipy.signal import butter, sosfilt

import torch_fx
from faust_probe import render_fx, save_fixture
from synth import DSP

SR = 44100
DUR = 17.904
FIX_SEC = 4.0                                    # what gets written to the fixture
OCTAVES = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
REV_GRID = ((0.5637385749104125, 0.9909437230668987), (0.85, 0.4), (0.2, 0.0),
            (0.95, 0.9), (0.5, 0.5), (0.15, 1.0), (0.98, 0.0))

FX_SRC = 'import("stdfaust.lib");\n' + DSP.split("// ---------------- shared effects ----------------")[1]

_MRSTFT = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[512, 1024, 2048, 4096],
    hop_sizes=[128, 256, 512, 1024],
    win_lengths=[512, 1024, 2048, 4096],
    w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0,
)


def probe(process: str, params: dict[str, float], x: np.ndarray, tail: float = 0.0) -> np.ndarray:
    return render_fx(FX_SRC + f"\nprocess = {process};", params, x, tail=tail)


def rel_l2(ref: np.ndarray, got: np.ndarray) -> float:
    ref = np.asarray(ref, np.float64)
    got = np.asarray(got, np.float64)[..., : ref.shape[-1]]
    return float(np.linalg.norm(ref - got) / (np.linalg.norm(ref) + 1e-30))


def mrstft(ref: np.ndarray, got: np.ndarray) -> float:
    a = torch.from_numpy(np.asarray(ref, np.float32).mean(axis=0)).view(1, 1, -1)
    b = torch.from_numpy(np.asarray(got, np.float32).mean(axis=0)).view(1, 1, -1)
    n = min(a.shape[-1], b.shape[-1])
    with torch.no_grad():
        return float(_MRSTFT(b[..., :n], a[..., :n]))


def lr_corr(x: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(x, np.float64)[0], np.asarray(x, np.float64)[1])[0, 1])


def octave(x: np.ndarray, fc: float) -> np.ndarray:
    hi = min(fc * math.sqrt(2.0), 0.99 * SR / 2)
    sos = butter(4, [fc / math.sqrt(2.0) / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
    return sosfilt(sos, np.asarray(x, np.float64))


def band_energy_db(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    """Per-octave energy of `got` relative to `ref`, in dB. What the spectral loss sees."""
    return np.array([10.0 * np.log10((octave(got, fc) ** 2).sum() / (octave(ref, fc) ** 2).sum())
                     for fc in OCTAVES])


def rt60(x: np.ndarray, fc: float) -> float:
    """Schroeder backward integration in one octave band, fitted between -5 and -35 dB."""
    b = octave(x, fc)
    e = np.cumsum(b[::-1] ** 2)[::-1]
    db = 10.0 * np.log10(e / (e[0] + 1e-30) + 1e-30)
    i0, i1 = int(np.argmax(db <= -5.0)), int(np.argmax(db <= -35.0))
    if i1 <= i0 + 100:
        return float("nan")
    t = np.arange(i0, i1) / SR
    return float(-60.0 / np.polyfit(t, db[i0:i1], 1)[0])


def f32_phasor_serial(freq: float, n: int) -> np.ndarray:
    """os.phasor written the slow honest way: one float32 add per sample, as Faust does."""
    incr = np.float32(np.float32(freq) / np.float32(SR))
    p = np.float32(0.0)
    out = np.empty(n, np.float32)
    for i in range(n):
        out[i] = p
        p = np.float32(np.float32(p + incr) % np.float32(1.0))
    return out


def exact_phasor(freq: float, n: int, sr: int = SR) -> np.ndarray:
    """A drift-free phasor, to stand in for torch_fx.faust_phasor in gradient checks."""
    return np.float32(np.mod(np.arange(n, dtype=np.float64) * (freq / sr), 1.0))


def test_signal(n: int) -> np.ndarray:
    """Broadband stereo with the pad's spectral tilt, decorrelated channels, fixed seed."""
    rng = np.random.default_rng(31337)
    w = rng.standard_normal((2, n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    shape = 1.0 / np.sqrt(1.0 + (f / 300.0) ** 2)
    x = np.fft.irfft(np.fft.rfft(w, axis=-1) * shape, n=n, axis=-1)
    env = np.minimum(1.0, np.arange(n) / (0.2 * SR))
    return (0.3 * x / np.abs(x).max() * env).astype(np.float32)


def main() -> None:
    patch = json.load(open("out/patch.json"))["params"]
    fixture: dict[str, np.ndarray] = {}
    nfix = int(FIX_SEC * SR)

    def keep(name: str, ref: np.ndarray, got: np.ndarray) -> None:
        fixture[f"{name}_faust"] = np.asarray(ref, np.float32)[..., :nfix]
        fixture[f"{name}_torch"] = np.asarray(got, np.float32)[..., :nfix]

    n_short = int(6.0 * SR)
    xs = test_signal(n_short)
    ts = torch.from_numpy(xs)

    print("=" * 74)
    print("chorus   (relative L2 against the isolated Faust chorus, 6 s test signal)")
    for rate in (patch["chRate"], 0.6, 4.0, 0.05):
        fast = torch_fx.faust_phasor(rate, n_short)
        slow = f32_phasor_serial(rate, n_short)
        drift = (fast[-1] - exact_phasor(rate, n_short)[-1] + 0.5) % 1.0 - 0.5
        print(f"  phasor emulation at {rate:.4f} Hz vs a per-sample float32 loop: "
              f"max abs diff={np.abs(fast.astype(np.float64) - slow.astype(np.float64)).max():.1e}"
              f"   phase drift after 6 s={drift:+.2e} cycles")
    for rate, depth in ((patch["chRate"], patch["chDepth"]), (0.6, 1.0), (4.0, 0.5)):
        ref = probe("chorus", {"chRate": rate, "chDepth": depth}, xs)
        with torch.no_grad():
            got = torch_fx.chorus(ts, torch.tensor(rate), torch.tensor(depth)).numpy()
        torch_fx.faust_phasor, keep_phasor = exact_phasor, torch_fx.faust_phasor
        with torch.no_grad():
            nodrift = torch_fx.chorus(ts, torch.tensor(rate), torch.tensor(depth)).numpy()
        torch_fx.faust_phasor = keep_phasor
        print(f"  chRate={rate:.4f} chDepth={depth:.4f}: relL2={rel_l2(ref, got):.3e}   "
              f"with a drift-free phasor instead={rel_l2(ref, nodrift):.3e}")
        if abs(rate - patch["chRate"]) < 1e-9:
            keep("chorus", ref, got)

    print("=" * 74)
    print("ping-pong delay   (relative L2; taps = truncation of the feedback recursion)")
    for time_s, fb in ((patch["dlyTime"], patch["dlyFb"]), (0.05, 0.8), (0.2, 0.5), (1.0, 0.8)):
        ref = probe("pingpong", {"dlyTime": time_s, "dlyFb": fb}, xs)
        with torch.no_grad():
            got = torch_fx.pingpong(ts, torch.tensor(time_s), torch.tensor(fb)).numpy()
        errs = []
        for cap in (1, 4, 12, 40, 64):
            torch_fx.DLY_MAX_TAPS = cap
            with torch.no_grad():
                errs.append((cap, rel_l2(ref, torch_fx.pingpong(
                    ts, torch.tensor(time_s), torch.tensor(fb)).numpy())))
        torch_fx.DLY_MAX_TAPS = 64
        print(f"  dlyTime={time_s:.4f} dlyFb={fb:.4f}: relL2={rel_l2(ref, got):.3e}   "
              + "  ".join(f"taps<={c}:{e:.1e}" for c, e in errs))
        if abs(time_s - patch["dlyTime"]) < 1e-9:
            keep("delay", ref, got)

    print("=" * 74)
    print("tilt EQ   (relative L2 against fi.highshelf(2)+fi.lowshelf(2))")
    for tilt in (patch["tilt"], -1.0, 1.0, 0.0):
        ref = probe("par(i, 2, tiltEQ)", {"tilt": tilt}, xs)
        with torch.no_grad():
            got = torch_fx.tilt_eq(ts, torch.tensor(tilt)).numpy()
        print(f"  tilt={tilt:+.4f}: relL2={rel_l2(ref, got):.3e}")
        if abs(tilt - patch["tilt"]) < 1e-9:
            keep("tilt", ref, got)

    print("=" * 74)
    print("reverb   (approximation, not a port: zita is an 8x8 FDN)")
    imp = np.zeros((2, 8), np.float32)
    imp[:, 0] = 1.0
    n_ir = int(12.0 * SR)
    ir_in = torch.zeros(2, n_ir)
    ir_in[:, 0] = 1.0
    ratios: list[float] = []
    bands: list[np.ndarray] = []
    for size, damp in REV_GRID:
        zit = probe("re.zita_rev1_stereo(0, 200, 1200.0 + (1.0-revDamp)*8000.0, "
                    "revSize*8.0, revSize*4.0, ma.SR)",
                    {"revSize": size, "revDamp": damp}, imp, tail=12.0)
        with torch.no_grad():
            sur = torch_fx.reverb(ir_in, torch.tensor(size), torch.tensor(damp)).numpy()
        ez, es = float((zit.astype(np.float64) ** 2).sum()), float((sur.astype(np.float64) ** 2).sum())
        ratios.append(math.sqrt(es / ez))
        zm, sm = zit.mean(axis=0), sur.mean(axis=0)
        rz = [rt60(zm, fc) for fc in OCTAVES]
        rs = [rt60(sm, fc) for fc in OCTAVES]
        bands.append(band_energy_db(zm, sm))
        print(f"  revSize={size:.4f} revDamp={damp:.4f}  t60dc={size*8:.2f} t60m={size*4:.2f}"
              f"  f2={1200 + (1 - damp) * 8000:.0f}")
        print(f"    IR amplitude ratio surrogate/zita = {ratios[-1]:.3f}"
              f"    L/R corr  zita={lr_corr(zit):+.4f}  surrogate={lr_corr(sur):+.4f}")
        print("    RT60 zita     :", "  ".join(f"{v:5.2f}" for v in rz))
        print("    RT60 surrogate:", "  ".join(f"{v:5.2f}" for v in rs))
        print("    RT60 ratio    :", "  ".join(f"{b / a:5.2f}" for a, b in zip(rz, rs)))
        print("    band energy dB:", "  ".join(f"{v:+5.1f}" for v in bands[-1]))
        if abs(size - patch["revSize"]) < 1e-9:
            keep("reverb_ir", zit, sur)
    be = np.array(bands)
    print(f"  IR amplitude ratio over the grid: min={min(ratios):.3f} max={max(ratios):.3f}")
    print(f"  band energy error over the grid: rms={np.sqrt((be ** 2).mean()):.2f} dB  "
          f"max={np.abs(be).max():.2f} dB")

    zit = probe("re.zita_rev1_stereo(0, 200, 1200.0 + (1.0-revDamp)*8000.0, "
                "revSize*8.0, revSize*4.0, ma.SR)",
                {"revSize": patch["revSize"], "revDamp": patch["revDamp"]}, xs)
    with torch.no_grad():
        sur = torch_fx.reverb(ts, torch.tensor(patch["revSize"]),
                              torch.tensor(patch["revDamp"])).numpy()
    # a decorrelated-noise reverb can never match sample for sample, so the only
    # honest yardstick is what two independent realisations of the SAME reverb score
    seed = torch_fx.REV_SEED
    torch_fx.REV_SEED = seed + 1
    with torch.no_grad():
        sur2 = torch_fx.reverb(ts, torch.tensor(patch["revSize"]),
                               torch.tensor(patch["revDamp"])).numpy()
    torch_fx.REV_SEED = seed
    print(f"  on the 6 s test signal at the fitted patch: relL2={rel_l2(zit, sur):.3e}  "
          f"multi-res STFT={mrstft(zit, sur):.4f}")
    print(f"    floor from re-seeding the surrogate alone: relL2={rel_l2(sur, sur2):.3e}  "
          f"multi-res STFT={mrstft(sur, sur2):.4f}")
    print(f"    zita vs silence: multi-res STFT={mrstft(zit, np.zeros_like(zit) + 1e-9):.4f}")
    keep("reverb_wet", zit, sur)

    print("=" * 74)
    print("whole chain   (all fx sliders at out/patch.json, 17.904 s)")
    n_full = int(DUR * SR)
    xf = test_signal(n_full)
    tf = torch.from_numpy(xf)
    fx_names = ["chRate", "chDepth", "dlyTime", "dlyFb", "dlyWet",
                "revSize", "revDamp", "revWet", "tilt", "outGain"]
    fx = {k: patch[k] for k in fx_names}
    ref = probe("effect", fx, xf)
    p = {k: torch.tensor(v, dtype=torch.float32) for k, v in fx.items()}
    with torch.no_grad():
        got = torch_fx.effects(tf, p).numpy()
    print(f"  relL2={rel_l2(ref, got):.3e}   multi-res STFT={mrstft(ref, got):.4f}")
    print(f"  L/R corr  faust={lr_corr(ref):+.4f}  torch={lr_corr(got):+.4f}")
    print(f"  rms  faust={np.sqrt((ref.astype(np.float64)**2).mean()):.5f}  "
          f"torch={np.sqrt((got.astype(np.float64)**2).mean()):.5f}")
    keep("chain", ref, got)

    # the same chain with the reverb muted isolates how much of the end-to-end
    # divergence is the reverb approximation and how much is everything else
    dry = dict(fx, revWet=0.0)
    ref_dry = probe("effect", dry, xf)
    p_dry = {k: torch.tensor(v, dtype=torch.float32) for k, v in dry.items()}
    with torch.no_grad():
        got_dry = torch_fx.effects(tf, p_dry).numpy()
    print(f"  with revWet=0: relL2={rel_l2(ref_dry, got_dry):.3e}   "
          f"multi-res STFT={mrstft(ref_dry, got_dry):.4f}")
    keep("chain_dry", ref_dry, got_dry)

    # in the real synth the pre-effect channels are identical (process = ... <: _,_),
    # so all stereo width the render has must be made inside this chain
    xm = np.repeat(xf[:1], 2, axis=0)
    ref_m = probe("effect", fx, xm)
    with torch.no_grad():
        got_m = torch_fx.effects(torch.from_numpy(xm), p).numpy()
    print(f"  mono-identical input: L/R corr  faust={lr_corr(ref_m):+.4f}  "
          f"torch={lr_corr(got_m):+.4f}")
    orig, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    render, _ = librosa.load("out/render.wav", sr=SR, mono=False)
    print(f"  for reference: data/original.wav L/R corr={lr_corr(orig):+.4f}  "
          f"out/render.wav L/R corr={lr_corr(render):+.4f}")

    print("=" * 74)
    print("gradient   (autograd vs central difference of mean(y^2), 4 s signal)")
    print("  chRate is checked with the drift-free phasor: the float32 drift is a step")
    print("  function of chRate, so the difference quotient of the real one is noise")
    xg = torch.from_numpy(test_signal(int(4.0 * SR)))
    real_phasor = torch_fx.faust_phasor
    torch_fx.faust_phasor = exact_phasor
    p_grad = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True) for k, v in fx.items()}
    torch_fx.effects(xg, p_grad).pow(2).mean().backward()
    for k in fx_names:
        # chRate and dlyTime enter through a phase and a sample offset, so the
        # difference has to stay far inside one LFO radian / one delay sample
        h = {"chRate": 1e-4, "dlyTime": 1e-6}.get(k, max(abs(fx[k]), 1e-3) * 1e-3)
        fd = []
        for step in (h, -h):
            q = {n: torch.tensor(v + (step if n == k else 0.0), dtype=torch.float32)
                 for n, v in fx.items()}
            with torch.no_grad():
                fd.append(float(torch_fx.effects(xg, q).pow(2).mean()))
        num = (fd[0] - fd[1]) / (2 * h)
        ana = float(p_grad[k].grad)
        print(f"  d/d{k:<8} autograd={ana:+.5e}  finite-diff={num:+.5e}  "
              f"ratio={ana / num if abs(num) > 1e-14 else float('nan'):.3f}")
    torch_fx.faust_phasor = real_phasor

    with torch.no_grad():
        torch_fx.effects(tf, p)
    t0 = time.perf_counter()
    with torch.no_grad():
        torch_fx.effects(tf, p)
    fwd = time.perf_counter() - t0
    p_t = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True) for k, v in fx.items()}
    t0 = time.perf_counter()
    torch_fx.effects(tf, p_t).pow(2).mean().backward()
    print(f"  17.904 s chain on cpu, noise cache warm: forward {fwd:.2f}s, "
          f"forward+backward {time.perf_counter() - t0:.2f}s")

    fixture["test_signal"] = xf[:, :nfix]
    save_fixture("out/fixtures/fx.npz", **fixture)


if __name__ == "__main__":
    main()
