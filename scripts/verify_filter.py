"""Measure torch_filter.tv_lowpass against real Faust output. Nothing is assumed.

Faust probing is slow, so the Faust side of every check is rendered once into
out/fixtures/filter.npz and reused: the default run is fixture-only and re-measures
the torch side against those stored arrays. Pass --probe to re-render Faust.

Three checks, in increasing order of how much they matter:

  1. static fc: impulse response and swept-sine magnitude response at a grid of
     (fc, q). This is the coefficient test. It also compares against a double
     precision scipy biquad cascade, which separates *our* block-truncation error
     from Faust's own single-precision noise: at low fc the Faust reference is the
     less accurate of the two.
  2. moving fc: the real fitted filter envelope from out/patch.json driving the
     cutoff, with the fc curve itself read out of Faust so the comparison isolates
     the filter from the envelope module.
  3. adversarial fc: fA = 0.005 s with envAmt = 6000 Hz, which sweeps the cutoff
     across the whole range inside a couple of blocks. This is where holding fc
     constant per block is least defensible, so it is measured explicitly, plus a
     harsher variant with fD also at its box minimum.

A whole-signal L2 flatters the adversarial case, because a 5 ms attack is 0.2
percent of a 3 s clip and the rest of the clip has a nearly static cutoff. So every
moving case also reports the error inside the transient alone and the worst 20 ms
window anywhere in the signal, both normalised by the global reference RMS.

Structural checks cover failure modes the Faust comparison cannot see: batching
over voices must be bit-identical to filtering each voice alone, the analytic
gradients must match float64 central finite differences, and the module must run on
MPS as well as CPU. The FFT-length cap is swept separately, because it trades
low-fc accuracy against a spectrum buffer that scales with it.

Deep-stopband dB errors are reported against an explicit relative floor. Comparing
magnitudes 120 dB below the passband is meaningless for both references (Faust runs
in float32, and our FFT length bounds the aliasing floor), so the table reports the
error down to -60 dB and -80 dB and says so rather than quoting one flattering number.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import auraloss
import numpy as np
import scipy.signal as sps
import torch

import faust_probe as fp
from metrics import mel_dist
from torch_filter import BLOCK, MAX_NFFT, _fft_len, tv_lowpass

SR = 44100
KBD_REF = 261.6255
FIXTURE = "out/fixtures/filter.npz"

STATIC_DSP = """
import("stdfaust.lib");
fc = hslider("fc", 1000, 20, 20000, 0.001);
q  = hslider("q", 1.0, 0.1, 20, 0.001);
process = fi.resonlp(fc, q, 1.0) : fi.lowpass(2, fc);
"""

# The cutoff path copied verbatim from synth.py, with the gate synthesised from
# ba.time so no polyphony layer is involved.
SWEEP_HEAD = """
import("stdfaust.lib");
hold = hslider("hold", 1.0, 0.001, 20, 0.000001);
gate = ba.time < int(hold * ma.SR);
freq   = hslider("freq", 440, 20, 8000, 0.001);
cutoff = hslider("cutoff", 2000, 60, 12000, 0.01);
reso   = hslider("reso", 1.0, 0.5, 12, 0.001);
envAmt = hslider("envAmt", 0.0, -4000, 8000, 0.001);
kbdTrk = hslider("kbdTrk", 0.3, 0, 1, 0.001);
fA = hslider("fA", 0.3, 0.001, 4, 0.000001);
fD = hslider("fD", 0.5, 0.001, 4, 0.000001);
fS = hslider("fS", 0.6, 0, 1, 0.000001);
aR = hslider("aR", 1.2, 0.01, 6, 0.000001);
fenv = en.adsr(fA, fD, fS, aR, gate);
fc = max(30.0, min(16000.0, cutoff * pow(freq / 261.6255, kbdTrk) + envAmt * fenv));
"""
SWEEP_FC_DSP = SWEEP_HEAD + "process = fc / 16000.0;\n"
SWEEP_FILT_DSP = SWEEP_HEAD + "process = fi.resonlp(fc, reso, 1.0) : fi.lowpass(2, fc);\n"

STATIC_FC = (80.0, 400.0, 1200.0, 5000.0)
STATIC_Q = (0.5, 1.2, 4.0, 8.0)
STATIC_BLOCKS = (128, 256, 512, 1024)
BLOCKS = (128, 256, 512, 1024, 2048)
IMP_N = 32768
SWEEP_N = 4 * SR
# Each moving case: name, overrides on the fitted params, carrier freq, duration, gate.
MOVING = (
    ("fit_110", {}, 110.0, 3.0, 1.5),
    ("fit_440", {}, 440.0, 3.0, 1.5),
    ("adv_220", {"fA": 0.005, "envAmt": 6000.0}, 220.0, 3.0, 1.5),
    ("adv2_220", {"fA": 0.005, "fD": 0.02, "envAmt": 6000.0}, 220.0, 3.0, 1.5),
)

_mrstft = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[512, 1024, 2048, 4096],
    hop_sizes=[128, 256, 512, 1024],
    win_lengths=[512, 1024, 2048, 4096],
    w_sc=1.0,
    w_log_mag=1.0,
    w_lin_mag=0.0,
)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def scipy_cascade(x: np.ndarray, fc: float, q: float, sr: int = SR) -> np.ndarray:
    """Exact float64 reference for the static case, straight from the tf2s formulas."""
    c = 1.0 / np.tan(np.pi * fc / sr)
    csq = c * c
    y = np.asarray(x, dtype=np.float64)
    for a1 in (1.0 / q, np.sqrt(2.0)):
        d = 1.0 + a1 * c + csq
        b = np.array([1.0, 2.0, 1.0]) / d
        a = np.array([1.0, 2.0 * (1.0 - csq) / d, (1.0 - a1 * c + csq) / d])
        y = sps.lfilter(b, a, y)
    return y


def max_db_error(ref: np.ndarray, est: np.ndarray, floor_db: float,
                 lo: float = 20.0, hi: float = 16000.0) -> float:
    """Worst |dB| gap between two spectra, over bins within floor_db of ref's peak."""
    n = len(ref)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    dref = 20.0 * np.log10(np.abs(np.fft.rfft(ref)) + 1e-300)
    dest = 20.0 * np.log10(np.abs(np.fft.rfft(est)) + 1e-300)
    m = (f >= lo) & (f <= hi) & (dref > dref.max() - floor_db)
    if not m.any():
        return float("nan")
    return float(np.abs(dest[m] - dref[m]).max())


def worst_window(ref: np.ndarray, est: np.ndarray, win: float = 0.02,
                 sr: int = SR) -> tuple[float, float]:
    """Worst windowed error RMS relative to the *global* reference RMS, and its time.

    Normalising per window would divide by near-silence; normalising globally keeps
    the number comparable across windows and answers the question that matters, how
    large the local error is next to the signal the loss actually sees.
    """
    w = int(win * sr)
    m = len(ref) // w
    r = ref[: m * w].reshape(m, w)
    e = est[: m * w].reshape(m, w)
    err = np.sqrt(((e - r) ** 2).mean(axis=1)) / (np.sqrt((r**2).mean()) + 1e-30)
    i = int(err.argmax())
    return float(err[i]), i * win


def torch_static(x: np.ndarray, fc: float, q: float, block: int,
                 max_nfft: int = MAX_NFFT) -> np.ndarray:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).view(1, -1)
    fct = torch.full_like(xt, fc)
    y = tv_lowpass(xt, fct, torch.tensor(float(q)), block=block, max_nfft=max_nfft)
    return y[0].numpy().astype(np.float64)


def check_static(fx: dict[str, np.ndarray], probe: bool) -> list[dict]:
    imp = fp.impulse(IMP_N)
    swp = fp.sweep(SWEEP_N)
    rows: list[dict] = []
    print("\n=== 1. static fc (impulse + sweep) ===")
    print(f"{'fc':>7} {'q':>4} {'blk':>5} {'nfft':>6} {'imp relL2':>10} {'vs f64':>9} "
          f"{'imp dB-60':>10} {'imp dB-80':>10} {'swp relL2':>10} {'swp dB-60':>10}")
    for fc in STATIC_FC:
        for q in STATIC_Q:
            key = f"{int(fc)}_{q}"
            if probe:
                fx[f"imp_faust_{key}"] = fp.render_fx(STATIC_DSP, {"fc": fc, "q": q}, imp)[0]
                fx[f"swp_faust_{key}"] = fp.render_fx(STATIC_DSP, {"fc": fc, "q": q}, swp)[0]
            y_imp = fx[f"imp_faust_{key}"].astype(np.float64)
            y_swp = fx[f"swp_faust_{key}"].astype(np.float64)
            ref_imp = scipy_cascade(imp[0], fc, q)
            for block in STATIC_BLOCKS:
                t_imp = torch_static(imp[0], fc, q, block)
                t_swp = torch_static(swp, fc, q, block)
                row = {
                    "test": f"static fc={fc:g} q={q:g} block={block}",
                    "nfft": _fft_len(block, fc, q, SR, MAX_NFFT),
                    "imp_rel_l2_vs_faust": rel_l2(t_imp, y_imp),
                    "imp_rel_l2_vs_float64": rel_l2(t_imp, ref_imp),
                    "imp_max_db_60": max_db_error(y_imp, t_imp, 60.0),
                    "imp_max_db_80": max_db_error(y_imp, t_imp, 80.0),
                    "swp_rel_l2_vs_faust": rel_l2(t_swp, y_swp),
                    "swp_max_db_60": max_db_error(y_swp, t_swp, 60.0),
                    "swp_max_db_80": max_db_error(y_swp, t_swp, 80.0),
                }
                rows.append(row)
                print(f"{fc:7.0f} {q:4.1f} {block:5d} {row['nfft']:6d} "
                      f"{row['imp_rel_l2_vs_faust']:10.2e} {row['imp_rel_l2_vs_float64']:9.2e} "
                      f"{row['imp_max_db_60']:10.3f} {row['imp_max_db_80']:10.3f} "
                      f"{row['swp_rel_l2_vs_faust']:10.2e} {row['swp_max_db_60']:10.3f}")
    return rows


def check_nfft_cap() -> list[dict]:
    """What a tighter FFT-length cap costs, at the (fc, q) corners that need it most.

    nfft is chosen from the pole decay, so low fc with high q asks for the longest
    FFT and therefore the largest spectrum buffer. The cap is a memory knob; this
    measures the accuracy it buys. The reference here is the float64 scipy cascade
    rather than Faust, because the worst corner the search box allows (cutoff at its
    minimum 120 Hz keyboard-tracked down to the 30 Hz clamp, reso at 12) has no Faust
    fixture, and for static fc the double-precision recursion is exact anyway. The
    static grid above is what ties these coefficients to Faust.
    """
    imp = fp.impulse(IMP_N)[0]
    block = 1024
    rows: list[dict] = []
    print(f"\n=== FFT-length cap sweep vs float64 cascade (block={block}) ===")
    print(f"{'fc':>5} {'q':>5} {'max_nfft':>9} {'nfft':>6} {'imp relL2':>10} "
          f"{'spec GB at V=29,N=789566':>26}")
    for fc, q in ((80.0, 8.0), (30.0, 12.0)):
        ref = scipy_cascade(imp, fc, q)
        for cap in (2048, 4096, 8192, 16384, 32768):
            est = torch_static(imp, fc, q, block, max_nfft=cap)
            nfft = _fft_len(block, fc, q, SR, cap)
            frames = 29 * ((789566 + 2 * block) // (block // 2))
            gb = frames * (nfft // 2 + 1) * 8 / 1e9
            rows.append({"test": f"nfft cap {cap} fc={fc:g} q={q:g}", "nfft": nfft,
                         "imp_rel_l2_vs_float64": rel_l2(est, ref), "spec_gb": gb})
            print(f"{fc:5.0f} {q:5.1f} {cap:9d} {nfft:6d} {rel_l2(est, ref):10.2e} {gb:26.2f}")
    return rows


def naive_saw(n: int, freq: float, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    return (0.5 * (2.0 * (t * freq % 1.0) - 1.0)).astype(np.float32)


def moving_case(name: str, params: dict[str, float], freq: float, dur: float,
                hold: float, fx: dict[str, np.ndarray], blocks: tuple[int, ...],
                probe: bool) -> list[dict]:
    n = int(dur * SR)
    if probe:
        pr = dict(params)
        pr["freq"] = freq
        pr["hold"] = hold
        fc_pr = {k: v for k, v in pr.items() if k != "reso"}  # faust prunes the unused slider
        fx[f"mov_{name}_x"] = naive_saw(n, freq)
        fx[f"mov_{name}_fc"] = (
            fp.render_gen(SWEEP_FC_DSP, fc_pr, dur)[0][:n] * 16000.0).astype(np.float32)
        fx[f"mov_{name}_faust"] = fp.render_fx(SWEEP_FILT_DSP, pr, fx[f"mov_{name}_x"])[0][:n]
    x = fx[f"mov_{name}_x"].astype(np.float32)
    fc = fx[f"mov_{name}_fc"].astype(np.float64)
    y = fx[f"mov_{name}_faust"].astype(np.float64)

    xt = torch.from_numpy(x).view(1, -1)
    fct = torch.from_numpy(fc.astype(np.float32)).view(1, -1)
    qt = torch.tensor(float(params["reso"]))
    yt_ref = torch.from_numpy(y.astype(np.float32)).view(1, 1, -1)
    # The transient is the attack plus the decay, the only stretch where fc moves
    # fast enough for the block-constant assumption to be doing any work.
    tr = int(min(dur, 4.0 * (params["fA"] + params["fD"])) * SR)

    rows: list[dict] = []
    print(f"\n--- moving fc: {name} (freq={freq:g}, fc {fc.min():.0f}..{fc.max():.0f} Hz, "
          f"q={params['reso']:.3f}, transient 0..{tr / SR * 1e3:.0f} ms) ---")
    print(f"{'blk':>5} {'nfft':>6} {'relL2':>9} {'mrstft':>9} {'mel_dist':>9} "
          f"{'trans relL2':>11} {'worst 20ms':>10} {'at s':>6} {'fwd s':>8}")
    for block in blocks:
        t0 = time.perf_counter()
        est = tv_lowpass(xt, fct, qt, block=block)
        dt = time.perf_counter() - t0
        e = est[0].numpy().astype(np.float64)
        wrst, at = worst_window(y, e)
        row = {
            "test": f"moving {name} block={block}",
            "nfft": _fft_len(block, float(fc.min()), float(params["reso"]), SR, MAX_NFFT),
            "rel_l2": rel_l2(e, y),
            "mrstft": float(_mrstft(est.view(1, 1, -1), yt_ref)),
            "mel_dist": mel_dist(e, y),
            "transient_rel_l2": rel_l2(e[:tr], y[:tr]),
            "worst_20ms_rel_rms": wrst,
            "worst_20ms_at_s": at,
            "fwd_s": dt,
        }
        rows.append(row)
        print(f"{block:5d} {row['nfft']:6d} {row['rel_l2']:9.2e} {row['mrstft']:9.4f} "
              f"{row['mel_dist']:9.4f} {row['transient_rel_l2']:11.2e} "
              f"{wrst:10.2e} {at:6.2f} {dt:8.3f}")
        if block == 512:
            # guards against a timeline offset between render_gen (fc curve) and
            # render_fx (filtered audio): the best lag must be 0.
            best = min(range(-64, 65), key=lambda k: rel_l2(np.roll(e, k), y))
            print(f"      best lag = {best} samples (0 means the two probes are aligned)")
    return rows


V_REAL, N_REAL = 29, 789566
TIME_LIMIT_S = 300.0


def time_one(fc_lo: float, fc_hi: float, q: float, device: str, block: int,
             reps: int = 2) -> dict:
    """Best-of-reps wall time for one (device, block) at the real render shape."""
    dev = torch.device(device)
    sync = torch.mps.synchronize if device == "mps" else lambda: None
    torch.manual_seed(0)
    x = (torch.randn(V_REAL, N_REAL) * 0.1).to(dev)
    ramp = torch.linspace(0.0, 1.0, N_REAL, device=dev).expand(V_REAL, N_REAL)
    fwd, bwd = float("inf"), float("inf")
    for _ in range(reps):
        fc = (fc_lo + (fc_hi - fc_lo) * ramp).clone().requires_grad_(True)
        qt = torch.tensor(q, device=dev, requires_grad=True)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            tv_lowpass(x, fc.detach(), qt.detach(), block=block)
        sync()
        fwd = min(fwd, time.perf_counter() - t0)
        t0 = time.perf_counter()
        tv_lowpass(x, fc, qt, block=block).pow(2).mean().backward()
        sync()
        bwd = min(bwd, time.perf_counter() - t0)
    nfft = _fft_len(block, fc_lo, q, SR, MAX_NFFT)
    frames = V_REAL * ((N_REAL + 2 * block) // (block // 2))
    return {"test": f"timing {device} block={block}", "nfft": nfft, "fwd_s": fwd,
            "fwd_bwd_s": bwd, "spec_mb": frames * (nfft // 2 + 1) * 8 / 1e6}


def check_timing(fc_lo: float, fc_hi: float, q: float, devices: tuple[str, ...]) -> list[dict]:
    """Wall time per (device, block), each measured in its own process.

    In-process this sweep is not measurable: the spectrum buffer is hundreds of MB
    per config, the allocator holds on to it, and a first attempt at running the
    whole suite in one process drove the machine into swap and stalled for a quarter
    of an hour inside a Metal command buffer. One subprocess per config both frees
    the memory and stops a single pathological config from poisoning the others.
    """
    print(f"\n=== block-size cost at the real render size (V={V_REAL}, N={N_REAL}) ===")
    print(f"{'device':>7} {'blk':>5} {'nfft':>6} {'fwd s':>8} {'fwd+bwd s':>10} {'spec MB':>9}")
    rows: list[dict] = []
    for device in devices:
        for block in BLOCKS:
            cmd = [sys.executable, __file__, "--time-one", device, str(block),
                   str(fc_lo), str(fc_hi), str(q)]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=TIME_LIMIT_S, check=True)
                row = json.loads(res.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                row = {"test": f"timing {device} block={block}",
                       "status": f"did not finish within {TIME_LIMIT_S:.0f} s"}
            except subprocess.CalledProcessError as exc:
                row = {"test": f"timing {device} block={block}",
                       "status": "failed", "error": exc.stderr[-200:]}
            rows.append(row)
            if "fwd_s" in row:
                print(f"{device:>7} {block:5d} {row['nfft']:6d} {row['fwd_s']:8.3f} "
                      f"{row['fwd_bwd_s']:10.3f} {row['spec_mb']:9.0f}")
            else:
                print(f"{device:>7} {block:5d} {row['status']}")
    return rows


def check_gradients(fc_curve: np.ndarray, q: float) -> dict:
    n = len(fc_curve)
    x = torch.from_numpy(naive_saw(n, 220.0)).view(1, -1)
    fc = torch.from_numpy(fc_curve.astype(np.float32)).view(1, -1).requires_grad_(True)
    qt = torch.tensor(q, requires_grad=True)
    loss = tv_lowpass(x, fc, qt).pow(2).mean()
    loss.backward()
    gq = float(qt.grad)
    gfc = fc.grad
    out = {
        "test": "gradients",
        "grad_q": gq,
        "grad_q_finite_nonzero": bool(np.isfinite(gq) and gq != 0.0),
        "grad_fc_absmean": float(gfc.abs().mean()),
        "grad_fc_finite": bool(torch.isfinite(gfc).all()),
        "grad_fc_nonzero_frac": float((gfc != 0).float().mean()),
    }
    print("\n=== gradients ===")
    print(f"d/dq = {gq:.6g} (finite and non-zero: {out['grad_q_finite_nonzero']})")
    print(f"d/dfc: mean|g| = {out['grad_fc_absmean']:.3e}, all finite: {out['grad_fc_finite']}, "
          f"non-zero fraction: {out['grad_fc_nonzero_frac']:.3f}")
    return out


def check_grad_direction(fc_curve: np.ndarray, q: float) -> list[dict]:
    """Central finite differences in float64. A gradient that flows but points the
    wrong way is worse than no gradient, so the analytic value is checked against
    the loss surface itself, not merely tested for being non-zero."""
    n = len(fc_curve)
    x = torch.from_numpy(naive_saw(n, 220.0).astype(np.float64)).view(1, -1)
    fc0 = torch.from_numpy(fc_curve.astype(np.float64)).view(1, -1)

    def loss_of(fcv: torch.Tensor, qv: torch.Tensor) -> torch.Tensor:
        return tv_lowpass(x, fcv, qv).pow(2).mean()

    fc = fc0.clone().requires_grad_(True)
    qt = torch.tensor(float(q), dtype=torch.float64, requires_grad=True)
    loss_of(fc, qt).backward()
    ana_q = float(qt.grad)
    # d/deps of a uniform shift of the whole cutoff curve, so one scalar FD probe
    # exercises every element of the fc gradient at once.
    ana_fc = float(fc.grad.sum())

    rows: list[dict] = []
    print("\n=== gradient vs central finite differences (float64) ===")
    for name, ana, h in (("q", ana_q, 1e-4), ("fc_shift", ana_fc, 1e-3)):
        with torch.no_grad():
            if name == "q":
                lp = loss_of(fc0, torch.tensor(q + h, dtype=torch.float64))
                lm = loss_of(fc0, torch.tensor(q - h, dtype=torch.float64))
            else:
                lp = loss_of(fc0 + h, torch.tensor(float(q), dtype=torch.float64))
                lm = loss_of(fc0 - h, torch.tensor(float(q), dtype=torch.float64))
        fd = float((lp - lm) / (2.0 * h))
        rel = abs(ana - fd) / (abs(fd) + 1e-300)
        rows.append({"test": f"fd grad {name}", "analytic": ana, "finite_diff": fd,
                     "rel_err": rel, "pass": bool(rel < 1e-3)})
        print(f"d/d{name:9s} analytic {ana: .6e}  fd {fd: .6e}  rel err {rel:.2e}")
    return rows


def check_batch(fx: dict[str, np.ndarray], q: float) -> list[dict]:
    """Batching over voices must be exactly equivalent to filtering each voice alone.

    The overlap-add folds a (V*M, nfft) buffer back through a view, which is where
    a voice/frame index swap would hide. Nothing else in the suite runs V > 1.
    """
    names = ["mov_fit_110", "mov_fit_440", "mov_adv_220"]
    xs = np.stack([fx[f"{k}_x"] for k in names]).astype(np.float32)
    fcs = np.stack([fx[f"{k}_fc"] for k in names]).astype(np.float32)
    qt = torch.tensor(float(q))
    batched = tv_lowpass(torch.from_numpy(xs), torch.from_numpy(fcs), qt)
    rows: list[dict] = []
    print("\n=== batching over voices (V=3 vs V=1) ===")
    for i, k in enumerate(names):
        alone = tv_lowpass(torch.from_numpy(xs[i : i + 1]), torch.from_numpy(fcs[i : i + 1]), qt)
        d = float((batched[i] - alone[0]).abs().max())
        rows.append({"test": f"batch row {k}", "max_abs_diff": d, "pass": bool(d == 0.0)})
        print(f"{k:14s} max|batched - alone| = {d:.3e}")
    return rows


def check_device(fx: dict[str, np.ndarray], q: float) -> list[dict]:
    """Does the module run on MPS, and does it agree with CPU there?"""
    rows: list[dict] = []
    print("\n=== device support ===")
    x = torch.from_numpy(fx["mov_fit_440_x"].astype(np.float32)).view(1, -1)
    fc = torch.from_numpy(fx["mov_fit_440_fc"].astype(np.float32)).view(1, -1)
    qt = torch.tensor(float(q))
    ref = tv_lowpass(x, fc, qt)
    if not torch.backends.mps.is_available():
        print("mps: unavailable, not tested")
        return [{"test": "device mps", "status": "unavailable"}]
    try:
        got = tv_lowpass(x.to("mps"), fc.to("mps"), qt.to("mps")).cpu()
        d = rel_l2(got[0].numpy(), ref[0].numpy())
        rows.append({"test": "device mps", "status": "ok", "rel_l2_vs_cpu": d})
        print(f"mps: ok, relL2 vs cpu = {d:.3e}")
    except (RuntimeError, NotImplementedError) as exc:
        rows.append({"test": "device mps", "status": "failed", "error": str(exc)[:200]})
        print(f"mps: FAILED: {str(exc)[:200]}")
    return rows


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--time-one":
        dev, block, lo, hi, q = sys.argv[2:7]
        print(json.dumps(time_one(float(lo), float(hi), float(q), dev, int(block))))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="re-render the Faust references instead of reusing the fixture")
    args = ap.parse_args()

    with open("out/patch.json") as fh:
        patch = json.load(fh)["params"]
    fitted = {k: patch[k] for k in ("cutoff", "reso", "envAmt", "kbdTrk", "fA", "fD", "fS", "aR")}
    print("fitted filter params:", {k: round(v, 5) for k, v in fitted.items()})
    print(f"default BLOCK={BLOCK}, MAX_NFFT={MAX_NFFT}, probe={args.probe}")

    fx: dict[str, np.ndarray] = {}
    if not args.probe:
        with np.load(FIXTURE, allow_pickle=True) as z:
            fx = {k: z[k] for k in z.files if k != "measurements"}
        print(f"reusing {len(fx)} Faust arrays from {FIXTURE}")

    rows = check_static(fx, args.probe)
    rows += check_nfft_cap()

    print("\n=== 2/3. moving fc: fitted envelope, then adversarial ===")
    for name, over, freq, dur, hold in MOVING:
        rows += moving_case(name, {**fitted, **over}, freq, dur, hold, fx, BLOCKS, args.probe)

    q = float(fitted["reso"])
    rows += check_batch(fx, q)
    rows += check_device(fx, q)

    fc_fit = fx["mov_fit_440_fc"].astype(np.float64)
    devices = ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",)
    rows += check_timing(float(fc_fit.min()), float(fc_fit.max()), q, devices)
    grad = check_gradients(fc_fit[: 4 * SR], q)
    rows += check_grad_direction(fc_fit[: SR // 2], q)

    fx["measurements"] = np.array(json.dumps({"rows": rows, "gradients": grad}))
    fp.save_fixture(FIXTURE, **fx)


if __name__ == "__main__":
    main()
