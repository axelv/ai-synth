"""Measure the surrogate gap: torch_synth against the Faust it stands in for.

The gap is the number that decides whether the port is usable. A gradient step is
only worth taking if the surrogate's disagreement with Faust is small compared to
the improvement the gradient is chasing, so everything here is measured on
rendered audio at out/patch.json and reported next to the 1.5564 that CMA-ES
reached.

Steps are separate subcommands because the full-length renders are large: one
process doing all of them at once pushes this machine into swap and the timings
stop meaning anything. Intermediate renders go to out/fixtures/torch_synth.npz.

    uv run python scripts/verify_torch_synth.py faust    # authoritative render + loss
    uv run python scripts/verify_torch_synth.py life     # per-voice cut instants
    uv run python scripts/verify_torch_synth.py voices   # one Faust render per note
    uv run python scripts/verify_torch_synth.py torch --device cpu
    uv run python scripts/verify_torch_synth.py torch --device mps
    uv run python scripts/verify_torch_synth.py dry      # split the gap by stage
    uv run python scripts/verify_torch_synth.py grad     # all 27 gradients finite
    uv run python scripts/verify_torch_synth.py slope --start seed   # is it a descent dir
    uv run python scripts/verify_torch_synth.py compare --device cpu # gap + figure

Scalar summaries go to stdout, everything larger to out/torch_synth_results.json and
out/fixtures/torch_synth.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import auraloss
import librosa
import numpy as np
import soundfile as sf
import torch

import torch_synth
from metrics import env_l1, mel_dist
from stage2 import load_notes
from synth import PARAMS, PadRenderer, denorm
from torch_common import SR, Patch, SpectralLoss, default_n_samples, get_device
from torch_synth import TorchPad, VoiceLife

FIX = "out/fixtures/torch_synth.npz"
RESULTS = "out/torch_synth_results.json"
PATCH = "out/patch.json"
DUR = 17.904
FX_OFF = {"chDepth": 0.0, "dlyWet": 0.0, "revWet": 0.0, "tilt": 0.0, "outGain": 1.0}


# ---------------------------------------------------------------- shared


def normalized() -> np.ndarray:
    return np.asarray(json.load(open(PATCH))["normalized"], dtype=float)


def store(**arrays: np.ndarray) -> None:
    os.makedirs(os.path.dirname(FIX), exist_ok=True)
    have = dict(np.load(FIX)) if os.path.exists(FIX) else {}
    have.update({k: np.asarray(v) for k, v in arrays.items()})
    np.savez(FIX, **have)


def load(name: str) -> np.ndarray:
    return np.load(FIX)[name]


def record(**vals) -> None:
    have = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    have.update(vals)
    with open(RESULTS, "w") as fh:
        json.dump(have, fh, indent=2, sort_keys=True)


def mrstft_pair() -> auraloss.freq.MultiResolutionSTFTLoss:
    """The same resolutions the stage-2 objective uses, so distances are comparable."""
    return auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048, 4096],
        hop_sizes=[128, 256, 512, 1024],
        win_lengths=[512, 1024, 2048, 4096],
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
    )


def objective_loss(mono: np.ndarray, target: np.ndarray, w_env: float = 0.35) -> float:
    """stage2.Objective.__call__ reproduced exactly, so 1.5564 is reproducible."""
    n = min(len(mono), len(target))
    pred = torch.from_numpy(np.ascontiguousarray(mono[:n])).float().view(1, 1, -1)
    tgt = torch.from_numpy(np.ascontiguousarray(target[:n])).float().view(1, 1, -1)

    def env(x: torch.Tensor) -> torch.Tensor:
        f = x.view(1, 1, -1).unfold(-1, 1024, 512)
        e = f.pow(2).mean(-1).sqrt()
        return e / (e.mean() + 1e-9)

    with torch.no_grad():
        return float(mrstft_pair()(pred, tgt)) + w_env * float((env(pred) - env(tgt)).abs().mean())


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n], b[..., :n]
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-20))


# ---------------------------------------------------------------- faust


def step_faust() -> None:
    """Authoritative render at patch.json, plus the same render with the fx bypassed."""
    z = normalized()
    notes = load_notes()
    r = PadRenderer(n_voices=24)
    r.set_notes(notes)
    from bend2 import bend_curve

    r.set_bend(bend_curve(int(DUR * SR) + SR))

    vals = denorm(z)
    r.set_params(vals)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        aud = r.render(DUR)
        times.append(time.perf_counter() - t0)
    target, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    loss = objective_loss(aud.mean(axis=0), target)

    r.set_params({**vals, **FX_OFF})
    dry = r.render(DUR)

    store(faust=aud.astype(np.float32), faust_dry=dry.astype(np.float32))
    record(faust_loss=loss, render_seconds_faust=min(times), faust_n=int(aud.shape[1]))
    print(f"faust loss {loss:.4f}  (patch.json says {json.load(open(PATCH))['loss']:.4f})")
    print(f"faust render seconds min {min(times):.3f} of {['%.3f' % t for t in times]}")
    print(f"faust n {aud.shape[1]}, dry rms {float(np.sqrt((dry ** 2).mean())):.5f}")


def step_life() -> None:
    """Where dawdreamer stops computing each voice, measured one note at a time.

    A voice stops being computed partway through its release and its output is exactly
    zero from then on, so with the effects bypassed the last nonzero sample of a
    single-note render is the cut index. Read off Faust rather than predicted: the
    instant moves with the output level but is not a monotone function of pitch or of
    the release length. Notes whose note-off is closer to the render end than aR are
    truncated by the render instead, and are excluded from the summary.
    """
    z = normalized()
    notes = load_notes()
    r = PadRenderer(n_voices=24)
    from bend2 import bend_curve

    r.set_bend(bend_curve(int(DUR * SR) + SR))
    r.set_params({**denorm(z), **FX_OFF})

    n = default_n_samples()
    ends = np.zeros(len(notes), dtype=np.int64)
    offs = np.zeros(len(notes), dtype=np.int64)
    for i, note in enumerate(notes):
        r.set_notes([note])
        a = r.render(DUR).mean(axis=0)
        nz = np.nonzero(np.abs(a) > 0.0)[0]
        ends[i] = int(nz[-1]) + 1 if nz.size else 0
        offs[i] = int(round((note[2] + note[3]) * SR))

    a_r = denorm(z)["aR"]
    n_r = int(round(a_r * SR))
    tail = ends - offs
    fits = (offs + n_r) < n                    # the render itself does not truncate these
    t = tail[fits]
    store(voice_end=ends, note_off=offs)
    record(
        voice_life_aR_samples=n_r,
        voice_life_notes_measurable=int(fits.sum()),
        voice_life_frac_of_aR_min=float(t.min() / n_r),
        voice_life_frac_of_aR_median=float(np.median(t) / n_r),
        voice_life_frac_of_aR_max=float(t.max() / n_r),
    )
    print(f"aR = {a_r:.4f} s = {n_r} samples; {int(fits.sum())} of {len(notes)} notes have "
          f"their whole release inside the render")
    print(f"tail after note-off as a fraction of aR: min {t.min() / n_r:.3f} "
          f"median {np.median(t) / n_r:.3f} max {t.max() / n_r:.3f}")


# ---------------------------------------------------------------- torch


def build_pad(device: torch.device, with_life: bool = False) -> TorchPad:
    life = None
    if with_life:
        life = VoiceLife(torch.from_numpy(load("voice_end")).to(device))
    return TorchPad(load_notes(), default_n_samples(), device, voice_life=life)


def step_torch(device_name: str, repeats: int = 2) -> None:
    device = torch.device(device_name)
    z = normalized()
    pad = build_pad(device, with_life=False)
    patch = Patch(z).to(device)
    p = {k: v.detach() for k, v in patch.values().items()}

    times = []
    for _ in range(repeats):
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            y = pad.render(p)
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    a = y.detach().cpu().numpy()

    target, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    loss = objective_loss(a.mean(axis=0), target)

    # the same render WITH the measured voice-lifetime gate, to price that quirk
    pad_nolife = build_pad(device, with_life=True)
    with torch.no_grad():
        a_nl = pad_nolife.render(p).detach().cpu().numpy()
    loss_nl = objective_loss(a_nl.mean(axis=0), target)

    store(**{f"torch_{device_name}": a.astype(np.float32),
             f"torch_life_{device_name}": a_nl.astype(np.float32)})
    record(**{f"torch_loss_{device_name}": loss,
              f"torch_loss_with_life_gate_{device_name}": loss_nl,
              f"render_seconds_torch_{device_name}": min(times)})
    print(f"[{device_name}] torch loss {loss:.4f}  (with the voice-life gate {loss_nl:.4f})")
    print(f"[{device_name}] render seconds min {min(times):.3f} of {['%.3f' % t for t in times]}")
    print(f"[{device_name}] rms {float(np.sqrt((a ** 2).mean())):.5f} "
          f"finite {bool(np.isfinite(a).all())}")


def step_grad(device_name: str = "cpu") -> None:
    """One backward pass through the loss; every logit must get a finite gradient."""
    device = torch.device(device_name)
    pad = build_pad(device, with_life=False)
    patch = Patch(normalized()).to(device)
    loss_fn = SpectralLoss()

    t0 = time.perf_counter()
    loss = loss_fn(pad(patch))
    fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    loss.backward()
    bwd = time.perf_counter() - t0

    g = patch.logits.grad.detach().cpu().numpy()
    names = [p.name for p in PARAMS]
    zero = [names[i] for i in range(len(names)) if g[i] == 0.0]
    nonfinite = [names[i] for i in range(len(names)) if not np.isfinite(g[i])]
    record(
        grad_loss=float(loss),
        grad_forward_seconds=fwd,
        grad_backward_seconds=bwd,
        grad_zero=zero,
        grad_nonfinite=nonfinite,
        grad_abs_min=float(np.abs(g).min()),
        grad_abs_max=float(np.abs(g).max()),
        grad=dict(zip(names, [float(v) for v in g])),
    )
    print(f"[{device_name}] SpectralLoss {float(loss):.4f}  forward {fwd:.2f}s backward {bwd:.2f}s")
    print(f"grad |g| min {np.abs(g).min():.4g} max {np.abs(g).max():.4g}")
    print(f"zero grads: {zero or 'none'}   non-finite: {nonfinite or 'none'}")


# ---------------------------------------------------------------- decomposition


def step_voices() -> None:
    """One Faust render per note with the effects bypassed, so each voice is isolated.

    The summed dry signal cannot tell an oscillator error from an envelope error, and
    the surrogate gap is only actionable if it is attributed to a stage.
    """
    z = normalized()
    notes = load_notes()
    r = PadRenderer(n_voices=24)
    from bend2 import bend_curve

    r.set_bend(bend_curve(int(DUR * SR) + SR))
    r.set_params({**denorm(z), **FX_OFF})
    n = default_n_samples()
    out = np.zeros((len(notes), n), dtype=np.float32)
    for i, note in enumerate(notes):
        r.set_notes([note])
        out[i] = r.render(DUR).mean(axis=0)[:n]
    store(faust_voices=out)
    print(f"stored faust_voices {out.shape}, rms {float(np.sqrt((out ** 2).mean())):.5f}")


def step_dry(device_name: str = "cpu") -> None:
    """Split the gap into (osc, env, filter) versus (chorus, delay, reverb, tilt)."""
    device = torch.device(device_name)
    z = normalized()
    p = {k: v.detach() for k, v in Patch(z).to(device).values().items()}

    faust_v = load("faust_voices")
    n = faust_v.shape[1]
    with torch.no_grad():
        v_nl = build_pad(device, with_life=False).voice_output(p).cpu().numpy()
        v_life = build_pad(device, with_life=True).voice_output(p).cpu().numpy()

    mr = mrstft_pair()

    def mrs(a: np.ndarray, b: np.ndarray) -> float:
        with torch.no_grad():
            return float(mr(torch.from_numpy(np.ascontiguousarray(a)).view(1, 1, -1),
                            torch.from_numpy(np.ascontiguousarray(b)).view(1, 1, -1)))


    # tilt = 0 is NOT a bypass in Faust: fi.highshelf(2, 0 dB, 1200) is
    # lowband + highband of a 2nd-order Butterworth crossover, whose sum has a zero
    # at the crossover, so the fx-off Faust render carries deep notches at 300 and
    # 1200 Hz. The torch dry signal has to go through the same chain to be comparable.
    p_off = dict(p)
    for k, v in FX_OFF.items():
        p_off[k] = torch.tensor(float(v), device=device)

    def through_fx_off(mono: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            st = torch.from_numpy(mono).to(device).expand(2, -1).contiguous()
            return torch_synth.torch_fx.effects(st, p_off, SR).mean(dim=0).cpu().numpy()

    dry_f = load("faust_dry").mean(axis=0)[:n]
    dry_t = through_fx_off(v_nl.sum(axis=0))
    dry_t_life = through_fx_off(v_life.sum(axis=0))

    v_nl_fx = np.stack([through_fx_off(v_nl[i]) for i in range(len(faust_v))])
    v_life_fx = np.stack([through_fx_off(v_life[i]) for i in range(len(faust_v))])
    per_voice = np.array([rel_l2(v_nl_fx[i], faust_v[i]) for i in range(len(faust_v))])
    per_voice_life = np.array([rel_l2(v_life_fx[i], faust_v[i]) for i in range(len(faust_v))])
    per_rms = np.array([float(np.sqrt((v_nl_fx[i] ** 2).mean())
                              / (np.sqrt((faust_v[i] ** 2).mean()) + 1e-20))
                        for i in range(len(faust_v))])
    per_mrstft = np.array([mrs(v_nl_fx[i], faust_v[i]) for i in range(len(faust_v))])

    # the fx stage alone: run Faust's own dry signal through the torch effects
    with torch.no_grad():
        stereo = torch.from_numpy(load("faust_dry")[:, :n].copy()).to(device)
        fx_t = torch_synth.torch_fx.effects(stereo, p, SR).cpu().numpy()
    full_f = load("faust").mean(axis=0)[:n]
    fx_gap = mrs(fx_t.mean(axis=0), full_f)

    worst = int(np.argmax(per_mrstft))
    record(
        dry_mrstft=mrs(dry_t, dry_f),
        dry_mrstft_life=mrs(dry_t_life, dry_f),
        dry_rel_l2=rel_l2(dry_t, dry_f),
        dry_rms_ratio=float(np.sqrt((dry_t ** 2).mean()) / np.sqrt((dry_f ** 2).mean())),
        dry_mel=mel_dist(dry_f, dry_t),
        fx_only_mrstft=fx_gap,
        fx_only_rel_l2=rel_l2(fx_t.mean(axis=0), full_f),
        voice_mrstft_median=float(np.median(per_mrstft)),
        voice_mrstft_max=float(per_mrstft.max()),
        voice_mrstft_worst_note=worst,
        voice_rel_l2_median=float(np.median(per_voice)),
        voice_rel_l2_median_life=float(np.median(per_voice_life)),
        voice_rms_ratio_median=float(np.median(per_rms)),
        voice_rms_ratio_min=float(per_rms.min()),
        voice_rms_ratio_max=float(per_rms.max()),
    )
    store(per_voice_mrstft=per_mrstft, per_voice_rel_l2=per_voice, per_voice_rms_ratio=per_rms)
    print(f"dry sum: mrstft {mrs(dry_t, dry_f):.4f} (with life gate {mrs(dry_t_life, dry_f):.4f})"
          f"  rel L2 {rel_l2(dry_t, dry_f):.4f}  rms ratio "
          f"{np.sqrt((dry_t ** 2).mean()) / np.sqrt((dry_f ** 2).mean()):.4f}")
    print(f"fx only (faust dry -> torch fx vs faust full): mrstft {fx_gap:.4f}  "
          f"rel L2 {rel_l2(fx_t.mean(axis=0), full_f):.4f}")
    print(f"per voice mrstft: median {np.median(per_mrstft):.4f} max {per_mrstft.max():.4f} "
          f"(note {worst}, pitch {load_notes()[worst][0]})")
    print(f"per voice rel L2 median {np.median(per_voice):.4f} "
          f"(with life gate {np.median(per_voice_life):.4f})")
    print(f"per voice rms ratio: min {per_rms.min():.3f} median {np.median(per_rms):.3f} "
          f"max {per_rms.max():.3f}")


# ---------------------------------------------------------------- compare


def step_compare(device_name: str = "mps") -> None:
    faust = load("faust")
    torch_a = load(f"torch_{device_name}")
    torch_nl = load(f"torch_life_{device_name}")
    n = min(faust.shape[1], torch_a.shape[1])
    fm, tm = faust.mean(axis=0)[:n], torch_a.mean(axis=0)[:n]

    mr = mrstft_pair()
    with torch.no_grad():
        gap_mrstft = float(mr(torch.from_numpy(tm.copy()).view(1, 1, -1),
                              torch.from_numpy(fm.copy()).view(1, 1, -1)))
        gap_nolife = float(mr(torch.from_numpy(torch_nl.mean(axis=0)[:n].copy()).view(1, 1, -1),
                              torch.from_numpy(fm.copy()).view(1, 1, -1)))
    gap_mel = mel_dist(fm, tm)

    # Reference floor: two legitimate Faust renderings of the SAME patch and notes
    # (the polyphonic render, and the sum of 29 one-note renders) decorrelate over
    # 18 s because saw2ptr accumulates its phase in float32. Any render-to-render
    # distance has to be read against this, not against zero.
    floor_sum = load("faust_voices").sum(axis=0)[:n]
    floor_poly = load("faust_dry").mean(axis=0)[:n]
    with torch.no_grad():
        floor_mrstft = float(mr(torch.from_numpy(floor_sum.copy()).view(1, 1, -1),
                                torch.from_numpy(floor_poly.copy()).view(1, 1, -1)))
    dry_f = load("faust_dry").mean(axis=0)[:n]
    res = json.load(open(RESULTS))
    record(
        faust_vs_faust_mrstft=floor_mrstft,
        faust_vs_faust_mel=mel_dist(floor_poly, floor_sum),
        faust_vs_faust_rel_l2=rel_l2(floor_sum, floor_poly),
        env_l1_torch_vs_faust=env_l1(fm, tm),
        mrstft_torch_vs_faust=gap_mrstft,
        mrstft_torch_vs_faust_with_life_gate=gap_nolife,
        mel_torch_vs_faust=gap_mel,
        rel_l2_torch_vs_faust=rel_l2(tm, fm),
        rms_ratio_torch_faust=float(np.sqrt((tm ** 2).mean()) / np.sqrt((fm ** 2).mean())),
        corr_torch_faust=float(np.corrcoef(tm, fm)[0, 1]),
    )
    print(f"gap mrstft {gap_mrstft:.4f} (with the voice-life gate {gap_nolife:.4f})")
    print(f"faust-vs-faust floor: mrstft {floor_mrstft:.4f}  mel "
          f"{mel_dist(floor_poly, floor_sum):.4f}  rel L2 {rel_l2(floor_sum, floor_poly):.4f}")
    print(f"env_l1 torch vs faust {env_l1(fm, tm):.4f}")
    print(f"gap mel_dist {gap_mel:.4f}   rel L2 {rel_l2(tm, fm):.4f}   "
          f"rms ratio {np.sqrt((tm ** 2).mean()) / np.sqrt((fm ** 2).mean()):.4f}")
    print(f"losses: faust {res['faust_loss']:.4f}  torch {res[f'torch_loss_{device_name}']:.4f}")
    print(f"dry faust rms {float(np.sqrt((dry_f ** 2).mean())):.5f}")

    sf.write("out/torch_render.wav", torch_a[:, :n].T, SR)
    figure(device_name)
    print("wrote out/torch_render.wav and out/torch_vs_faust.png")


def figure(device_name: str) -> None:
    """Spectrograms of original / faust / torch, the per-bin difference, per-band error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig, _ = librosa.load("data/original.wav", sr=SR, mono=True)
    faust = load("faust").mean(axis=0)
    tor = load(f"torch_{device_name}").mean(axis=0)
    n = min(len(orig), len(faust), len(tor))
    sigs = [("original", orig[:n]), ("faust", faust[:n]), (f"torch ({device_name})", tor[:n])]

    mags = [(lab, np.abs(librosa.stft(y, n_fft=2048, hop_length=512))) for lab, y in sigs]
    ref = float(max(m.max() for _, m in mags))
    specs = [(lab, librosa.amplitude_to_db(m, ref=ref)) for lab, m in mags]

    fig, ax = plt.subplots(5, 1, figsize=(12, 15), constrained_layout=True)
    for a, (lab, S) in zip(ax, specs):
        im = librosa.display.specshow(S, sr=SR, hop_length=512, x_axis="time", y_axis="log",
                                      ax=a, vmin=-80, vmax=0, cmap="magma")
        a.set_title(f"{lab}  (dB re the loudest of the three)")
    fig.colorbar(im, ax=ax[2])

    diff = specs[2][1] - specs[1][1]
    audible = (specs[1][1] > -70) | (specs[2][1] > -70)
    d = librosa.display.specshow(np.where(audible, diff, 0.0), sr=SR, hop_length=512,
                                 x_axis="time", y_axis="log", ax=ax[3],
                                 vmin=-15, vmax=15, cmap="coolwarm")
    ax[3].set_title("torch minus faust, dB per STFT bin (bins below -70 dB blanked)")
    fig.colorbar(d, ax=ax[3])

    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    band = {}
    for lab, S in specs:
        band[lab] = S.mean(axis=1)
        ax[4].semilogx(freqs, band[lab], label=lab, lw=1.0)
    err = band[specs[2][0]] - band["faust"]
    ax[4].semilogx(freqs, err, label="torch - faust", lw=1.4, color="k")
    ax[4].axhline(0.0, color="0.6", lw=0.5)
    ax[4].set_xlim(20, SR / 2)
    ax[4].set_xlabel("Hz")
    ax[4].set_ylabel("mean dB")
    ax[4].legend(fontsize=8)
    ax[4].set_title("per-band mean level over the clip, and the torch - faust error")
    fig.savefig("out/torch_vs_faust.png", dpi=110)
    plt.close(fig)

    keep = band["faust"] > band["faust"].max() - 60.0
    worst = int(np.argmax(np.abs(np.where(keep, err, 0.0))))
    record(
        band_err_max_db=float(np.abs(err[keep]).max()),
        band_err_max_db_at_hz=float(freqs[worst]),
        band_err_rms_db=float(np.sqrt((err[keep] ** 2).mean())),
    )
    print(f"per-band mean level error within 60 dB of the faust peak: max "
          f"{np.abs(err[keep]).max():.2f} dB at {freqs[worst]:.0f} Hz, "
          f"rms {np.sqrt((err[keep] ** 2).mean()):.2f} dB")


# ---------------------------------------------------------------- gradient


def step_slope(device_name: str = "cpu", delta: float = 0.03, start: str = "patch") -> None:
    """The only test that matters for usability: is the surrogate gradient a descent
    direction for the FAUST loss?

    Two measurements. First, the surrogate's analytic gradient in normalized space
    against a central finite difference of the true Faust loss, one Faust render pair
    per parameter. Second, an actual step along the negative surrogate gradient,
    re-rendered through PadRenderer, which is what stage 2 would really do.

    Run it at more than one point. out/patch.json is a converged CMA-ES optimum where
    the true gradient is small and the loss surface is not smooth at these step sizes,
    so agreement there says nothing about whether the surrogate can find its way
    towards an optimum. --start seed uses stage2.seeded_start, which is far from one.
    """
    device = torch.device(device_name)
    if start == "patch":
        z = normalized()
    elif start == "seed":
        from stage2 import seeded_start

        z = seeded_start()
    else:
        z = np.asarray(json.load(open(start))["normalized"], dtype=float)
    notes = load_notes()
    names = [p.name for p in PARAMS]

    loss_fn = SpectralLoss()
    zt = np.clip(z, 1e-4, 1 - 1e-4)
    grads = {}
    for tag, with_life in (("nolife", False), ("life", True)):
        patch = Patch(z).to(device)
        loss_fn(build_pad(device, with_life=with_life)(patch)).backward()
        # chain rule back to normalized space, where the Faust difference is taken
        grads[tag] = patch.logits.grad.detach().cpu().numpy() / (zt * (1.0 - zt))
    g_torch = grads["nolife"]

    r = PadRenderer(n_voices=24)
    r.set_notes(notes)
    from bend2 import bend_curve

    r.set_bend(bend_curve(int(DUR * SR) + SR))
    target, _ = librosa.load("data/original.wav", sr=SR, mono=True)

    def faust_loss(x: np.ndarray) -> float:
        r.set_params(denorm(np.clip(x, 0.0, 1.0)))
        return objective_loss(r.render(DUR).mean(axis=0), target)

    base = faust_loss(z)
    g_faust = np.zeros(len(z))
    for i in range(len(z)):
        hi, lo = z.copy(), z.copy()
        hi[i] = min(1.0, z[i] + delta)
        lo[i] = max(0.0, z[i] - delta)
        g_faust[i] = (faust_loss(hi) - faust_loss(lo)) / (hi[i] - lo[i])

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))

    cos = cosine(g_torch, g_faust)
    cos_life = cosine(grads["life"], g_faust)
    agree = [names[i] for i in range(len(z)) if np.sign(g_torch[i]) == np.sign(g_faust[i])]
    d = -g_torch / (np.linalg.norm(g_torch) + 1e-20)
    steps = {eta: faust_loss(z + eta * d) for eta in (0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0)}

    record(**{f"slope_{start}": {
        "delta": delta,
        "cosine": cos,
        "cosine_with_life_gate": cos_life,
        "sign_agree": len(agree),
        "sign_disagree": [names[i] for i in range(len(z))
                          if np.sign(g_torch[i]) != np.sign(g_faust[i])],
        "base_faust_loss": base,
        "faust_after_step": {str(k): v for k, v in steps.items()},
        "best_loss": min(steps.values()),
        "grad_norm_torch": float(np.linalg.norm(g_torch)),
        "grad_norm_faust": float(np.linalg.norm(g_faust)),
        "grad_normalized_torch": dict(zip(names, [float(v) for v in g_torch])),
        "grad_normalized_faust": dict(zip(names, [float(v) for v in g_faust])),
    }})
    print(f"[{start}] cosine(surrogate grad, faust FD grad) = {cos:+.4f} "
          f"(with the voice-life gate {cos_life:+.4f})  sign agreement {len(agree)}/{len(z)}")
    print(f"[{start}] grad norm: surrogate {np.linalg.norm(g_torch):.3f} "
          f"faust FD {np.linalg.norm(g_faust):.3f}")
    print(f"[{start}] faust loss {base:.4f}; after a unit step along -grad_surrogate: "
          + "  ".join(f"{k}: {v:.4f}" for k, v in steps.items()))


# ---------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["faust", "life", "voices", "torch", "grad", "dry", "slope", "compare"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--start", default="patch",
                    help="slope step only: 'patch', 'seed', or a patch json path")
    args = ap.parse_args()
    dev = args.device or ("mps" if get_device().type == "mps" else "cpu")
    if args.step == "faust":
        step_faust()
    elif args.step == "life":
        step_life()
    elif args.step == "torch":
        step_torch(dev)
    elif args.step == "voices":
        step_voices()
    elif args.step == "grad":
        step_grad(dev)
    elif args.step == "dry":
        step_dry(dev)
    elif args.step == "slope":
        step_slope(dev, start=args.start)
    else:
        step_compare(dev)


if __name__ == "__main__":
    main()
