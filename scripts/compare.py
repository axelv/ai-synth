"""Deliverable: side-by-side mel spectrograms + loss curve."""

from __future__ import annotations


import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = 44100


def running_corr(a: np.ndarray, win: int = 4096, hop: int = 2048) -> np.ndarray:
    """Per-frame corr(L, R). The stage-2 loss is mono, so this is measured, never fitted."""
    f = np.lib.stride_tricks.sliding_window_view(a, win, axis=-1)[:, ::hop]
    f = f - f.mean(axis=-1, keepdims=True)
    num = (f[0] * f[1]).sum(axis=-1)
    den = np.sqrt((f[0] ** 2).sum(axis=-1) * (f[1] ** 2).sum(axis=-1))
    return num / np.maximum(den, 1e-20)


def main() -> None:
    os_, _ = librosa.load("data/original.wav", sr=SR, mono=False)
    rs, _ = librosa.load("out/render.wav", sr=SR, mono=False)
    n = min(os_.shape[1], rs.shape[1])
    os_, rs = os_[:, :n], rs[:, :n]
    o, r = os_.mean(axis=0), rs.mean(axis=0)

    Mo = librosa.power_to_db(librosa.feature.melspectrogram(y=o, sr=SR, n_mels=128, fmax=12000), ref=np.max)
    Mr = librosa.power_to_db(librosa.feature.melspectrogram(y=r, sr=SR, n_mels=128, fmax=12000), ref=np.max)

    try:
        hist = np.load("out/loss_history.npy")
    except OSError:
        hist = np.array([])

    fig = plt.figure(figsize=(15, 14), constrained_layout=True)
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 3, 2, 2, 2])

    ax0 = fig.add_subplot(gs[0])
    librosa.display.specshow(Mo, sr=SR, x_axis="time", y_axis="mel", fmax=12000, ax=ax0, vmin=-80, vmax=0)
    ax0.set(title="ORIGINAL — mel spectrogram")

    ax1 = fig.add_subplot(gs[1])
    librosa.display.specshow(Mr, sr=SR, x_axis="time", y_axis="mel", fmax=12000, ax=ax1, vmin=-80, vmax=0)
    ax1.set(title="MATCHED RENDER — mel spectrogram")

    ax2 = fig.add_subplot(gs[2])
    to = librosa.times_like(librosa.feature.rms(y=o, hop_length=512)[0], sr=SR, hop_length=512)
    ax2.plot(to, librosa.feature.rms(y=o, hop_length=512)[0], label="original", lw=1.4)
    ax2.plot(to, librosa.feature.rms(y=r, hop_length=512)[0], label="render", lw=1.4, alpha=0.85)
    ax2.set(title="loudness envelope (RMS)", xlabel="time (s)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[3])
    co, cr = running_corr(os_), running_corr(rs)
    tc = np.arange(len(co)) * 2048 / SR
    ax3.plot(tc, co, label=f"original (mean {co.mean():.3f})", lw=1.2)
    ax3.plot(tc, cr, label=f"render (mean {cr.mean():.3f})", lw=1.2, alpha=0.85)
    ax3.set(title="L/R correlation per 93 ms frame (1 is mono)", xlabel="time (s)", ylim=(-1.05, 1.05))
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[4])
    if len(hist):
        ax4.plot(hist, lw=1.0)
        ax4.set(title="CMA-ES best loss per generation (out/loss_history.npy)",
                xlabel="generation", ylabel="MRSTFT + env")
        ax4.grid(alpha=0.3)
    fig.savefig("out/comparison.png", dpi=110)
    print("wrote out/comparison.png")

    # long-term average spectra, a good read on timbral match
    fig2, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for sig, lbl in ((o, "original"), (r, "render")):
        S = np.abs(librosa.stft(sig, n_fft=8192, hop_length=2048))
        avg = 20 * np.log10(np.median(S, axis=1) + 1e-8)
        fr = librosa.fft_frequencies(sr=SR, n_fft=8192)
        ax.semilogx(fr[1:], avg[1:], label=lbl, lw=1.2)
    ax.set(xlim=(25, 16000), xlabel="Hz", ylabel="dB", title="long-term average spectrum")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig2.savefig("out/spectrum_compare.png", dpi=110)
    print("wrote out/spectrum_compare.png")


if __name__ == "__main__":
    main()
