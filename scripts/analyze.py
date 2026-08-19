"""Stage 0: inspect the clip before assuming the single-synth premise."""

from __future__ import annotations

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = 44100
CLIP = "data/original.wav"


def main() -> None:
    y, sr = librosa.load(CLIP, sr=SR, mono=True)
    dur = len(y) / sr
    print(f"duration {dur:.2f}s  sr {sr}")

    # Harmonic / percussive split tells us whether drums are present.
    harm, perc = librosa.effects.hpss(y, margin=(1.0, 5.0))
    e_h = float(np.sum(harm**2))
    e_p = float(np.sum(perc**2))
    print(f"harmonic energy {e_h:.1f}  percussive energy {e_p:.1f}  perc/total {e_p / (e_h + e_p):.3f}")

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    print(f"tempo estimate {tempo:.2f} BPM   beats {len(beats)}")
    if len(beats) > 1:
        ibi = np.diff(beats)
        print(f"  inter-beat: mean {ibi.mean():.4f}s  std {ibi.std():.4f}s")

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time", backtrack=False)
    print(f"onsets detected: {len(onsets)}")
    print("  times:", np.round(onsets, 3).tolist())

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    print(f"centroid mean {cent.mean():.0f} Hz  rolloff95 mean {roll.mean():.0f} Hz  flatness mean {flat.mean():.4f}")

    # Energy in time, to spot the arrangement (intro / drop / tail).
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    t_rms = librosa.times_like(rms, sr=sr, hop_length=512)
    for lo in range(0, int(dur) + 1, 2):
        m = (t_rms >= lo) & (t_rms < lo + 2)
        if m.any():
            print(f"  {lo:2d}-{lo + 2:2d}s  rms {rms[m].mean():.4f}  peak {rms[m].max():.4f}")

    fig, ax = plt.subplots(3, 1, figsize=(16, 11), constrained_layout=True)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096, hop_length=512)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=ax[0])
    ax[0].set(title="full mix (log-f STFT)")
    ax[0].vlines(onsets, 0, sr / 2, color="w", alpha=0.6, lw=0.7)

    M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=12000), ref=np.max)
    librosa.display.specshow(M, sr=sr, x_axis="time", y_axis="mel", fmax=12000, ax=ax[1])
    ax[1].set(title="mel")

    C = librosa.feature.chroma_cqt(y=harm, sr=sr)
    librosa.display.specshow(C, sr=sr, x_axis="time", y_axis="chroma", ax=ax[2])
    ax[2].set(title="chroma (harmonic component)")
    fig.savefig("out/00_analysis.png", dpi=110)
    print("wrote out/00_analysis.png")


if __name__ == "__main__":
    main()
