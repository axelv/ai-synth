"""Time the machine parts of stage 1: the greedy chord search and the f0 tracker.

transcribe.main() writes out/chords.json, so nothing here calls it; fit_region is timed
on one region and the rest extrapolated from the render count.

Run: PYTHONPATH=scripts uv run python .scratch/app-design-flow/research/bench_stage1.py
"""

from __future__ import annotations

import json
import time

import librosa
import numpy as np

import transcribe
from synth import PadRenderer

R = {}
SR = 44100

t = time.perf_counter()
y, _ = librosa.load("data/original.wav", sr=SR, mono=True)
R["librosa_load_44100_mono_s"] = time.perf_counter() - t
print(f"librosa.load(sr=44100, mono): {R['librosa_load_44100_mono_s']:.2f} s "
      f"({len(y)/SR:.2f} s of audio)")

freqs = librosa.fft_frequencies(sr=SR, n_fft=16384)
r = PadRenderer(n_voices=24)
r.set_params(transcribe.neutral_params())

# one region of the greedy add-one-note search
t0, t1 = transcribe.REGIONS[2]
t = time.perf_counter()
notes = transcribe.fit_region(r, y, t0, t1, freqs)
dt = time.perf_counter() - t
R["fit_region_one_s"] = dt
R["fit_region_notes"] = notes
R["spec_cache_entries_after_one_region"] = len(transcribe._spec_cache)
print(f"fit_region [{t0}-{t1}]: {dt:.1f} s, {len(transcribe._spec_cache)} candidate "
      f"renders, notes {notes}")
R["all_regions_est_s"] = dt * len(transcribe.REGIONS)
print(f"  -> 6 regions, no cache sharing: ~{dt*len(transcribe.REGIONS):.0f} s "
      f"(upper bound; the cache is shared in main(), so the real figure is lower)")

# f0 tracker: harmonic-sum salience + viterbi over the intro
import pitch_track  # noqa: E402

t = time.perf_counter()
S = np.abs(librosa.stft(y, n_fft=pitch_track.N_FFT, hop_length=pitch_track.HOP))
f = librosa.fft_frequencies(sr=SR, n_fft=pitch_track.N_FFT)
R["stft_8192_s"] = time.perf_counter() - t

grid = np.geomspace(pitch_track.F_LO, pitch_track.F_HI, 600)
t = time.perf_counter()
sal = pitch_track.salience(S, f, grid)
R["salience_s"] = time.perf_counter() - t
t = time.perf_counter()
path = pitch_track.viterbi(sal, grid)
R["viterbi_s"] = time.perf_counter() - t
print(f"pitch track: stft {R['stft_8192_s']:.2f} s, salience {R['salience_s']:.2f} s, "
      f"viterbi {R['viterbi_s']:.2f} s")

json.dump(R, open(".scratch/app-design-flow/research/bench_stage1.json", "w"), indent=1)
print("wrote .scratch/app-design-flow/research/bench_stage1.json")
