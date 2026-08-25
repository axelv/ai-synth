"""Losses built from quantities that map one-to-one onto the synth's own controls.

The premise. The incumbent measures waveform agreement bin by bin, so every knob is
entangled with every bin and a -85 dB change in a near-empty bin outranks the largest
real improvement the project ever found. These candidates measure the synth instead:
a long-term band spectrum where the 26 EQ gains and the tilt live, a per-band peak-to-
valley ratio where the unison detune lives, a brightness trajectory where the filter
ADSR lives, an envelope modulation spectrum where lfoRate/chRate/detune live, an
envelope autocorrelation where dlyTime/dlyFb live, and a per-band tail decay rate where
revSize/revDamp/aR live. Nothing else can move them, which is the point: a change no
control can produce has nowhere to show up.

Insensitivity to the inaudible is structural, not a tuning constant. Every block is
derived from band or bin powers floored at -70 dB relative to the target's own peak, so
anything sitting at -85 dB is pinned to the floor for both signals and contributes
exactly zero. The measured pedestal (one parameter moved 1e-8, spectrum changed 5.4e-5
relative) moves an above-floor band by under 0.0005 dB and a below-floor band not at all.
The two blocks that measure a RELATIVE quantity need one more thing, an absolute gate;
both times that was left out it reintroduced the incumbent's exact failure, and both are
recorded below.

Four candidates, sharing one front end but reading different physics:

- `synth_controls`  static timbre: band LTAS, per-band peak-to-valley, brightness
                    trajectory, and six named scalars. Reads the oscillator, filter and
                    EQ, and is blind to the effects' timing.
- `mod_spectrum`    the modulation lens: per-band envelope modulation spectra. Reads
                    unison beating, vibrato and chorus, which a static spectrum averages
                    away entirely.
- `decay_profile`   the time-structure lens: envelope autocorrelation over the delay's
                    lag range plus per-band tail decay rates. Reads delay and reverb.
- `synth_full`      all of the above, weighted. The one meant to replace the incumbent;
                    the other three exist so the bake-off can attribute what carries it.

The three lenses are each rank-deficient on their own (a modulation spectrum cannot see
outGain), so `mod_spectrum` and `decay_profile` carry a light static-spectrum anchor.
Without it they would score a wildly wrong patch as perfect and the `discrim` screen
would be measuring nothing.

Front end, and why it is two-tier. The static blocks read a Welch LTAS at n_fft=8192,
because 5.4 Hz resolves both the 43.65 Hz partial spacing of the lowest note and the
few-Hz width of a detuned unison cluster, which is what the peak-to-valley statistic is
for; non-overlapping, so 96 frames cover the whole clip for 5 ms. The time blocks read
n_fft=1024 at hop 256, because a 172 Hz frame rate is what puts unison beat rates inside
the modulation band. Both are hand-framed through scipy.fft in float32 rather than
librosa.stft: 18 ms against librosa's 46 ms, on a machine where the incumbent measures
400 ms, and the budget is 50 ms per call.

Measured on 17.9 s at 44.1 kHz, load average 39: 13 to 23 ms per call, worst decile 52
ms for `synth_full`. Against a synthetic pad with a realistic -55 dB noise floor, the
-85 dB pedestal costs 0.0001 to 0.0005 of each candidate's own dynamic range, against
0.028 for the incumbent and 0.037 for pow03 on the identical pair.

Tried and rejected:

- A single 2048/256 STFT for everything. 30 to 46 ms on its own, most of the budget, and
  its 21.5 Hz resolution smears the sub octave and the detune clusters into each other,
  which destroys the peak-to-valley statistic that is the only direct read on detune.
- The modulation block without an audibility weight. It charged 0.19 for the -85 dB
  pedestal, a fifth of what a real control change costs, and all of it came from four
  bands sitting more than 60 dB below the loudest: there the pad has nothing, so the
  added noise was 100% of the band's envelope and its modulation spectrum was rewritten
  outright. Relative measures need an absolute gate, and this is the second time in this
  file that exact mistake showed up.
- The envelope autocorrelation normalised by its own zero lag. Same failure again: a flat
  envelope divided by its own dust, moving 0.65 out of a range of 5.3 for the same
  inaudible perturbation. Fixed by an absolute 0.5 dB floor on the normaliser and by not
  dividing by the zero lag at all.
- argmax of that autocorrelation as a scalar delay-time estimate. It is a step function
  of dlyTime, so it hands the optimiser a flat landscape with a cliff. The whole curve
  over the lag range is smooth and says the same thing.
- Per-frame spectral rolloff by thresholded index lookup, for the same reason. It is
  interpolated inside the band it lands in here, so it moves continuously with cutoff.
- np.polyfit for the two spectral-slope scalars. It solves a system per call; the slope
  of a least-squares line is one precomputable dot product (`_slope_basis`).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.fft as sfft

from losses import SR, match, register

# --- front end geometry -------------------------------------------------------------
_LT_FFT = 8192          # 5.4 Hz: resolves partials at 43.65 Hz and detune clusters
_ENV_FFT = 1024
_ENV_HOP = 256          # 172 Hz frames: unison beat rates land inside the modulation band
_FLOOR_DB = -70.0       # relative to the target's peak band power

# --- modulation band edges ----------------------------------------------------------
# 0.25 Hz is below the slowest chorus (chRate lo 0.05 Hz is slower than the clip resolves
# reliably); 60 Hz is above the fastest unison beat (60 cents across 7 voices at the
# clip's top pitch). Log spaced because every rate control here is multiplicative.
_MOD_LO, _MOD_HI, _N_MOD = 0.25, 60.0, 20
_MOD_FLOOR = 1e-3       # relative modulation depth; -60 dB of AM is not audible

# --- delay and decay geometry -------------------------------------------------------
_ACF_LO_S, _ACF_HI_S = 0.02, 1.25   # dlyTime spans 0.05 to 1.0 s
_TAIL_S = 2.5                       # the release, where reverb and aR are alone


def _hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def _power_frames(x: np.ndarray, n_fft: int, hop: int, win: np.ndarray) -> np.ndarray:
    """(T, n_fft//2+1) power spectrogram. Hand-framed and float32 through scipy.fft:
    measured 10 ms against librosa.stft's 46 ms at the same geometry, and the whole
    per-call budget is 50 ms. float32 is the render's own precision and the floor sits
    70 dB above where it would matter."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    f = np.lib.stride_tricks.sliding_window_view(np.ascontiguousarray(x, np.float32),
                                                 n_fft)[::hop]
    return np.abs(sfft.rfft(f * win, axis=-1)) ** 2


def _bandmat(freqs: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Row-normalised third-octave supports, matching the EQ bank's measured 0.508 oct."""
    fb = np.zeros((len(centres), len(freqs)), dtype=np.float32)
    for i, fc in enumerate(centres):
        m = (freqs >= fc * 2 ** -0.254) & (freqs <= fc * 2 ** 0.254)
        if not m.any():                     # bands narrower than the bin spacing
            m = np.zeros(len(freqs), dtype=bool)
            m[int(np.argmin(np.abs(freqs - fc)))] = True
        fb[i] = m / m.sum()
    return fb


def _slope_basis(u: np.ndarray) -> np.ndarray:
    """Row vector v with v @ y = the least-squares slope of y against u. np.polyfit does
    the same thing by solving a system per call, and this block is called a few hundred
    times."""
    d = u - u.mean()
    return d / (d @ d)


def _band_slices(freqs: np.ndarray, centres: np.ndarray) -> list[np.ndarray]:
    """Bin index arrays per band, for the within-band order statistics."""
    out = []
    for fc in centres:
        idx = np.flatnonzero((freqs >= fc * 2 ** -0.254) & (freqs <= fc * 2 ** 0.254))
        out.append(idx)
    return out


class _Analyzer:
    """One front end, several readouts. Everything target-dependent is fixed at build."""

    def __init__(self, target: np.ndarray, sr: int) -> None:
        import synth  # local, as in losses.band26_env: keeps the registry import cheap

        self.sr = sr
        centres = synth.eq_band_freqs()
        self.oct = np.log2(centres)

        self.lt_win = _hann(_LT_FFT)
        self.env_win = _hann(_ENV_FFT)
        lt_f = np.fft.rfftfreq(_LT_FFT, 1.0 / sr)
        env_f = np.fft.rfftfreq(_ENV_FFT, 1.0 / sr)
        self.fb_lt = _bandmat(lt_f, centres)
        self.fb_env = _bandmat(env_f, centres)
        self.slices = _band_slices(lt_f, centres)

        # Reference level, and therefore the floor, is a property of the TARGET alone, so
        # the same audio always scores the same however it is paired.
        lt_bins = _power_frames(target, _LT_FFT, _LT_FFT, self.lt_win).mean(0)
        lt = self.fb_lt @ lt_bins
        ev = self.fb_env @ _power_frames(target, _ENV_FFT, _ENV_HOP, self.env_win).T
        frel = 10.0 ** (_FLOOR_DB / 10.0)
        self.floor_lt = float(lt.max()) * frel
        self.floor_env = float(ev.max()) * frel
        # Separate reference for the per-bin statistic: a single bin at a partial sits
        # well above its band's average, so sharing the band floor would clip the valleys
        # the peak-to-valley term exists to measure.
        self.floor_bin = float(lt_bins.max()) * frel

        self.t_ref = ev.shape[1]
        fps = sr / _ENV_HOP
        mf = np.fft.rfftfreq(self.t_ref, 1.0 / fps)
        edges = np.geomspace(_MOD_LO, _MOD_HI, _N_MOD + 1)
        idx = np.searchsorted(mf, edges)
        # Collapse edges that fall in the same bin, so every modulation band is non-empty
        # and the feature has the same length whatever the clip duration.
        idx = np.unique(np.clip(idx, 1, len(mf) - 1))
        self.mod_edges = idx[:-1]
        self.mod_widths = np.diff(idx).astype(float)

        self.acf_lags = np.arange(int(_ACF_LO_S * fps), int(_ACF_HI_S * fps) + 1)
        self.tail_n = min(int(_TAIL_S * fps), self.t_ref)
        self.tail_basis = _slope_basis(np.arange(self.tail_n) / fps)   # slope in dB/s

        # Bands whose tail is audible in the TARGET. A decay rate measured on silence is
        # noise, and it would be compared against the pred's noise.
        tail_db = 10.0 * np.log10(ev[:, -self.tail_n:] + self.floor_env)
        self.tail_mask = (tail_db.mean(1) > 10.0 * np.log10(self.floor_env) + 6.0)

        self.mod_win = np.hanning(self.t_ref)
        self.m_lo = self.oct < np.log2(80.0)
        self.m_mid = (self.oct >= np.log2(160.0)) & (self.oct < np.log2(640.0))
        self.m_s1 = (self.oct >= np.log2(300.0)) & (self.oct < np.log2(2000.0))
        self.m_s2 = (self.oct >= np.log2(2000.0)) & (self.oct < np.log2(9000.0))
        self.fit_s1 = _slope_basis(self.oct[self.m_s1])
        self.fit_s2 = _slope_basis(self.oct[self.m_s2])

        # Audibility weight for the two blocks that read a band's SHAPE rather than its
        # level. Measured, not assumed: without it, adding white noise 85 dB down moved
        # the modulation block by 0.19, and all of that sat in the four bands more than 60
        # dB below the loudest one, where the pad has nothing and the noise IS the
        # envelope. Full weight down to -40 dB, none below -60. The hole it leaves (a band
        # the target is silent in is free) is closed by `spec`, which is unweighted.
        tspec = 10.0 * np.log10(lt + self.floor_lt)
        w = np.clip((tspec - tspec.max() + 60.0) / 20.0, 0.0, 1.0)
        # Renormalised so a target with few audible bands does not simply score lower,
        # bounded at 4x so a single surviving band cannot dominate.
        self.band_w = (w / max(float(w.mean()), 0.25))[:, None]

    # ---------------- readouts ----------------

    def _ptv(self, lt_bins: np.ndarray) -> np.ndarray:
        """Per band, top-decile bin level minus the median bin level, in dB.

        How far the partials stand above whatever fills the gaps between them, which is
        the one statistic that reads unison detune, uniMix and the wet effects directly.
        The project already measured that those gaps are load-bearing: a pure-tone bank at
        exactly the fitted partial amplitudes scored WORSE than the render it replaced,
        because it left the gaps empty. This measures the gaps without letting them
        explode, because both terms are order statistics over a floored band.
        """
        bdb = 10.0 * np.log10(lt_bins + self.floor_bin)
        ptv = np.zeros(len(self.slices))
        for i, idx in enumerate(self.slices):
            if len(idx) >= 8:                       # a band too narrow to have an inside
                b = bdb[idx]
                k = max(1, len(b) // 10)
                ptv[i] = np.partition(b, -k)[-k:].mean() - np.median(b)
        return ptv

    def blocks(self, x: np.ndarray, need: frozenset[str]) -> dict[str, np.ndarray]:
        """Only the requested blocks. The lens candidates each want a subset, and neither
        the per-bin order statistics nor the modulation transform is free."""
        out: dict[str, np.ndarray] = {}

        lt_bins = _power_frames(x, _LT_FFT, _LT_FFT, self.lt_win).mean(0)

        # -- static timbre: the EQ bank, the tilt, the filter's shelf, outGain ----------
        spec = 10.0 * np.log10(self.fb_lt @ lt_bins + self.floor_lt)
        if "spec" in need:
            out["spec"] = spec
        if "ptv" in need:
            out["ptv"] = self._ptv(lt_bins)
        if not need & {"bright", "bshape", "mod", "acf", "tail", "scal"}:
            return out

        ev = self.fb_env @ _power_frames(x, _ENV_FFT, _ENV_HOP, self.env_win).T
        # Length guard. The corpus renders all match the target, so this only ever fires
        # on a truncated render; padding with each band's own mean keeps the modulation
        # and autocorrelation blocks from seeing a fake edge.
        if ev.shape[1] > self.t_ref:
            ev = ev[:, :self.t_ref]
        elif ev.shape[1] < self.t_ref:
            pad = np.repeat(ev.mean(1, keepdims=True), self.t_ref - ev.shape[1], 1)
            ev = np.concatenate([ev, pad], axis=1)
        evf = ev + self.floor_env

        # -- brightness trajectory: cutoff, envAmt, kbdTrk and the filter ADSR ----------
        tot = evf.sum(0)
        cen = (self.oct @ evf) / tot
        if "bright" in need:
            # Rolloff interpolated inside the band it lands in, because the index itself
            # is a step function of cutoff and would give the optimiser a cliff.
            c = np.cumsum(evf, axis=0) / tot
            j = np.clip((c < 0.85).sum(0), 1, len(self.oct) - 1)
            c0 = np.take_along_axis(c, (j - 1)[None], 0)[0]
            c1 = np.take_along_axis(c, j[None], 0)[0]
            frac = np.clip((0.85 - c0) / np.maximum(c1 - c0, 1e-12), 0.0, 1.0)
            roll = self.oct[j - 1] + frac * (self.oct[j] - self.oct[j - 1])
            out["bright"] = np.stack([cen, roll])

        # -- per-band envelope shape: the amplitude ADSR, and the filter ADSR's imprint on
        # the high bands. Level removed per band, so this is shape only and the EQ gains
        # cannot pay for an envelope error.
        bshape = 10.0 * np.log10(evf)
        bshape = bshape - bshape.mean(1, keepdims=True)
        if "bshape" in need:
            out["bshape"] = bshape * self.band_w

        # -- modulation spectrum: lfoRate, lfoAmt, chRate, chDepth, detune beating -------
        if "mod" in need:
            amp = np.sqrt(evf)
            rel = amp / amp.mean(1, keepdims=True) - 1.0
            M = np.abs(np.fft.rfft(rel * self.mod_win, axis=-1))
            agg = np.add.reduceat(M, self.mod_edges, axis=-1)[:, :len(self.mod_widths)]
            out["mod"] = 20.0 * np.log10(agg / self.mod_widths + _MOD_FLOOR) * self.band_w

        # -- envelope autocorrelation: dlyTime and dlyFb, which repeat the whole envelope
        tdb = 10.0 * np.log10(tot)
        if "acf" in need:
            # Normalised by the envelope's own fluctuation but with an ABSOLUTE floor of
            # half a dB, and NOT renormalised by the zero lag afterwards. Either omission
            # hands a flat envelope its full-scale correlation curve back, computed from
            # numerical dust: on a bare sine, -85 dB of added noise moved this block by
            # 0.65 out of a 5.3 range before the floor and by 7e-5 after it. Below half a
            # dB there is no rhythm to correlate and the block should fade out, not blow up.
            z = (tdb - tdb.mean())
            z = z / max(float(np.sqrt((z * z).mean())), 0.5)
            Z = np.abs(np.fft.rfft(z, 2 * self.t_ref)) ** 2
            out["acf"] = np.fft.irfft(Z, 2 * self.t_ref)[self.acf_lags] / self.t_ref

        # -- per-band tail decay rate: revSize, revDamp, revWet and aR ------------------
        if "tail" in need:
            tail = bshape[:, -self.tail_n:]
            slope = (tail - tail.mean(1, keepdims=True)) @ self.tail_basis
            out["tail"] = np.clip(slope, -80.0, 20.0) * self.tail_mask

        # -- named scalars, each one a control read directly ----------------------------
        if "scal" in need:
            out["scal"] = np.array([
                spec[self.m_lo].mean() - spec[self.m_mid].mean(),   # subLvl balance
                self.fit_s1 @ spec[self.m_s1],                      # dB/oct, 300-2000 Hz
                self.fit_s2 @ spec[self.m_s2],                      # dB/oct, filter skirt
                spec.max() - spec.mean(),                           # reso prominence
                float(np.percentile(tdb, 95) - np.median(tdb)),     # amp env dynamics
                float(cen.max() - cen.min()),                       # filter env depth, oct
            ])
        return out


# Reciprocal scales: what counts as one unit of error in each block's own units. Chosen
# from the physics, not fitted. 6 dB is a clearly audible band error, half an octave a
# clearly audible brightness error, 10 dB/s a clearly different reverb tail.
_SCALE = {
    "spec": 1 / 6.0,
    "ptv": 1 / 6.0,
    "bright": 1 / 0.5,
    "bshape": 1 / 6.0,
    "mod": 1 / 6.0,
    "acf": 1 / 0.2,
    "tail": 1 / 10.0,
    "scal": 1 / np.array([6.0, 3.0, 3.0, 6.0, 6.0, 0.5]),
}


def _make(target: np.ndarray, sr: int, weights: dict[str, float]) -> Callable[[np.ndarray], float]:
    ana = _Analyzer(target, sr)
    need = frozenset(weights)
    ref = ana.blocks(target, need)

    def score(pred: np.ndarray) -> float:
        p, _ = match(pred, target)
        b = ana.blocks(p, need)
        tot = 0.0
        for k, w in weights.items():
            d = np.abs((b[k] - ref[k]) * _SCALE[k])
            tot += w * float(d.mean())
        return tot if np.isfinite(tot) else 1e6

    return score


@register("synth_controls")
def synth_controls(target: np.ndarray, sr: int = SR) -> Callable[[np.ndarray], float]:
    """Static timbre read as controls: band LTAS, peak-to-valley, brightness, scalars.

    Covers everything that survives time-averaging: the 26 EQ gains, tilt, outGain,
    cutoff/reso/kbdTrk, subLvl, sqrMix, detune and uniMix (through the peak-to-valley
    term) and the filter envelope's depth (through the brightness trajectory's range).
    Deliberately blind to delay time and reverb size, which is what `decay_profile` is
    for; a candidate that sees everything cannot tell the bake-off which lens paid.
    """
    return _make(target, sr, {"spec": 1.0, "ptv": 0.5, "bright": 0.6, "scal": 0.5})


@register("mod_spectrum")
def mod_spectrum(target: np.ndarray, sr: int = SR) -> Callable[[np.ndarray], float]:
    """Per-band envelope modulation spectra: unison beating, vibrato, chorus.

    detune sets the beat rates between the seven unison voices, lfoRate/lfoAmt a single
    coherent rate, chRate/chDepth a slow one. All three are invisible to a long-term
    spectrum and nearly invisible to a frame-wise STFT distance, because they change
    WHEN energy arrives in a band and not how much. Reading them as a spectrum of the
    band envelopes makes each one a location on an axis rather than a phase agreement.

    Carries a light static-spectrum anchor: without it a patch with the right wobble and
    the wrong timbre scores zero, and the bake-off's `discrim` screen would measure the
    anchor's absence rather than the modulation block's quality.
    """
    return _make(target, sr, {"mod": 1.0, "spec": 0.35})


@register("decay_profile")
def decay_profile(target: np.ndarray, sr: int = SR) -> Callable[[np.ndarray], float]:
    """Time structure: envelope autocorrelation plus per-band tail decay rates.

    The autocorrelation over 0.02 to 1.25 s covers dlyTime's whole range, and its height
    reads dlyFb and dlyWet, all three of which move nothing a long-term spectrum can see.
    The tail slopes are a per-band decay rate over the last 2.5 s, which is where reverb
    is unaccompanied: revSize sets the rate, revDamp sets its slant across bands, and aR
    sets where the voices stop contributing to it.

    Same static anchor as `mod_spectrum`, for the same reason.
    """
    return _make(target, sr, {"acf": 1.0, "tail": 0.8, "bshape": 0.5, "spec": 0.35})


@register("synth_full")
def synth_full(target: np.ndarray, sr: int = SR) -> Callable[[np.ndarray], float]:
    """Every block, weighted. The candidate actually proposed as the objective.

    The three lenses above each read a subset of the 29 macros, so each is rank-deficient
    on its own and could be minimised by a patch that is wrong on the axes it cannot see.
    The static block keeps the weight it has in `synth_controls`, since it is the only one
    that pins absolute level; the two time lenses are scaled to about 0.6 of their solo
    weight so that neither of them can outvote the timbre, which is what the patch is
    mostly made of. Read the gap between this and each lens as that lens's contribution.
    """
    return _make(target, sr, {
        "spec": 1.0, "ptv": 0.5, "bright": 0.6, "scal": 0.5,
        "bshape": 0.5, "mod": 0.6, "acf": 0.6, "tail": 0.5,
    })
