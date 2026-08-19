"""Differentiable time-varying lowpass: fi.resonlp(fc, q, 1) : fi.lowpass(2, fc).

A sample-by-sample biquad recursion in PyTorch would be 790k sequential steps per
render, which would make the surrogate slower than the Faust it is meant to replace.
Instead the chain is treated as locally LTI: fc is held constant over a block, the
exact digital cascade response is evaluated on an FFT grid, and the blocks are
overlap-added. The envelope driving fc is slow (fA is 0.16 s in the fitted patch),
so fc barely moves inside a block.

Two properties make this cheap approximation tight rather than hand-wavy:

  * The analysis window is a periodic Hann at 50 percent hop, which sums to exactly
    1.0 across frames, and no synthesis window is applied. So for *static* fc the
    overlap-add is algebraically exact linear filtering, not an approximation:
    sum_j (w_j x) * h == (sum_j w_j x) * h == x * h. The only error is time-aliasing
    from an FFT shorter than the impulse response.
  * That aliasing is bounded analytically. The cascade has four poles whose slowest
    decay rate follows from fc and q in closed form, so the FFT length is derived
    from the data (see _fft_len) instead of guessed. Low fc with high q is where the
    bound gets expensive; max_nfft caps the cost and _fft_len reports the cap by
    returning a shorter length than requested.

The coefficient formulas are transcribed from filters.lib (tf2s, resonlp,
lowpass0_highpass1). What that buys, all measured by verify_filter.py against real
Faust renders rather than asserted:

  * static fc, the coefficient test: 5e-7 relative L2 on the impulse response at
    fc = 5000 Hz, 1e-4 at 400 Hz, 7e-3 at 80 Hz with q = 8. The rise is time
    aliasing, not the coefficients: at 80 Hz the cascade rings for ~300 ms and
    _fft_len hits max_nfft. Magnitude response agrees within 0.45 dB everywhere
    down to 60 dB below the passband peak.
  * moving fc with the fitted envelope: mrstft 0.015 to 0.023 at block 512, which
    is about 1.5 percent of the stage-2 loss the surrogate has to reproduce.

Two honest limits. First, fc moving fast is where the block-constant assumption
costs the most, and it costs it exactly during the attack: with fA = 0.005 s and
envAmt = 6000 the first 20 ms carry an error of 38 percent of the signal RMS.
Second, _fft_len takes the *global* minimum of fc, so one voice dipping low sets
the FFT length for the whole batch, and below roughly 150 Hz with q above 4 no
affordable FFT is long enough: at fc = 30, q = 12 the error is 9 percent even at
nfft = 16384. The fitted patch sits at fc >= 242 Hz with q = 0.51, well inside the
region where the port is tight, but a search that walks cutoff down to its box
floor with high reso walks out of it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

SR = 44100
# Window length; hop is half of it. 512 is where the measured tradeoff sits: on MPS
# at the real render shape (V=29, N=789566) forward+backward costs 1.29 s against
# 0.97 s at 1024, and buys the fitted-envelope transient error down from 1.8e-2 to
# 1.0e-2 relative L2. Halving again to 256 costs 1.62 s for a further third, on a
# 460 MB spectrum buffer instead of 368 MB. Every block size in 128..2048 is
# measured in verify_filter.py, so the caller can trade differently.
BLOCK = 512
MAX_NFFT = 16384
TAIL_EPS = 1e-4     # residual impulse-response amplitude tolerated at FFT wrap

SQRT2 = math.sqrt(2.0)
# fi.lowpass(2, fc) is tf2s(0, 0, 1, a1s, 1, 2*pi*fc) with a1s = -2*cos(-3*pi/4).
BUTTER_A1 = SQRT2


# Allowed nfft/hop ratios. nfft must be a whole number of hops for the overlap-add
# below, and 5-smooth so the FFT stays fast; measured cost is proportional to
# frames*nfft and insensitive to the radix, so 1536 beats 2048 by a quarter.
_RATIOS = tuple(sorted(
    2**a * 3**b * 5**c
    for a in range(12) for b in range(3) for c in range(2)
    if 2 <= 2**a * 3**b * 5**c <= 512
))


def _slowest_rate(fc: float, q: float) -> float:
    """Slowest pole decay rate of the cascade, in nepers per second."""
    wc = 2.0 * math.pi * fc
    if q >= 0.5:
        reson = wc / (2.0 * q)
    else:  # overdamped: the pole nearest the origin decays slowest
        reson = wc * (1.0 / (2.0 * q) - math.sqrt(1.0 / (4.0 * q * q) - 1.0))
    return min(reson, wc / SQRT2)


def _fft_len(block: int, fc_min: float, q: float, sr: int, max_nfft: int) -> int:
    """Shortest allowed FFT length holding time-aliasing below TAIL_EPS.

    Returns a length shorter than the bound asks for when max_nfft binds, which is
    what happens for low fc with high q: the impulse response there is hundreds of
    milliseconds long and the honest FFT would cost more than the whole render.
    """
    hop = block // 2
    need = block + int(math.ceil(sr * math.log(1.0 / TAIL_EPS) / _slowest_rate(fc_min, q)))
    cap = max(max_nfft, block)
    chosen = block
    for r in _RATIOS:
        n = r * hop
        if n > cap:
            break
        chosen = n
        if n >= need:
            break
    return chosen


def cascade_response(fc: torch.Tensor, q: torch.Tensor, nfft: int, sr: int = SR) -> torch.Tensor:
    """Exact z-domain response of resonlp(fc,q,1) : lowpass(2,fc) on an rfft grid.

    fc is (..., ) in Hz, q is a 0-dim tensor. Returns (..., nfft//2+1) complex.
    """
    w = 2.0 * math.pi * torch.arange(nfft // 2 + 1, device=fc.device, dtype=fc.dtype) / nfft
    zi = torch.polar(torch.ones_like(w), -w)
    zi2 = zi * zi
    num = (1.0 + zi) ** 4  # (1 + 2z + z^2) from each section, shared numerator

    c = 1.0 / torch.tan(math.pi * fc / sr)
    csq = (c * c).unsqueeze(-1)
    c = c.unsqueeze(-1)
    mid = 2.0 * (1.0 - csq)

    # tf2s scales all five coefficients by d; keeping d in the denominator instead
    # of normalising avoids two divisions per section.
    a1_reson = 1.0 / q
    den = (
        (1.0 + a1_reson * c + csq) + mid * zi + (1.0 - a1_reson * c + csq) * zi2
    ) * (
        (1.0 + BUTTER_A1 * c + csq) + mid * zi + (1.0 - BUTTER_A1 * c + csq) * zi2
    )
    return num / den


def tv_lowpass(
    x: torch.Tensor,
    fc: torch.Tensor,
    q: torch.Tensor,
    sr: int = SR,
    block: int = BLOCK,
    max_nfft: int = MAX_NFFT,
) -> torch.Tensor:
    """(V, N) input, (V, N) cutoff in Hz already clamped to [30, 16000], scalar q."""
    if x.shape != fc.shape:
        raise ValueError(f"x {tuple(x.shape)} and fc {tuple(fc.shape)} must match")
    if block & (block - 1) or block < 4:
        raise ValueError(f"block must be a power of two >= 4, got {block}")
    v, n = x.shape
    hop = block // 2
    nfft = _fft_len(block, float(fc.detach().min()), float(q.detach()), sr, max_nfft)

    # Pad by one hop on the left so sample 0 is already covered by two windows, and
    # far enough on the right that the last sample is too and unfold divides evenly.
    pad_r = block + hop
    pad_r += -(n + hop + pad_r - block) % hop
    xp = F.pad(x, (hop, pad_r))
    fcp = F.pad(fc.unsqueeze(0), (hop, pad_r), mode="replicate").squeeze(0)

    win = torch.hann_window(block, periodic=True, device=x.device, dtype=x.dtype)
    frames = xp.unfold(-1, block, hop) * win                     # (V, M, block)
    fc_frames = fcp.unfold(-1, block, hop)
    fc_block = (fc_frames * win).sum(-1) / win.sum()             # (V, M)

    h = cascade_response(fc_block.reshape(-1), q, nfft, sr)
    spec = torch.fft.rfft(frames.reshape(-1, block), n=nfft)
    out = torch.fft.irfft(spec * h, n=nfft)

    # Overlap-add. F.fold does this but measured 5x slower than adding nfft/hop
    # shifted slices, because nfft is a whole number of hops: frame m contributes
    # its r-th hop-sized chunk to output hop-block m+r.
    m = out.shape[0] // v
    r = nfft // hop
    chunks = out.view(v, m, r, hop)
    y = sum(F.pad(chunks[:, :, k], (0, 0, k, r - 1 - k)) for k in range(r))
    return y.reshape(v, -1)[:, hop : hop + n]
