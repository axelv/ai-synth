"""Does render throughput scale across processes? CMA-ES popsize 16 is embarrassingly
parallel, so this decides whether a generation costs 16 renders or 16/N.

Each worker builds its own PadRenderer (its own Faust engine, its own MIDI) and renders
the full clip N_REP times. Reported as renders per second, aggregate.

Run: PYTHONPATH=scripts uv run python .scratch/app-design-flow/research/bench_par.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time

N_REP = 6


def worker(seed: int, q) -> None:
    import numpy as np
    import stage2
    import synth
    from bend2 import bend_curve

    r = synth.PadRenderer(n_voices=24, dsp=synth.DSP)
    r.set_notes(stage2.load_notes())
    r.set_bend(bend_curve(int(stage2.DUR * stage2.SR) + stage2.SR))
    rng = np.random.default_rng(seed)
    x = stage2.seeded_start()
    t0 = time.perf_counter()
    for _ in range(N_REP):
        z = np.clip(x + rng.normal(0, 0.02, len(x)), 0, 1)
        r.set_params(synth.denorm(z))
        r.render(stage2.DUR)
    q.put(time.perf_counter() - t0)


def run(n: int) -> dict:
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(i, q)) for i in range(n)]
    t0 = time.perf_counter()
    for p in ps:
        p.start()
    times = [q.get() for _ in ps]
    for p in ps:
        p.join()
    wall = time.perf_counter() - t0
    return {"workers": n, "wall_s": wall, "worker_render_loops_s": times,
            "renders": n * N_REP, "renders_per_s": n * N_REP / wall,
            "s_per_render_in_worker": sum(times) / (n * N_REP)}


if __name__ == "__main__":
    mp.set_start_method("spawn")
    out = []
    for n in (1, 2, 4, 6, 8):
        r = run(n)
        out.append(r)
        print(f"{n} workers: {r['renders']} renders in {r['wall_s']:.1f} s wall "
              f"-> {r['renders_per_s']:.2f} renders/s "
              f"({r['s_per_render_in_worker']:.3f} s per render inside a worker)")
    base = out[0]["renders_per_s"]
    for r in out:
        print(f"  speedup x{r['renders_per_s']/base:.2f} at {r['workers']} workers")
    json.dump(out, open(".scratch/app-design-flow/research/bench_par.json", "w"), indent=1)
    print("wrote .scratch/app-design-flow/research/bench_par.json")
