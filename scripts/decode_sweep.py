#!/usr/bin/env python3
"""Перебор декодеров границ офлайн по снятым вероятностям.

Два семейства: пики (порог, минимальный интервал, выраженность) и штрафной DP по
логитам — разрезы выбираются так, чтобы сумма (logit − λ) была максимальна при
минимальной длине шага. Второе — тот же принцип, что у ядрового change-point, только
над выученной вероятностью, и число шагов решает штраф λ, а не порог.

    python scripts/decode_sweep.py work/scores/base_atomic85.npz
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

from praxis.metrics import Segment, f1
from praxis.pipeline.learned import peaks_above


def penalised_cuts(scores: np.ndarray, lam: float, minimum: int) -> list[int]:
    """DP: best[t] = лучшая сумма с последним разрезом в t; разрезы не ближе minimum."""
    logit = np.log(np.clip(scores, 1e-4, 1 - 1e-4) / (1 - np.clip(scores, 1e-4, 1 - 1e-4)))
    gain = logit - lam
    n = len(scores)
    best = np.full(n, -np.inf)
    prev = np.full(n, -1)
    running_best, running_arg = -np.inf, -1  # лучший best[j] для j ≤ t - minimum
    for t in range(minimum, n - minimum):
        j = t - minimum
        if j >= 0 and best[j] > running_best:
            running_best, running_arg = best[j], j
        base = max(0.0, running_best)
        best[t] = base + gain[t]
        prev[t] = running_arg if running_best > 0 else -1
    if not np.isfinite(best).any() or best.max() <= 0:
        return []
    t = int(best.argmax())
    cuts = []
    while t >= 0:
        cuts.append(t)
        t = int(prev[t])
    return sorted(cuts)


def steps_from_cuts(cuts: list[int], count: int, fps: float, offset: float, duration: float, min_sec: float) -> list[Segment]:
    edges = [0, *cuts, count]
    out = []
    for a, b in zip(edges, edges[1:]):
        start, end = offset + a / fps, min(duration, offset + b / fps)
        if end - start >= min_sec:
            out.append(Segment(start, end, "step", None))
    return out


def main() -> None:
    data = np.load(sys.argv[1], allow_pickle=True)
    clips = list(zip(data["names"], data["scores"], data["fps"], data["offsets"], data["truths"], data["durations"]))
    print(f"роликов: {len(clips)}, эталон шагов/ролик: {np.mean([len(t) for *_, t, _ in clips]):.1f}")
    results = []

    def score(make_cuts, label):
        f1s, dcount = [], []
        for _, s, fps, off, truth, dur in clips:
            truth_segments = [Segment(a, b, "step", None) for a, b in truth]
            cuts = make_cuts(np.asarray(s, dtype=np.float32), fps)
            pred = steps_from_cuts(cuts, len(s), fps, off, dur, 0.5)
            f1s.append(f1(truth_segments, pred, 0.5, "none"))
            dcount.append(len(pred) - len(truth_segments))
        results.append((float(np.mean(f1s)), float(np.mean(dcount)), label))

    for thr, gap, prom in itertools.product((0.3, 0.5, 0.7, 0.85), (0.5, 1.0, 1.5), (0.0, 0.1, 0.2)):
        score(lambda s, fps, thr=thr, gap=gap, prom=prom: peaks_above(s, thr, max(1, int(gap * fps)), prom),
              f"пики thr={thr} gap={gap}с prom={prom}")
    for lam, gap in itertools.product((0.5, 1.0, 2.0, 3.0, 4.0), (0.5, 1.0, 1.5)):
        score(lambda s, fps, lam=lam, gap=gap: penalised_cuts(s, lam, max(1, int(gap * fps))),
              f"DP λ={lam} gap={gap}с")

    results.sort(key=lambda r: -r[0])
    print("\nлучшие декодеры (средний F1@0.5 по роликам, Δшагов = наши − эталон):")
    for f, d, label in results[:12]:
        print(f"  {f:.3f}  Δ{d:+.1f}  {label}")
    base = [r for r in results if r[2] == "пики thr=0.7 gap=0.5с prom=0.0"][0]
    print(f"\nтекущее декодирование: {base[0]:.3f}  Δ{base[1]:+.1f}  {base[2]}")


if __name__ == "__main__":
    main()
