#!/usr/bin/env python3
"""Таблица F1@0.5 по снятым вероятностям: один декодер для всех наборов.

decode_sweep подбирает лучший декодер под каждый файл; здесь наоборот — декодер
фиксирован (как в приложении), и сравниваются модели на нескольких наборах сразу.
Файлы задаются как метка=путь; метка модели берётся из имени файла до «_».

    python scripts/score_table.py --thr 0.7 --gap 1.0 work/scores/base_85.npz work/scores/fine604-s2_85.npz
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from praxis.metrics import Segment, f1
from praxis.pipeline.learned import peaks_above
from scripts.decode_sweep import steps_from_cuts


def score(path: Path, thr: float, gap: float, min_sec: float) -> tuple[float, float, int]:
    data = np.load(path, allow_pickle=True)
    f1s, delta = [], []
    for s, fps, off, truth, dur in zip(
        data["scores"], data["fps"], data["offsets"], data["truths"], data["durations"]
    ):
        truth_segments = [Segment(a, b, "step", None) for a, b in truth]
        cuts = peaks_above(np.asarray(s, dtype=np.float32), thr, max(1, int(gap * fps)), 0.0)
        pred = steps_from_cuts(cuts, len(s), float(fps), float(off), float(dur), min_sec)
        f1s.append(f1(truth_segments, pred, 0.5, "none"))
        delta.append(len(pred) - len(truth_segments))
    return float(np.mean(f1s)), float(np.mean(delta)), len(f1s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F1@0.5 моделей на наборах при фиксированном декодере"
    )
    parser.add_argument("dumps", nargs="+", type=Path, help="файлы <модель>_<набор>.npz")
    parser.add_argument("--thr", type=float, default=0.7)
    parser.add_argument(
        "--gap", type=float, default=1.0, help="минимальный интервал между пиками, с"
    )
    parser.add_argument("--min-sec", type=float, default=0.5)
    args = parser.parse_args()
    table: dict[str, dict[str, tuple[float, float, int]]] = defaultdict(dict)
    sets: list[str] = []
    for path in args.dumps:
        model, _, subset = path.stem.partition("_")
        if subset not in sets:
            sets.append(subset)
        table[model][subset] = score(path, args.thr, args.gap, args.min_sec)
    print(
        f"декодер: пики, порог {args.thr}, интервал {args.gap} с, мин. шаг {args.min_sec} с; F1@0.5 (Δшагов)"
    )
    print("| модель | " + " | ".join(sets) + " |")
    print("| --- |" + " --- |" * len(sets))
    for model, row in table.items():
        cells = [f"{row[s][0]:.3f} ({row[s][1]:+.1f})" if s in row else "—" for s in sets]
        print(f"| {model} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
