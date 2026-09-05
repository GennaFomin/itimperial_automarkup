#!/usr/bin/env python3
"""Усреднение снятых вероятностей нескольких моделей в один файл для decode_sweep.

Ансамбль по чекпоинтам — это среднее вероятностей границ по кадрам, ровно то, что
делает /load с несколькими именами. Здесь то же самое офлайн: любые сочетания уже
снятых моделей меряются без повторного прогона роликов через сервис.

    python scripts/ensemble_scores.py work/scores/a_85.npz work/scores/b_85.npz --out work/scores/a+b_85.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def average(dumps: list[dict]) -> dict:
    """Среднее по роликам с одинаковыми именами; порядок и сетка — как у первого файла."""
    first = dumps[0]
    names = list(first["names"])
    index = [{name: k for k, name in enumerate(d["names"])} for d in dumps]
    scores = []
    for k, name in enumerate(names):
        rows = []
        for d, table in zip(dumps, index):
            if name not in table:
                raise SystemExit(f"в одном из файлов нет ролика {name}")
            row = np.asarray(d["scores"][table[name]], dtype=np.float32)
            if len(row) != len(first["scores"][k]):
                raise SystemExit(f"разная длина ряда у {name}: сетки признаков не совпадают")
            rows.append(row)
        scores.append(np.mean(rows, axis=0))
    return {
        "names": np.array(names),
        "scores": np.array(scores, dtype=object),
        "fps": first["fps"],
        "offsets": first["offsets"],
        "truths": first["truths"],
        "durations": first["durations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dumps", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    merged = average([dict(np.load(p, allow_pickle=True)) for p in args.dumps])
    np.savez(args.out, **merged)
    print(f"усреднено {len(args.dumps)} файлов, роликов {len(merged['names'])} → {args.out}")


if __name__ == "__main__":
    main()
