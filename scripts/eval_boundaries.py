#!/usr/bin/env python3
"""Разбор качества границ отдельно от сегментов.

Зачем отдельный замер. step-F1 смешивает две разные вещи: насколько точно поставлена
каждая граница и сколько их поставлено всего. Метод может выигрывать, потому что бьёт
точнее, а может — потому что бьёт чаще и случайно накрывает больше. По одному числу это
неразличимо, а лечится по-разному.

Здесь границы считаются как точки. Для набора допусков печатаются:

* recall — доля эталонных границ, рядом с которыми нашлась предсказанная;
* precision — доля предсказанных границ, попавших рядом с эталонной;
* смещение — знаковая ошибка попавших, чтобы отличить систематический сдвиг от разброса.

    python scripts/eval_boundaries.py --gt data/pool_atomic/gt \\
        --pred kernel=/tmp/pred_kernel learned=/tmp/pred_learned
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from praxis.schema import Annotation

TOLERANCES = (0.25, 0.5, 1.0)


def boundaries(annotation: Annotation) -> list[float]:
    """Моменты смены: начала и концы шагов без краёв ролика."""
    edges = sorted({step.start_sec for step in annotation.steps}
                   | {step.end_sec for step in annotation.steps})
    duration = annotation.video.duration_sec
    return [edge for edge in edges if 0.05 < edge < duration - 0.05]


def match_points(truth: list[float], predicted: list[float], tolerance: float):
    """Жадное сопоставление точек по возрастанию расстояния, один к одному."""
    pairs = sorted(
        ((abs(p - t), pi, ti) for pi, p in enumerate(predicted) for ti, t in enumerate(truth)),
        key=lambda item: item[0],
    )
    used_p, used_t, matched = set(), set(), []
    for distance, pi, ti in pairs:
        if distance > tolerance or pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matched.append(predicted[pi] - truth[ti])
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", nargs="+", required=True, help="имя=каталог")
    args = parser.parse_args()

    references = sorted(args.gt.glob("*.json"))
    truths = {
        path.stem: Annotation.model_validate_json(path.read_text(encoding="utf-8"))
        for path in references
    }

    print(f"\nроликов: {len(truths)}, эталонных границ: "
          f"{sum(len(boundaries(a)) for a in truths.values())}\n")
    header = f"{'способ':<22}{'границ':>8}"
    for tolerance in TOLERANCES:
        header += f"{f'±{tolerance}с recall':>15}{'precision':>11}"
    print(header + f"{'смещение':>11}")

    for entry in args.pred:
        name, _, directory = entry.partition("=")
        total_pred = 0
        hits = {t: 0 for t in TOLERANCES}
        offsets: list[float] = []
        total_truth = 0

        for stem, truth in truths.items():
            path = Path(directory) / f"{stem}.json"
            if not path.exists():
                total_truth += len(boundaries(truth))
                continue
            guess = Annotation.model_validate_json(path.read_text(encoding="utf-8"))
            true_points, pred_points = boundaries(truth), boundaries(guess)
            total_truth += len(true_points)
            total_pred += len(pred_points)
            for tolerance in TOLERANCES:
                matched = match_points(true_points, pred_points, tolerance)
                hits[tolerance] += len(matched)
                if tolerance == 0.5:
                    offsets += matched

        row = f"{name:<22}{total_pred:>8}"
        for tolerance in TOLERANCES:
            recall = hits[tolerance] / max(total_truth, 1)
            precision = hits[tolerance] / max(total_pred, 1)
            row += f"{recall:>15.3f}{precision:>11.3f}"
        bias = statistics.fmean(offsets) if offsets else 0.0
        print(row + f"{bias:>+11.3f}")


if __name__ == "__main__":
    main()
