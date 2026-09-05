#!/usr/bin/env python3
"""Продуктовые метрики детектора границ по снятым вероятностям.

Кроме step-F1 при IoU 0.5 (метрика кейса) считаются метрики самих границ — те, что
описывают качество разметки для человека, который её правит: доля эталонных границ,
рядом с которыми (в пределах допуска) стоит наш разрез, доля наших разрезов, у которых
есть эталон, и средняя ошибка совпавших границ. Допуск 2 с — требование кейса,
1 с и 0.5 с — строже.

    python scripts/boundary_report.py work/scores/final_85.npz --thr 0.5 --gap 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from praxis.metrics import Segment, f1
from praxis.pipeline.learned import peaks_above
from scripts.decode_sweep import steps_from_cuts


def boundary_match(
    truth: list[float], ours: list[float], tolerance: float
) -> tuple[int, int, list[float]]:
    """Жадное одно-к-одному сопоставление границ по близости; возвращает совпадения
    с каждой стороны и ошибки совпавших пар."""
    pairs = sorted((abs(t - o), i, j) for i, t in enumerate(truth) for j, o in enumerate(ours))
    used_t: set[int] = set()
    used_o: set[int] = set()
    errors: list[float] = []
    for err, i, j in pairs:
        if err > tolerance:
            break
        if i in used_t or j in used_o:
            continue
        used_t.add(i)
        used_o.add(j)
        errors.append(err)
    return len(used_t), len(used_o), errors


def report(
    path: Path, thr: float, gap: float, min_sec: float, tolerances: tuple[float, ...]
) -> dict:
    data = np.load(path, allow_pickle=True)
    step_f1 = {0.5: [], 0.25: []}
    hits = {
        t: [0, 0, 0, 0] for t in tolerances
    }  # найдено эталонных, всего эталонных, верных наших, всего наших
    errors = {t: [] for t in tolerances}
    count_delta = []
    for s, fps, off, truth, dur in zip(
        data["scores"], data["fps"], data["offsets"], data["truths"], data["durations"]
    ):
        s = np.asarray(s, dtype=np.float32)
        cuts = peaks_above(s, thr, max(1, int(gap * fps)), 0.0)
        pred = steps_from_cuts(cuts, len(s), float(fps), float(off), float(dur), min_sec)
        truth_segments = [Segment(a, b, "step", None) for a, b in truth]
        for iou in step_f1:
            step_f1[iou].append(f1(truth_segments, pred, iou, "none"))
        count_delta.append(len(pred) - len(truth_segments))
        # Границы — внутренние моменты смены: начала и концы шагов без краёв ролика.
        truth_b = sorted({m for a, b in truth for m in (a, b) if 0.0 < m < float(dur)})
        ours_b = [float(off) + c / float(fps) for c in cuts]
        for t in tolerances:
            found, correct, errs = boundary_match(truth_b, ours_b, t)
            hits[t][0] += found
            hits[t][1] += len(truth_b)
            hits[t][2] += correct
            hits[t][3] += len(ours_b)
            errors[t].extend(errs)
    out = {
        "clips": len(count_delta),
        "step_f1_iou50": float(np.mean(step_f1[0.5])),
        "step_f1_iou25": float(np.mean(step_f1[0.25])),
        "count_delta": float(np.mean(count_delta)),
    }
    for t in tolerances:
        found, total, correct, ours = hits[t]
        recall = found / max(1, total)
        precision = correct / max(1, ours)
        out[t] = {
            "recall": recall,
            "precision": precision,
            "f1": 2 * precision * recall / max(1e-9, precision + recall),
            "mae": float(np.mean(errors[t])) if errors[t] else float("nan"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="продуктовые метрики детектора границ")
    parser.add_argument("dumps", nargs="+", type=Path)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--gap", type=float, default=0.5)
    parser.add_argument("--min-sec", type=float, default=0.5)
    args = parser.parse_args()
    tolerances = (0.5, 1.0, 2.0)
    print(f"декодер: пики выше {args.thr}, интервал {args.gap} с, мин. шаг {args.min_sec} с")
    print(
        "| набор | роликов | step-F1@0.5 | step-F1@0.25 | Δшагов | границы ±0.5 с P / R / F1 | ±1 с P / R / F1 | ±2 с P / R / F1 | MAE ±2 с |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for path in args.dumps:
        r = report(path, args.thr, args.gap, args.min_sec, tolerances)
        cells = [
            path.stem,
            str(r["clips"]),
            f"{r['step_f1_iou50']:.3f}",
            f"{r['step_f1_iou25']:.3f}",
            f"{r['count_delta']:+.1f}",
        ]
        for t in tolerances:
            b = r[t]
            cells.append(f"{b['precision']:.2f} / {b['recall']:.2f} / {b['f1']:.2f}")
        cells.append(f"{r[2.0]['mae']:.2f} с")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
