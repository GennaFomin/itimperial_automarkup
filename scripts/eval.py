#!/usr/bin/env python3
"""Метрики качества разметки: step-F1, ошибка границ, точность действий и объектов.

Считает то, что записано в целях кейса, и отдельно то, что нужно нам самим, чтобы
понимать, где именно ломается пайплайн:

* **step-F1@IoU** в двух вариантах — со сверкой меток (как оценивают нас) и без неё
  (чистое качество нарезки). Разница между ними сразу показывает, что чинить: границы
  или семантику.
* **ошибка границ** по сопоставленным сегментам: среднее и 95-й процентиль.
* **точность** отдельно по действию и по объекту — они ломаются по разным причинам.
* **edit score** и **пофреймовая точность** — стандартные метрики temporal action
  segmentation, по ним нас можно сравнить с литературой.

    python scripts/eval.py --gt devset/gt --pred out/pred
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from praxis.schema import Annotation, Level

IOU_THRESHOLDS = (0.1, 0.25, 0.5)
FRAME_STEP = 1 / 30


@dataclass
class Segment:
    start: float
    end: float
    action: str
    object: str | None

    @property
    def label(self) -> str:
        return f"{self.action}|{self.object or ''}"


def segments(annotation: Annotation) -> list[Segment]:
    return [
        Segment(step.start_sec, step.end_sec, step.action, step.object)
        for step in annotation.at_level(Level.coarse)
    ]


def iou(left: Segment, right: Segment) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union > 0 else 0.0


def match(truth: list[Segment], predicted: list[Segment], threshold: float, labelled: bool):
    """Жадное сопоставление предсказаний с эталоном, как принято в temporal action segmentation."""
    used = [False] * len(truth)
    pairs: list[tuple[int, int]] = []
    for predicted_index, prediction in enumerate(predicted):
        scores = [
            iou(prediction, reference)
            if (not labelled or prediction.label == reference.label)
            else 0.0
            for reference in truth
        ]
        if not scores:
            break
        best = max(range(len(scores)), key=lambda index: scores[index])
        if scores[best] >= threshold and not used[best]:
            used[best] = True
            pairs.append((predicted_index, best))
    return pairs


def f1(truth: list[Segment], predicted: list[Segment], threshold: float, labelled: bool) -> float:
    true_positive = len(match(truth, predicted, threshold, labelled))
    if not true_positive:
        return 0.0
    precision = true_positive / len(predicted)
    recall = true_positive / len(truth)
    return 2 * precision * recall / (precision + recall)


def edit_score(truth: list[Segment], predicted: list[Segment]) -> float:
    """Насколько последовательность шагов совпадает по порядку, без учёта времени."""
    reference = [segment.label for segment in truth]
    hypothesis = [segment.label for segment in predicted]
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distance = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distance[row][0] = row
    for column in range(columns):
        distance[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            cost = 0 if reference[row - 1] == hypothesis[column - 1] else 1
            distance[row][column] = min(
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
                distance[row - 1][column - 1] + cost,
            )
    longest = max(len(reference), len(hypothesis)) or 1
    return 1 - distance[-1][-1] / longest


def frame_accuracy(truth: list[Segment], predicted: list[Segment], duration: float) -> float:
    def label_at(items: list[Segment], time: float) -> str:
        for segment in items:
            if segment.start <= time < segment.end:
                return segment.label
        return "∅"

    times = [index * FRAME_STEP for index in range(int(duration / FRAME_STEP))]
    if not times:
        return 0.0
    hits = sum(1 for time in times if label_at(truth, time) == label_at(predicted, time))
    return hits / len(times)


def evaluate(pairs: list[tuple[Path, Path]]) -> dict:
    per_clip = []
    boundary_errors: list[float] = []
    action_hits = object_hits = both_hits = matched_total = 0
    latencies: list[float] = []

    for gt_path, pred_path in pairs:
        truth_annotation = Annotation.model_validate_json(gt_path.read_text(encoding="utf-8"))
        pred_annotation = Annotation.model_validate_json(pred_path.read_text(encoding="utf-8"))
        truth = segments(truth_annotation)
        predicted = segments(pred_annotation)
        duration = truth_annotation.video.duration_sec

        clip = {
            "clip": gt_path.stem,
            "gt_steps": len(truth),
            "pred_steps": len(predicted),
            "edit": round(edit_score(truth, predicted), 3),
            "frame_accuracy": round(frame_accuracy(truth, predicted, duration), 3),
        }
        for threshold in IOU_THRESHOLDS:
            clip[f"f1@{threshold}"] = round(f1(truth, predicted, threshold, labelled=True), 3)
            clip[f"f1@{threshold}_nolabel"] = round(
                f1(truth, predicted, threshold, labelled=False), 3
            )
        per_clip.append(clip)

        if pred_annotation.provenance.processing_sec:
            latencies.append(pred_annotation.provenance.processing_sec)

        for predicted_index, truth_index in match(truth, predicted, 0.5, labelled=False):
            prediction, reference = predicted[predicted_index], truth[truth_index]
            boundary_errors.append(abs(prediction.start - reference.start))
            boundary_errors.append(abs(prediction.end - reference.end))
            matched_total += 1
            action_hits += prediction.action == reference.action
            object_hits += prediction.object == reference.object
            both_hits += (
                prediction.action == reference.action and prediction.object == reference.object
            )

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 3) if values else 0.0

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return round(ordered[min(int(fraction * len(ordered)), len(ordered) - 1)], 3)

    summary = {
        "clips": len(per_clip),
        "boundary_mae_sec": mean(boundary_errors),
        "boundary_p95_sec": percentile(boundary_errors, 0.95),
        "matched_segments": matched_total,
        "action_accuracy": round(action_hits / matched_total, 3) if matched_total else 0.0,
        "object_accuracy": round(object_hits / matched_total, 3) if matched_total else 0.0,
        "action_object_accuracy": round(both_hits / matched_total, 3) if matched_total else 0.0,
        "edit": mean([clip["edit"] for clip in per_clip]),
        "frame_accuracy": mean([clip["frame_accuracy"] for clip in per_clip]),
        "processing_sec_mean": mean(latencies),
        "processing_sec_max": round(max(latencies), 3) if latencies else 0.0,
    }
    for threshold in IOU_THRESHOLDS:
        summary[f"f1@{threshold}"] = mean([clip[f"f1@{threshold}"] for clip in per_clip])
        summary[f"f1@{threshold}_nolabel"] = mean(
            [clip[f"f1@{threshold}_nolabel"] for clip in per_clip]
        )
    return {"summary": summary, "per_clip": per_clip}


def targets(summary: dict) -> list[str]:
    """Цели кейса — печатаем рядом с фактом, чтобы не сверять руками."""
    checks = [
        ("step-F1 >= 0.75", summary["f1@0.5"] >= 0.75, f"{summary['f1@0.5']:.3f}"),
        (
            "границы <= 2 c",
            summary["boundary_mae_sec"] <= 2.0,
            f"{summary['boundary_mae_sec']:.3f}",
        ),
        (
            "действие+объект >= 0.80",
            summary["action_object_accuracy"] >= 0.80,
            f"{summary['action_object_accuracy']:.3f}",
        ),
        (
            "обработка <= 120 c",
            summary["processing_sec_max"] <= 120,
            f"{summary['processing_sec_max']:.1f}",
        ),
    ]
    return [f"  {'✓' if ok else '✗'} {name}: {value}" for name, ok, value in checks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="каталог с эталонной разметкой")
    parser.add_argument("--pred", type=Path, required=True, help="каталог с предсказаниями")
    parser.add_argument("--json", type=Path, help="куда сохранить полный отчёт")
    args = parser.parse_args()

    pairs = []
    for gt_path in sorted(args.gt.glob("*.json")):
        pred_path = args.pred / gt_path.name
        if pred_path.exists():
            pairs.append((gt_path, pred_path))
        else:
            print(f"нет предсказания для {gt_path.name}")
    if not pairs:
        raise SystemExit("не нашлось ни одной пары эталон/предсказание")

    report = evaluate(pairs)
    summary = report["summary"]

    print(f"\nроликов: {summary['clips']}, сопоставлено сегментов: {summary['matched_segments']}\n")
    print(f"{'метрика':<28} {'со сверкой меток':>18} {'только нарезка':>16}")
    for threshold in IOU_THRESHOLDS:
        print(
            f"{'step-F1 @ IoU ' + str(threshold):<28}"
            f"{summary[f'f1@{threshold}']:>18.3f}{summary[f'f1@{threshold}_nolabel']:>16.3f}"
        )
    print()
    print(f"{'ошибка границ, среднее':<28}{summary['boundary_mae_sec']:>18.3f} с")
    print(f"{'ошибка границ, p95':<28}{summary['boundary_p95_sec']:>18.3f} с")
    print(f"{'точность действия':<28}{summary['action_accuracy']:>18.3f}")
    print(f"{'точность объекта':<28}{summary['object_accuracy']:>18.3f}")
    print(f"{'точность пары':<28}{summary['action_object_accuracy']:>18.3f}")
    print(f"{'edit score':<28}{summary['edit']:>18.3f}")
    print(f"{'пофреймовая точность':<28}{summary['frame_accuracy']:>18.3f}")
    print(f"{'обработка, среднее':<28}{summary['processing_sec_mean']:>18.3f} с")
    print("\nцели кейса:")
    print("\n".join(targets(summary)))

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nполный отчёт: {args.json}")


if __name__ == "__main__":
    main()
