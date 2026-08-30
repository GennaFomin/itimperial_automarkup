"""Метрики качества разметки.

Считает то, что записано в целях кейса, и отдельно то, что нужно нам самим, чтобы
понимать, где именно ломается пайплайн:

* **step-F1@IoU** в двух вариантах — со сверкой меток (как оценивают нас) и без неё
  (чистое качество нарезки). Разница между вариантами сразу показывает, что чинить:
  границы или семантику; по одной цифре F1 это неразличимо.
* **ошибка границ** по сопоставленным сегментам: среднее и 95-й процентиль.
* **точность** отдельно по действию и по объекту — они ломаются по разным причинам.
* **edit score** и **пофреймовая точность** — стандартные метрики temporal action
  segmentation, по ним нас можно сравнить с литературой.
"""

from __future__ import annotations

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
    """Оценка по парам файлов эталон/предсказание."""
    return evaluate_annotations(
        [
            (
                Annotation.model_validate_json(gt.read_text(encoding="utf-8")),
                Annotation.model_validate_json(pred.read_text(encoding="utf-8")),
                gt.stem,
            )
            for gt, pred in pairs
        ]
    )


def evaluate_annotations(items: list[tuple[Annotation, Annotation, str]]) -> dict:
    """Оценка по уже загруженным разметкам — этим пользуется подбор параметров."""
    per_clip = []
    boundary_errors: list[float] = []
    action_hits = object_hits = both_hits = matched_total = 0
    latencies: list[float] = []

    for truth_annotation, pred_annotation, name in items:
        truth = segments(truth_annotation)
        predicted = segments(pred_annotation)
        duration = truth_annotation.video.duration_sec

        clip = {
            "clip": name,
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


