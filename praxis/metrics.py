"""Метрики качества разметки.

Считает то, что записано в целях кейса, и отдельно то, что нужно нам самим, чтобы
понимать, где именно ломается пайплайн:

* **метрика кейса** — `case_f1` с precision и recall: сопоставление один-к-одному при
  IoU >= 0.5 **и верном действии**. Предмет в матчинг не входит: правило кейса требует
  только класс действия. Это единственное число, по которому нас оценивают.
* **step-F1@IoU** ещё в двух вариантах — по паре «действие+предмет» (строже кейса) и
  вовсе без сверки меток (чистая нарезка). Разница между тремя вариантами сразу
  показывает, что чинить: границы, глагол или предмет.
* **ошибка границ** по сопоставленным сегментам: среднее и 95-й процентиль.
* **точность** отдельно по действию и по объекту — они ломаются по разным причинам.
* **edit score** и **пофреймовая точность** — стандартные метрики temporal action
  segmentation, по ним нас можно сравнить с литературой.
"""

from __future__ import annotations

import random
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

    def key(self, match_on: str) -> str | None:
        """Что должно совпасть, чтобы сегменты считались одним и тем же шагом."""
        if match_on == "action":
            return self.action
        if match_on == "pair":
            return self.label
        return None  # "none" — сверяем только время


def segments(annotation: Annotation) -> list[Segment]:
    return [
        Segment(step.start_sec, step.end_sec, step.action, step.object)
        for step in annotation.at_level(Level.coarse)
    ]


def iou(left: Segment, right: Segment) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union > 0 else 0.0


def match(truth: list[Segment], predicted: list[Segment], threshold: float, match_on: str):
    """Сопоставление один-к-одному, как требует правило кейса.

    Пары набираются по убыванию IoU, а не по порядку предсказаний: иначе ранний сегмент
    занимает эталон, которому лучше подошёл бы поздний, и пара теряется на ровном месте.
    При пороге 0.5 разницы нет — перекрытий внутри уровня схема не допускает, — но при
    мягких порогах порядок решает.
    """
    candidates = []
    for predicted_index, prediction in enumerate(predicted):
        for truth_index, reference in enumerate(truth):
            if prediction.key(match_on) != reference.key(match_on):
                continue
            score = iou(prediction, reference)
            if score >= threshold:
                candidates.append((score, predicted_index, truth_index))

    candidates.sort(key=lambda item: -item[0])
    used_truth: set[int] = set()
    used_predicted: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, predicted_index, truth_index in candidates:
        if predicted_index in used_predicted or truth_index in used_truth:
            continue
        used_predicted.add(predicted_index)
        used_truth.add(truth_index)
        pairs.append((predicted_index, truth_index))
    return pairs


def counts(
    truth: list[Segment], predicted: list[Segment], threshold: float, match_on: str
) -> tuple[int, int, int]:
    """Совпало, лишнее, пропущенное. Из этих трёх чисел считаются precision, recall и F1."""
    true_positive = len(match(truth, predicted, threshold, match_on))
    return true_positive, len(predicted) - true_positive, len(truth) - true_positive


def prf(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float, float]:
    """precision, recall и F1 из трёх счётчиков."""
    predicted = true_positive + false_positive
    actual = true_positive + false_negative
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    total = precision + recall
    return precision, recall, 2 * precision * recall / total if total else 0.0


def f1(truth: list[Segment], predicted: list[Segment], threshold: float, match_on: str) -> float:
    return prf(*counts(truth, predicted, threshold, match_on))[2]


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


def evaluate(pairs: list[tuple[Path, Path | None]]) -> dict:
    """Оценка по парам файлов эталон/предсказание.

    Предсказание может отсутствовать: ролик, на котором пайплайн упал, обязан попасть в
    знаменатель полным промахом, иначе падение улучшает метрику.
    """
    return evaluate_annotations(
        [
            (
                Annotation.model_validate_json(gt.read_text(encoding="utf-8")),
                Annotation.model_validate_json(pred.read_text(encoding="utf-8"))
                if pred is not None
                else None,
                gt.stem,
            )
            for gt, pred in pairs
        ]
    )


def evaluate_annotations(items: list[tuple[Annotation, Annotation | None, str]]) -> dict:
    """Оценка по уже загруженным разметкам — этим пользуется подбор параметров."""
    per_clip = []
    boundary_errors: list[float] = []
    action_hits = object_hits = both_hits = matched_total = 0
    latencies: list[float] = []
    # Пул по всему набору для метрики кейса: «затем считаем precision, recall и F1»
    # естественнее читается как общий счёт, а не среднее по роликам.
    case_tp = case_fp = case_fn = 0
    missing = 0

    for truth_annotation, pred_annotation, name in items:
        truth = segments(truth_annotation)
        predicted = segments(pred_annotation) if pred_annotation is not None else []
        missing += pred_annotation is None
        duration = truth_annotation.video.duration_sec

        true_positive, false_positive, false_negative = counts(truth, predicted, 0.5, "action")
        case_tp += true_positive
        case_fp += false_positive
        case_fn += false_negative

        clip = {
            "clip": name,
            "gt_steps": len(truth),
            "pred_steps": len(predicted),
            "case_f1": round(prf(true_positive, false_positive, false_negative)[2], 3),
            "edit": round(edit_score(truth, predicted), 3),
            "frame_accuracy": round(frame_accuracy(truth, predicted, duration), 3),
        }
        for threshold in IOU_THRESHOLDS:
            clip[f"f1@{threshold}"] = round(f1(truth, predicted, threshold, "pair"), 3)
            clip[f"f1@{threshold}_nolabel"] = round(f1(truth, predicted, threshold, "none"), 3)
        per_clip.append(clip)

        if pred_annotation is not None and pred_annotation.provenance.processing_sec:
            latencies.append(pred_annotation.provenance.processing_sec)

        for predicted_index, truth_index in match(truth, predicted, 0.5, "none"):
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

    def interval(values: list[float]) -> list[float]:
        """Границы 90% для среднего по роликам — бутстрэп с пересэмплированием роликов.

        На сорока роликах разброс среднего сравним с разницей между вариантами пайплайна.
        Без интервала легко принять шум за улучшение и закрепить в коде случайность.
        """
        if len(values) < 2:
            return [mean(values), mean(values)]
        generator = random.Random(20260901)
        count = len(values)
        means = sorted(
            statistics.fmean(generator.choices(values, k=count)) for _ in range(2000)
        )
        return [round(means[100], 3), round(means[1900], 3)]

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return round(ordered[min(int(fraction * len(ordered)), len(ordered) - 1)], 3)

    case_precision, case_recall, case_f1 = prf(case_tp, case_fp, case_fn)
    summary = {
        "clips": len(per_clip),
        "clips_without_prediction": missing,
        # Метрика кейса: один-к-одному, IoU >= 0.5, верное действие. Пул по всему набору.
        "case_f1": round(case_f1, 3),
        "case_precision": round(case_precision, 3),
        "case_recall": round(case_recall, 3),
        "case_f1_macro": mean([clip["case_f1"] for clip in per_clip]),
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
    summary["ci90"] = {
        key: interval([clip[key] for clip in per_clip])
        for key in ["case_f1", *(f"f1@{t}_nolabel" for t in IOU_THRESHOLDS)]
    }
    return {"summary": summary, "per_clip": per_clip}


