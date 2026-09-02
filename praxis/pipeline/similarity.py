"""Границы по матрице самоподобия — два метода, которым не нужно обучение.

Зачем. Обученные сегментаторы публикуют веса, но прибиты к своей таксономии: MS-TCN на
50 Salads знает 17 своих классов, на Assembly101 — 202 своих. Под чужой словарь их не
переставить. Зато сама задача «где кончилось одно и началось другое» класс-агностична, и
на ней работают методы вообще без обучаемых параметров:

* **recursive** — рекурсивный разбор матрицы самоподобия в духе UBoCo (F1 0.703 на
  Kinetics-GEBD без обучения): ищем разрез, максимально разделяющий похожее внутри
  блоков и непохожее между ними, и рекурсивно повторяем внутри частей.
* **kernel** — точный change-point с ядровой стоимостью и штрафом за сегмент, как у
  Perochon и Oudre (ICASSP 2023): на EPIC-KITCHENS-55 их unsupervised обгонял обученный
  метод по f1@50. Отличается от нашего motion-dp тем, что стоимость считается по ядру
  Грама, а не по расстоянию до среднего сегмента.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from praxis import config
from praxis.pipeline.base import Perception, PipelineResult
from praxis.pipeline.segment import normalise, reduce_dimensions
from praxis.schema import Step, VideoMeta
from praxis.vocab import Vocabulary


def gram(features: np.ndarray) -> np.ndarray:
    """Матрица самоподобия: косинус между всеми парами моментов."""
    matrix = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    return matrix @ matrix.T


def contrast(similarity: np.ndarray, start: int, end: int, split: int) -> float:
    """Насколько разрез в точке split разделяет отрезок [start, end).

    Внутри блоков кадры должны быть похожи, между блоками — нет. Разность этих двух
    средних и есть мера того, что здесь действительно граница.
    """
    left, right = similarity[start:split, start:split], similarity[split:end, split:end]
    across = similarity[start:split, split:end]
    if not left.size or not right.size or not across.size:
        return -1.0
    inside = (left.sum() + right.sum()) / (left.size + right.size)
    return float(inside - across.mean())


def recursive_split(
    similarity: np.ndarray, start: int, end: int, minimum: int, threshold: float
) -> list[int]:
    """Рекурсивный разбор: режем там, где контраст максимален, пока он значим."""
    if end - start < 2 * minimum:
        return []
    scores = [
        (contrast(similarity, start, end, split), split)
        for split in range(start + minimum, end - minimum + 1)
    ]
    if not scores:
        return []
    best, position = max(scores)
    if best < threshold:
        return []
    return [
        *recursive_split(similarity, start, position, minimum, threshold),
        position,
        *recursive_split(similarity, position, end, minimum, threshold),
    ]


def kernel_costs(similarity: np.ndarray) -> np.ndarray:
    """Ядровая стоимость каждого отрезка [i, j): сумма диагонали минус блок, делённый на длину.

    Это стандартная стоимость kernel change-point detection. Считается через
    интегральные суммы, поэтому вся таблица строится за O(T^2), а не за O(T^3).
    """
    length = similarity.shape[0]
    padded = np.zeros((length + 1, length + 1))
    padded[1:, 1:] = np.cumsum(np.cumsum(similarity, axis=0), axis=1)
    diagonal = np.concatenate([[0.0], np.cumsum(np.diag(similarity))])

    costs = np.full((length + 1, length + 1), np.inf)
    for i in range(length):
        j = np.arange(i + 1, length + 1)
        block = padded[j, j] - padded[i, j] - padded[j, i] + padded[i, i]
        costs[i, i + 1 :] = (diagonal[j] - diagonal[i]) - block / (j - i)
    return costs


def penalised_segmentation(costs: np.ndarray, penalty: float, minimum: int) -> list[int]:
    """Оптимальное разбиение при штрафе за каждый сегмент — точное DP за O(T^2)."""
    length = costs.shape[0] - 1
    best = np.full(length + 1, np.inf)
    best[0] = 0.0
    previous = np.zeros(length + 1, dtype=int)
    for end in range(minimum, length + 1):
        starts = np.arange(0, end - minimum + 1)
        values = best[starts] + costs[starts, end] + penalty
        index = int(np.argmin(values))
        best[end] = values[index]
        previous[end] = starts[index]

    bounds: list[int] = []
    cursor = length
    while cursor > 0:
        bounds.append(int(previous[cursor]))
        cursor = previous[cursor]
    return sorted(set(bounds) - {0})


@dataclass
class SimilaritySegmenter:
    """Нарезка по матрице самоподобия. mode: recursive или kernel."""

    mode: str = "recursive"

    @property
    def name(self) -> str:
        return f"tsm-{self.mode}"

    def run(
        self,
        video_path: Path,
        meta: VideoMeta,
        vocabulary: Vocabulary,
        perception: Perception,
    ) -> PipelineResult:
        features = reduce_dimensions(normalise(perception.appearance), config.COMPONENTS)
        if len(features) < 4:
            return PipelineResult(
                steps=[self._whole(meta, vocabulary)], models={"segmenter": self.name}
            )

        similarity = gram(features)
        minimum = max(2, int(config.MIN_SEGMENT_SEC * perception.fps))
        if self.mode == "kernel":
            cuts = penalised_segmentation(
                kernel_costs(similarity), config.TSM_PENALTY, minimum
            )
        else:
            cuts = recursive_split(similarity, 0, len(features), minimum, config.TSM_THRESHOLD)

        edges = [0, *sorted(cuts), len(features)]
        steps = []
        for index in range(len(edges) - 1):
            first, last = edges[index], edges[index + 1]
            # Простой отрезок — это пауза, а не шаг: кейс разрешает пробелы между шагами.
            if np.mean(perception.motion[first:last]) < np.mean(perception.motion) * config.IDLE_RATIO:
                continue
            start = perception.offset + first / perception.fps
            end = min(meta.duration_sec, perception.offset + last / perception.fps)
            if end - start < config.MIN_SEGMENT_SEC:
                continue
            steps.append(
                Step(
                    id=len(steps),
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                    action=vocabulary.actions[0],
                    object=None,
                    keyframe_sec=round((start + end) / 2, 3),
                    confidence=None,
                )
            )
        if not steps:
            steps = [self._whole(meta, vocabulary)]
        return PipelineResult(steps=steps, models={"segmenter": self.name})

    @staticmethod
    def _whole(meta: VideoMeta, vocabulary: Vocabulary) -> Step:
        return Step(
            id=0,
            start_sec=0.0,
            end_sec=meta.duration_sec,
            action=vocabulary.actions[0],
            object=None,
            keyframe_sec=round(meta.duration_sec / 2, 3),
            confidence=None,
        )
