"""TW-FINCH — классический unsupervised-метод нарезки, без единого обученного параметра.

Зачем. В литературе по temporal action segmentation это базовая линия для случая, когда
меток нет вовсе: кадры кластеризуются по признакам с поправкой на время, и связные куски
одного кластера становятся шагами. Метод параметрический ровно в одном месте — число
кластеров, — и это удобно: гранулярность задаётся снаружи, как и требует кейс.

Оригинал: Sarfraz и др., «Temporally-Weighted Hierarchical Clustering for Unsupervised
Action Segmentation», CVPR 2021. Идея FINCH: соединить каждый кадр с ближайшим соседом,
компоненты связности этого графа и есть кластеры; повторяя, получаем иерархию. Временной
вес добавляет к близости признаков близость по времени, иначе одинаковые по виду куски
из разных мест ролика склеиваются в один шаг.
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


def temporal_similarity(features: np.ndarray, weight: float) -> np.ndarray:
    """Косинус между кадрами, умноженный на близость по времени.

    Без временного веса кластеризация склеивает похожие по виду куски из разных частей
    ролика: человек дважды берёт одну и ту же деталь, и это становится одним шагом.
    """
    matrix = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    similarity = matrix @ matrix.T
    positions = np.arange(len(features))[:, None] / max(len(features) - 1, 1)
    distance = np.abs(positions - positions.T)
    return similarity * (1.0 - weight * distance)


def finch_partition(similarity: np.ndarray) -> np.ndarray:
    """Один шаг FINCH: связь каждого элемента с ближайшим соседом, затем компоненты связности."""
    scores = similarity.copy()
    np.fill_diagonal(scores, -np.inf)
    neighbour = scores.argmax(axis=1)

    parent = np.arange(len(neighbour))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, other in enumerate(neighbour):
        left, right = root(index), root(int(other))
        if left != right:
            parent[left] = right

    labels = np.array([root(index) for index in range(len(neighbour))])
    _, compact = np.unique(labels, return_inverse=True)
    return compact


def cluster_to_target(features: np.ndarray, target: int, weight: float) -> np.ndarray:
    """Иерархия FINCH до нужного числа кластеров: сливаем, пока их больше цели."""
    labels = np.arange(len(features))
    current = features.copy()
    mapping = labels.copy()
    while len(np.unique(mapping)) > target and len(current) > target:
        partition = finch_partition(temporal_similarity(current, weight))
        if len(np.unique(partition)) >= len(current):
            break
        mapping = partition[mapping]
        current = np.stack(
            [features[mapping == index].mean(axis=0) for index in range(mapping.max() + 1)]
        )
    return mapping


def smooth_labels(labels: np.ndarray, window: int) -> np.ndarray:
    """Убирает покадровое дрожание меток: кластер меняется на один кадр и возвращается.

    Без этого связные куски рассыпаются на десятки обрывков — та же болезнь
    пересегментации, от которой в MS-TCN лечит сглаживающий член функции потерь.
    """
    if window < 3:
        return labels
    half = window // 2
    result = labels.copy()
    for index in range(len(labels)):
        first, last = max(0, index - half), min(len(labels), index + half + 1)
        values, counts = np.unique(labels[first:last], return_counts=True)
        result[index] = values[counts.argmax()]
    return result


@dataclass
class FinchSegmenter:
    """Нарезка кластеризацией кадров с временным весом."""

    weight: float = 0.5

    @property
    def name(self) -> str:
        return "tw-finch"

    def run(
        self,
        video_path: Path,
        meta: VideoMeta,
        vocabulary: Vocabulary,
        perception: Perception,
    ) -> PipelineResult:
        features = reduce_dimensions(normalise(perception.appearance), config.COMPONENTS)
        target = max(2, config.MAX_SEGMENTS)
        if len(features) <= target:
            return PipelineResult(
                steps=[self._whole(meta, vocabulary)], models={"segmenter": self.name}
            )

        labels = cluster_to_target(features, target, self.weight)
        labels = smooth_labels(labels, max(3, int(config.MIN_SEGMENT_SEC * perception.fps)))
        # Шаги — связные куски одного кластера, а не сам кластер: один и тот же кластер
        # может встретиться в ролике дважды, и это два разных шага.
        edges = [0, *(index for index in range(1, len(labels)) if labels[index] != labels[index - 1]),
                 len(labels)]

        steps = []
        for position in range(len(edges) - 1):
            first, last = edges[position], edges[position + 1]
            start = perception.offset + first / perception.fps
            end = min(meta.duration_sec, perception.offset + last / perception.fps)
            if end - start < config.MIN_SEGMENT_SEC:
                continue
            steps.append(
                Step(
                    id=len(steps), start_sec=round(start, 3), end_sec=round(end, 3),
                    action=vocabulary.actions[0], object=None,
                    keyframe_sec=round((start + end) / 2, 3), confidence=None,
                )
            )
        if not steps:
            steps = [self._whole(meta, vocabulary)]
        return PipelineResult(steps=steps, models={"segmenter": self.name})

    @staticmethod
    def _whole(meta: VideoMeta, vocabulary: Vocabulary) -> Step:
        return Step(
            id=0, start_sec=0.0, end_sec=meta.duration_sec,
            action=vocabulary.actions[0], object=None,
            keyframe_sec=round(meta.duration_sec / 2, 3), confidence=None,
        )
