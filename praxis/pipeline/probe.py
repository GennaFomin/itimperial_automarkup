"""Классификатор действия по признакам движения — то, чего языковая модель не умеет.

Зачем он вообще. Атомарные глаголы различаются направлением движения, а не картинкой:
«поднял» и «положил» дают почти одинаковые кадры и противоположный смысл. Мы измерили,
что языковая модель порядок кадров надёжно не воспринимает — подписи времени и контекст
соседей ничего не дали. А видеоэнкодер, обученный на Kinetics, кодирует именно движение:
для этого его и обучали.

Отсюда разделение труда: **движение решает глагол, языковая модель решает предмет.**

Классификатор нарочно простой — ближайший центроид по косинусу с опцией логистической
регрессии. Причина не в лени: размеченных примеров у нас будут десятки, а не тысячи, и
на таком объёме линейная модель поверх сильных признаков бьёт всё остальное. Обучение
занимает миллисекунды, поэтому его можно переобучать прямо во время разметки, как только
человек поправил несколько шагов.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def pool_segment(features: np.ndarray, fps: float, offset: float, start: float, end: float) -> np.ndarray:
    """Вектор сегмента: среднее по нему плюс разность концов.

    Одного среднего мало: у «поднял» и «положил» средние почти совпадают. Разность
    последнего и первого вектора несёт направление изменения — именно она их и разделяет.
    """
    first = max(0, int(round((start - offset) * fps)))
    last = min(len(features), max(first + 1, int(round((end - offset) * fps))))
    window = features[first:last]
    if not len(window):
        window = features[max(0, first - 1) : first + 1] or features[:1]
    mean = window.mean(axis=0)
    delta = window[-1] - window[0]
    return np.concatenate([mean, delta])


def normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=-1, keepdims=True) + 1e-8)


@dataclass
class NearestCentroid:
    """Ближайший центроид по косинусу. Требует единиц примеров на класс."""

    labels: list[str] = field(default_factory=list)
    centroids: np.ndarray | None = None

    def fit(self, vectors: np.ndarray, labels: list[str]) -> "NearestCentroid":
        self.labels = sorted(set(labels))
        index = {label: position for position, label in enumerate(self.labels)}
        sums = np.zeros((len(self.labels), vectors.shape[1]), dtype=np.float64)
        counts = np.zeros(len(self.labels))
        for vector, label in zip(normalise(vectors), labels):
            sums[index[label]] += vector
            counts[index[label]] += 1
        self.centroids = normalise(sums / np.maximum(counts, 1)[:, None])
        return self

    def scores(self, vectors: np.ndarray) -> np.ndarray:
        return normalise(vectors) @ self.centroids.T

    def predict(self, vectors: np.ndarray) -> list[str]:
        return [self.labels[index] for index in self.scores(vectors).argmax(axis=1)]


@dataclass
class LinearProbe:
    """Многоклассовая логистическая регрессия на градиентном спуске.

    Своя реализация вместо sklearn: одна матрица весов и сотня шагов Адама — это
    двадцать строк, а тянуть ради них зависимость в образ, который поедет на площадку,
    незачем.
    """

    steps: int = 400
    learning_rate: float = 0.2
    weight_decay: float = 1e-3
    labels: list[str] = field(default_factory=list)
    weights: np.ndarray | None = None

    def fit(self, vectors: np.ndarray, labels: list[str]) -> "LinearProbe":
        self.labels = sorted(set(labels))
        index = {label: position for position, label in enumerate(self.labels)}
        features = normalise(vectors)
        target = np.zeros((len(labels), len(self.labels)))
        for row, label in enumerate(labels):
            target[row, index[label]] = 1.0

        weights = np.zeros((features.shape[1], len(self.labels)))
        moment = np.zeros_like(weights)
        velocity = np.zeros_like(weights)
        for step in range(1, self.steps + 1):
            logits = features @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            gradient = features.T @ (probabilities - target) / len(features)
            gradient += self.weight_decay * weights
            moment = 0.9 * moment + 0.1 * gradient
            velocity = 0.999 * velocity + 0.001 * gradient**2
            weights -= self.learning_rate * (moment / (1 - 0.9**step)) / (
                np.sqrt(velocity / (1 - 0.999**step)) + 1e-8
            )
        self.weights = weights
        return self

    def scores(self, vectors: np.ndarray) -> np.ndarray:
        return normalise(vectors) @ self.weights

    def predict(self, vectors: np.ndarray) -> list[str]:
        return [self.labels[index] for index in self.scores(vectors).argmax(axis=1)]
