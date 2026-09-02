"""Тривиальные и классические способы нарезки — линейка, без которой числа не читаются.

Кейсодатель формулирует это прямо: «сначала baseline… цель — получить первый полный
прогон и метрику». Без нижней планки любое число выглядит достижением. А планка тут
неожиданно высокая: в T-PIVOT равномерная нарезка без всякой модели дала IoU 49.1
против 52.0 у GPT-4o с бинарным поиском по времени.

Методы:

* **single** — один шаг на весь ролик. Абсолютный ноль.
* **uniform** — равномерная нарезка на K частей. Не смотрит на видео вовсе.
* **motion** — разрезы в локальных минимумах движения, без всяких признаков.
* **peaks** — разрезы там, где соседние векторы признаков сильнее всего расходятся;
  классическая детекция смены, но без оптимизации по всему ролику.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from praxis import config
from praxis.pipeline.base import Perception, PipelineResult
from praxis.pipeline.segment import boundary_scores, normalise, reduce_dimensions
from praxis.schema import Step, VideoMeta
from praxis.vocab import Vocabulary


@dataclass
class BaselineSegmenter:
    mode: str = "uniform"
    parts: int = 3

    @property
    def name(self) -> str:
        return f"baseline-{self.mode}"

    def run(
        self,
        video_path: Path,
        meta: VideoMeta,
        vocabulary: Vocabulary,
        perception: Perception,
    ) -> PipelineResult:
        cuts = self._cuts(perception)
        edges = [0.0, *cuts, meta.duration_sec]
        steps = []
        for index in range(len(edges) - 1):
            start, end = edges[index], edges[index + 1]
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
            steps = [
                Step(
                    id=0,
                    start_sec=0.0,
                    end_sec=meta.duration_sec,
                    action=vocabulary.actions[0],
                    object=None,
                    keyframe_sec=round(meta.duration_sec / 2, 3),
                    confidence=None,
                )
            ]
        return PipelineResult(steps=steps, models={"segmenter": self.name})

    def _cuts(self, perception: Perception) -> list[float]:
        duration = len(perception.motion) / perception.fps
        if self.mode == "single":
            return []
        if self.mode == "uniform":
            return [duration * (i + 1) / self.parts for i in range(self.parts - 1)]

        if self.mode == "motion":
            # Локальные минимумы движения: между действиями человек обычно замирает.
            signal = perception.motion
            scores = boundary_scores(signal)
        else:
            features = reduce_dimensions(normalise(perception.appearance), config.COMPONENTS)
            # Расхождение соседних векторов — простейшая детекция смены содержимого.
            diff = np.linalg.norm(np.diff(features, axis=0), axis=1)
            scores = np.concatenate([[0.0], diff])

        minimum = max(1, int(config.MIN_SEGMENT_SEC * perception.fps))
        order = np.argsort(-scores)
        chosen: list[int] = []
        for index in order:
            if len(chosen) >= self.parts - 1:
                break
            if index < minimum or index > len(scores) - minimum:
                continue
            if any(abs(index - other) < minimum for other in chosen):
                continue
            chosen.append(int(index))
        return sorted(perception.offset + index / perception.fps for index in chosen)
