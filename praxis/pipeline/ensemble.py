"""Ансамбль двух детекторов границ: обучаемого и ядрового change-point.

Зачем. Замеры на четырёх доменах показали, что ни один из двух не выигрывает везде:
обучаемый точнее там, где плотность границ похожа на обучающую (атомарная сборка),
ядровой — там, где шаги длиннее и реже. Граница, которую подтверждают оба, надёжнее,
чем любая из них по отдельности; граница, которую видит только один, — кандидат,
которому нужен второй голос.

Схема: ядровой метод даёт разбиение как обычно; обучаемый детектор даёт вероятность
смены на каждом кадре. Итог — разрезы ядрового, у которых рядом (в пределах допуска)
есть уверенность детектора выше порога, плюс уверенные пики детектора, которых у
ядрового нет вовсе. Число шагов при этом не задаётся руками.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from praxis import config
from praxis.pipeline.base import Perception, PipelineResult
from praxis.pipeline.learned import encode, peaks_above, post
from praxis.pipeline.segment import normalise, reduce_dimensions
from praxis.pipeline.similarity import gram, kernel_costs, penalised_segmentation
from praxis.schema import Step, VideoMeta
from praxis.vocab import Vocabulary


@dataclass
class EnsembleSegmenter:
    @property
    def name(self) -> str:
        return "ensemble"

    def run(self, video_path: Path, meta: VideoMeta, vocabulary: Vocabulary,
            perception: Perception) -> PipelineResult:
        fps = perception.fps
        minimum = max(2, int(config.MIN_SEGMENT_SEC * fps))
        features = reduce_dimensions(normalise(perception.appearance), config.COMPONENTS)
        if len(features) < 4:
            return PipelineResult(steps=[self._whole(meta, vocabulary)], models={"segmenter": self.name})

        kernel_cuts = penalised_segmentation(kernel_costs(gram(features)), config.TSM_PENALTY, minimum)
        try:
            scores = np.array(post("/predict", {"samples": [encode(perception.appearance)]})["scores"][0])
        except Exception:  # noqa: BLE001 — без детектора остаёмся на ядровом
            scores = None

        if scores is None:
            cuts = sorted(kernel_cuts)
        else:
            tolerance = max(1, int(config.ENSEMBLE_TOLERANCE_SEC * fps))
            # Разрез ядрового подтверждается, если детектор рядом с ним уверен.
            confirmed = [
                cut for cut in kernel_cuts
                if scores[max(0, cut - tolerance): cut + tolerance + 1].max() >= config.ENSEMBLE_CONFIRM
            ]
            # Уверенные пики детектора, которых у ядрового нет.
            strong = [
                peak for peak in peaks_above(scores, config.ENSEMBLE_STRONG, minimum)
                if all(abs(peak - cut) > tolerance for cut in kernel_cuts)
            ]
            cuts = sorted(set(confirmed) | set(strong))
            # Убираем разрезы ближе минимальной длины друг к другу.
            filtered: list[int] = []
            for cut in cuts:
                if not filtered or cut - filtered[-1] >= minimum:
                    filtered.append(cut)
            cuts = filtered

        edges = [0, *cuts, len(features)]
        steps = []
        for index in range(len(edges) - 1):
            first, last = edges[index], edges[index + 1]
            if config.IDLE_RATIO > 0 and np.mean(perception.motion[first:last]) < np.mean(perception.motion) * config.IDLE_RATIO:
                continue
            start = perception.offset + first / fps
            end = min(meta.duration_sec, perception.offset + last / fps)
            if end - start < config.MIN_SEGMENT_SEC:
                continue
            steps.append(Step(id=len(steps), start_sec=round(start, 3), end_sec=round(end, 3),
                              action=vocabulary.actions[0], object=None,
                              keyframe_sec=round((start + end) / 2, 3), confidence=None))
        if not steps:
            steps = [self._whole(meta, vocabulary)]
        return PipelineResult(steps=steps, models={"segmenter": self.name})

    @staticmethod
    def _whole(meta: VideoMeta, vocabulary: Vocabulary) -> Step:
        return Step(id=0, start_sec=0.0, end_sec=meta.duration_sec, action=vocabulary.actions[0],
                    object=None, keyframe_sec=round(meta.duration_sec / 2, 3), confidence=None)
