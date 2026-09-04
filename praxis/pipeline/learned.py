"""Границы от обучаемого детектора — MS-TCN, переделанный под открытый словарь.

Классифицирующая голова из статей нам не годится: её веса привязаны к таксономии
обучающего набора, а у нас списка классов нет. Но граница класс-агностична, поэтому
голова обучается предсказывать не класс кадра, а вероятность смены действия. Такую можно
обучить на любом размеченном наборе и применить к любому домену.

Сама сеть живёт на GPU-машине за HTTP, как и остальные модели: здесь только отправка
признаков и превращение вероятностей в шаги.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from praxis import config
from praxis.pipeline.base import Perception, PipelineResult
from praxis.schema import Step, VideoMeta
from praxis.vocab import Vocabulary


def encode(features: np.ndarray) -> dict:
    matrix = features.astype(np.float16)
    return {
        "features": base64.b64encode(matrix.tobytes()).decode(),
        "frames": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
    }


def post(path: str, payload: dict, timeout: float | None = None) -> dict:
    request = urllib.request.Request(
        config.TAS_BASE_URL.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout or config.VLM_TIMEOUT) as response:
        return json.loads(response.read())


def peaks_above(scores: np.ndarray, level: float, minimum: int) -> list[int]:
    """Локальные максимумы выше порога, не ближе minimum кадров друг к другу."""
    chosen: list[int] = []
    for index in np.argsort(-scores):
        if scores[index] < level:
            break
        if index < minimum or index > len(scores) - minimum:
            continue
        if any(abs(index - other) < minimum for other in chosen):
            continue
        chosen.append(int(index))
    return sorted(chosen)


@dataclass
class LearnedSegmenter:
    """Нарезка по вероятностям обучаемого детектора границ."""

    @property
    def name(self) -> str:
        return "learned-boundaries"

    def run(
        self,
        video_path: Path,
        meta: VideoMeta,
        vocabulary: Vocabulary,
        perception: Perception,
    ) -> PipelineResult:
        if not config.TAS_BASE_URL:
            return self._fallback(video_path, meta, vocabulary, perception, "адрес сервиса не задан")
        try:
            answer = post("/predict", {"samples": [encode(perception.appearance)]})
            scores = np.array(answer["scores"][0], dtype=np.float32)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError, KeyError) as error:
            return self._fallback(video_path, meta, vocabulary, perception, f"недоступен: {error}")

        minimum = max(1, int(config.MIN_SEGMENT_SEC * perception.fps))
        cuts = peaks_above(scores, config.TAS_THRESHOLD, minimum)
        edges = [0, *cuts, len(scores)]

        steps = []
        for index in range(len(edges) - 1):
            first, last = edges[index], edges[index + 1]
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

    def _fallback(
        self, video_path: Path, meta: VideoMeta, vocabulary: Vocabulary,
        perception: Perception, reason: str,
    ) -> PipelineResult:
        """Без детектора режет ядровой change-point — второй по качеству метод, которому
        не нужен GPU сверх признаков. Один шаг на весь ролик здесь не годится: это
        выглядело бы как результат, а не как отказ. Причина уходит в предупреждения."""
        from praxis.pipeline.similarity import SimilaritySegmenter

        result = SimilaritySegmenter(mode="kernel").run(video_path, meta, vocabulary, perception)
        result.models["segmenter"] = "tsm-kernel"
        result.models["segmenter_status"] = f"детектор границ {reason}: нарезка ядровым change-point"
        return result

    @staticmethod
    def _whole(meta: VideoMeta, vocabulary: Vocabulary) -> Step:
        return Step(
            id=0, start_sec=0.0, end_sec=meta.duration_sec,
            action=vocabulary.actions[0], object=None,
            keyframe_sec=round(meta.duration_sec / 2, 3), confidence=None,
        )
