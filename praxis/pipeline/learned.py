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


def stack_motion(appearance: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Признаки энкодера плюс полоса движения последним каналом."""
    count = len(appearance)
    band = np.asarray(motion, dtype=np.float32).ravel()
    if len(band) >= count:
        band = band[:count]
    elif len(band):
        band = np.pad(band, (0, count - len(band)), mode="edge")
    else:
        band = np.zeros(count, np.float32)
    return np.concatenate([appearance.astype(np.float32), band[:, None]], axis=1)


def difference_channels(matrix: np.ndarray, lags: tuple[int, ...] = (1, 2, 4)) -> np.ndarray:
    """Разности признаков во времени как дополнительные каналы (идея DDM-Net).

    Граница — это изменение, и модели проще увидеть готовую разность, чем вычислить её
    самой из двух соседних векторов: в DDM-Net это дало +8.5 F1 против сырых признаков,
    тогда как оптический поток — меньше одного пункта.
    """
    matrix = matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1) + 1e-6
    extra = []
    for lag in lags:
        shifted = np.vstack([np.repeat(matrix[:1], lag, axis=0), matrix[:-lag]])
        extra.append(np.abs(matrix - shifted).mean(axis=1))
        shifted_norms = np.concatenate([np.repeat(norms[:1], lag), norms[:-lag]])
        extra.append((matrix * shifted).sum(axis=1) / (norms * shifted_norms))
    return np.concatenate([matrix, np.stack(extra, axis=1).astype(np.float32)], axis=1)


def post(path: str, payload: dict, timeout: float | None = None) -> dict:
    request = urllib.request.Request(
        config.TAS_BASE_URL.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout or config.VLM_TIMEOUT) as response:
        return json.loads(response.read())


def peaks_above(
    scores: np.ndarray, level: float, minimum: int, prominence: float = 0.0
) -> list[int]:
    """Локальные максимумы выше порога, не ближе minimum кадров друг к другу.

    prominence — насколько пик обязан возвышаться над минимумом в своей окрестности
    (±minimum кадров). Плато высокой вероятности с рябью даёт ложные пики у простого
    локального максимума; выраженность отсеивает их, оставляя настоящие подъёмы.
    """
    chosen: list[int] = []
    for index in np.argsort(-scores):
        if scores[index] < level:
            break
        if index < minimum or index > len(scores) - minimum:
            continue
        if any(abs(index - other) < minimum for other in chosen):
            continue
        if prominence > 0:
            low = float(scores[max(0, index - minimum): index + minimum + 1].min())
            if float(scores[index]) - low < prominence:
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
        # Детектор обучен на признаках видеоэнкодера. Если те не получены и на входе
        # запасные признаки серых блоков, звать его бессмысленно: он отвечает 500 на
        # чужую размерность. Сразу берём ядровой метод — он работает на любых признаках.
        if perception.degraded:
            return self._fallback(video_path, meta, vocabulary, perception, "признаки просели")
        try:
            matrix = (
                stack_motion(perception.appearance, perception.motion)
                if config.TAS_MOTION else perception.appearance
            )
            if config.TAS_DIFF:
                matrix = difference_channels(matrix)
            answer = post("/predict", {"samples": [encode(matrix)], "tta": config.TAS_TTA})
            scores = np.array(answer["scores"][0], dtype=np.float32)
        except urllib.error.HTTPError as error:
            # Тело ответа несёт причину («ждёт 768 признаков, получено 769»): без него
            # предупреждение говорило бы только код, и искать поломку пришлось бы в логах.
            body = error.read().decode("utf-8", "replace")[:200] if hasattr(error, "read") else ""
            return self._fallback(video_path, meta, vocabulary, perception, f"недоступен: {error.code} {body}")
        except (urllib.error.URLError, OSError, TimeoutError, ValueError, KeyError) as error:
            return self._fallback(video_path, meta, vocabulary, perception, f"недоступен: {error}")

        minimum = max(1, int(config.MIN_SEGMENT_SEC * perception.fps))
        cuts = peaks_above(scores, config.TAS_THRESHOLD, minimum, config.TAS_PROMINENCE)
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
