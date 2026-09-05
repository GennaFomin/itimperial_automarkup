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


def post(path: str, payload: dict, timeout: float | None = None, base_url: str | None = None) -> dict:
    request = urllib.request.Request(
        (base_url or config.TAS_BASE_URL).rstrip("/") + path,
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


def activity_segments(
    scores: np.ndarray, motion: np.ndarray, fps: float, threshold: float, minimum: int,
    prominence: float, level: float, low: float, smooth_sec: float, close_sec: float,
    gaps: np.ndarray | None = None, gap_threshold: float = 0.5, gap_min_sec: float = 0.5,
) -> list[tuple[int, int]]:
    """Шаги только там, где есть движение: активные области сглаженной полосы движения
    (вход выше пол + level·(среднее − пол), выход ниже пол + low·(вход − пол)), провалы
    короче close_sec закрываются, разрезы детектора делят области изнутри. Между
    областями — паузы.

    Полоса движения не зависит от домена: на снятых нами роликах детектор видит смены с
    уверенностью 0.3–0.5, а неподвижные руки и удерживаемый предмет видны по движению
    всегда. Без полосы (пустой или нулевой) активность считается сплошной.

    gaps — вероятность паузы по кадрам от обученного детектора пауз (переходы с движением,
    ожидание — то, что в размеченных наборах не считается действием): участки выше
    gap_threshold длиннее gap_min_sec тоже становятся паузами."""
    count = len(scores)
    cuts = peaks_above(scores, threshold, minimum, prominence)
    band = np.asarray(motion, dtype=np.float32).ravel()
    active = np.ones(count, dtype=bool)
    if len(band) == count and np.isfinite(band).all() and band.max() > 0:
        kernel = max(1, int(round(smooth_sec * fps)))
        if kernel > 1:
            pad = np.pad(band, (kernel // 2, kernel - 1 - kernel // 2), mode="edge")
            band = np.convolve(pad, np.ones(kernel) / kernel, mode="valid")
        # Уровень отсчитывается от шумового пола (20-й перцентиль), а не от нуля: у
        # оригиналов 720p пол выше, чем у сжатых копий, и доля среднего попадала бы в шум.
        floor = float(np.percentile(band, 20))
        spread = float(band.mean()) - floor
        if spread > 1e-6:  # ровная полоса не даёт свидетельств неподвижности
            high = floor + level * spread
            low_level = floor + (high - floor) * low
            on = False
            for index, value in enumerate(band):
                if not on and value > high:
                    on = True
                elif on and value < low_level:
                    on = False
                active[index] = on
    if gaps is not None and len(gaps) == count:
        pause = np.asarray(gaps, dtype=np.float32) > gap_threshold
        index = 0
        while index < count:
            if pause[index]:
                stop = index
                while stop < count and pause[stop]:
                    stop += 1
                if (stop - index) / fps >= gap_min_sec:
                    active[index:stop] = False
                index = stop
            else:
                index += 1
    if active.all():
        edges = [0, *cuts, count]
        return list(zip(edges, edges[1:]))
    regions: list[list[int]] = []
    index = 0
    while index < count:
        if active[index]:
            stop = index
            while stop < count and active[stop]:
                stop += 1
            if regions and index - regions[-1][1] <= int(round(close_sec * fps)):
                regions[-1][1] = stop
            else:
                regions.append([index, stop])
            index = stop
        else:
            index += 1
    segments: list[tuple[int, int]] = []
    for first, last in regions:
        edges = [first, *[cut for cut in cuts if first < cut < last], last]
        segments.extend(zip(edges, edges[1:]))
    return segments


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

        minimum = max(1, int(config.TAS_PEAK_GAP_SEC * perception.fps))
        gaps = None
        if config.IDLE_MODE == "activity" and config.GAP_BASE_URL:
            try:
                gaps = np.array(post("/predict", {"samples": [encode(matrix)]}, base_url=config.GAP_BASE_URL)["scores"][0], dtype=np.float32)
            except (urllib.error.URLError, OSError, TimeoutError, ValueError, KeyError):
                gaps = None  # без детектора пауз остаётся неподвижность по полосе движения
        if config.IDLE_MODE == "activity":
            segments = activity_segments(
                scores, perception.motion, perception.fps, config.TAS_THRESHOLD, minimum,
                config.TAS_PROMINENCE, config.ACTIVITY_LEVEL, config.ACTIVITY_LOW,
                config.ACTIVITY_SMOOTH_SEC, config.ACTIVITY_CLOSE_SEC,
                gaps, config.GAP_THRESHOLD, config.GAP_MIN_SEC,
            )
        else:
            cuts = peaks_above(scores, config.TAS_THRESHOLD, minimum, config.TAS_PROMINENCE)
            edges = [0, *cuts, len(scores)]
            segments = list(zip(edges, edges[1:]))

        steps = []
        for first, last in segments:
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
