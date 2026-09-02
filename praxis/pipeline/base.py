"""Интерфейс сегментатора.

Всё, что делает разметку, прячется за этим интерфейсом: заглушка сегодня, физика с DP
и VLM завтра. Приложение об их внутренностях не знает.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from praxis.schema import Step, VideoMeta
from praxis.vocab import Vocabulary


@dataclass
class Perception:
    """Что удалось увидеть в ролике до всякой семантики.

    Считается один раз на ролик и передаётся сегментатору: и полоса движения для
    таймлайна, и признаки для разбиения берутся из одного прохода ffmpeg.
    """

    fps: float
    motion: "np.ndarray"
    appearance: "np.ndarray"
    # Доли кадра (слева, сверху, ширина, высота), в которых вообще что-то происходит.
    # Нужны семантической стадии: на широком плане стола детали слишком мелкие.
    crop: tuple[float, float, float, float] | None = None
    # Сдвиг первого признака от начала ролика. У видеопризнаков вектор описывает окно
    # кадров, и его момент — середина окна, а не ноль.
    offset: float = 0.0
    # Чем пришлось пожертвовать: недоступный сервис признаков роняет качество молча,
    # и прогон обязан сказать об этом вслух, а не притвориться успешным.
    degraded: tuple[str, ...] = ()
    # Сколько секунд из этого прогона отработали удалённые модели: основа стоимости.
    remote_sec: float = 0.0


@dataclass
class PipelineResult:
    steps: list[Step]
    models: dict[str, str] = field(default_factory=dict)


class Segmenter(Protocol):
    name: str

    def run(
        self,
        video_path: Path,
        meta: VideoMeta,
        vocabulary: Vocabulary,
        perception: Perception,
    ) -> PipelineResult: ...


def get_segmenter(name: str) -> Segmenter:
    if name == "stub":
        from praxis.pipeline.stub import StubSegmenter

        return StubSegmenter()
    if name == "motion-dp":
        from praxis.pipeline.physical import PhysicalSegmenter

        return PhysicalSegmenter()
    if name.startswith("baseline-"):
        from praxis.pipeline.baselines import BaselineSegmenter

        return BaselineSegmenter(mode=name.removeprefix("baseline-"))
    if name in {"tsm-recursive", "tsm-kernel"}:
        from praxis.pipeline.similarity import SimilaritySegmenter

        return SimilaritySegmenter(mode=name.removeprefix("tsm-"))
    if name in {"vlm-direct", "vlm-bisect"}:
        from praxis.pipeline.vlm_bounds import VlmBoundarySegmenter

        return VlmBoundarySegmenter(mode=name.removeprefix("vlm-"))
    raise ValueError(f"неизвестный сегментатор: {name}")
