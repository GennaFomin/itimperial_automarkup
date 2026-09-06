"""Режим «без авторазметки»: ролик готовится к редактору, шаги ставит человек."""

from __future__ import annotations

from pathlib import Path

from praxis.pipeline.base import Perception, PipelineResult
from praxis.schema import VideoMeta
from praxis.vocab import Vocabulary


class ManualSegmenter:
    """Ни одной границы и ни одной метки: пустая дорожка для ручной разметки.

    Нужен, когда автоматика мешает — например, чтобы замерить скорость ручной
    разметки на чистом ролике или разметить то, на чём модель заведомо ошибается.
    Декодирование при этом проходит как обычно: плеер, полоса движения и кадры
    редактору нужны в любом случае.
    """

    @property
    def name(self) -> str:
        return "manual"

    def run(
        self,
        video_path: Path,
        meta: VideoMeta,
        vocabulary: Vocabulary,
        perception: Perception,
    ) -> PipelineResult:
        return PipelineResult(steps=[], models={"segmenter": "manual"})
