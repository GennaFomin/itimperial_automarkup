"""Заглушка сегментатора: равномерная нарезка и метки по кругу из словаря.

Существует ровно для того, чтобы сквозной путь «загрузка → разметка → редактор →
экспорт» работал до появления настоящего пайплайна. Уверенность занижена намеренно:
это не предсказание, а рыба.
"""

from __future__ import annotations

from pathlib import Path

from praxis.pipeline.base import PipelineResult
from praxis.schema import Source, Step, VideoMeta
from praxis.vocab import Vocabulary

STUB_CONFIDENCE = 0.3


class StubSegmenter:
    name = "stub"

    def run(
        self,
        video_path: Path,
        meta: VideoMeta,
        vocabulary: Vocabulary,
        motion: list[float],
    ) -> PipelineResult:
        count = max(2, min(5, round(meta.duration_sec / 6)))
        span = meta.duration_sec / count
        pairs = self._pairs(vocabulary, count)

        steps = []
        for index in range(count):
            start = index * span
            end = meta.duration_sec if index == count - 1 else (index + 1) * span
            action, obj = pairs[index]
            steps.append(
                Step(
                    id=index,
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                    action=action,
                    object=obj,
                    keyframe_sec=round((start + end) / 2, 3),
                    confidence=STUB_CONFIDENCE,
                    source=Source.auto,
                )
            )
        return PipelineResult(steps=steps, models={"segmenter": "stub"})

    @staticmethod
    def _pairs(vocabulary: Vocabulary, count: int) -> list[tuple[str, str | None]]:
        available: list[tuple[str, str | None]] = []
        for action in vocabulary.actions:
            objects = vocabulary.objects_for(action)
            available.append((action, objects[0] if objects else None))
        return [available[index % len(available)] for index in range(count)]
