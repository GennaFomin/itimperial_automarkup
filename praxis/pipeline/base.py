"""Интерфейс сегментатора.

Всё, что делает разметку, прячется за этим интерфейсом: заглушка сегодня, физика с DP
и VLM завтра. Приложение об их внутренностях не знает.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from praxis.schema import Step, VideoMeta
from praxis.vocab import Vocabulary


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
        motion: list[float],
    ) -> PipelineResult: ...


def get_segmenter(name: str) -> Segmenter:
    if name == "stub":
        from praxis.pipeline.stub import StubSegmenter

        return StubSegmenter()
    raise ValueError(f"неизвестный сегментатор: {name}")
