"""Сегментатор на физике движения и динамическом программировании.

Границы шагов ставит не языковая модель, а сам ролик: разрезы попадают в моменты, где
движение затихает, а число отрезков выбирается точным перебором со штрафом за каждый
лишний. Названия действий этот сегментатор не придумывает — их даёт следующая стадия,
поэтому уверенность здесь отражает только качество нарезки.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from praxis import config
from praxis.pipeline.base import Perception, PipelineResult
from praxis.pipeline.segment import boundary_scores, pick_keyframe, segment
from praxis.schema import Source, Step, VideoMeta
from praxis.vocab import Vocabulary


class PhysicalSegmenter:
    name = "motion-dp"

    def __init__(
        self,
        penalty: float | None = None,
        boundary_weight: float | None = None,
        max_segments: int | None = None,
        min_segment_sec: float | None = None,
    ) -> None:
        self.penalty = config.SEGMENT_PENALTY if penalty is None else penalty
        self.boundary_weight = (
            config.BOUNDARY_WEIGHT if boundary_weight is None else boundary_weight
        )
        self.max_segments = config.MAX_SEGMENTS if max_segments is None else max_segments
        self.min_segment_sec = (
            config.MIN_SEGMENT_SEC if min_segment_sec is None else min_segment_sec
        )

    def run(
        self,
        video_path: Path,
        meta: VideoMeta,
        vocabulary: Vocabulary,
        perception: Perception,
    ) -> PipelineResult:
        appearance = perception.appearance
        motion = perception.motion
        fps = perception.fps

        if len(appearance) < 4:
            return PipelineResult(steps=[self._whole_clip(meta, vocabulary)], models=self._models())

        bounds = segment(
            appearance,
            motion,
            fps=fps,
            penalty=self.penalty,
            boundary_weight=self.boundary_weight,
            max_segments=self.max_segments,
            min_segment_sec=self.min_segment_sec,
        )
        scores = boundary_scores(motion)

        steps: list[Step] = []
        for index, (start, end) in enumerate(bounds):
            start_sec = round(start / fps, 3)
            end_sec = round(min(end / fps, meta.duration_sec), 3)
            if end_sec - start_sec < 0.4:
                continue
            keyframe = pick_keyframe(appearance, motion, start, end)
            steps.append(
                Step(
                    id=index,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    action=vocabulary.actions[0],
                    object=None,
                    keyframe_sec=round(min(keyframe / fps, end_sec), 3),
                    confidence=self._confidence(appearance, scores, start, end),
                    source=Source.auto,
                )
            )

        if not steps:
            steps = [self._whole_clip(meta, vocabulary)]
        else:
            # Границы отрезков округлялись по отдельности — сшиваем их встык и дотягиваем
            # последний до конца ролика, чтобы разметка покрывала таймлайн без дыр.
            for left, right in zip(steps, steps[1:]):
                right.start_sec = left.end_sec
            steps[-1].end_sec = round(meta.duration_sec, 3)
            steps[0].start_sec = 0.0

        return PipelineResult(steps=steps, models=self._models())

    @staticmethod
    def _models() -> dict[str, str]:
        return {"segmenter": "motion-dp", "features": "gray 9x16 + motion"}

    @staticmethod
    def _whole_clip(meta: VideoMeta, vocabulary: Vocabulary) -> Step:
        """Запасной вариант: один шаг на весь ролик. Пустую разметку не отдаём никогда."""
        return Step(
            id=0,
            start_sec=0.0,
            end_sec=round(meta.duration_sec, 3),
            action=vocabulary.actions[0],
            object=None,
            keyframe_sec=round(meta.duration_sec / 2, 3),
            confidence=0.1,
            source=Source.auto,
        )

    @staticmethod
    def _confidence(appearance: np.ndarray, scores: np.ndarray, start: int, end: int) -> float:
        """Насколько отрезок похож на настоящий шаг: тихие границы и однородная середина."""
        edges = [scores[min(start, len(scores) - 1)], scores[min(end - 1, len(scores) - 1)]]
        block = appearance[start:end]
        spread = float(block.std()) / (float(appearance.std()) or 1.0)
        value = 0.25 + 0.5 * float(np.mean(edges)) - 0.25 * min(spread, 2.0)
        return round(float(np.clip(value, 0.05, 0.95)), 3)
