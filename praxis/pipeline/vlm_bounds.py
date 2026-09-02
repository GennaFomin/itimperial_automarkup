"""Границы шагов ставит языковая модель — прямой конкурент разбиению по признакам.

Зачем это существует. Спор «границы должна ставить VLM или отдельный алгоритм» решается
измерением, а не обсуждением, и для измерения нужны обе реализации на одном наборе и
под одной метрикой. Здесь живёт вариант с VLM в двух видах, как его обычно и предлагают:

* **direct** — модель получает подписанные временем кадры всего ролика и сразу отдаёт
  список шагов с таймкодами. Один запрос на ролик.
* **bisect** — модель отвечает только на бинарный вопрос «между этими двумя кадрами
  действие сменилось?», а границы ищутся бинпоиском по времени. Запросов больше, зато
  каждый простой.

Оба варианта сравниваются с `PhysicalSegmenter` в scripts/compare_bounds.py.
"""

from __future__ import annotations

import base64
import json
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from praxis import config, media
from praxis.pipeline.base import Perception, PipelineResult
from praxis.schema import Step, VideoMeta
from praxis.vocab import Vocabulary


@dataclass
class VlmBoundarySegmenter:
    """Нарезка ролика языковой моделью. mode: direct или bisect."""

    mode: str = "direct"
    frames: int = 24  # сколько кадров показать модели в режиме direct
    min_gap_sec: float = 0.4  # точность, до которой бинпоиск уточняет границу
    anchors: int = 12  # опорные точки, между которыми ищется смена действия

    @property
    def name(self) -> str:
        return f"vlm-{self.mode}"

    def run(
        self,
        video_path: Path,
        meta: VideoMeta,
        vocabulary: Vocabulary,
        perception: Perception,
    ) -> PipelineResult:
        try:
            steps = (
                self._direct(video_path, meta, vocabulary)
                if self.mode == "direct"
                else self._bisect(video_path, meta, vocabulary)
            )
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as error:
            return PipelineResult(
                steps=[self._whole_clip(meta, vocabulary)],
                models={"segmenter": self.name, "segmenter_status": f"недоступен: {error}"},
            )
        if not steps:
            steps = [self._whole_clip(meta, vocabulary)]
        return PipelineResult(steps=steps, models={"segmenter": self.name})

    # --- прямой запрос -------------------------------------------------------------

    def _direct(self, video_path: Path, meta: VideoMeta, vocabulary: Vocabulary) -> list[Step]:
        times = [
            meta.duration_sec * (index + 0.5) / self.frames for index in range(self.frames)
        ]
        answer = self._post(
            "/segment",
            {
                "frames": self._encode(video_path, times),
                "times": [round(value, 2) for value in times],
                "actions": vocabulary.actions,
                "objects": vocabulary.objects,
                "domain": config.DOMAIN or vocabulary.description or None,
            },
        )
        return self._to_steps(answer.get("steps", []), meta, vocabulary)

    # --- бинпоиск ------------------------------------------------------------------

    def _bisect(self, video_path: Path, meta: VideoMeta, vocabulary: Vocabulary) -> list[Step]:
        """Сначала находим пары опорных точек, между которыми действие сменилось, потом
        уточняем каждую границу делением пополам."""
        anchors = [
            meta.duration_sec * index / (self.anchors - 1) for index in range(self.anchors)
        ]
        anchors = [min(max(value, 0.05), meta.duration_sec - 0.05) for value in anchors]

        changed = self._compare(
            video_path, [(anchors[i], anchors[i + 1]) for i in range(len(anchors) - 1)]
        )
        boundaries: list[float] = []
        for index, flag in enumerate(changed):
            if not flag:
                continue
            left, right = anchors[index], anchors[index + 1]
            # Делим пополам, пока интервал не станет меньше требуемой точности.
            while right - left > self.min_gap_sec:
                middle = (left + right) / 2
                if self._compare(video_path, [(left, middle)])[0]:
                    right = middle
                else:
                    left = middle
            boundaries.append(round((left + right) / 2, 2))

        edges = [0.0, *sorted(boundaries), meta.duration_sec]
        raw = [
            {"start_sec": edges[i], "end_sec": edges[i + 1], "action": None, "object": None}
            for i in range(len(edges) - 1)
            if edges[i + 1] - edges[i] > config.MIN_SEGMENT_SEC
        ]
        return self._to_steps(raw, meta, vocabulary)

    def _compare(self, video_path: Path, pairs: list[tuple[float, float]]) -> list[bool]:
        moments = [value for pair in pairs for value in pair]
        encoded = self._encode(video_path, moments)
        payload = {
            "pairs": [
                {
                    "left": encoded[2 * index],
                    "right": encoded[2 * index + 1],
                    "left_sec": round(pair[0], 2),
                    "right_sec": round(pair[1], 2),
                }
                for index, pair in enumerate(pairs)
            ],
            "domain": config.DOMAIN or None,
        }
        return self._post("/compare", payload).get("changed", [False] * len(pairs))

    # --- общее ---------------------------------------------------------------------

    def _encode(self, video_path: Path, times: list[float]) -> list[str]:
        encoded = []
        with tempfile.TemporaryDirectory() as directory:
            for index, at in enumerate(times):
                path = Path(directory) / f"{index}.jpg"
                media.extract_frame(video_path, at, path, width=config.VLM_FRAME_WIDTH)
                encoded.append(base64.b64encode(path.read_bytes()).decode())
        return encoded

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            config.VLM_BASE_URL.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=config.VLM_TIMEOUT) as response:
            return json.loads(response.read())

    def _to_steps(self, raw: list[dict], meta: VideoMeta, vocabulary: Vocabulary) -> list[Step]:
        steps: list[Step] = []
        for item in sorted(raw, key=lambda value: value["start_sec"]):
            start = max(0.0, float(item["start_sec"]))
            end = min(meta.duration_sec, float(item["end_sec"]))
            if end - start < config.MIN_SEGMENT_SEC:
                continue
            if steps and start < steps[-1].end_sec:
                start = steps[-1].end_sec  # пересечения запрещены схемой
            if end - start < config.MIN_SEGMENT_SEC:
                continue
            steps.append(
                Step(
                    id=len(steps),
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                    action=item.get("action") or vocabulary.actions[0],
                    object=item.get("object"),
                    keyframe_sec=round((start + end) / 2, 3),
                    confidence=None,
                )
            )
        return steps

    @staticmethod
    def _whole_clip(meta: VideoMeta, vocabulary: Vocabulary) -> Step:
        return Step(
            id=0,
            start_sec=0.0,
            end_sec=meta.duration_sec,
            action=vocabulary.actions[0],
            object=None,
            keyframe_sec=round(meta.duration_sec / 2, 3),
            confidence=None,
        )
