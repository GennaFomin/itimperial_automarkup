"""Стадия именования: как называется то, что уже нарезано.

Границы к этому моменту стоят, и языковая модель их не двигает — она отвечает только на
вопрос «что здесь делают и с чем». Ответ притягивается к закрытому словарю, а если сервис
недоступен или отвечает мусором, шаги остаются без меток: пустая разметка не выдаётся
никогда, и весь ролик не разваливается из-за одной неудачной стадии.
"""

from __future__ import annotations

import base64
import json
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from praxis import config, media
from praxis.schema import Step, VideoMeta
from praxis.vocab import Vocabulary


@dataclass
class NamingResult:
    steps: list[Step]
    models: dict[str, str] = field(default_factory=dict)


class Namer(Protocol):
    name: str

    def name_steps(
        self, video_path: Path, meta: VideoMeta, steps: list[Step], vocabulary: Vocabulary
    ) -> NamingResult: ...


class NullNamer:
    """Семантики нет: шаги уходят в редактор без меток, человек проставит их сам."""

    name = "none"

    def name_steps(
        self, video_path: Path, meta: VideoMeta, steps: list[Step], vocabulary: Vocabulary
    ) -> NamingResult:
        return NamingResult(steps=steps, models={"namer": "none"})


class RemoteVlmNamer:
    """Клиент к сервису с видеомоделью (scripts/serve_vlm.py на GPU-машине)."""

    name = "vlm"

    def __init__(
        self,
        base_url: str,
        frames_per_step: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.frames_per_step = frames_per_step or config.VLM_FRAMES
        self.timeout = timeout or config.VLM_TIMEOUT

    def name_steps(
        self, video_path: Path, meta: VideoMeta, steps: list[Step], vocabulary: Vocabulary
    ) -> NamingResult:
        if not steps:
            return NamingResult(steps=steps, models={"namer": "vlm", "namer_status": "нет шагов"})

        try:
            payload = {
                "segments": [
                    {"id": step.id, "frames": self._frames(video_path, step)} for step in steps
                ],
                "actions": vocabulary.actions,
                "objects": vocabulary.objects,
                "pairs": vocabulary.pairs,
            }
            answer = self._post("/annotate", payload)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as error:
            # Сервис недоступен — отдаём нарезку без меток, а не падаем.
            return NamingResult(
                steps=steps, models={"namer": "vlm", "namer_status": f"недоступен: {error}"}
            )

        by_id = {item["id"]: item for item in answer.get("results", [])}
        for step in steps:
            named = by_id.get(step.id)
            if not named:
                continue
            if named.get("action") and vocabulary.has_action(named["action"]):
                step.action = named["action"]
            obj = named.get("object")
            step.object = obj if obj and vocabulary.has_object(obj) else None
            if named.get("confidence") is not None:
                step.confidence = round(float(named["confidence"]), 3)

        return NamingResult(
            steps=steps,
            models={
                "namer": "vlm",
                "vlm": answer.get("model", config.VLM_MODEL),
                "vlm_sec": str(answer.get("elapsed_sec", "")),
            },
        )

    def _frames(self, video_path: Path, step: Step) -> list[str]:
        """Кадры, равномерно разбросанные внутри шага, плюс его ключевой кадр."""
        span = step.end_sec - step.start_sec
        offsets = [
            step.start_sec + span * (index + 0.5) / self.frames_per_step
            for index in range(self.frames_per_step)
        ]
        if step.keyframe_sec is not None:
            offsets.append(step.keyframe_sec)

        encoded: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            for index, at in enumerate(sorted(set(round(value, 2) for value in offsets))):
                path = Path(directory) / f"{index}.jpg"
                media.extract_frame(video_path, at, path, width=config.VLM_FRAME_WIDTH)
                encoded.append(base64.b64encode(path.read_bytes()).decode())
        return encoded

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())


def get_namer() -> Namer:
    """Какой источник семантики использовать. Пустой адрес — работаем без неё."""
    return RemoteVlmNamer(config.VLM_BASE_URL) if config.VLM_BASE_URL else NullNamer()
