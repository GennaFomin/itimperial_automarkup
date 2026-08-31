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
    # Подсказки для редактора: {id шага: [{action, object}, ...]}. В экспорт не идут —
    # это средство ускорить проверку, а не часть разметки.
    alternatives: dict[int, list[dict]] = field(default_factory=dict)


class Namer(Protocol):
    name: str

    def name_steps(
        self,
        video_path: Path,
        meta: VideoMeta,
        steps: list[Step],
        vocabulary: Vocabulary,
        crop: tuple[float, float, float, float] | None = None,
    ) -> NamingResult: ...


class NullNamer:
    """Семантики нет: шаги уходят в редактор без меток, человек проставит их сам."""

    name = "none"

    def name_steps(
        self,
        video_path: Path,
        meta: VideoMeta,
        steps: list[Step],
        vocabulary: Vocabulary,
        crop: tuple[float, float, float, float] | None = None,
    ) -> NamingResult:
        return NamingResult(steps=steps, models={"namer": "none"})


class HttpNamer:
    """Общая часть клиентов: нарезка кадров сегмента и запрос к сервису на GPU-машине."""

    name = "http"

    def __init__(
        self,
        base_url: str,
        frames_per_step: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.frames_per_step = frames_per_step or config.VLM_FRAMES
        self.timeout = timeout or config.VLM_TIMEOUT

    def _frames(
        self,
        video_path: Path,
        step: Step,
        crop: tuple[float, float, float, float] | None = None,
    ) -> list[str]:
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
                media.extract_frame(
                    video_path, at, path, width=config.VLM_FRAME_WIDTH, crop=crop
                )
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


class RemoteVlmNamer(HttpNamer):
    """Клиент к генеративной видеомодели (scripts/serve_vlm.py)."""

    name = "vlm"

    def name_steps(
        self,
        video_path: Path,
        meta: VideoMeta,
        steps: list[Step],
        vocabulary: Vocabulary,
        crop: tuple[float, float, float, float] | None = None,
    ) -> NamingResult:
        if not steps:
            return NamingResult(steps=steps, models={"namer": "vlm", "namer_status": "нет шагов"})

        try:
            payload = {
                "segments": [
                    {"id": step.id, "frames": self._frames(video_path, step, crop)}
                    for step in steps
                ],
                "actions": vocabulary.actions,
                "objects": vocabulary.objects,
                "pairs": vocabulary.pairs,
                "domain": config.DOMAIN or vocabulary.description or None,
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
            alternatives={
                step.id: by_id[step.id].get("alternatives", [])
                for step in steps
                if step.id in by_id and by_id[step.id].get("alternatives")
            },
            models={
                "namer": "vlm",
                "vlm": answer.get("model", config.VLM_MODEL),
                "vlm_sec": str(answer.get("elapsed_sec", "")),
            },
        )



class ClipNamer(HttpNamer):
    """Клиент к классификатору по закрытому словарю (scripts/serve_clip.py).

    Отдаёт не свободный текст, а распределение по 202 допустимым парам: значение вне
    словаря невозможно по построению, а уверенность берётся из самого распределения.
    """

    name = "siglip"

    def name_steps(
        self,
        video_path: Path,
        meta: VideoMeta,
        steps: list[Step],
        vocabulary: Vocabulary,
        crop: tuple[float, float, float, float] | None = None,
    ) -> NamingResult:
        if not steps:
            return NamingResult(steps=steps, models={"namer": self.name, "namer_status": "нет шагов"})

        pairs = (
            [[action, obj] for action, objects in vocabulary.pairs.items() for obj in objects]
            if vocabulary.pairs
            else [[action, ""] for action in vocabulary.actions]
        )
        try:
            answer = self._post(
                "/classify",
                {
                    "segments": [
                        {"id": step.id, "frames": self._frames(video_path, step, crop)}
                        for step in steps
                    ],
                    "pairs": pairs,
                    "mode": config.CLIP_MODE,
                    "verb_weight": config.CLIP_VERB_WEIGHT,
                    "noun_weight": config.CLIP_NOUN_WEIGHT,
                },
            )
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as error:
            return NamingResult(
                steps=steps,
                models={"namer": self.name, "namer_status": f"недоступен: {error}"},
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
            alternatives={
                step.id: [
                    {"action": item["action"], "object": item["object"]}
                    for item in by_id[step.id].get("top", [])[1:4]
                ]
                for step in steps
                if step.id in by_id
            },
            models={
                "namer": self.name,
                "classifier": answer.get("model", ""),
                "classifier_sec": str(answer.get("elapsed_sec", "")),
            },
        )


def merge_adjacent(steps: list[Step], labelled: bool = True) -> list[Step]:
    """Склеивает соседние шаги с одинаковой меткой.

    Сегментатор работает по картинке и не знает, что два соседних куска — это одно и то же
    действие: он видит, что кадры изменились, и режет. Понять, что «place tray» и следующий
    «place tray» — один шаг, можно только после того, как метки проставлены. Поэтому склейка
    живёт здесь, а не в сегментаторе.
    """
    # Без настоящих меток склеивать нельзя: у всех шагов стоит одна и та же заглушка,
    # и склейка схлопнула бы весь ролик в один шаг, уничтожив заодно и нарезку.
    if not labelled:
        return steps

    ordered = sorted(steps, key=lambda step: step.start_sec)
    merged: list[Step] = []
    for step in ordered:
        previous = merged[-1] if merged else None
        same_label = (
            previous is not None
            and previous.action == step.action
            and previous.object == step.object
            and previous.level == step.level
            and abs(previous.end_sec - step.start_sec) < 0.05
        )
        if not same_label:
            merged.append(step.model_copy())
            continue

        # Ключевой кадр берём у более длинного куска: он представительнее.
        longer = previous if previous.duration_sec >= step.duration_sec else step
        previous.end_sec = step.end_sec
        previous.keyframe_sec = longer.keyframe_sec
        previous.confidence = max(
            previous.confidence or 0.0, step.confidence or 0.0
        ) or None

    for index, step in enumerate(merged):
        step.id = index
    return merged


def get_namer() -> Namer:
    """Какой источник семантики использовать. Пустые адреса — работаем без неё."""
    choice = config.NAMER
    if choice in {"auto", "vlm"} and config.VLM_BASE_URL:
        return RemoteVlmNamer(config.VLM_BASE_URL)
    if choice in {"auto", "siglip"} and config.CLIP_BASE_URL:
        return ClipNamer(config.CLIP_BASE_URL)
    return NullNamer()
