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

    def _track(
        self,
        video_path: Path,
        steps: list[Step],
        fallback: tuple[float, float, float, float] | None,
    ) -> dict[int, tuple[float, float, float, float]]:
        """Рамка движущегося предмета на каждый шаг. При отказе трекера — общая зона."""
        try:
            answer = self._post(
                "/track",
                {
                    "segments": [
                        {"id": step.id, "frames": self._frames(video_path, step, None, width=512)}
                        for step in steps
                    ],
                    "grid": config.TRACK_GRID,
                },
                base_url=config.TRACK_BASE_URL,
            )
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return {}

        boxes: dict[int, tuple[float, float, float, float]] = {}
        for item in answer.get("results", []):
            box = item.get("box")
            if not box or item.get("shift", 0.0) < config.TRACK_MIN_SHIFT:
                continue
            left, top, width, height = box
            # Расширяем рамку: предмет полезнее видеть вместе с руками и опорой.
            margin = config.TRACK_MARGIN
            left = max(0.0, left - width * margin)
            top = max(0.0, top - height * margin)
            width = min(1.0 - left, width * (1 + 2 * margin))
            height = min(1.0 - top, height * (1 + 2 * margin))
            if width > 0.05 and height > 0.05:
                boxes[item["id"]] = (left, top, width, height)
        return boxes

    @staticmethod
    def _shortlist(named: dict, vocabulary: Vocabulary) -> list[tuple[str, str]]:
        """Гипотезы для оценки: ответ модели, её альтернативы и перекрёстные комбинации."""
        actions, objects = [], []
        if named.get("action"):
            actions.append(named["action"])
        if named.get("object"):
            objects.append(named["object"])
        for alternative in named.get("alternatives") or []:
            if alternative.get("action") and alternative["action"] not in actions:
                actions.append(alternative["action"])
            if alternative.get("object") and alternative["object"] not in objects:
                objects.append(alternative["object"])

        candidates: list[tuple[str, str]] = []
        for action in actions[:3]:
            for noun in objects[:3]:
                if vocabulary.is_valid_pair(action, noun) and (action, noun) not in candidates:
                    candidates.append((action, noun))
        return candidates

    def _frames(
        self,
        video_path: Path,
        step: Step,
        crop: tuple[float, float, float, float] | None = None,
        width: int | None = None,
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
                    video_path, at, path, width=width or config.VLM_FRAME_WIDTH, crop=crop
                )
                encoded.append(base64.b64encode(path.read_bytes()).decode())
        return encoded

    def _post(self, path: str, payload: dict, base_url: str | None = None) -> dict:
        request = urllib.request.Request(
            (base_url.rstrip("/") if base_url else self.base_url) + path,
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

        # Область предмета от трекера, если он поднят: кадры для каждого шага режутся по
        # своей рамке, а не по общей рабочей зоне. Языковая модель тогда смотрит на предмет
        # крупно, а не ищет его на общем плане.
        boxes = self._track(video_path, steps, crop) if config.TRACK_BASE_URL else {}

        try:
            payload = {
                "segments": [
                    {
                        "id": step.id,
                        "frames": self._frames(video_path, step, boxes.get(step.id, crop)),
                    }
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

        # Второй проход с контекстом: модель переспрашивается, зная, что она же сказала про
        # соседние шаги. «Открыл» и «закрыл» на отдельных кадрах неразличимы — их
        # различает только порядок, и его надо дать модели явно.
        if config.VLM_CONTEXT and len(steps) > 1 and by_id:
            def label(step_id: int) -> str | None:
                item = by_id.get(step_id)
                if not item or not item.get("action"):
                    return None
                return f"{item['action']} {item.get('object') or ''}".strip()

            ordered = sorted(steps, key=lambda step: step.start_sec)
            segments = []
            for index, step in enumerate(ordered):
                segments.append(
                    {
                        "id": step.id,
                        "frames": self._frames(video_path, step, crop),
                        "previous": label(ordered[index - 1].id) if index else None,
                        "following": (
                            label(ordered[index + 1].id) if index + 1 < len(ordered) else None
                        ),
                    }
                )
            try:
                second = self._post(
                    "/annotate",
                    {
                        "segments": segments,
                        "actions": vocabulary.actions,
                        "objects": vocabulary.objects,
                        "pairs": vocabulary.pairs,
                        "domain": config.DOMAIN or vocabulary.description or None,
                    },
                )
                for item in second.get("results", []):
                    if item.get("action"):
                        by_id[item["id"]] = item
            except (urllib.error.URLError, OSError, TimeoutError, ValueError):
                pass  # остаёмся с первым проходом

        # Ещё один проход: модель не придумывает ответ, а оценивает короткий список гипотез.
        # Свободная генерация заставляет её вспоминать слово из длинного списка; оценка
        # правдоподобия превращает задачу в «сравни картинку с гипотезой», а это ей даётся
        # заметно лучше. Список собирается из её же первого ответа и альтернатив, включая
        # перекрёстные комбинации — так проверяются обе оси, и действие, и предмет.
        if config.VLM_RESCORE and by_id:
            segments = []
            for step in steps:
                named = by_id.get(step.id)
                if not named:
                    continue
                candidates = self._shortlist(named, vocabulary)
                if len(candidates) > 1:
                    segments.append(
                        {
                            "id": step.id,
                            "frames": self._frames(video_path, step, crop),
                            "candidates": [list(pair) for pair in candidates],
                        }
                    )
            if segments:
                try:
                    rescored = self._post(
                        "/annotate",
                        {
                            "segments": segments,
                            "actions": vocabulary.actions,
                            "objects": vocabulary.objects,
                            "pairs": vocabulary.pairs,
                            "domain": config.DOMAIN or vocabulary.description or None,
                            "mode": "score",
                        },
                    )
                    for item in rescored.get("results", []):
                        by_id[item["id"]] = item
                except (urllib.error.URLError, OSError, TimeoutError, ValueError):
                    pass  # остаёмся с результатом свободной генерации

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

        # Второй проход с контекстом: модель переспрашивается, зная, что она же сказала про
        # соседние шаги. «Открыл» и «закрыл» на отдельных кадрах неразличимы — их
        # различает только порядок, и его надо дать модели явно.
        if config.VLM_CONTEXT and len(steps) > 1 and by_id:
            def label(step_id: int) -> str | None:
                item = by_id.get(step_id)
                if not item or not item.get("action"):
                    return None
                return f"{item['action']} {item.get('object') or ''}".strip()

            ordered = sorted(steps, key=lambda step: step.start_sec)
            segments = []
            for index, step in enumerate(ordered):
                segments.append(
                    {
                        "id": step.id,
                        "frames": self._frames(video_path, step, crop),
                        "previous": label(ordered[index - 1].id) if index else None,
                        "following": (
                            label(ordered[index + 1].id) if index + 1 < len(ordered) else None
                        ),
                    }
                )
            try:
                second = self._post(
                    "/annotate",
                    {
                        "segments": segments,
                        "actions": vocabulary.actions,
                        "objects": vocabulary.objects,
                        "pairs": vocabulary.pairs,
                        "domain": config.DOMAIN or vocabulary.description or None,
                    },
                )
                for item in second.get("results", []):
                    if item.get("action"):
                        by_id[item["id"]] = item
            except (urllib.error.URLError, OSError, TimeoutError, ValueError):
                pass  # остаёмся с первым проходом

        # Ещё один проход: модель не придумывает ответ, а оценивает короткий список гипотез.
        # Свободная генерация заставляет её вспоминать слово из длинного списка; оценка
        # правдоподобия превращает задачу в «сравни картинку с гипотезой», а это ей даётся
        # заметно лучше. Список собирается из её же первого ответа и альтернатив, включая
        # перекрёстные комбинации — так проверяются обе оси, и действие, и предмет.
        if config.VLM_RESCORE and by_id:
            segments = []
            for step in steps:
                named = by_id.get(step.id)
                if not named:
                    continue
                candidates = self._shortlist(named, vocabulary)
                if len(candidates) > 1:
                    segments.append(
                        {
                            "id": step.id,
                            "frames": self._frames(video_path, step, crop),
                            "candidates": [list(pair) for pair in candidates],
                        }
                    )
            if segments:
                try:
                    rescored = self._post(
                        "/annotate",
                        {
                            "segments": segments,
                            "actions": vocabulary.actions,
                            "objects": vocabulary.objects,
                            "pairs": vocabulary.pairs,
                            "domain": config.DOMAIN or vocabulary.description or None,
                            "mode": "score",
                        },
                    )
                    for item in rescored.get("results", []):
                        by_id[item["id"]] = item
                except (urllib.error.URLError, OSError, TimeoutError, ValueError):
                    pass  # остаёмся с результатом свободной генерации

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
