"""Контракт разметки: единый формат для пайплайна, редактора и экспорта.

Всё остальное в проекте общается через эти структуры. Инварианты проверяются здесь,
поэтому невалидная разметка не может попасть ни в редактор, ни в экспорт.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator

# Допуск на сравнение таймкодов: миллисекунда. Кадр даже при 240 fps длиннее.
EPS = 1e-3


class Level(str, Enum):
    """Уровень иерархии. Крупные шаги и вложенные в них подшаги."""

    coarse = "coarse"
    fine = "fine"


class Source(str, Enum):
    """Откуда взялся шаг: пайплайн, правка пользователя или ручное добавление."""

    auto = "auto"
    edited = "edited"
    manual = "manual"


class Step(BaseModel):
    id: int = Field(ge=0)
    level: Level = Level.coarse
    parent_id: int | None = None
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    action: str = Field(min_length=1)
    object: str | None = None
    keyframe_sec: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: Source = Source.auto
    # Человек посмотрел на этот шаг и подтвердил его. Отличается от source: правка
    # означает, что шаг меняли, а проверка — что на него смотрели глазами.
    verified: bool = False

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @model_validator(mode="after")
    def _check(self) -> Step:
        if self.end_sec <= self.start_sec + EPS:
            raise ValueError(f"шаг {self.id}: end_sec должен быть больше start_sec")
        if self.keyframe_sec is not None and not (
            self.start_sec - EPS <= self.keyframe_sec <= self.end_sec + EPS
        ):
            raise ValueError(f"шаг {self.id}: ключевой кадр вне границ шага")
        if self.level is Level.fine and self.parent_id is None:
            raise ValueError(f"шаг {self.id}: подшаг обязан ссылаться на родителя")
        if self.level is Level.coarse and self.parent_id is not None:
            raise ValueError(f"шаг {self.id}: у крупного шага не может быть родителя")
        return self


class VideoMeta(BaseModel):
    id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    duration_sec: float = Field(gt=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Provenance(BaseModel):
    """Чем и когда сделана разметка. Уходит в экспорт: без этого результат невоспроизводим."""

    app_version: str
    pipeline: str
    vocabulary: str
    models: dict[str, str] = Field(default_factory=dict)
    backend: str | None = None
    processing_sec: float | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Annotation(BaseModel):
    video: VideoMeta
    steps: list[Step] = Field(default_factory=list)
    provenance: Provenance

    @model_validator(mode="after")
    def _check(self) -> Annotation:
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("идентификаторы шагов не уникальны")
        by_id = {s.id: s for s in self.steps}

        for s in self.steps:
            if s.end_sec > self.video.duration_sec + EPS:
                raise ValueError(f"шаг {s.id} выходит за длительность видео")

        for level in Level:
            segments = sorted(
                (s for s in self.steps if s.level is level), key=lambda s: s.start_sec
            )
            for left, right in zip(segments, segments[1:]):
                if right.start_sec < left.end_sec - EPS:
                    raise ValueError(
                        f"шаги {left.id} и {right.id} пересекаются на уровне {level.value}"
                    )

        for s in self.steps:
            if s.parent_id is None:
                continue
            parent = by_id.get(s.parent_id)
            if parent is None or parent.level is not Level.coarse:
                raise ValueError(
                    f"шаг {s.id}: родитель {s.parent_id} не найден среди крупных шагов"
                )
            if s.start_sec < parent.start_sec - EPS or s.end_sec > parent.end_sec + EPS:
                raise ValueError(f"шаг {s.id} не вложен в родительский шаг {parent.id}")

        return self

    def at_level(self, level: Level) -> list[Step]:
        return sorted((s for s in self.steps if s.level is level), key=lambda s: s.start_sec)


CSV_COLUMNS = [
    "video_id",
    "filename",
    "step_id",
    "level",
    "parent_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "action",
    "object",
    "keyframe_sec",
    "confidence",
    "source",
    "verified",
]


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def to_csv(annotation: Annotation, include_verified: bool = True) -> str:
    """Плоский экспорт: одна строка на шаг, иерархия выражена через parent_id."""
    columns = CSV_COLUMNS if include_verified else [c for c in CSV_COLUMNS if c != "verified"]
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=columns, lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    for step in sorted(annotation.steps, key=lambda s: (s.start_sec, s.id)):
        writer.writerow(
            {
                "video_id": annotation.video.id,
                "filename": annotation.video.filename,
                "step_id": step.id,
                "level": step.level.value,
                "parent_id": "" if step.parent_id is None else step.parent_id,
                "start_sec": _fmt(step.start_sec),
                "end_sec": _fmt(step.end_sec),
                "duration_sec": _fmt(step.duration_sec),
                "action": step.action,
                "object": "" if step.object is None else step.object,
                "keyframe_sec": _fmt(step.keyframe_sec),
                "confidence": _fmt(step.confidence),
                "source": step.source.value,
                "verified": int(step.verified),
            }
        )
    return buffer.getvalue()


def steps_from_csv(text: str) -> list[Step]:
    """Разбор экспортированного CSV обратно в шаги — нужен для проверок round-trip."""
    steps: list[Step] = []
    for row in csv.DictReader(io.StringIO(text)):
        steps.append(
            Step(
                id=int(row["step_id"]),
                level=Level(row["level"]),
                parent_id=int(row["parent_id"]) if row["parent_id"] else None,
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                action=row["action"],
                object=row["object"] or None,
                keyframe_sec=float(row["keyframe_sec"]) if row["keyframe_sec"] else None,
                confidence=float(row["confidence"]) if row["confidence"] else None,
                source=Source(row["source"]),
                verified=bool(int(row.get("verified") or 0)),
            )
        )
    return steps


def to_json(annotation: Annotation, include_verified: bool = True, indent: int | None = 2) -> str:
    """JSON-экспорт. Отметку о проверке можно погасить, если её не ждут на приёмке."""
    payload = annotation.model_dump(mode="json")
    if not include_verified:
        for step in payload["steps"]:
            step.pop("verified", None)
    return json.dumps(payload, ensure_ascii=False, indent=indent)
