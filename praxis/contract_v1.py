"""Конверсия между внутренней моделью praxis и контрактом кейсодателя (contracts.md).

Здесь нет ни HTTP, ни базы — только чистые функции, поэтому конверсия проверяется
тестами напрямую, без поднятия приложения.

Две модели расходятся сильнее, чем кажется на первый взгляд:

    praxis                          контракт
    steps[]                         segments[]
    start_sec: float                start_ms: int
    action: str                     action: {value, confidence}
    object: str | None              object: {value, confidence}, где None → "unknown"
    confidence: float | None        по полю отдельно
    id: int                         id: "seg_001"
    source: auto|edited|manual      origin: model|human
    level, parent_id, verified      —

Направление «наружу» простое. Опасное направление — обратное: контракт не переносит
`level`, `parent_id`, `verified`, `source` и `provenance`, поэтому правку нельзя
собирать из одного лишь присланного массива — иначе каждое сохранение обедняло бы
разметку. Всё, чего в контракте нет, берётся из сохранённого документа.
"""

from __future__ import annotations

import colorsys
import hashlib
import re

from praxis import config
from praxis.schema import (
    EPS,
    SCHEMA_VERSION,
    Annotation,
    Level,
    Source,
    Step,
    _ms,
    model_version,
)
from praxis.vocab import Vocabulary

SEGMENT_RE = re.compile(r"^seg_(\d+)$")

# Контракт (§1) требует, чтобы неизвестное было значением словаря, а не null.
# Внутри praxis неизвестный объект — это None, поэтому пара конверсий симметрична.
UNKNOWN = "unknown"

# Допуск обратной конверсии, в секундах. Ровно половина миллисекунды: округление
# `sec → ms → sec` не может сдвинуть значение больше чем на неё.
ROUND_TRIP_EPS = 0.0005


def segment_id(step_id: int) -> str:
    return f"seg_{step_id:03d}"


def step_id(segment_id_value: str) -> int | None:
    """Обратный разбор. None — идентификатор не наш (сегмент создан человеком)."""
    match = SEGMENT_RE.match(segment_id_value or "")
    return int(match.group(1)) if match else None


def _sec(ms: int | None) -> float | None:
    return None if ms is None else round(ms / 1000, 3)


def _keep(base: float | None, incoming: float | None) -> float | None:
    """Вернуть исходное значение, если правка — это лишь артефакт округления.

    Прогноз с `end_sec = 5.1004` уходит наружу как 5100 мс и возвращается как 5.100.
    Без этой проверки `diff_steps` засчитал бы правку границы там, где человек ничего
    не трогал, и телеметрия правок — та самая, на которой держится аргументация
    «в три раза быстрее», — начала бы врать в нашу пользу.
    """
    if base is None or incoming is None:
        return incoming
    return base if abs(base - incoming) <= ROUND_TRIP_EPS else incoming


# ---------------------------------------------------------------- наружу


def to_segment(step: Step, duration_ms: int) -> dict:
    """Шаг praxis в виде сегмента контракта.

    Конец клампится по длительности: `round(sec * 1000)` может дать миллисекунду
    сверх ролика и нарушить инвариант §3.1, который UI и валидатор проверяют.
    """
    start = _ms(step.start_sec) or 0
    end = min(_ms(step.end_sec) or 0, duration_ms)
    keyframe = _ms(step.keyframe_sec)
    if keyframe is not None:
        keyframe = min(max(keyframe, start), end)

    return {
        "id": segment_id(step.id),
        "start_ms": start,
        "end_ms": end,
        # Уверенности границ у пайплайна нет: разрез — это выброс в ядровой
        # статистике, а не вероятность. null честнее выдуманного числа.
        "boundary_confidence": None,
        "action": {"value": step.action, "confidence": step.confidence},
        # Одно и то же число не дублируется во второе поле: confidence оценивает
        # пару «действие+объект» целиком, и выдать его за независимую уверенность
        # по объекту значило бы соврать ровно там, где контракт просит точности.
        "object": {"value": step.object or UNKNOWN, "confidence": None},
        "keyframe_ms": keyframe,
        "keyframe_confidence": None,
    }


def prediction_id(prediction_json: str) -> str:
    """Идентификатор прогноза, выводимый из его содержимого.

    Отдельной колонки под него нет, а хранить ещё одну — лишняя миграция ради
    строки. Содержательный хеш даёт то же свойство, которого требует §1: он
    стабилен внутри прогона и меняется, как только ролик пересчитали заново.
    """
    return "p_" + hashlib.sha1(prediction_json.encode("utf-8")).hexdigest()[:8]


def capabilities() -> dict:
    """Что пайплайн умеет на самом деле — машиночитаемо, рядом с прогнозом.

    Половина полей уверенности приходит как null. Без такого блока это выглядит
    как недоделка; с ним видно, что null — измеренное свойство пайплайна.
    """
    return {
        "boundary_confidence": False,
        "object_confidence": False,
        "keyframe_confidence": False,
        # Число есть, но оценивает связку «действие+объект», а не глагол отдельно.
        "action_confidence": "pair",
        # У сегментаторов, кроме motion-dp, ключевой кадр — середина отрезка.
        "keyframe_source": "midpoint" if config.PIPELINE != "motion-dp" else "selected",
        "open_vocabulary": bool(config.OPEN_VOCABULARY),
    }


def to_prediction(annotation: Annotation, record: dict, raw_json: str) -> dict:
    """Документ прогноза в форме контракта §3."""
    duration_ms = _ms(annotation.video.duration_sec) or 0
    provenance = annotation.provenance

    return {
        "schema_version": SCHEMA_VERSION,
        "prediction_id": prediction_id(raw_json),
        "job_id": record["id"],
        "model_version": model_version(annotation),
        "vocab_version": provenance.vocabulary,
        "created_at": provenance.created_at.isoformat(),
        "video": {
            "duration_ms": duration_ms,
            "fps": annotation.video.fps,
            "width": annotation.video.width,
            "height": annotation.video.height,
        },
        "segments": [
            to_segment(step, duration_ms) for step in annotation.at_level(Level.coarse)
        ],
        "stats": {
            "latency_ms": provenance.latency_ms,
            # Валюта задаётся ставкой PRAXIS_GPU_HOUR_COST и не обязана быть долларом,
            # поэтому сумма едет вместе с ней, а не под именем cost_usd.
            "cost": provenance.cost,
            "stages_ms": provenance.stages_ms,
        },
        "errors": _errors_from_warnings(annotation),
        "capabilities": capabilities(),
    }


def _errors_from_warnings(annotation: Annotation) -> list[dict]:
    """Деградация прогона в форме §3 «Частичный результат».

    Предупреждение praxis — это «прогон прошёл, но не в полную силу»: сервис
    признаков или именования был недоступен. Контракт называет такое состояние
    отдельно, и терять его нельзя — иначе деградировавший прогон неотличим
    от полноценного.
    """
    out = [
        {
            "stage": "recognize" if "именован" in warning else "decode",
            "code": "DEGRADED",
            "message": warning,
            "segment_ids": [],
        }
        for warning in annotation.provenance.warnings
    ]

    missing = [
        segment_id(step.id)
        for step in annotation.at_level(Level.coarse)
        if step.keyframe_sec is None
    ]
    if missing:
        out.append(
            {
                "stage": "keyframe",
                "code": "INTERNAL",
                "message": f"ключевой кадр не посчитан для {len(missing)} сегментов",
                "segment_ids": missing,
            }
        )
    return out


# ---------------------------------------------------------------- словарь


def color_for(value: str) -> str:
    """Цвет класса, выводимый из его названия.

    Контракт (§6) требует цвета с сервера, чтобы они не разъезжались между
    таймлайном и скриншотом. Но словарь praxis — плоский список строк без
    оформления, а при открытой лексике классы вообще заранее неизвестны.
    Хеш даёт то же нужное свойство: один и тот же класс всегда одного цвета,
    в любом прогоне и на любой машине, и новый класс не требует правки YAML.
    """
    if value == UNKNOWN:
        return "#9aa3ad"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    hue = int(digest[:4], 16) / 0xFFFF
    # Насыщенность и светлота фиксированы: подобраны под тёмный фон редактора,
    # чтобы любой цвет оставался читаемым и не спорил с оранжевым акцентом.
    red, green, blue = colorsys.hls_to_rgb(hue, 0.58, 0.62)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def to_vocab_doc(vocabulary: Vocabulary) -> dict:
    """Словарь в форме §6: идентификатор, подпись и цвет.

    Подпись берётся из необязательной карты `labels` в YAML, а если её нет — это
    сама строка. Машинного перевода здесь быть не должно: при открытой лексике
    модель отвечает на языке PRAXIS_LANGUAGE, и её ответ уже является подписью.
    """
    labels = getattr(vocabulary, "labels", {}) or {}

    actions = [
        {"id": value, "label_ru": labels.get(value, value), "color": color_for(value)}
        for value in vocabulary.actions
    ]
    objects = [{"id": value, "label_ru": labels.get(value, value)} for value in vocabulary.objects]

    if not any(item["id"] == UNKNOWN for item in actions):
        actions.append({"id": UNKNOWN, "label_ru": "Неизвестно", "color": color_for(UNKNOWN)})
    if not any(item["id"] == UNKNOWN for item in objects):
        objects.append({"id": UNKNOWN, "label_ru": "Неизвестно"})

    return {
        "version": str(vocabulary.version),
        "name": vocabulary.name,
        # При открытой лексике список — подсказка, а не ограничение. UI по этому
        # флагу может разрешить свободный ввод вместо строгого выбора.
        "open": bool(config.OPEN_VOCABULARY),
        "actions": actions,
        "objects": objects,
        "pairs": vocabulary.pairs,
    }


# ---------------------------------------------------------------- обратно


def annotation_from_review(
    segments: list[dict],
    base: Annotation,
    prediction: Annotation | None,
    verified_ids: set[str],
) -> tuple[Annotation, list[str], dict[str, str]]:
    """Собрать полный документ разметки из присланной правки.

    `base` — то, что лежит сейчас (правка или прогноз): из него берётся всё, чего
    в контракте нет. `prediction` нужен, чтобы отличить настоящую правку от
    нетронутого сегмента и проставить `source` честно.

    Возвращает документ, список замечаний и карту новых идентификаторов, чтобы
    клиент мог связать свои временные id с присвоенными.
    """
    by_id = {step.id: step for step in base.steps}
    predicted = {step.id: step for step in (prediction.steps if prediction else [])}
    problems: list[str] = []
    id_map: dict[str, str] = {}

    used = {step.id for step in base.steps}
    next_id = (max(used) + 1) if used else 0

    steps: list[Step] = []
    kept_ids: set[int] = set()

    for incoming in segments:
        raw_id = str(incoming.get("id") or "")
        parsed = step_id(raw_id)
        origin = incoming.get("origin", "model")

        existing = by_id.get(parsed) if parsed is not None else None
        if existing is None:
            # Сегмент создан человеком: у него идентификатор вида seg_new_x1,
            # которому надо выдать собственный целочисленный номер.
            new_id = next_id
            next_id += 1
            id_map[raw_id] = segment_id(new_id)
        else:
            new_id = existing.id
            kept_ids.add(new_id)

        start = _keep(existing.start_sec if existing else None, _sec(incoming.get("start_ms")))
        end = _keep(existing.end_sec if existing else None, _sec(incoming.get("end_ms")))
        keyframe = _keep(
            existing.keyframe_sec if existing else None, _sec(incoming.get("keyframe_ms"))
        )

        action = incoming.get("action") or (existing.action if existing else UNKNOWN)
        obj = incoming.get("object")
        obj = None if obj in (None, "", UNKNOWN) else obj

        step = Step(
            id=new_id,
            # Правка приходит плоской, поэтому всё, что она создаёт, — верхний уровень.
            level=existing.level if existing else Level.coarse,
            parent_id=existing.parent_id if existing else None,
            start_sec=start if start is not None else 0.0,
            end_sec=end if end is not None else 0.0,
            action=action,
            object=obj,
            keyframe_sec=keyframe,
            # Уверенности человек не выдаёт (§4): у правленого сегмента она
            # остаётся модельной, у созданного руками её нет вовсе.
            confidence=existing.confidence if existing else None,
            source=_source_for(existing, predicted.get(new_id), origin, start, end, action, obj),
            verified=raw_id in verified_ids
            or (segment_id(new_id) in verified_ids)
            or (existing.verified if existing and raw_id not in verified_ids else False),
        )
        steps.append(step)

    # Подшаги переносятся, только если их родитель уцелел и не сдвинулся: иначе
    # вложенность стала бы неправдой, а клампить чужие границы — значит выдумывать
    # данные. Сейчас ни один сегментатор их не производит, но правило нужно, чтобы
    # двухуровневая разметка не исчезала молча при первой же правке.
    for step in base.steps:
        if step.level is not Level.fine or step.parent_id is None:
            continue
        parent_before = by_id.get(step.parent_id)
        parent_after = next((s for s in steps if s.id == step.parent_id), None)
        unchanged = (
            parent_after is not None
            and parent_before is not None
            and abs(parent_after.start_sec - parent_before.start_sec) <= EPS
            and abs(parent_after.end_sec - parent_before.end_sec) <= EPS
        )
        if unchanged:
            steps.append(step)
        else:
            problems.append(f"подшаг {step.id} отброшен: родительский шаг изменён")

    annotation = Annotation(
        video=base.video,
        steps=sorted(steps, key=lambda s: (s.start_sec, s.id)),
        # Происхождение описывает прогон, который дал прогноз, и не может
        # приходить из браузера.
        provenance=base.provenance,
    )
    return annotation, problems, id_map


def _source_for(
    existing: Step | None,
    predicted: Step | None,
    origin: str,
    start: float | None,
    end: float | None,
    action: str,
    obj: str | None,
) -> Source:
    """Откуда взялся шаг. Вычисляется, а не принимается от клиента."""
    if origin == "human" or existing is None:
        return Source.manual
    if predicted is None:
        return existing.source
    changed = (
        abs((start or 0) - predicted.start_sec) > EPS
        or abs((end or 0) - predicted.end_sec) > EPS
        or action != predicted.action
        or obj != predicted.object
    )
    return Source.edited if changed else existing.source
