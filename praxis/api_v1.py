"""HTTP-слой контракта кейсодателя (contracts.md, schema_version 1.0).

Отдельный слой, а не переписанный `/api/videos`: внутренний API богаче контракта —
у него иерархия шагов, отметка проверки, происхождение прогона, — и обеднять его
ради внешней формы неправильно. Поэтому контракт живёт рядом как проекция, а не
вместо.

Вся конверсия вынесена в `praxis.contract_v1` и проверяется без HTTP. Здесь
остаётся только маршрутизация, доступ к хранилищу и коды ошибок.

Расхождения с контрактом, которые нельзя закрыть кодом, перечислены в
docs/CONTRACT_V1.md — в частности, отсутствующие уверенности по полям.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from praxis import config, contract_v1, errors, jobs, media, store
from praxis.errors import ContractError, UploadRejected
from praxis.schema import Annotation, to_contract_csv, to_contract_json
from praxis.vocab import check_annotation, load_vocabulary

router = APIRouter(prefix="/api/v1", tags=["contract"])


# Контракт называет состояния иначе, чем внутренняя модель, и одно из них
# внутренней модели неизвестно вовсе.
STATUS = {"queued": "queued", "processing": "running", "failed": "failed"}

# Стадии praxis: сегментация и именование слиты в одну, поэтому `proposals` и
# `keyframe` не возникают никогда. Показывать их «пройденными» было бы враньём —
# UI честно оставит их незажжёнными.
STAGE = {None: None, "decode": "decode", "recognize": "recognize", "done": "validate"}


class ReviewSegmentIn(BaseModel):
    id: str
    origin: str = "model"
    start_ms: int
    end_ms: int
    action: str
    object: str | None = None
    keyframe_ms: int | None = None


class ReviewIn(BaseModel):
    schema_version: str = "1.0"
    prediction_id: str | None = None
    reviewer: str = "anonymous"
    submitted_at: datetime | None = None
    segments: list[ReviewSegmentIn] = Field(default_factory=list)
    time_spent_ms: int = 0
    # Расширения контракта, оба необязательные и аддитивные.
    # Какие сегменты человек подтвердил глазами: в контракте такого поля нет,
    # а заказчику важно отличать проверенное от просто нетронутого.
    # None и пустой список — разные утверждения: первое означает «клиент про
    # проверку ничего не говорит», второе — «проверенных нет». Без различия
    # снятую человеком отметку невозможно было бы сохранить.
    verified_ids: list[str] | None = None
    # Замер разметки с нуля — знаменатель KPI «в три раза быстрее». Такая правка
    # намеренно не сохраняется поверх настоящей.
    mode: str = "review"


class ActivityIn(BaseModel):
    mode: str = "review"
    seconds: float


def _record(job_id: str) -> dict:
    record = store.get_video(job_id)
    if record is None:
        raise ContractError(404, errors.JOB_NOT_FOUND, "Задача не найдена", {"job_id": job_id})
    return record


def _annotation(payload: str | None, job_id: str, record: dict, kind: str) -> Annotation:
    if not payload:
        raise ContractError(
            409,
            errors.NOT_READY,
            f"Разметка «{kind}» ещё не готова",
            {"status": record["status"], "stage": record["stage"]},
        )
    return Annotation.model_validate_json(payload)


def _job_view(record: dict) -> dict:
    """Статус задания в форме §2."""
    warnings = json.loads(record["warnings"]) if record["warnings"] else []
    status = record["status"]
    if status == "done":
        # Прогон мог пройти не в полную силу: сервис признаков или именования был
        # недоступен. Контракт называет это отдельным состоянием, и потерять его
        # нельзя — иначе деградировавший прогон неотличим от полноценного.
        contract_status = "done_with_errors" if warnings else "done"
    else:
        contract_status = STATUS.get(status, status)

    error = None
    if status == "failed":
        error = {
            "code": errors.classify_failure(record["error"]),
            "message": errors.first_line(record["error"]) or "Обработка не удалась",
            "details": {},
        }

    return {
        "job_id": record["id"],
        "status": contract_status,
        "stage": STAGE.get(record["stage"], None),
        "progress": _progress(record),
        "created_at": record["created_at"],
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "error": error,
        # Сверх контракта, но необходимо экрану задач: без этого список пришлось бы
        # собирать вторым запросом на каждый ролик.
        "filename": record["filename"],
        "duration_ms": round((record["duration_sec"] or 0) * 1000),
        "warnings": warnings,
        "reviewed": bool(record["review"]),
    }


def _progress(record: dict) -> float:
    from praxis.api import STAGE_PROGRESS

    if record["status"] == "queued":
        return 0.0
    return STAGE_PROGRESS.get(record["stage"], 0.0)


# ------------------------------------------------------------------ задания


@router.post("/jobs", status_code=202)
async def create_job(background: BackgroundTasks, file: UploadFile | None = None) -> dict:
    if file is None:
        raise ContractError(
            422,
            errors.UNSUPPORTED_FORMAT,
            "Нужен файл: приём по ссылке не поддерживается",
            {"reason": "video_url"},
        )
    job_id = uuid.uuid4().hex[:12]
    raw = await file.read()
    try:
        meta = jobs.prepare_upload(job_id, file.filename or "video.mp4", raw)
    except UploadRejected as rejected:
        raise ContractError(422, rejected.code, rejected.message, rejected.details) from rejected
    except media.MediaError as error:
        raise ContractError(422, errors.DECODE_FAILED, str(error)) from error

    store.create_video(job_id, file.filename or "video.mp4", meta)
    background.add_task(jobs.process_video, job_id)
    return {
        "job_id": job_id,
        "status": "queued",
        "created_at": store.get_video(job_id)["created_at"],
    }


@router.get("/jobs")
async def list_jobs() -> list[dict]:
    """Список заданий. Расширение контракта: экрану задач нужен один запрос вместо N."""
    return [_job_view(record) for record in store.list_videos()]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    return _job_view(_record(job_id))


@router.get("/jobs/{job_id}/prediction")
async def get_prediction(job_id: str) -> dict:
    """Прогноз модели. Неизменяем: на нём считаются метрики (§3)."""
    record = _record(job_id)
    payload = record["prediction"]
    annotation = _annotation(payload, job_id, record, "prediction")
    return contract_v1.to_prediction(annotation, record, payload)


@router.get("/jobs/{job_id}/annotation")
async def get_annotation(job_id: str, source: str = "current") -> dict:
    """Актуальная разметка: правка человека, если она есть, иначе прогноз.

    Расширение контракта, и оно обязательное. Без него редактор, открытый
    повторно, показывал бы прогноз поверх уже сохранённой правки, а следующее
    сохранение стирало бы её — на задаче, которую список отмечает как
    проверенную. Форма ответа та же, что у прогноза, чтобы клиент читал их
    одним кодом.
    """
    record = _record(job_id)
    if source not in {"current", "review", "prediction"}:
        raise ContractError(
            422, errors.INVALID_REVIEW, "source: current, review или prediction"
        )
    payload = (
        store.annotation_json(record) if source == "current" else record[source]
    )
    annotation = _annotation(payload, job_id, record, source)
    document = contract_v1.to_prediction(annotation, record, record["prediction"] or payload)
    document["source"] = "review" if record["review"] and source != "prediction" else "prediction"
    return document


@router.post("/jobs/{job_id}/review")
async def post_review(job_id: str, review: ReviewIn) -> dict:
    record = _record(job_id)
    prediction_json = record["prediction"]
    prediction = _annotation(prediction_json, job_id, record, "prediction")
    base = Annotation.model_validate_json(store.annotation_json(record) or prediction_json)

    # Правка привязана к конкретному прогону (§1). Если ролик пересчитали, молча
    # записать её поверх нового прогноза значит потерять и то и другое.
    expected = contract_v1.prediction_id(prediction_json)
    if review.prediction_id and review.prediction_id != expected:
        raise ContractError(
            409,
            errors.NOT_READY,
            "Правка относится к другому прогону: ролик пересчитан",
            {"expected": expected, "got": review.prediction_id},
        )

    try:
        annotation, problems, id_map = contract_v1.annotation_from_review(
            [segment.model_dump() for segment in review.segments],
            base=base,
            prediction=prediction,
            verified_ids=None if review.verified_ids is None else set(review.verified_ids),
        )
    except ValueError as error:
        raise ContractError(
            422, errors.INVALID_REVIEW, "Разметка нарушает инварианты", {"violations": [str(error)]}
        ) from error

    problems += check_annotation(annotation, load_vocabulary(config.VOCAB_PATH))
    saved_at = datetime.now(timezone.utc).isoformat()

    if review.mode == "scratch":
        # Замер ручной разметки не должен затирать настоящую правку: иначе
        # `annotation_json` начнёт отдавать пустышку, а работа человека исчезнет.
        store.log_event(
            job_id,
            "scratch_review",
            {"steps": len(annotation.steps), "reviewer": review.reviewer},
        )
    else:
        store.save_review(job_id, annotation.model_dump_json())

    # Время в событие пишет только таймер активности (POST /activity). Здесь оно
    # сохраняется как справочное «по часам»: сложить оба замера в один счётчик
    # значило бы удвоить время и превратить speedup в мусор.
    store.log_event(
        job_id,
        "save",
        {
            "steps": len(annotation.steps),
            "wall_ms": review.time_spent_ms,
            "mode": review.mode,
            "reviewer": review.reviewer,
        },
    )

    return {
        "review_id": f"r_{job_id}_{len(store.events(job_id, 'save'))}",
        "saved_at": saved_at,
        "problems": problems,
        "id_map": id_map,
    }


@router.get("/jobs/{job_id}/export")
async def export(
    job_id: str,
    format: str = Query("json", pattern="^(json|csv)$"),
    source: str = Query("review"),
) -> PlainTextResponse:
    record = _record(job_id)
    if source not in {"review", "prediction"}:
        raise ContractError(422, errors.INVALID_REVIEW, "source: review или prediction")

    payload = record["prediction"] if source == "prediction" else store.annotation_json(record)
    annotation = _annotation(payload, job_id, record, source)
    store.log_event(job_id, "export", {"format": format, "source": source})

    if format == "json":
        return PlainTextResponse(
            to_contract_json(annotation),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.json"'},
        )
    # BOM обязателен по §5: без него Excel ломает кириллицу в названиях действий.
    return PlainTextResponse(
        "﻿" + to_contract_csv(annotation),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.csv"'},
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    record = _record(job_id)
    if record["status"] in {"done", "failed", "cancelled"}:
        # Идемпотентность (§2): по уже завершённому заданию отмена — не ошибка.
        return _job_view(record)
    store.update_video(job_id, cancel_requested=1)
    return _job_view(_record(job_id))


# ------------------------------------------------------------------ словарь и метрики


@router.get("/vocab")
async def get_vocab() -> dict:
    return contract_v1.to_vocab_doc(load_vocabulary(config.VOCAB_PATH))


@router.get("/limits")
async def get_limits() -> dict:
    """Ограничения на вход. Расширение: чтобы смена лимита не требовала релиза фронта."""
    return {
        "max_duration_ms": round(config.MAX_DURATION_SEC * 1000),
        "min_height": config.MIN_HEIGHT,
        "allowed_extensions": sorted(config.ALLOWED_SUFFIXES),
        "job_timeout_ms": round(config.JOB_TIMEOUT_SEC * 1000),
    }


@router.get("/stats")
async def get_stats() -> dict:
    return store.review_stats()


@router.post("/jobs/{job_id}/activity")
async def post_activity(job_id: str, activity: ActivityIn) -> dict:
    """Активные секунды работы человека — единственный источник времени для KPI.

    Присылается дельтами по ходу работы, а не одной суммой в конце: вкладку
    закрывают, и замер, накопленный только в памяти, теряется целиком.
    """
    _record(job_id)
    kind = "scratch_seconds" if activity.mode == "scratch" else "review_seconds"
    store.log_event(job_id, kind, {"seconds": round(activity.seconds, 3)})
    return {"logged": True}


# ------------------------------------------------------------------ медиа


@router.get("/jobs/{job_id}/media")
async def get_media(job_id: str) -> FileResponse:
    _record(job_id)
    return FileResponse(store.video_dir(job_id) / "source.mp4", media_type="video/mp4")


@router.get("/jobs/{job_id}/frame")
async def get_frame(job_id: str, ms: int = Query(ge=0)) -> FileResponse:
    """Кадр по времени в миллисекундах — контракт не знает секунд (§1).

    Дисковый кэш общий с `/api/videos/{id}/frame`: ключ там уже считается в
    миллисекундах, поэтому два эндпоинта греют один и тот же набор файлов.
    """
    _record(job_id)
    directory = store.video_dir(job_id)
    path = directory / "frames" / f"t_{ms:09d}.jpg"
    if not path.exists():
        try:
            media.extract_frame(directory / "source.mp4", ms / 1000, path)
        except media.MediaError as error:
            raise ContractError(404, errors.DECODE_FAILED, str(error)) from error
    return FileResponse(path, media_type="image/jpeg")


def is_contract_path(request: Request) -> bool:
    """Относится ли запрос к контрактному слою — по нему выбирается формат ошибки."""
    return request.url.path.startswith(router.prefix)
