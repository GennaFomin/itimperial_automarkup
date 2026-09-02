"""Обработка ролика: от загруженного файла до сохранённой разметки."""

from __future__ import annotations

import base64
import json
import tempfile
import time
import urllib.error
import urllib.request
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from praxis import config, media, store
from praxis.pipeline.base import Perception, get_segmenter
from praxis.pipeline.naming import get_namer, merge_adjacent
from praxis.schema import Annotation, Provenance, VideoMeta
from praxis.vocab import load_vocabulary


def _remote_embeddings(source: Path) -> np.ndarray | None:
    """Покадровые эмбеддинги с GPU-машины.

    Ролик уезжает целиком и в уменьшенном виде: пять мегабайт по сети дешевле, чем двести
    отдельных кадров. Если сервис недоступен, возвращаем None и работаем на серых блоках —
    качество ниже, но система не встаёт.
    """
    if config.FEATURES != "embed" or not config.CLIP_BASE_URL:
        return None

    with tempfile.TemporaryDirectory() as directory:
        small = Path(directory) / "small.mp4"
        try:
            media.transcode_for_upload(source, small)
            payload = {
                "video": base64.b64encode(small.read_bytes()).decode(),
                "fps": float(config.MOTION_FPS),
            }
            request = urllib.request.Request(
                config.CLIP_BASE_URL.rstrip("/") + "/embed",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=config.VLM_TIMEOUT) as response:
                answer = json.loads(response.read())
        except (urllib.error.URLError, OSError, TimeoutError, ValueError, media.MediaError):
            return None

    if not answer.get("count"):
        return None
    raw = np.frombuffer(base64.b64decode(answer["embeddings"]), dtype=np.float16)
    return raw.reshape(answer["count"], answer["dim"]).astype(np.float32)


def _video_features(source: Path) -> dict | None:
    """Признаки окон кадров с видеоэнкодера: они видят движение, а не только содержимое."""
    if config.FEATURES != "video" or not config.VIDEO_BASE_URL:
        return None

    with tempfile.TemporaryDirectory() as directory:
        small = Path(directory) / "small.mp4"
        try:
            media.transcode_for_upload(source, small, height=256)
            payload = {
                "video": base64.b64encode(small.read_bytes()).decode(),
                "fps": config.VIDEO_FPS,
                "window": config.VIDEO_WINDOW,
                "stride": config.VIDEO_STRIDE,
            }
            request = urllib.request.Request(
                config.VIDEO_BASE_URL.rstrip("/") + "/embed",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=config.VLM_TIMEOUT) as response:
                answer = json.loads(response.read())
        except (urllib.error.URLError, OSError, TimeoutError, ValueError, media.MediaError):
            return None

    if not answer.get("count"):
        return None
    raw = np.frombuffer(base64.b64decode(answer["embeddings"]), dtype=np.float16)
    answer["matrix"] = raw.reshape(answer["count"], answer["dim"]).astype(np.float32)
    return answer


def _resample(values: np.ndarray, source_fps: float, target_fps: float, count: int) -> np.ndarray:
    """Переносит сигнал на сетку признаков: усреднение по интервалу каждого окна."""
    if not len(values):
        return np.zeros(count, dtype=np.float32)
    result = np.zeros(count, dtype=np.float32)
    span = source_fps / target_fps
    for index in range(count):
        start = int(index * span)
        end = max(start + 1, int((index + 1) * span))
        window = values[start : min(end, len(values))]
        result[index] = float(np.mean(window)) if len(window) else float(values[-1])
    return result


def perceive(source: Path) -> Perception:
    """Один проход по кадрам: и полоса движения для таймлайна, и признаки для нарезки."""
    frames = media.gray_frames(source)
    motion = media.motion_from_frames(frames)
    fps = float(config.MOTION_FPS)
    offset = 0.0

    # Признаки заказаны, но сервис не ответил — дальше будет заметно хуже, и это надо
    # донести до прогона, а не спрятать в тихий фолбэк.
    wanted_remote = (config.FEATURES == "video" and config.VIDEO_BASE_URL) or (
        config.FEATURES == "embed" and config.CLIP_BASE_URL
    )

    video = _video_features(source)
    if video is not None:
        appearance = video["matrix"]
        motion = _resample(motion, config.MOTION_FPS, video["fps"], len(appearance))
        fps = float(video["fps"])
        offset = float(video.get("offset_sec", 0.0))
        return Perception(
            fps=fps,
            motion=motion,
            appearance=appearance,
            crop=_crop(frames),
            offset=offset,
        )

    appearance = _remote_embeddings(source)
    if appearance is not None:
        # Длины могут разойтись на кадр-другой из-за округления частоты.
        length = min(len(appearance), len(motion))
        appearance, motion = appearance[:length], motion[:length]
    else:
        appearance = media.appearance_from_frames(frames)
        degraded = ("сервис признаков недоступен: нарезка на серых блоках",) if wanted_remote else ()
        return Perception(
            fps=fps,
            motion=motion,
            appearance=appearance,
            crop=_crop(frames),
            degraded=degraded,
        )
    # Три попытки улучшить признаки провалились на валидационном наборе: карта движения по
    # блокам дала F1@0.5 0.605 вместо 0.715, обрезка по рабочей зоне — 0.642, сглаживание
    # по времени — 0.700. Остался простой вид кадра целиком: дело не в признаках.
    return Perception(fps=fps, motion=motion, appearance=appearance, crop=_crop(frames))


def _crop(frames: np.ndarray) -> tuple[float, float, float, float]:
    """Доли кадра, в которых вообще что-то происходит: нужны семантической стадии."""
    top, bottom, left, right = media.active_region(frames)
    height, width = (frames.shape[1], frames.shape[2]) if len(frames) else (1, 1)
    return (left / width, top / height, (right - left) / width, (bottom - top) / height)


@dataclass
class ClipResult:
    """Разметка плюс подсказки редактору. Подсказки в экспорт не идут."""

    annotation: Annotation
    alternatives: dict[int, list[dict]]


def annotate_clip(
    source: Path, meta: VideoMeta, perception: Perception, started: float | None = None
) -> ClipResult:
    """Ядро разметки без веб-обвязки: используется и фоновой задачей, и пакетным прогоном."""
    started = time.perf_counter() if started is None else started
    vocabulary = load_vocabulary(config.VOCAB_PATH)
    segmenter = get_segmenter(config.PIPELINE)
    result = segmenter.run(source, meta, vocabulary, perception)

    # Границы уже стоят — языковая модель только называет то, что нарезано.
    namer = get_namer()
    named = namer.name_steps(source, meta, result.steps, vocabulary, perception.crop)

    # Соседние куски с одинаковой меткой — это один шаг, разрезанный по смене картинки.
    # Понять это можно только после именования, поэтому склейка здесь, а не в сегментаторе.
    alternatives = dict(named.alternatives)
    labelled = named.models.get("namer") not in {None, "none"} and "namer_status" not in named.models
    steps = merge_adjacent(named.steps, labelled=labelled)

    # Всё, что просело в этом прогоне, собирается в одном месте и уезжает в происхождение.
    warnings = list(perception.degraded)
    if named.models.get("namer_status"):
        warnings.append(f"именование: {named.models['namer_status']}")

    annotation = Annotation(
        video=meta,
        steps=steps,
        provenance=Provenance(
            app_version=_version(),
            pipeline=segmenter.name,
            vocabulary=vocabulary.name,
            models={**result.models, **named.models},
            backend=config.VLM_BASE_URL or "local",
            processing_sec=round(time.perf_counter() - started, 3),
            warnings=warnings,
        ),
    )
    return ClipResult(
        annotation=annotation,
        alternatives={step.id: alternatives.get(step.id, []) for step in steps},
    )


def process_video(video_id: str) -> None:
    """Синхронная обработка одного ролика. Вызывается из фоновой задачи."""
    started = time.perf_counter()
    record = store.get_video(video_id)
    if record is None:
        return

    store.update_video(video_id, status="processing", error=None)
    try:
        directory = store.video_dir(video_id)
        source = directory / "source.mp4"
        meta = VideoMeta(
            id=video_id,
            filename=record["filename"],
            duration_sec=record["duration_sec"],
            fps=record["fps"],
            width=record["width"],
            height=record["height"],
        )

        perception = perceive(source)
        motion = [round(float(value), 4) for value in perception.motion]
        strip = media.filmstrip(source, meta.duration_sec, directory / "strip")

        result = annotate_clip(source, meta, perception, started)
        annotation = result.annotation
        elapsed = annotation.provenance.processing_sec

        store.update_video(
            video_id,
            status="done",
            warnings=json.dumps(annotation.provenance.warnings, ensure_ascii=False),
            processing_sec=elapsed,
            prediction=annotation.model_dump_json(),
            motion=json.dumps(motion),
            filmstrip=json.dumps(strip),
            alternatives=json.dumps(result.alternatives),
        )
    except Exception as error:  # noqa: BLE001 — статус задачи важнее типа ошибки
        store.update_video(
            video_id,
            status="failed",
            error=f"{error}\n{traceback.format_exc(limit=3)}",
            processing_sec=round(time.perf_counter() - started, 3),
        )


def prepare_upload(video_id: str, filename: str, raw: bytes) -> dict:
    """Сохраняет загруженный файл и проверяет требования кейса к ролику."""
    suffix = Path(filename).suffix.lower()
    if suffix not in config.ALLOWED_SUFFIXES:
        raise ValueError(f"поддерживаются только {', '.join(sorted(config.ALLOWED_SUFFIXES))}")

    directory = store.video_dir(video_id)
    source = directory / "source.mp4"
    source.write_bytes(raw)

    meta = media.probe(source)
    if meta["duration_sec"] > config.MAX_DURATION_SEC:
        raise ValueError(
            f"ролик длиннее {config.MAX_DURATION_SEC:.0f} с (получено {meta['duration_sec']:.1f} с)"
        )
    if meta["height"] < config.MIN_HEIGHT:
        raise ValueError(f"нужно не меньше {config.MIN_HEIGHT}p (получено {meta['height']}p)")
    return meta


def _version() -> str:
    from praxis import __version__

    return __version__
