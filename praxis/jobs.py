"""Обработка ролика: от загруженного файла до сохранённой разметки."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from praxis import config, media, store
from praxis.pipeline.base import get_segmenter
from praxis.schema import Annotation, Provenance, VideoMeta
from praxis.vocab import load_vocabulary


def annotate_clip(
    source: Path, meta: VideoMeta, motion: list[float], started: float | None = None
) -> Annotation:
    """Ядро разметки без веб-обвязки: используется и фоновой задачей, и пакетным прогоном."""
    started = time.perf_counter() if started is None else started
    vocabulary = load_vocabulary(config.VOCAB_PATH)
    segmenter = get_segmenter(config.PIPELINE)
    result = segmenter.run(source, meta, vocabulary, motion)

    return Annotation(
        video=meta,
        steps=result.steps,
        provenance=Provenance(
            app_version=_version(),
            pipeline=segmenter.name,
            vocabulary=vocabulary.name,
            models=result.models,
            backend=config.VLM_BASE_URL or "local",
            processing_sec=round(time.perf_counter() - started, 3),
        ),
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

        motion = media.motion_signal(source)
        strip = media.filmstrip(source, meta.duration_sec, directory / "strip")

        annotation = annotate_clip(source, meta, motion, started)
        elapsed = annotation.provenance.processing_sec

        store.update_video(
            video_id,
            status="done",
            processing_sec=elapsed,
            annotation=annotation.model_dump_json(),
            motion=json.dumps(motion),
            filmstrip=json.dumps(strip),
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
