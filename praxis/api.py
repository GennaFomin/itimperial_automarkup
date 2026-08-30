"""HTTP-слой: загрузка, статус, чтение и правка разметки, экспорт, кадры, события."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from praxis import config, jobs, media, store
from praxis.schema import Annotation, to_csv
from praxis.vocab import check_annotation, load_vocabulary


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    yield


app = FastAPI(title="Praxis", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EventIn(BaseModel):
    kind: str
    payload: dict = {}


def _video_or_404(video_id: str) -> dict:
    record = store.get_video(video_id)
    if record is None:
        raise HTTPException(404, "видео не найдено")
    return record


def _public(record: dict) -> dict:
    return {
        "id": record["id"],
        "filename": record["filename"],
        "duration_sec": record["duration_sec"],
        "fps": record["fps"],
        "width": record["width"],
        "height": record["height"],
        "status": record["status"],
        "error": record["error"],
        "processing_sec": record["processing_sec"],
        "motion": json.loads(record["motion"]) if record["motion"] else [],
        "filmstrip": json.loads(record["filmstrip"]) if record["filmstrip"] else [],
    }


@app.post("/api/videos")
async def upload_video(background: BackgroundTasks, file: UploadFile) -> dict:
    video_id = uuid.uuid4().hex[:12]
    raw = await file.read()
    try:
        meta = jobs.prepare_upload(video_id, file.filename or "video.mp4", raw)
    except (ValueError, media.MediaError) as error:
        raise HTTPException(400, str(error)) from error

    store.create_video(video_id, file.filename or "video.mp4", meta)
    background.add_task(jobs.process_video, video_id)
    return {"id": video_id, **meta, "status": "queued"}


@app.get("/api/videos")
async def list_videos() -> list[dict]:
    return [_public(record) for record in store.list_videos()]


@app.get("/api/videos/{video_id}")
async def get_video(video_id: str) -> dict:
    return _public(_video_or_404(video_id))


@app.get("/api/videos/{video_id}/annotation")
async def get_annotation(video_id: str) -> dict:
    record = _video_or_404(video_id)
    if not record["annotation"]:
        raise HTTPException(409, f"разметка ещё не готова (статус: {record['status']})")
    annotation = Annotation.model_validate_json(record["annotation"])
    problems = check_annotation(annotation, load_vocabulary(config.VOCAB_PATH))
    return {"annotation": annotation.model_dump(mode="json"), "problems": problems}


@app.put("/api/videos/{video_id}/annotation")
async def save_annotation(video_id: str, annotation: Annotation) -> dict:
    _video_or_404(video_id)
    if annotation.video.id != video_id:
        raise HTTPException(400, "идентификатор видео в разметке не совпадает с адресом")
    store.update_video(video_id, annotation=annotation.model_dump_json())
    store.log_event(video_id, "save", {"steps": len(annotation.steps)})
    problems = check_annotation(annotation, load_vocabulary(config.VOCAB_PATH))
    return {"saved": True, "problems": problems}


@app.get("/api/videos/{video_id}/export.json")
async def export_json(video_id: str) -> FileResponse:
    record = _video_or_404(video_id)
    if not record["annotation"]:
        raise HTTPException(409, "разметки нет")
    annotation = Annotation.model_validate_json(record["annotation"])
    path = store.video_dir(video_id) / "annotation.json"
    path.write_text(annotation.model_dump_json(indent=2), encoding="utf-8")
    store.log_event(video_id, "export", {"format": "json"})
    return FileResponse(path, media_type="application/json", filename=f"{video_id}.json")


@app.get("/api/videos/{video_id}/export.csv")
async def export_csv(video_id: str) -> PlainTextResponse:
    record = _video_or_404(video_id)
    if not record["annotation"]:
        raise HTTPException(409, "разметки нет")
    annotation = Annotation.model_validate_json(record["annotation"])
    store.log_event(video_id, "export", {"format": "csv"})
    return PlainTextResponse(
        to_csv(annotation),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{video_id}.csv"'},
    )


@app.get("/api/videos/{video_id}/media")
async def get_media(video_id: str) -> FileResponse:
    _video_or_404(video_id)
    return FileResponse(store.video_dir(video_id) / "source.mp4", media_type="video/mp4")


@app.get("/api/videos/{video_id}/frame")
async def get_frame(video_id: str, t: float = Query(ge=0)) -> FileResponse:
    """Кадр на произвольной секунде с кэшем на диске — редактор дёргает его при правках."""
    _video_or_404(video_id)
    directory = store.video_dir(video_id)
    path = directory / "frames" / f"t_{int(round(t * 1000)):09d}.jpg"
    if not path.exists():
        try:
            media.extract_frame(directory / "source.mp4", t, path)
        except media.MediaError as error:
            raise HTTPException(404, str(error)) from error
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/videos/{video_id}/strip/{name}")
async def get_strip(video_id: str, name: str) -> FileResponse:
    _video_or_404(video_id)
    path = store.video_dir(video_id) / "strip" / name
    if ".." in name or not path.exists():
        raise HTTPException(404, "превью не найдено")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/vocabulary")
async def get_vocabulary() -> dict:
    return load_vocabulary(config.VOCAB_PATH).model_dump()


@app.post("/api/videos/{video_id}/events")
async def post_event(video_id: str, event: EventIn) -> dict:
    _video_or_404(video_id)
    store.log_event(video_id, event.kind, event.payload)
    return {"logged": True}


@app.get("/api/stats")
async def get_stats() -> dict:
    return store.review_stats()


# Собранный фронт отдаётся тем же процессом — в докере это один контейнер, без nginx.
# Монтируется последним, чтобы не перехватывать /api.
if config.WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=config.WEB_DIST, html=True), name="web")
