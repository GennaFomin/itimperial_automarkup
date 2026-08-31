"""Сквозная проверка: загрузка → обработка → правка → экспорт."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from praxis import config, store
from praxis.schema import Annotation, steps_from_csv


def make_video(path: Path, seconds: float = 12, size: str = "1280x720") -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}:rate=30",
            "-t",
            str(seconds),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def videos(tmp_path_factory) -> dict[str, Path]:
    directory = tmp_path_factory.mktemp("clips")
    return {
        "ok": make_video(directory / "ok.mp4", seconds=12),
        "too_long": make_video(directory / "long.mp4", seconds=35),
        "too_small": make_video(directory / "small.mp4", seconds=5, size="640x480"),
    }


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    """Клиент с заглушкой сегментатора: эти тесты про HTTP-цикл, а не про качество нарезки.

    Настоящий сегментатор на синтетическом ролике честно отдаёт один шаг — резать там
    нечего, — поэтому проверять на нём правку границ бессмысленно.
    """
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "work" / "praxis.db")
    monkeypatch.setattr(config, "PIPELINE", "stub")
    with TestClient(app=_app()) as test_client:
        yield test_client


@pytest.fixture
def real_pipeline_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "work" / "praxis.db")
    monkeypatch.setattr(config, "PIPELINE", "motion-dp")
    with TestClient(app=_app()) as test_client:
        yield test_client


def _app():
    from praxis.api import app

    return app


def upload(client: TestClient, path: Path):
    with path.open("rb") as handle:
        return client.post("/api/videos", files={"file": (path.name, handle, "video/mp4")})


def test_full_cycle_upload_edit_export(client, videos):
    created = upload(client, videos["ok"])
    assert created.status_code == 200, created.text
    video_id = created.json()["id"]

    record = client.get(f"/api/videos/{video_id}").json()
    assert record["status"] == "done", record["error"]
    assert record["height"] == 720
    # ffmpeg округляет частоту, поэтому кадров может быть на один-два меньше заказанных.
    assert config.FILMSTRIP_COUNT - 2 <= len(record["filmstrip"]) <= config.FILMSTRIP_COUNT
    assert len(record["motion"]) > 10

    payload = client.get(f"/api/videos/{video_id}/annotation").json()
    annotation = Annotation.model_validate(payload["annotation"])
    assert payload["problems"] == []
    assert len(annotation.steps) >= 2
    assert annotation.steps[-1].end_sec == pytest.approx(record["duration_sec"], abs=0.05)
    assert annotation.provenance.pipeline == "stub"

    # Правка: двигаем границу и меняем метку — так же, как это сделает редактор.
    annotation.steps[0].end_sec = 3.0
    annotation.steps[1].start_sec = 3.0
    annotation.steps[0].action = "screw"
    annotation.steps[0].object = "chassis"
    saved = client.put(
        f"/api/videos/{video_id}/annotation", json=annotation.model_dump(mode="json")
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["problems"] == []

    exported = client.get(f"/api/videos/{video_id}/export.json").json()
    assert exported["steps"][0]["end_sec"] == 3.0
    assert Annotation.model_validate(exported)

    csv_text = client.get(f"/api/videos/{video_id}/export.csv").text
    assert steps_from_csv(csv_text) == annotation.steps


def test_editor_endpoints_serve_media_and_frames(client, videos):
    video_id = upload(client, videos["ok"]).json()["id"]

    frame = client.get(f"/api/videos/{video_id}/frame", params={"t": 4.2})
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"

    strip_name = client.get(f"/api/videos/{video_id}").json()["filmstrip"][0]
    assert client.get(f"/api/videos/{video_id}/strip/{strip_name}").status_code == 200

    ranged = client.get(f"/api/videos/{video_id}/media", headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206, "плеер не сможет перематывать без Range-запросов"


def test_rejects_videos_outside_case_requirements(client, videos):
    too_long = upload(client, videos["too_long"])
    assert too_long.status_code == 400 and "длиннее" in too_long.json()["detail"]

    too_small = upload(client, videos["too_small"])
    assert too_small.status_code == 400 and "720p" in too_small.json()["detail"]


def test_review_time_is_recorded_for_the_kpi(client, videos):
    video_id = upload(client, videos["ok"]).json()["id"]
    client.post(
        f"/api/videos/{video_id}/events",
        json={"kind": "review_seconds", "payload": {"seconds": 42.5}},
    )
    stats = client.get("/api/stats").json()
    assert stats["videos"] == 1
    assert stats["total_sec"] == pytest.approx(42.5)


def test_vocabulary_is_served_for_dropdowns(client):
    vocabulary = client.get("/api/vocabulary").json()
    assert len(vocabulary["actions"]) == 11
    assert "wheel" in vocabulary["pairs"]["attach"]


def test_store_survives_restart(client, videos, monkeypatch):
    video_id = upload(client, videos["ok"]).json()["id"]
    assert store.get_video(video_id)["status"] == "done"


def test_real_pipeline_produces_valid_annotation(real_pipeline_client, videos):
    """Шаги не пересекаются, лежат внутри ролика и никогда не отдаются пустым списком.

    Сплошного покрытия таймлайна не требуется и не должно быть: между действиями бывают
    паузы, и помечать их действием — значит выдумывать разметку.
    """
    video_id = upload(real_pipeline_client, videos["ok"]).json()["id"]
    record = real_pipeline_client.get(f"/api/videos/{video_id}").json()
    assert record["status"] == "done", record["error"]

    payload = real_pipeline_client.get(f"/api/videos/{video_id}/annotation").json()
    annotation = Annotation.model_validate(payload["annotation"])

    assert annotation.provenance.pipeline == "motion-dp"
    assert len(annotation.steps) >= 1, "пустая разметка недопустима ни при каких входных данных"
    assert len(annotation.steps) <= config.MAX_SEGMENTS

    steps = sorted(annotation.steps, key=lambda step: step.start_sec)
    assert steps[0].start_sec >= 0.0
    assert steps[-1].end_sec <= record["duration_sec"] + 0.05
    for left, right in zip(steps, steps[1:]):
        assert left.end_sec <= right.start_sec + 0.001, "шаги не должны пересекаться"
    for step in steps:
        assert step.keyframe_sec is not None
        assert step.start_sec <= step.keyframe_sec <= step.end_sec
