"""Контрактный слой /api/v1: форма ответов, коды ошибок и обратимость правки.

Проверяется именно то, что легко сломать незаметно: единицы времени, коды вместо
русских строк, честные null там, где пайплайн ничего не измеряет, и — главное —
что отправленный назад прогноз не выглядит как правка человека.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from praxis import config, contract_v1, store
from praxis.schema import Annotation, diff_steps
from tests.test_api import make_video  # общий генератор роликов


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "work" / "praxis.db")
    monkeypatch.setattr(config, "PIPELINE", "stub")
    from praxis.api import app

    store.init_db()
    with TestClient(app=app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict:
    directory = tmp_path_factory.mktemp("v1clips")
    return {
        "ok": make_video(directory / "ok.mp4", seconds=12),
        "too_long": make_video(directory / "long.mp4", seconds=int(config.MAX_DURATION_SEC) + 5),
        "too_small": make_video(directory / "small.mp4", seconds=5, size="640x480"),
    }


def upload(client: TestClient, path) -> str:
    with path.open("rb") as handle:
        response = client.post("/api/v1/jobs", files={"file": (path.name, handle, "video/mp4")})
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def wait_done(client: TestClient, job_id: str) -> dict:
    for _ in range(200):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in {"done", "done_with_errors", "failed", "cancelled"}:
            return job
    raise AssertionError("задание не завершилось")


def test_prediction_satisfies_contract_invariants(client, clips):
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    prediction = client.get(f"/api/v1/jobs/{job_id}/prediction").json()

    assert prediction["schema_version"] == "1.0"
    assert prediction["job_id"] == job_id
    assert prediction["prediction_id"].startswith("p_")
    duration_ms = prediction["video"]["duration_ms"]
    assert isinstance(duration_ms, int)

    previous_end = -1
    seen = set()
    for segment in prediction["segments"]:
        assert isinstance(segment["start_ms"], int) and isinstance(segment["end_ms"], int)
        assert 0 <= segment["start_ms"] < segment["end_ms"] <= duration_ms
        assert segment["start_ms"] >= previous_end, "сегменты пересекаются или не отсортированы"
        previous_end = segment["end_ms"]
        assert segment["id"] not in seen, "идентификаторы не уникальны"
        seen.add(segment["id"])
        if segment["keyframe_ms"] is not None:
            assert segment["start_ms"] <= segment["keyframe_ms"] <= segment["end_ms"]
        # Объект всегда строка: неизвестное — значение словаря, а не null (§1).
        assert isinstance(segment["object"]["value"], str)


def test_missing_confidences_are_null_not_invented(client, clips):
    """Пайплайн не измеряет уверенность границ и объекта — эти поля обязаны быть null.

    Тест существует, чтобы их не «починили» константой: выдуманное число выглядит
    как измерение и молча портит и разметку, и доверие к метрике.
    """
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    prediction = client.get(f"/api/v1/jobs/{job_id}/prediction").json()

    for segment in prediction["segments"]:
        assert segment["boundary_confidence"] is None
        assert segment["keyframe_confidence"] is None
        assert segment["object"]["confidence"] is None

    assert prediction["capabilities"]["boundary_confidence"] is False
    assert prediction["capabilities"]["action_confidence"] == (
        "self-reported" if config.OPEN_VOCABULARY else "pair"
    )


def test_review_round_trip_is_a_noop(client, clips):
    """Прогноз, отправленный назад без изменений, не должен считаться правкой.

    Время в контракте целое в миллисекундах, а внутри — секунды с плавающей точкой.
    Без защиты от округления `5.1004 → 5100 → 5.100` записалось бы как сдвиг
    границы, и телеметрия правок начала бы завышать работу человека — ровно ту
    величину, на которой строится KPI «в три раза быстрее».
    """
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    prediction = client.get(f"/api/v1/jobs/{job_id}/prediction").json()

    before = Annotation.model_validate_json(store.get_video(job_id)["prediction"])
    response = client.post(
        f"/api/v1/jobs/{job_id}/review",
        json={
            "prediction_id": prediction["prediction_id"],
            "reviewer": "tester",
            "segments": [
                {
                    "id": segment["id"],
                    "origin": "model",
                    "start_ms": segment["start_ms"],
                    "end_ms": segment["end_ms"],
                    "action": segment["action"]["value"],
                    "object": segment["object"]["value"],
                    "keyframe_ms": segment["keyframe_ms"],
                }
                for segment in prediction["segments"]
            ],
            "time_spent_ms": 1000,
        },
    )
    assert response.status_code == 200, response.text

    after = Annotation.model_validate_json(store.get_video(job_id)["review"])
    changes = diff_steps(before.steps, after.steps)
    assert changes == {
        "boundary": 0,
        "action": 0,
        "object": 0,
        "keyframe": 0,
        "added": 0,
        "removed": 0,
    }, f"round-trip выдумал правки: {changes}"


def test_review_keeps_verified_and_assigns_ids_for_new_segments(client, clips):
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    prediction = client.get(f"/api/v1/jobs/{job_id}/prediction").json()
    first = prediction["segments"][0]

    response = client.post(
        f"/api/v1/jobs/{job_id}/review",
        json={
            "prediction_id": prediction["prediction_id"],
            "segments": [
                {
                    "id": first["id"],
                    "origin": "model",
                    "start_ms": first["start_ms"],
                    "end_ms": first["end_ms"],
                    "action": first["action"]["value"],
                    "object": first["object"]["value"],
                    "keyframe_ms": first["keyframe_ms"],
                },
                {
                    "id": "seg_new_a1",
                    "origin": "human",
                    "start_ms": prediction["video"]["duration_ms"] - 2000,
                    "end_ms": prediction["video"]["duration_ms"] - 500,
                    "action": "проверить",
                    "object": "деталь",
                    "keyframe_ms": prediction["video"]["duration_ms"] - 1200,
                },
            ],
            "verified_ids": [first["id"]],
            "time_spent_ms": 5000,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["id_map"]["seg_new_a1"].startswith("seg_")

    saved = Annotation.model_validate_json(store.get_video(job_id)["review"])
    by_action = {step.action: step for step in saved.steps}
    assert "проверить" in by_action
    assert by_action["проверить"].source.value == "manual"
    assert any(step.verified for step in saved.steps), "отметка проверки не сохранилась"


def test_scratch_review_does_not_overwrite_the_annotation(client, clips):
    """Замер разметки с нуля не должен стирать настоящую правку."""
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    prediction = client.get(f"/api/v1/jobs/{job_id}/prediction").json()

    client.post(
        f"/api/v1/jobs/{job_id}/review",
        json={"segments": [], "mode": "scratch", "time_spent_ms": 3000},
    )
    assert store.get_video(job_id)["review"] is None
    # Прогноз при этом на месте и по-прежнему отдаётся.
    assert client.get(f"/api/v1/jobs/{job_id}/prediction").json()["segments"] == prediction[
        "segments"
    ]


def test_review_from_a_stale_run_is_rejected(client, clips):
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    response = client.post(
        f"/api/v1/jobs/{job_id}/review",
        json={"prediction_id": "p_deadbeef", "segments": [], "time_spent_ms": 0},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "NOT_READY"


def test_upload_limits_map_to_contract_codes(client, clips):
    for clip, code in (("too_long", "VIDEO_TOO_LONG"), ("too_small", "VIDEO_TOO_SMALL")):
        with clips[clip].open("rb") as handle:
            response = client.post(
                "/api/v1/jobs", files={"file": (clips[clip].name, handle, "video/mp4")}
            )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == code
        assert response.json()["error"]["details"], "детали ошибки пусты"

    with clips["ok"].open("rb") as handle:
        response = client.post("/api/v1/jobs", files={"file": ("clip.avi", handle, "video/x-msvideo")})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_errors_use_the_contract_envelope_only_under_v1(client):
    missing = client.get("/api/v1/jobs/nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "JOB_NOT_FOUND"

    # Внутренний API не меняется: на нём стоят его собственные тесты и его клиент.
    legacy = client.get("/api/videos/nope")
    assert legacy.status_code == 404
    assert "detail" in legacy.json() and "error" not in legacy.json()


def test_prediction_is_not_ready_before_the_run_finishes(client, clips):
    job_id = upload(client, clips["ok"])
    response = client.get(f"/api/v1/jobs/{job_id}/prediction")
    if response.status_code == 409:
        assert response.json()["error"]["code"] == "NOT_READY"
    else:
        assert response.status_code == 200, "прогон успел закончиться — это тоже валидно"


def test_vocab_has_unknown_and_stable_colors(client):
    vocab = client.get("/api/v1/vocab").json()
    assert any(item["id"] == "unknown" for item in vocab["actions"])
    assert any(item["id"] == "unknown" for item in vocab["objects"])
    for action in vocab["actions"]:
        assert action["color"].startswith("#") and len(action["color"]) == 7
    # Цвет выводится из названия, поэтому один и тот же класс красится одинаково
    # в любом прогоне и на любой машине.
    assert contract_v1.color_for("attach") == contract_v1.color_for("attach")
    assert contract_v1.color_for("attach") != contract_v1.color_for("detach")


def test_limits_come_from_config(client):
    limits = client.get("/api/v1/limits").json()
    assert limits["max_duration_ms"] == int(config.MAX_DURATION_SEC * 1000)
    assert limits["min_height"] == config.MIN_HEIGHT
    assert ".mp4" in limits["allowed_extensions"]


def test_export_csv_has_bom_and_contract_header(client, clips):
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    response = client.get(f"/api/v1/jobs/{job_id}/export", params={"format": "csv"})
    assert response.status_code == 200
    text = response.content.decode("utf-8")
    assert text.startswith("﻿"), "нет BOM — Excel сломает кириллицу"
    header = text.lstrip("﻿").splitlines()[0]
    assert header == "start_ms,end_ms,action,object,keyframe_ms,confidence,model_version"


def test_export_json_can_serve_the_untouched_prediction(client, clips):
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    client.post(
        f"/api/v1/jobs/{job_id}/review",
        json={"segments": [], "time_spent_ms": 0},
    )
    predicted = client.get(
        f"/api/v1/jobs/{job_id}/export", params={"format": "json", "source": "prediction"}
    ).json()
    reviewed = client.get(
        f"/api/v1/jobs/{job_id}/export", params={"format": "json", "source": "review"}
    ).json()
    assert predicted["steps"], "прогноз должен остаться доступным после правки"
    assert reviewed["steps"] == []


def test_activity_feeds_the_speedup_metric(client, clips):
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    client.post(f"/api/v1/jobs/{job_id}/activity", json={"mode": "review", "seconds": 40})
    client.post(f"/api/v1/jobs/{job_id}/activity", json={"mode": "scratch", "seconds": 160})

    stats = client.get("/api/v1/stats").json()
    assert stats["median_sec"] == pytest.approx(40, abs=0.01)
    assert stats["scratch"]["median_sec"] == pytest.approx(160, abs=0.01)
    assert stats["speedup"] == pytest.approx(4.0, abs=0.01)


def test_submit_does_not_double_count_time(client, clips):
    """Время считает только таймер активности.

    Если бы `time_spent_ms` из отправки тоже становился секундами, один и тот же
    интервал попал бы в статистику дважды и speedup перестал бы что-либо значить.
    """
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    client.post(f"/api/v1/jobs/{job_id}/activity", json={"mode": "review", "seconds": 30})
    client.post(
        f"/api/v1/jobs/{job_id}/review",
        json={"segments": [], "time_spent_ms": 999_000},
    )
    assert client.get("/api/v1/stats").json()["median_sec"] == pytest.approx(30, abs=0.01)


def test_job_view_reports_status_stage_and_timestamps(client, clips):
    job_id = upload(client, clips["ok"])
    job = wait_done(client, job_id)
    assert job["status"] in {"done", "done_with_errors"}
    assert job["stage"] == "validate"
    assert job["progress"] == 1.0
    assert job["started_at"] and job["finished_at"]
    assert job["error"] is None


def test_degraded_run_is_reported_as_done_with_errors(tmp_path, monkeypatch, clips):
    """Прогон без сервиса признаков проходит, но не в полную силу.

    Контракт называет это состояние отдельно; если бы оно приходило как обычный
    `done`, деградировавший результат был бы неотличим от полноценного.
    """
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "work" / "praxis.db")
    monkeypatch.setattr(config, "PIPELINE", "tsm-kernel")
    monkeypatch.setattr(config, "FEATURES", "video")
    monkeypatch.setattr(config, "VIDEO_BASE_URL", "http://127.0.0.1:9")
    from praxis.api import app

    store.init_db()
    with TestClient(app=app) as client:
        job_id = upload(client, clips["ok"])
        job = wait_done(client, job_id)
        assert job["status"] == "done_with_errors"
        assert job["warnings"], "деградация должна быть названа, а не скрыта"

        prediction = client.get(f"/api/v1/jobs/{job_id}/prediction").json()
        codes = {error["code"] for error in prediction["errors"]}
        assert "DEGRADED" in codes


def test_cancel_is_idempotent_on_finished_jobs(client, clips):
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    first = client.post(f"/api/v1/jobs/{job_id}/cancel")
    second = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert first.status_code == second.status_code == 200
    assert first.json()["status"] == second.json()["status"]


def test_frame_is_addressed_in_milliseconds(client, clips):
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    response = client.get(f"/api/v1/jobs/{job_id}/frame", params={"ms": 4000})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    # Кэш общий с внутренним эндпоинтом: ключ там уже считается в миллисекундах.
    cached = store.video_dir(job_id) / "frames" / "t_000004000.jpg"
    assert cached.exists()


def test_media_is_served_for_the_player(client, clips):
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    response = client.get(f"/api/v1/jobs/{job_id}/media")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"


def test_jobs_list_serves_the_task_screen(client, clips):
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    jobs_list = client.get("/api/v1/jobs").json()
    assert any(item["job_id"] == job_id for item in jobs_list)
    entry = next(item for item in jobs_list if item["job_id"] == job_id)
    assert entry["duration_ms"] > 0 and entry["filename"]

def test_editor_reopens_on_the_saved_review_not_the_prediction(client, clips):
    """Повторно открытая задача обязана показывать работу человека.

    Редактор заполняет дорожку из /annotation. Если бы он читал /prediction, то
    поверх сохранённой правки показал бы модель, а следующее сохранение стёрло
    бы её — на задаче, которую список уже отмечает проверенной.
    """
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    prediction = client.get(f"/api/v1/jobs/{job_id}/prediction").json()
    first = prediction["segments"][0]

    client.post(
        f"/api/v1/jobs/{job_id}/review",
        json={
            "prediction_id": prediction["prediction_id"],
            "segments": [
                {
                    "id": first["id"],
                    "origin": "model",
                    "start_ms": first["start_ms"],
                    "end_ms": first["end_ms"],
                    "action": "переименовано человеком",
                    "object": first["object"]["value"],
                    "keyframe_ms": first["keyframe_ms"],
                }
            ],
            "verified_ids": [first["id"]],
            "time_spent_ms": 1000,
        },
    )

    current = client.get(f"/api/v1/jobs/{job_id}/annotation").json()
    assert current["source"] == "review"
    assert [s["action"]["value"] for s in current["segments"]] == ["переименовано человеком"]
    # Отметка проверки возвращается вместе с сегментом, иначе редактор попросил
    # бы проверить заново то, что человек уже просмотрел.
    assert current["segments"][0]["verified"] is True

    # Прогноз при этом остаётся нетронутым — на нём считаются метрики.
    again = client.get(f"/api/v1/jobs/{job_id}/prediction").json()
    assert [s["action"]["value"] for s in again["segments"]] == [
        s["action"]["value"] for s in prediction["segments"]
    ]


def test_verification_can_be_taken_back(client, clips):
    """Снятую человеком отметку надо сохранять, а не накапливать.

    Список подтверждённых авторитетен целиком: если бы он только добавлял,
    ошибочно отмеченный шаг нельзя было бы вернуть в непроверенные.
    """
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    prediction = client.get(f"/api/v1/jobs/{job_id}/prediction").json()
    segments = [
        {
            "id": s["id"],
            "origin": "model",
            "start_ms": s["start_ms"],
            "end_ms": s["end_ms"],
            "action": s["action"]["value"],
            "object": s["object"]["value"],
            "keyframe_ms": s["keyframe_ms"],
        }
        for s in prediction["segments"]
    ]
    body = {"prediction_id": prediction["prediction_id"], "segments": segments, "time_spent_ms": 0}

    client.post(f"/api/v1/jobs/{job_id}/review", json={**body, "verified_ids": [segments[0]["id"]]})
    assert client.get(f"/api/v1/jobs/{job_id}/annotation").json()["segments"][0]["verified"] is True

    client.post(f"/api/v1/jobs/{job_id}/review", json={**body, "verified_ids": []})
    assert client.get(f"/api/v1/jobs/{job_id}/annotation").json()["segments"][0]["verified"] is False


def test_internal_api_still_answers_422_on_broken_annotation(client, clips):
    """Обработчик ошибок контракта не должен ломать внутренний API.

    Pydantic кладёт в подробности само исключение, которое json не сериализует:
    без приведения любая невалидная разметка возвращала бы 500 вместо 422.
    """
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    response = client.put(
        f"/api/videos/{job_id}/annotation",
        json={
            "video": {
                "id": job_id,
                "filename": "clip.mp4",
                "duration_sec": 12,
                "fps": 30,
                "width": 1280,
                "height": 720,
            },
            # Конец раньше начала — шаг невалиден.
            "steps": [{"id": 0, "start_sec": 5, "end_sec": 1, "action": "a"}],
            "provenance": {"app_version": "0", "pipeline": "stub", "vocabulary": "v"},
        },
    )
    assert response.status_code == 422, response.text
    assert "detail" in response.json()


def test_review_accepts_free_labels_when_vocabulary_is_open(client, clips, monkeypatch):
    """Словаря у заказчика нет: человек обязан иметь право вписать любой глагол и объект,
    и бэкенд не должен отклонять такую правку как «вне списка»."""
    monkeypatch.setattr(config, "OPEN_VOCABULARY", True)
    job_id = upload(client, clips["ok"])
    wait_done(client, job_id)
    prediction = client.get(f"/api/v1/jobs/{job_id}/prediction").json()

    segments = [
        {
            "id": segment["id"],
            "origin": "model",
            "start_ms": segment["start_ms"],
            "end_ms": segment["end_ms"],
            "action": segment["action"]["value"],
            "object": segment["object"]["value"],
            "keyframe_ms": segment["keyframe_ms"],
        }
        for segment in prediction["segments"]
    ]
    segments[0]["action"], segments[0]["object"] = "повернул бутылку", "бутылка с водой"
    response = client.post(
        f"/api/v1/jobs/{job_id}/review",
        json={"prediction_id": prediction["prediction_id"], "reviewer": "tester",
              "segments": segments, "time_spent_ms": 1000},
    )
    assert response.status_code == 200, response.text

    saved = Annotation.model_validate_json(store.get_video(job_id)["review"])
    assert saved.steps[0].action == "повернул бутылку"
    assert saved.steps[0].object == "бутылка с водой"
    exported = client.get(f"/api/v1/jobs/{job_id}/export?format=json").json()
    assert exported["steps"][0]["action"] == "повернул бутылку"

