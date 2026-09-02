"""Хранилище: SQLite для состояния задач, разметки и событий редактора.

Видео и кадры лежат на диске, всё остальное — здесь. События нужны не для красоты:
из них считается фактическое время работы над роликом, то есть KPI кейса.

Прогноз модели и правка человека хранятся в разных колонках и никогда не смешиваются:
`prediction` пишется прогоном один раз и дальше неизменяем, `review` появляется при
первой правке. Иначе метрики модели пересчитать нечем — после одного прохода редактора
эталона просто не остаётся, — а сама пара «что предсказали → что оказалось верным»
и есть тот проверенный датасет, ради которого кейс затевался.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from praxis import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id             TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    duration_sec   REAL,
    fps            REAL,
    width          INTEGER,
    height         INTEGER,
    status         TEXT NOT NULL,
    error          TEXT,
    processing_sec REAL,
    prediction     TEXT,
    review         TEXT,
    warnings       TEXT,
    motion         TEXT,
    filmstrip      TEXT,
    alternatives   TEXT,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    kind     TEXT NOT NULL,
    payload  TEXT,
    at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_video ON events(video_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with connect() as connection:
        connection.executescript(SCHEMA)
        # Мягкие миграции: база, созданная более ранней версией, не должна ломаться.
        # Каждый шаг падает на уже применённой базе и это нормально.
        for statement in (
            "ALTER TABLE videos ADD COLUMN alternatives TEXT",
            "ALTER TABLE videos ADD COLUMN prediction TEXT",
            "ALTER TABLE videos ADD COLUMN review TEXT",
            "ALTER TABLE videos ADD COLUMN warnings TEXT",
            "UPDATE videos SET prediction = annotation WHERE prediction IS NULL",
            "ALTER TABLE videos DROP COLUMN annotation",
        ):
            try:
                connection.execute(statement)
            except sqlite3.OperationalError:
                pass


def create_video(video_id: str, filename: str, meta: dict) -> None:
    with connect() as connection:
        connection.execute(
            "INSERT INTO videos (id, filename, duration_sec, fps, width, height, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)",
            (
                video_id,
                filename,
                meta["duration_sec"],
                meta["fps"],
                meta["width"],
                meta["height"],
                now(),
            ),
        )


def update_video(video_id: str, **fields) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with connect() as connection:
        connection.execute(
            f"UPDATE videos SET {assignments} WHERE id = ?", (*fields.values(), video_id)
        )


def annotation_json(record: dict) -> str | None:
    """Актуальная разметка: правка человека, если она есть, иначе прогноз модели."""
    return record["review"] or record["prediction"]


def save_review(video_id: str, payload: str) -> None:
    """Правка пишется только в свою колонку — прогноз остаётся нетронутым навсегда."""
    update_video(video_id, review=payload)


def get_video(video_id: str) -> dict | None:
    with connect() as connection:
        row = connection.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    return dict(row) if row else None


def list_videos() -> list[dict]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def log_event(video_id: str, kind: str, payload: dict | None = None) -> None:
    with connect() as connection:
        connection.execute(
            "INSERT INTO events (video_id, kind, payload, at) VALUES (?, ?, ?, ?)",
            (video_id, kind, json.dumps(payload or {}, ensure_ascii=False), now()),
        )


def _seconds_by_video(kind: str) -> dict[str, float]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT video_id, payload FROM events WHERE kind = ?", (kind,)
        ).fetchall()
    per_video: dict[str, float] = {}
    for row in rows:
        seconds = float(json.loads(row["payload"]).get("seconds", 0))
        per_video[row["video_id"]] = per_video.get(row["video_id"], 0.0) + seconds
    return per_video


def _summarise(per_video: dict[str, float]) -> dict:
    values = sorted(per_video.values())
    if not values:
        return {"videos": 0, "total_sec": 0.0, "median_sec": 0.0, "per_video": {}}
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {
        "videos": len(values),
        "total_sec": round(sum(values), 1),
        "median_sec": round(median, 1),
        "per_video": {key: round(value, 1) for key, value in per_video.items()},
    }


def events(video_id: str, kind: str | None = None) -> list[dict]:
    """Журнал по ролику. Append-only: прогон перезаписывает разметку, история остаётся."""
    query = "SELECT * FROM events WHERE video_id = ?"
    params: tuple = (video_id,)
    if kind:
        query += " AND kind = ?"
        params += (kind,)
    with connect() as connection:
        return [dict(row) for row in connection.execute(query + " ORDER BY id", params)]


def review_stats() -> dict:
    """Сколько человек потратил на правку и сколько — на разметку с нуля.

    Кейс требует сокращения ручной работы втрое. Одного времени правки для этого мало:
    нужно, с чем сравнивать, поэтому редактор умеет и то и другое, а отношение медиан
    считается здесь.
    """
    review = _seconds_by_video("review_seconds")
    scratch = _seconds_by_video("scratch_seconds")
    result = _summarise(review)
    result["scratch"] = _summarise(scratch)
    if result["median_sec"] and result["scratch"]["median_sec"]:
        result["speedup"] = round(result["scratch"]["median_sec"] / result["median_sec"], 2)
    return result


def video_dir(video_id: str) -> Path:
    path = config.WORK_DIR / video_id
    path.mkdir(parents=True, exist_ok=True)
    return path
