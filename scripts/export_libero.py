#!/usr/bin/env python3
"""LIBERO-10 с подзадачами → ролики и покадровые метки для роботного корпуса.

Набор `KeWangRobotics/libero_10_subtasks` хранит кадры внутри parquet (LeRobot v3):
камера agentview 256×256 при 10 fps и строка подзадачи на каждый кадр. Здесь каждый
эпизод собирается в mp4 (ffmpeg, libx264, высота 720 — как ждёт пайплайн) и рядом
кладётся <эпизод>.subtasks.json = {"fps": 10, "labels": [...]}; дальше работает
`make_robot_devset.py --dataset subtasks`. Последние N эпизодов — держ-аут.

    python scripts/export_libero.py --in data/libero_10_subtasks --out data/raw_libero --holdout 60
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def group_episodes(rows: list[dict]) -> dict[int, list[dict]]:
    """Кадры по эпизодам, внутри эпизода — по номеру кадра."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["episode_index"])].append(row)
    return {k: sorted(v, key=lambda r: int(r["frame_index"])) for k, v in sorted(groups.items())}


def split_holdout(episodes: list[int], holdout: int) -> tuple[list[int], list[int]]:
    if holdout <= 0:
        return list(episodes), []
    return list(episodes[:-holdout]), list(episodes[-holdout:])


def image_bytes(cell) -> bytes:
    """Ячейка изображения в LeRobot v3 — словарь {bytes, path} или сырые байты."""
    if isinstance(cell, dict):
        return cell["bytes"]
    return bytes(cell)


def write_episode(frames: list[dict], fps: float, out: Path, stem: str, image_key: str) -> None:
    (out).mkdir(parents=True, exist_ok=True)
    video = out / f"{stem}.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
         "-vf", "scale=-2:720", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-an", str(video)],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None
    for row in frames:
        ffmpeg.stdin.write(image_bytes(row[image_key]))
    ffmpeg.stdin.close()
    ffmpeg.wait()
    (out / f"{stem}.subtasks.json").write_text(
        json.dumps({"fps": fps, "labels": [str(r["subtask"]) for r in frames]}), encoding="utf-8"
    )


def main() -> None:
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=60)
    parser.add_argument("--image-key", default="images.agentview_rgb")
    args = parser.parse_args()

    info = json.loads((args.source / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 10))
    total = int(info.get("total_episodes", 0))
    first_holdout = total - args.holdout if total else None
    columns = ["episode_index", "frame_index", "subtask", args.image_key]

    # Потоково: эпизоды в файлах идут по порядку, поэтому в памяти живёт один эпизод,
    # а не весь набор — целиком это десятки гигабайт кадров.
    current: int | None = None
    buffer: list[dict] = []
    written = {"train": 0, "holdout": 0}

    def flush() -> None:
        if current is None or not buffer:
            return
        frames = sorted(buffer, key=lambda r: int(r["frame_index"]))
        name = "holdout" if first_holdout is not None and current >= first_holdout else "train"
        write_episode(frames, fps, args.out / name, f"libero_{current:04d}", args.image_key)
        written[name] += 1

    for chunk in sorted((args.source / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(chunk)
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                episode = int(row["episode_index"])
                if episode != current:
                    flush()
                    current, buffer = episode, []
                buffer.append(row)
    flush()
    print(f"train: {written['train']} эпизодов, holdout: {written['holdout']} → {args.out}")


if __name__ == "__main__":
    main()
