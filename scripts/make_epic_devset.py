#!/usr/bin/env python3
"""Второй валидационный набор — из EPIC-KITCHENS-100.

Зачем он нужен. На Assembly101 именование не работает, и причина, по всей видимости, в
самой таксономии: детали игрушечных машин визуально почти неразличимы, а их названия
(`rocker panel`, `interior`, `base`) лежат вне текстового распределения моделей. Проверить
это можно только на другом домене с тем же устройством разметки — пара (глагол, объект) с
границами. EPIC-KITCHENS даёт ровно это, но предметы там обиходные: тарелка, кран, нож.

Если на кухне те же модели работают заметно лучше, значит пайплайн исправен, а трудность
была в данных. Если нет — значит проблема в подходе, и это тоже надо знать.

Разметка публична, видео качаются range-запросами, как и в Assembly101.

    python scripts/make_epic_devset.py --out data/devset_epic --clips 16
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ANNOTATIONS = (
    "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/"
    "master/EPIC_100_validation.csv"
)
BASE_55 = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"
BASE_100 = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"


def seconds(timestamp: str) -> float:
    hours, minutes, rest = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def video_url(video_id: str) -> str:
    """У расширения EPIC-100 и у исходных записей EPIC-55 разные адреса."""
    participant, _, number = video_id.partition("_")
    if int(number) >= 100:
        return f"{BASE_100}/{participant}/videos/{video_id}.MP4"
    return f"{BASE_55}/videos/test/{participant}/{video_id}.MP4"


def load_rows() -> list[dict]:
    with urllib.request.urlopen(ANNOTATIONS, timeout=120) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def windows(rows: list[dict], min_dur: float, max_dur: float, min_steps: int, max_steps: int):
    """Подряд идущие действия одного видео, укладывающиеся в требования кейса."""
    by_video: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)

    found = []
    for video_id, items in by_video.items():
        items.sort(key=lambda row: seconds(row["start_timestamp"]))
        index = 0
        while index < len(items):
            chunk = [items[index]]
            for candidate in items[index + 1 : index + max_steps]:
                gap = seconds(candidate["start_timestamp"]) - seconds(chunk[-1]["stop_timestamp"])
                span = seconds(candidate["stop_timestamp"]) - seconds(chunk[0]["start_timestamp"])
                if gap > 1.5 or span > max_dur:
                    break
                chunk.append(candidate)

            duration = seconds(chunk[-1]["stop_timestamp"]) - seconds(chunk[0]["start_timestamp"])
            if len(chunk) >= min_steps and min_dur <= duration <= max_dur:
                found.append((video_id, chunk))
                index += len(chunk)
            else:
                index += 1
    return found


def cut(url: str, start: float, duration: float, out: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            url,
            "-t",
            f"{duration:.3f}",
            "-vf",
            "scale=-2:720",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out),
        ],
        check=True,
        timeout=900,
    )


def probe(path: Path) -> dict:
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
    ).stdout
    data = json.loads(output)
    stream = data["streams"][0]
    num, _, den = stream["avg_frame_rate"].partition("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(float(num) / float(den), 3),
        "duration_sec": round(float(data["format"]["duration"]), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=16)
    parser.add_argument("--verbs", type=int, default=20, help="сколько глаголов в словаре")
    parser.add_argument("--nouns", type=int, default=60, help="сколько объектов в словаре")
    parser.add_argument("--min-dur", type=float, default=6.0)
    parser.add_argument("--max-dur", type=float, default=28.0)
    args = parser.parse_args()

    rows = load_rows()
    print(f"сегментов в валидации EPIC-100: {len(rows)}")

    # Закрытый словарь из самых частых значений — так его и задаёт заказчик, а не
    # объединением всего, что встречалось в датасете.
    verbs = [v for v, _ in collections.Counter(r["verb"] for r in rows).most_common(args.verbs)]
    nouns = [n for n, _ in collections.Counter(r["noun"] for r in rows).most_common(args.nouns)]
    allowed_verbs, allowed_nouns = set(verbs), set(nouns)

    candidates = [
        (video_id, chunk)
        for video_id, chunk in windows(rows, args.min_dur, args.max_dur, 2, 5)
        if all(r["verb"] in allowed_verbs and r["noun"] in allowed_nouns for r in chunk)
    ]
    # Разные видео, чтобы набор не выродился в одну кухню.
    seen: set[str] = set()
    chosen = []
    for video_id, chunk in candidates:
        if video_id in seen:
            continue
        seen.add(video_id)
        chosen.append((video_id, chunk))
        if len(chosen) >= args.clips:
            break

    print(f"окон найдено: {len(candidates)}, взято: {len(chosen)} из разных видео")

    clips_dir, gt_dir = args.out / "clips", args.out / "gt"
    clips_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    pairs: dict[str, set] = collections.defaultdict(set)
    made = 0
    for video_id, chunk in chosen:
        offset = seconds(chunk[0]["start_timestamp"])
        duration = seconds(chunk[-1]["stop_timestamp"]) - offset
        clip = clips_dir / f"{video_id}_{int(offset)}.mp4"
        try:
            cut(video_url(video_id), offset, duration, clip)
            meta = probe(clip)
        except Exception as error:  # noqa: BLE001 — недоступное видео пропускаем
            print(f"  пропуск {video_id}: {str(error)[:90]}")
            continue

        steps = []
        previous_end = 0.0
        for index, row in enumerate(chunk):
            # В EPIC соседние нарративы иногда перекрываются на десятые доли секунды:
            # это независимые описания, а не разбиение. Приводим к разбиению, обрезая
            # начало по концу предыдущего шага.
            start = round(max(seconds(row["start_timestamp"]) - offset, previous_end), 3)
            end = round(min(seconds(row["stop_timestamp"]) - offset, meta["duration_sec"]), 3)
            if end <= start + 0.05:
                continue
            previous_end = end
            steps.append(
                {
                    "id": index,
                    "level": "coarse",
                    "parent_id": None,
                    "start_sec": start,
                    "end_sec": end,
                    "action": row["verb"],
                    "object": row["noun"],
                    "keyframe_sec": round((start + end) / 2, 3),
                    "confidence": None,
                    "source": "manual",
                }
            )
            pairs[row["verb"]].add(row["noun"])

        (gt_dir / f"{clip.stem}.json").write_text(
            json.dumps(
                {
                    "video": {"id": clip.stem, "filename": clip.name, **meta},
                    "steps": steps,
                    "provenance": {
                        "app_version": "devset",
                        "pipeline": "epic100-gt",
                        "vocabulary": "epic100-top",
                        "models": {},
                        "backend": None,
                        "processing_sec": None,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        made += 1
        print(f"  {clip.name}: {duration:.1f} с, шагов {len(steps)}", flush=True)

    vocabulary = args.out / "vocab_epic.yaml"
    lines = [
        "# Закрытый словарь из самых частых действий и предметов EPIC-KITCHENS-100.",
        "# Домен обиходный: тарелка, кран, нож — в отличие от деталей игрушечных машин.",
        "name: epic100-top",
        "version: 1",
        'description: "Top verbs and nouns of EPIC-KITCHENS-100"',
        "",
        "actions:",
        *[f'  - "{verb}"' for verb in verbs],
        "",
        "objects:",
        *[f'  - "{noun}"' for noun in nouns],
    ]
    vocabulary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nготово: {made} роликов, словарь {len(verbs)} глаголов и {len(nouns)} объектов")
    print(f"словарь: {vocabulary}")


if __name__ == "__main__":
    main()
