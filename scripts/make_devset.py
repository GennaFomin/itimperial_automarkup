#!/usr/bin/env python3
"""Валидационный набор из Assembly101: окна 5–30 секунд с готовой эталонной разметкой.

Assembly101 размечен ровно тем, что требует кейс: пара (глагол, объект) с границами по
кадрам. Значит валидационный набор можно нарезать, не размечая руками ни одного ролика.

Исходные записи весят до двух гигабайт каждая, но качать их целиком не нужно: ffmpeg
умеет искать по HTTP, поэтому из подписанной ссылки вытягиваются только те байты, что
попадают в окно. Проверено: окно 10 секунд из файла 1.78 ГБ — 38 секунд и 1.8 МБ на диске.

    HF_TOKEN=... python scripts/make_devset.py --sessions 8 --out data/devset
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

REPO = "cvml-nus/assembly101"
# Номера кадров в аннотациях привязаны к 30 fps, хотя исходники сняты на 60.
LABEL_FPS = 30.0


def fetch(path: str) -> Path:
    return Path(hf_hub_download(REPO, path, repo_type="dataset"))


def signed_url(path: str) -> str:
    """Прямая ссылка на CDN: по ней ffmpeg ходит range-запросами вместо полной загрузки."""
    metadata = get_hf_file_metadata(
        hf_hub_url(REPO, path, repo_type="dataset"), token=os.environ.get("HF_TOKEN")
    )
    return metadata.location


def action_map() -> dict[str, tuple[str, str]]:
    """action_cls -> (глагол, объект) по таксономии coarse."""
    with fetch("annotations/coarse-annotations/actions.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        return {
            row["action_cls"].strip(): (row["verb_cls"].strip(), row["noun_cls"].strip())
            for row in csv.DictReader(f)
        }


def read_split(name: str) -> list[str]:
    path = fetch(f"annotations/coarse-annotations/coarse_splits/{name}")
    return [
        line.split("\t")[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_labels(path: Path) -> list[dict]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part for part in line.split("\t") if part.strip()]
        if len(parts) < 3:
            continue
        segments.append(
            {
                "start": int(parts[0]) / LABEL_FPS,
                "end": int(parts[1]) / LABEL_FPS,
                "action": parts[2].strip(),
            }
        )
    return sorted(segments, key=lambda segment: segment["start"])


def windows(
    segments: list[dict],
    min_dur: float,
    max_dur: float,
    min_steps: int,
    max_steps: int,
    max_gap: float,
) -> list[list[dict]]:
    """Подряд идущие шаги, укладывающиеся в требования кейса к ролику.

    Большие паузы между шагами отбрасываются: в окне должно быть действие, а не ожидание.
    """
    result: list[list[dict]] = []
    index = 0
    while index < len(segments):
        chunk = [segments[index]]
        for candidate in segments[index + 1 : index + max_steps]:
            if candidate["start"] - chunk[-1]["end"] > max_gap:
                break
            if candidate["end"] - chunk[0]["start"] > max_dur:
                break
            chunk.append(candidate)

        duration = chunk[-1]["end"] - chunk[0]["start"]
        if len(chunk) >= min_steps and min_dur <= duration <= max_dur:
            result.append(chunk)
            index += len(chunk)
        else:
            index += 1
    return result


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


def cut(source: str, start: float, duration: float, out: Path) -> None:
    """Режет окно из локального файла или прямо из удалённого по подписанной ссылке."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            source,
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
    )


def annotation(clip: Path, chunk: list[dict], offset: float, actions: dict) -> dict:
    meta = probe(clip)
    steps = []
    for index, segment in enumerate(chunk):
        start = round(max(segment["start"] - offset, 0.0), 3)
        end = round(min(segment["end"] - offset, meta["duration_sec"]), 3)
        verb, noun = actions.get(segment["action"], (segment["action"], None))
        steps.append(
            {
                "id": index,
                "level": "coarse",
                "parent_id": None,
                "start_sec": start,
                "end_sec": end,
                "action": verb,
                "object": noun,
                "keyframe_sec": round((start + end) / 2, 3),
                "confidence": None,
                "source": "manual",
            }
        )
    return {
        "video": {"id": clip.stem, "filename": clip.name, **meta},
        "steps": steps,
        "provenance": {
            "app_version": "devset",
            "pipeline": "assembly101-coarse-gt",
            "vocabulary": "assembly101-coarse",
            "models": {},
            "backend": None,
            "processing_sec": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="val_coarse_assembly.txt")
    parser.add_argument("--view", default="C10095_rgb", help="экзоцентрическая камера")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--clips-per-session", type=int, default=3)
    parser.add_argument("--min-dur", type=float, default=5.0)
    parser.add_argument("--max-dur", type=float, default=28.0)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-gap", type=float, default=2.0)
    args = parser.parse_args()

    clips_dir = args.out / "clips"
    gt_dir = args.out / "gt"
    clips_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    actions = action_map()
    entries = read_split(args.split)
    # Берём с шагом по всему сплиту: так в наборе окажутся разные игрушки, а не подряд одна.
    stride = max(1, len(entries) // args.sessions)
    label_files = entries[::stride][: args.sessions]

    total = 0
    for label_file in label_files:
        session = (
            label_file.replace("assembly_", "").replace("disassembly_", "").removesuffix(".txt")
        )
        print(f"[{session}]", flush=True)
        try:
            labels = fetch(f"annotations/coarse-annotations/coarse_labels/{label_file}")
            source = signed_url(f"recordings/{session}/{args.view}.mp4")
        except Exception as error:  # noqa: BLE001 — пропускаем недоступную запись и идём дальше
            print(f"  пропуск: {error}", flush=True)
            continue

        segments = read_labels(labels)
        chosen = windows(
            segments, args.min_dur, args.max_dur, args.min_steps, args.max_steps, args.max_gap
        )[: args.clips_per_session]

        for number, chunk in enumerate(chosen):
            offset = chunk[0]["start"]
            duration = chunk[-1]["end"] - offset
            clip = clips_dir / f"{session[-19:]}_{number}.mp4"
            cut(source, offset, duration, clip)
            payload = annotation(clip, chunk, offset, actions)
            (gt_dir / f"{clip.stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total += 1
            print(f"  {clip.name}: {duration:.1f} с, шагов {len(chunk)}", flush=True)

    print(f"готово: {total} роликов в {args.out}")


if __name__ == "__main__":
    main()
