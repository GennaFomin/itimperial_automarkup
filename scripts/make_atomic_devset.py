#!/usr/bin/env python3
"""Валидационный набор атомарных действий из fine-grained разметки Assembly101.

Зачем отдельно от coarse. Кейсодатель показал ролики, где действия атомарные: поднял
предмет, повернул, перенёс, опустил. Это секунда-две на шаг и 5–12 шагов на ролик.
Наши прежние наборы имеют медиану шага 6 секунд и 2.2 шага на ролик — на такой
гранулярности выигрывает равномерная нарезка, и это говорит о наборе, а не о методе.

Fine-grained разметка Assembly101 — ровно нужный режим: 1380 классов действий из
24 глаголов, кадры при 30 fps, статичная камера от третьего лица, руки на столе.
Глаголы совпадают с описанием кейса: pick up, put down, position, rotate.

    HF_TOKEN=... python scripts/make_atomic_devset.py --out data/pool_atomic --clips 60
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

from praxis.schema import Annotation, Provenance, Step, VideoMeta

REPO = "cvml-nus/assembly101"
LABEL_FPS = 30.0


def fetch(path: str) -> Path:
    return Path(hf_hub_download(REPO, path, repo_type="dataset"))


def signed_url(path: str) -> str:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    import os

    metadata = get_hf_file_metadata(
        hf_hub_url(REPO, path, repo_type="dataset"), token=os.environ.get("HF_TOKEN")
    )
    return metadata.location


def load_actions(split: str, view: str) -> dict[str, list[dict]]:
    """Атомарные действия, сгруппированные по видео и упорядоченные по времени."""
    path = fetch(f"annotations/fine-grained-annotations/{split}.csv")
    by_video: dict[str, list[dict]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # В записи восемь статичных камер, разметка у них общая: берём один ракурс,
            # иначе набор восемь раз повторит одни и те же кадры.
            if not row["video"].endswith(f"{view}.mp4"):
                continue
            by_video[row["video"]].append(
                {
                    "start": int(row["start_frame"]) / LABEL_FPS,
                    "end": int(row["end_frame"]) / LABEL_FPS,
                    "verb": row["verb_cls"],
                    "noun": row["noun_cls"],
                }
            )
    for items in by_video.values():
        items.sort(key=lambda item: item["start"])
    return by_video


def drop_overlaps(actions: list[dict]) -> list[dict]:
    """Одна последовательность вместо двух.

    Fine-разметка Assembly101 ведётся по рукам, и обе руки работают одновременно —
    действия пересекаются во времени. Заказчику нужна одна цепочка шагов, поэтому
    берём непересекающиеся жадно по времени начала; выброшенные остаются в наборе
    допустимых меток, чтобы не наказывать модель за верный, но не выбранный ответ.
    """
    kept: list[dict] = []
    for item in actions:
        if kept and item["start"] < kept[-1]["end"]:
            continue
        kept.append(item)
    return kept


def allowed_labels(actions: list[dict], kept: list[dict], share: float = 0.5) -> list[list[list[str]]]:
    """Все метки, накрывающие каждый оставленный шаг более чем на половину его длины."""
    result = []
    for item in kept:
        span = max(item["end"] - item["start"], 1e-6)
        options = {(item["verb"], item["noun"])}
        for other in actions:
            overlap = min(item["end"], other["end"]) - max(item["start"], other["start"])
            if overlap > share * span:
                options.add((other["verb"], other["noun"]))
        result.append(sorted([verb, noun] for verb, noun in options))
    return result


def windows(actions: list[dict], span: float, min_steps: int, max_steps: int) -> list[list[dict]]:
    """Окна фиксированной длины с подходящим числом атомарных действий внутри.

    Берём непересекающиеся окна: одна запись даёт несколько роликов, но кадры в них
    не повторяются, иначе набор сам себя дублирует.
    """
    result, index = [], 0
    while index < len(actions):
        start = actions[index]["start"]
        chunk = [
            item for item in actions[index:] if item["end"] <= start + span and item["start"] >= start
        ]
        if min_steps <= len(chunk) <= max_steps:
            result.append(chunk)
            last = chunk[-1]["end"]
            while index < len(actions) and actions[index]["start"] < last:
                index += 1
        else:
            index += 1
    return result


def probe(path: Path) -> dict:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(output)
    stream, fmt = info["streams"][0], info["format"]
    numerator, _, denominator = stream["avg_frame_rate"].partition("/")
    return {
        "duration_sec": float(fmt["duration"]),
        "fps": float(numerator) / float(denominator or 1),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def cut(source: str, start: float, duration: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", source,
         "-t", f"{duration:.3f}", "-vf", "scale=-2:720", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(out)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--clips", type=int, default=60)
    parser.add_argument("--span", type=float, default=20.0, help="длина окна в секундах")
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--per-video", type=int, default=2, help="сколько окон с одной записи")
    parser.add_argument("--view", default="C10095_rgb", help="экзоцентрическая камера")
    args = parser.parse_args()

    clips_dir, gt_dir = args.out / "clips", args.out / "gt"
    clips_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    by_video = load_actions(args.split, args.view)
    print(f"записей в сплите {args.split}: {len(by_video)}", flush=True)

    verbs, nouns = set(), set()
    allowed: dict[str, list] = {}
    written = 0
    for video, actions in sorted(by_video.items()):
        if written >= args.clips:
            break
        sequence = drop_overlaps(actions)
        chunks = windows(sequence, args.span, args.min_steps, args.max_steps)[: args.per_video]
        if not chunks:
            continue
        try:
            source = signed_url(f"recordings/{video}")
        except Exception as error:  # noqa: BLE001 — недоступная запись не должна ронять сборку
            print(f"  {video}: пропуск ({error})", flush=True)
            continue

        session = video.split("/")[0].removeprefix("nusar-2021_action_both_")
        for order, chunk in enumerate(chunks):
            if written >= args.clips:
                break
            offset = chunk[0]["start"]
            span = chunk[-1]["end"] - offset
            name = f"{session}_{order}"
            target = clips_dir / f"{name}.mp4"
            try:
                cut(source, offset, span, target)
                meta = probe(target)
            except (subprocess.CalledProcessError, KeyError, ValueError) as error:
                print(f"  {name}: пропуск ({error})", flush=True)
                continue

            steps = []
            for index, item in enumerate(chunk):
                start = round(item["start"] - offset, 3)
                end = round(min(item["end"] - offset, meta["duration_sec"]), 3)
                if end - start < 0.2:
                    continue
                verbs.add(item["verb"])
                nouns.add(item["noun"])
                steps.append(
                    Step(
                        id=len(steps), start_sec=start, end_sec=end,
                        action=item["verb"], object=item["noun"],
                        keyframe_sec=round((start + end) / 2, 3), confidence=1.0,
                    )
                )
            if len(steps) < args.min_steps:
                target.unlink(missing_ok=True)
                continue

            annotation = Annotation(
                video=VideoMeta(id=name, filename=target.name, **meta),
                steps=steps,
                provenance=Provenance(
                    app_version="devset", pipeline="assembly101-fine",
                    vocabulary="assembly101-fine", models={},
                    backend="assembly101", processing_sec=0.0,
                ),
            )
            (gt_dir / f"{name}.json").write_text(
                annotation.model_dump_json(indent=1), encoding="utf-8"
            )
            allowed[name] = allowed_labels(actions, chunk)[: len(annotation.steps)]
            written += 1
            print(f"  {written}/{args.clips} {name}: {len(steps)} действий за {span:.1f} с", flush=True)

    (args.out / "allowed.json").write_text(
        json.dumps(allowed, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    lines = ["name: assembly101-fine", "description: человек собирает игрушку на столе", "actions:"]
    lines += [f"  - {verb}" for verb in sorted(verbs)]
    lines += ["objects:"] + [f"  - {noun}" for noun in sorted(nouns)]
    (args.out / "vocab_atomic.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nготово: {written} роликов, {len(verbs)} глаголов, {len(nouns)} предметов")


if __name__ == "__main__":
    main()
