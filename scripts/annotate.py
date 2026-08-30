#!/usr/bin/env python3
"""Пакетная разметка папки с роликами — без веб-приложения.

Нужен в двух местах: чтобы гонять метрики по валидационному набору и чтобы обработать
скрытый набор организаторов, если его дадут файлами.

    python scripts/annotate.py --in devset/clips --out devset/pred
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from praxis import jobs, media
from praxis.schema import VideoMeta

SUFFIXES = {".mp4", ".mov", ".MP4", ".MOV"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", dest="target", type=Path, required=True)
    args = parser.parse_args()

    args.target.mkdir(parents=True, exist_ok=True)
    clips = sorted(path for path in args.source.iterdir() if path.suffix in SUFFIXES)
    if not clips:
        raise SystemExit(f"в {args.source} нет роликов")

    durations: list[float] = []
    failed: list[str] = []
    for clip in clips:
        started = time.perf_counter()
        try:
            probed = media.probe(clip)
            meta = VideoMeta(
                id=clip.stem,
                filename=clip.name,
                **{k: probed[k] for k in ("duration_sec", "fps", "width", "height")},
            )
            annotation = jobs.annotate_clip(clip, meta, jobs.perceive(clip), started).annotation
        except Exception as error:  # noqa: BLE001 — один битый файл не должен ронять прогон
            failed.append(f"{clip.name}: {error}")
            print(f"{clip.name}: ПРОПУЩЕН ({error})")
            continue

        (args.target / f"{clip.stem}.json").write_text(
            annotation.model_dump_json(indent=2), encoding="utf-8"
        )
        elapsed = annotation.provenance.processing_sec or 0.0
        durations.append(elapsed)
        print(f"{clip.name}: {len(annotation.steps)} шагов, {elapsed:.2f} с")

    if failed:
        print(f"\nне обработано: {len(failed)}")

    if durations:
        print(
            f"\nроликов: {len(durations)}, среднее {sum(durations) / len(durations):.2f} с, "
            f"максимум {max(durations):.2f} с (предел кейса — 120 с)"
        )


if __name__ == "__main__":
    main()
