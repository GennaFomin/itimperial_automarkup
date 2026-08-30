#!/usr/bin/env python3
"""Подбор ручек сегментатора по валидационному набору.

Штраф за отрезок и вес физических границ подбирать на глаз бессмысленно: пересегментация
роняет step-F1 сильнее, чем любая ошибка в метках. Скрипт считает признаки один раз на
ролик, а дальше гоняет по сетке параметров и печатает таблицу — видно, что именно даёт
каждая ручка.

    python scripts/tune.py --clips data/devset/clips --gt data/devset/gt
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from praxis import jobs, media
from praxis.metrics import evaluate_annotations
from praxis.pipeline.base import Perception
from praxis.pipeline.physical import PhysicalSegmenter
from praxis.schema import Annotation, Provenance, VideoMeta
from praxis.vocab import load_vocabulary

SUFFIXES = {".mp4", ".mov"}


def load_clips(clips: Path, gt: Path) -> list[tuple[Path, VideoMeta, Perception, Annotation]]:
    """Кадры считаются один раз: дальше сетка параметров гоняется уже по признакам."""
    prepared = []
    for clip in sorted(path for path in clips.iterdir() if path.suffix.lower() in SUFFIXES):
        reference = gt / f"{clip.stem}.json"
        if not reference.exists():
            continue
        probed = media.probe(clip)
        meta = VideoMeta(
            id=clip.stem,
            filename=clip.name,
            **{key: probed[key] for key in ("duration_sec", "fps", "width", "height")},
        )
        prepared.append(
            (
                clip,
                meta,
                jobs.perceive(clip),
                Annotation.model_validate_json(reference.read_text(encoding="utf-8")),
            )
        )
    return prepared


def run_grid(prepared, penalties, weights, max_segments, minimums) -> list[dict]:
    vocabulary = load_vocabulary()
    provenance = Provenance(app_version="tune", pipeline="motion-dp", vocabulary=vocabulary.name)
    rows = []

    for penalty, weight, minimum in itertools.product(penalties, weights, minimums):
        segmenter = PhysicalSegmenter(penalty, weight, max_segments, minimum)
        items = []
        step_error = 0
        for clip, meta, perception, truth in prepared:
            result = segmenter.run(clip, meta, vocabulary, perception)
            prediction = Annotation(video=meta, steps=result.steps, provenance=provenance)
            items.append((truth, prediction, clip.stem))
            step_error += abs(len(result.steps) - len(truth.steps))

        summary = evaluate_annotations(items)["summary"]
        rows.append(
            {
                "penalty": penalty,
                "weight": weight,
                "minimum": minimum,
                "f1_01": summary["f1@0.1_nolabel"],
                "f1_025": summary["f1@0.25_nolabel"],
                "f1_05": summary["f1@0.5_nolabel"],
                "boundary": summary["boundary_mae_sec"],
                "steps_mae": round(step_error / len(prepared), 2),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        default=[0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6],
    )
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1])
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--minimums", type=float, nargs="+", default=[0.8])
    parser.add_argument("--json", type=Path, help="куда сохранить таблицу")
    args = parser.parse_args()

    started = time.perf_counter()
    prepared = load_clips(args.clips, args.gt)
    if not prepared:
        raise SystemExit("не нашлось пар ролик/эталон")
    print(
        f"роликов: {len(prepared)}, признаки посчитаны за {time.perf_counter() - started:.1f} с\n"
    )

    rows = run_grid(prepared, args.penalties, args.weights, args.max_segments, args.minimums)
    rows.sort(key=lambda row: (-row["f1_05"], -row["f1_025"], row["boundary"]))

    print(
        f"{'штраф':>7} {'физика':>7} {'мин, с':>7} {'F1@0.1':>8} {'F1@0.25':>8} "
        f"{'F1@0.5':>8} {'границы':>8} {'ошибка числа шагов':>19}"
    )
    for row in rows:
        print(
            f"{row['penalty']:>7.3f} {row['weight']:>7.3f} {row['minimum']:>7.1f} {row['f1_01']:>8.3f} "
            f"{row['f1_025']:>8.3f} {row['f1_05']:>8.3f} {row['boundary']:>8.3f} "
            f"{row['steps_mae']:>19.2f}"
        )

    best = rows[0]
    print(
        f"\nлучшее: PRAXIS_SEGMENT_PENALTY={best['penalty']} "
        f"PRAXIS_BOUNDARY_WEIGHT={best['weight']} PRAXIS_MIN_SEGMENT_SEC={best['minimum']}"
    )
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
