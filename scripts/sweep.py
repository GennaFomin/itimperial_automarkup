#!/usr/bin/env python3
"""Прогон нескольких способов нарезки на одном наборе с одной метрикой.

Кейсодатель требует воспроизводимый evaluation harness — вот он. Признаки считаются
один раз и лежат в кэше, поэтому добавить ещё один метод стоит секунды, а не минуты.
Метки в сравнении не участвуют: вопрос про время, и смешивать его с семантикой нельзя.

    python scripts/sweep.py --clips data/pool_val/clips --gt data/pool_val/gt \\
        --methods baseline-single baseline-uniform motion-dp tsm-kernel

Переменные окружения можно задавать на метод, через двоеточие:

    --methods "tsm-kernel:PRAXIS_TSM_PENALTY=2.0" "tsm-kernel:PRAXIS_TSM_PENALTY=4.0"
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

from praxis import config, jobs, media
from praxis.metrics import evaluate_annotations
from praxis.schema import Annotation, VideoMeta


def apply_overrides(spec: str) -> str:
    """Имя метода и переменные окружения из строки вида «name:VAR=1:VAR2=2»."""
    name, *pairs = spec.split(":")
    for pair in pairs:
        key, _, value = pair.partition("=")
        os.environ[key] = value
        # config читается при импорте, поэтому значение надо положить и туда.
        attribute = key.removeprefix("PRAXIS_")
        if hasattr(config, attribute):
            current = getattr(config, attribute)
            setattr(config, attribute, type(current)(value) if current is not None else value)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    references = sorted(args.gt.glob("*.json"))
    if args.limit:
        references = references[: args.limit]
    truths = {
        path.stem: Annotation.model_validate_json(path.read_text(encoding="utf-8"))
        for path in references
    }
    clips = {stem: args.clips / truths[stem].video.filename for stem in truths}
    missing = [stem for stem, path in clips.items() if not path.exists()]
    for stem in missing:
        truths.pop(stem)
    if missing:
        print(f"нет видео для {len(missing)} эталонов, пропущены")

    print(f"\nроликов: {len(truths)}, эталон: "
          f"{statistics.fmean(len(t.steps) for t in truths.values()):.1f} шага на ролик\n")
    print(f"{'метод':<34}{'F1@0.1':>8}{'F1@0.25':>9}{'F1@0.5':>8}{'90% для F1@0.5':>17}"
          f"{'границы':>9}{'шагов':>7}{'с/ролик':>9}")

    for spec in args.methods:
        name = apply_overrides(spec)
        config.PIPELINE = name
        items, latencies, counts = [], [], []
        for stem, truth in truths.items():
            source = clips[stem]
            probe = media.probe(source)
            meta = VideoMeta(
                id=stem, filename=source.name, duration_sec=probe["duration_sec"],
                fps=probe["fps"], width=probe["width"], height=probe["height"],
            )
            started = time.perf_counter()
            try:
                result = jobs.annotate_clip(source, meta, jobs.perceive(source))
            except Exception as error:  # noqa: BLE001 — один упавший ролик не рушит прогон
                print(f"  {stem}: {type(error).__name__}: {error}")
                items.append((truth, None, stem))
                continue
            latencies.append(time.perf_counter() - started)
            counts.append(len(result.annotation.steps))
            items.append((truth, result.annotation, stem))

        summary = evaluate_annotations(items)["summary"]
        low, high = summary["ci90"]["f1@0.5_nolabel"]
        print(
            f"{spec:<34}"
            f"{summary['f1@0.1_nolabel']:>8.3f}{summary['f1@0.25_nolabel']:>9.3f}"
            f"{summary['f1@0.5_nolabel']:>8.3f}{f'{low:.3f}–{high:.3f}':>17}"
            f"{summary['boundary_mae_sec']:>8.2f}с"
            f"{statistics.fmean(counts) if counts else 0:>7.1f}"
            f"{statistics.fmean(latencies) if latencies else 0:>9.1f}"
        )


if __name__ == "__main__":
    main()
