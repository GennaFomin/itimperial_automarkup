#!/usr/bin/env python3
"""Обучение детектора границ на размеченном наборе.

Признаки считаются локально (и лежат в кэше), эталонные границы берутся из разметки,
всё уезжает на GPU-машину одним запросом. Голова учится предсказывать не класс, а
момент смены действия — поэтому обученную на одном домене можно применять к другому.

    PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104 \\
        python scripts/train_boundaries.py --train data/train_atomic --epochs 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

from praxis import config, jobs
from praxis.pipeline.learned import encode, post
from praxis.schema import Annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    references = sorted((args.train / "gt").glob("*.json"))
    if args.limit:
        references = references[: args.limit]

    samples = []
    for position, path in enumerate(references, 1):
        truth = Annotation.model_validate_json(path.read_text(encoding="utf-8"))
        source = args.train / "clips" / truth.video.filename
        if not source.exists():
            continue
        perception = jobs.perceive(source)
        # Границы в кадрах сетки признаков: и начала, и концы шагов — всё это смены.
        moments = sorted(
            {step.start_sec for step in truth.steps} | {step.end_sec for step in truth.steps}
        )
        boundaries = [
            (moment - perception.offset) * perception.fps
            for moment in moments
            if 0 < moment < truth.video.duration_sec
        ]
        samples.append({**encode(perception.appearance), "boundaries": boundaries})
        if position % 10 == 0:
            print(f"  признаки {position}/{len(references)}", flush=True)

    if not samples:
        raise SystemExit("не набралось примеров")

    print(f"\nобучаю на {len(samples)} роликах, "
          f"{sum(len(s['boundaries']) for s in samples)} границ…", flush=True)
    answer = post(
        "/train",
        {"samples": samples, "epochs": args.epochs, "tolerance": args.tolerance},
        timeout=3600,
    )
    print(f"готово за {answer['elapsed_sec']} с: "
          f"потеря {answer['loss_first']} -> {answer['loss_last']}, "
          f"признаков {answer['dim']}")


if __name__ == "__main__":
    main()
