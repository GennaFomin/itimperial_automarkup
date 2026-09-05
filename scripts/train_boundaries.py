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

import numpy as np

from praxis import config, jobs
from praxis.pipeline.learned import difference_channels, encode, post, stack_motion
from praxis.schema import Annotation


def gap_frames(steps: list[tuple[float, float]], duration: float, offset: float, fps: float, count: int) -> list[float]:
    """Индексы кадров сетки признаков, лежащих вне всех шагов. Цель детектора пауз."""
    times = offset + np.arange(count) / fps
    inside = np.zeros(count, dtype=bool)
    for start, end in steps:
        inside |= (times >= start) & (times < end)
    inside |= times >= duration
    return [float(index) for index in np.flatnonzero(~inside)]


def check_corpus(total: int, kept: int) -> None:
    """Сбой сервиса признаков молча уменьшил бы корпус, а модель выглядела бы как обычная."""
    skipped = total - kept
    if skipped > max(3, total // 50):
        raise SystemExit(f"пропущено {skipped} из {total} роликов — корпус неполный, обучение отменено")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--name", default="", help="имя варианта: чекпоинт boundary-<name>.pt")
    parser.add_argument("--no-activate", action="store_true", help="не подменять живую модель")
    parser.add_argument("--augment", action="store_true", help="временная аугментация C2F-TCN")
    parser.add_argument("--stages", type=int, default=4, help="число стадий MS-TCN")
    parser.add_argument("--target", choices=["boundaries", "gaps"], default="boundaries",
                        help="boundaries — смены действий; gaps — кадры внутри пауз между шагами (детектор пауз)")
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
        # Не perceive: тот сначала декодирует весь ролик в серые кадры ради полосы
        # движения для таймлайна, а обучению нужны только признаки энкодера. На шести
        # сотнях роликов это разница между двадцатью минутами и двумя с половиной часами.
        video = jobs._video_features(source)
        if video is None:
            print(f"  пропуск без признаков: {source.name}")
            continue
        matrix = video["matrix"]
        # Полоса движения нужна только как 769-й канал: без него не декодируем ролик.
        perception = jobs.Perception(
            fps=float(video["fps"]),
            motion=(jobs.motion_band(source, float(video["fps"]), len(matrix))
                    if config.TAS_MOTION else np.zeros(len(matrix), dtype=np.float32)),
            appearance=matrix,
            offset=float(video.get("offset_sec", 0.0)),
        )
        if args.target == "gaps":
            # Кадры вне размеченных шагов: пауза между действиями, включая края ролика.
            boundaries = gap_frames(
                [(step.start_sec, step.end_sec) for step in truth.steps],
                truth.video.duration_sec, perception.offset, perception.fps, len(matrix),
            )
        else:
            # Границы в кадрах сетки признаков: и начала, и концы шагов — всё это смены.
            moments = sorted(
                {step.start_sec for step in truth.steps} | {step.end_sec for step in truth.steps}
            )
            boundaries = [
                (moment - perception.offset) * perception.fps
                for moment in moments
                if 0 < moment < truth.video.duration_sec
            ]
        matrix = (
            stack_motion(perception.appearance, perception.motion)
            if config.TAS_MOTION else perception.appearance
        )
        if config.TAS_DIFF:
            matrix = difference_channels(matrix)
        samples.append({**encode(matrix), "boundaries": boundaries})
        if position % 10 == 0:
            print(f"  признаки {position}/{len(references)}", flush=True)

    if not samples:
        raise SystemExit("не набралось примеров")
    check_corpus(len(references), len(samples))

    print(f"\nобучаю на {len(samples)} роликах, "
          f"{sum(len(s['boundaries']) for s in samples)} границ…", flush=True)
    answer = post(
        "/train",
        {"samples": samples, "epochs": args.epochs,
         "tolerance": 0 if args.target == "gaps" else args.tolerance,
         "name": args.name, "activate": not args.no_activate, "augment": args.augment,
         "stages": args.stages},
        timeout=3600,
    )
    print(f"готово за {answer['elapsed_sec']} с: "
          f"потеря {answer['loss_first']} -> {answer['loss_last']}, "
          f"признаков {answer['dim']}")


if __name__ == "__main__":
    main()
