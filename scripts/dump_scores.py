#!/usr/bin/env python3
"""Снять сырые вероятности границ с сервиса один раз, чтобы декодеры перебирать офлайн.

Пересегментация — свойство декодирования, а не модели: та же кривая вероятностей даёт
разное число шагов при разном пороге, минимальном интервале, выраженности пика или
штрафе за разрез. Гонять GPU ради каждого такого варианта незачем.

    python scripts/dump_scores.py --clips data/pool_atomic/clips --gt data/pool_atomic/gt --out work/scores_base.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from praxis import config, jobs
from praxis.pipeline.learned import difference_channels, encode, post, stack_motion
from praxis.schema import Annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    names, scores, fps, offsets, truths, durations = [], [], [], [], [], []
    for gt in sorted(args.gt.glob("*.json")):
        truth = Annotation.model_validate_json(gt.read_text(encoding="utf-8"))
        clip = args.clips / truth.video.filename
        if not clip.exists():
            continue
        # Признаки из кэша, без декодирования серых кадров: полоса движения нужна
        # только моделям с каналом движения.
        video = jobs._video_features(clip)
        if video is None:
            print(f"  нет признаков: {clip.name}")
            continue
        appearance = video["matrix"]
        perception = jobs.Perception(fps=float(video["fps"]), motion=np.ones(len(appearance)),
                                     appearance=appearance, offset=float(video.get("offset_sec", 0.0)))
        matrix = (stack_motion(appearance, jobs.motion_band(clip, perception.fps, len(appearance)))
                  if config.TAS_MOTION else appearance)
        if config.TAS_DIFF:
            matrix = difference_channels(matrix)
        answer = post("/predict", {"samples": [encode(matrix)]})
        names.append(gt.stem)
        scores.append(np.array(answer["scores"][0], dtype=np.float32))
        fps.append(perception.fps)
        offsets.append(perception.offset)
        truths.append([(s.start_sec, s.end_sec) for s in truth.steps])
        durations.append(truth.video.duration_sec)
    np.savez(args.out, names=np.array(names), scores=np.array(scores, dtype=object),
             fps=np.array(fps), offsets=np.array(offsets), truths=np.array(truths, dtype=object),
             durations=np.array(durations))
    print(f"снято {len(names)} роликов → {args.out}")


if __name__ == "__main__":
    main()
