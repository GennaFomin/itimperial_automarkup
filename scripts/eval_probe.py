#!/usr/bin/env python3
"""Кто лучше называет атомарное действие: признаки движения или языковая модель.

Границы в этом замере берутся эталонные — вопрос только про класс. Сравниваются:

* линейная модель и ближайший центроид поверх признаков видеоэнкодера, обученные на
  отдельном наборе;
* языковая модель в zero-shot на тех же сегментах (если указан --vlm);
* константа «самый частый глагол» — нижняя планка, без которой числа не читаются.

    python scripts/eval_probe.py --train data/train_atomic --val data/pool_atomic
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from praxis import config, jobs, media
from praxis.pipeline.probe import LinearProbe, NearestCentroid, pool_segment
from praxis.schema import Annotation


def collect(directory: Path, field: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Векторы сегментов, их метки и имя ролика — по эталонным границам."""
    vectors, labels, clips = [], [], []
    references = sorted((directory / "gt").glob("*.json"))
    for position, path in enumerate(references, 1):
        truth = Annotation.model_validate_json(path.read_text(encoding="utf-8"))
        source = directory / "clips" / truth.video.filename
        if not source.exists():
            continue
        perception = jobs.perceive(source)
        for step in truth.steps:
            vectors.append(
                pool_segment(
                    perception.appearance, perception.fps, perception.offset,
                    step.start_sec, step.end_sec,
                )
            )
            labels.append(getattr(step, field) or "")
            clips.append(truth.video.id)
        if position % 20 == 0:
            print(f"  {directory.name}: {position}/{len(references)}", flush=True)
    return np.array(vectors), labels, clips


def report(name: str, predicted: list[str], truth: list[str], elapsed: float | None = None) -> None:
    hits = sum(p == t for p, t in zip(predicted, truth))
    tail = f"{elapsed:8.1f} с" if elapsed is not None else ""
    print(f"{name:<34}{hits / max(len(truth), 1):>8.3f}{tail:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--field", default="action", choices=["action", "object"])
    parser.add_argument("--vlm", action="store_true", help="добавить замер языковой модели")
    args = parser.parse_args()

    print("считаю признаки обучающего набора…", flush=True)
    train_x, train_y, _ = collect(args.train, args.field)
    print("считаю признаки валидационного набора…", flush=True)
    val_x, val_y, val_clips = collect(args.val, args.field)
    if not len(train_x) or not len(val_x):
        raise SystemExit("не набралось сегментов")

    shared = sorted(set(train_y) & set(val_y))
    print(
        f"\nобучение: {len(train_x)} сегментов, {len(set(train_y))} классов"
        f"\nвалидация: {len(val_x)} сегментов, {len(set(val_y))} классов"
        f"\nобщих классов: {len(shared)}"
        f"\nроликов в валидации: {len(set(val_clips))}\n"
    )
    print(f"{'способ':<34}{'точность':>8}{'время':>12}")

    common = collections.Counter(train_y).most_common(1)[0][0]
    report(f"константа «{common}»", [common] * len(val_y), val_y)

    for model in (NearestCentroid(), LinearProbe()):
        started = time.perf_counter()
        model.fit(train_x, train_y)
        predicted = model.predict(val_x)
        report(type(model).__name__, predicted, val_y, time.perf_counter() - started)

    if args.vlm:
        from praxis.pipeline.naming import get_namer
        from praxis.vocab import load_vocabulary

        vocabulary = load_vocabulary(config.VOCAB_PATH)
        namer = get_namer()
        started = time.perf_counter()
        predicted: list[str] = []
        for path in sorted((args.val / "gt").glob("*.json")):
            truth = Annotation.model_validate_json(path.read_text(encoding="utf-8"))
            source = args.val / "clips" / truth.video.filename
            if not source.exists():
                continue
            blank = [
                step.model_copy(update={"action": vocabulary.actions[0], "object": None})
                for step in truth.steps
            ]
            named = namer.name_steps(source, truth.video, blank, vocabulary, None).steps
            predicted += [getattr(step, args.field) or "" for step in named]
        report("языковая модель", predicted, val_y, time.perf_counter() - started)


if __name__ == "__main__":
    main()
