#!/usr/bin/env python3
"""Подбор способа назначать метки по закрытому словарю.

Устройство сделано так, чтобы модель прогонялась ОДИН раз: сервис отдаёт сырые матрицы
сходства кадров с фразами глаголов и объектов, они кладутся в кэш, а дальше сотни
вариантов калибровки и весов перебираются мгновенно на numpy.

Почему калибровка вообще нужна. Контрастная модель систематически смещена: одни фразы
получают высокий скор на любой картинке (у нас так вышло с «attempt to detach» — он
побеждал в 19 сегментах из 50). Вычитание среднего по столбцу, посчитанного по всему
набору сегментов, убирает ровно это смещение. Приём трансдуктивный: он требует видеть
весь батч, поэтому живёт здесь, а не в сервисе.

    PRAXIS_CLIP_BASE_URL=http://127.0.0.1:8101 python scripts/tune_naming.py \
        --clips data/devset/clips --gt data/devset/gt
"""

from __future__ import annotations

import argparse
import base64
import collections
import itertools
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from praxis import jobs, media
from praxis.schema import Annotation
from praxis.vocab import load_vocabulary


def collect(clips: Path, gt: Path, base_url: str, frames: int, crop: bool) -> dict:
    """Один прогон модели по всем сегментам. Результат — сырые матрицы сходства."""
    vocabulary = load_vocabulary()
    verbs, nouns = vocabulary.actions, vocabulary.objects
    rows: list[dict] = []

    for reference in sorted(gt.glob("*.json")):
        truth = Annotation.model_validate_json(reference.read_text(encoding="utf-8"))
        clip = clips / truth.video.filename
        if not clip.exists():
            continue

        region = jobs.perceive(clip).crop if crop else None
        segments = []
        with tempfile.TemporaryDirectory() as directory:
            for step in truth.steps:
                span = step.end_sec - step.start_sec
                encoded = []
                for index in range(frames):
                    path = Path(directory) / f"{step.id}_{index}.jpg"
                    at = step.start_sec + span * (index + 0.5) / frames
                    media.extract_frame(clip, at, path, width=640, crop=region)
                    encoded.append(base64.b64encode(path.read_bytes()).decode())
                segments.append({"id": step.id, "frames": encoded})

            request = urllib.request.Request(
                base_url.rstrip("/") + "/scores",
                data=json.dumps({"segments": segments, "verbs": verbs, "nouns": nouns}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                answer = json.loads(response.read())

        by_id = {item["id"]: item for item in answer["results"]}
        for step in truth.steps:
            item = by_id[step.id]
            rows.append(
                {
                    "clip": truth.video.id,
                    "toy": truth.video.id.split("_")[0],
                    "verb": step.action,
                    "noun": step.object,
                    "verb_mean": item["verb_mean"],
                    "verb_max": item["verb_max"],
                    "noun_mean": item["noun_mean"],
                    "noun_max": item["noun_max"],
                    "verb_diff": item["verb_first_last"],
                }
            )
        print(f"  {truth.video.id}: {len(truth.steps)} сегментов", flush=True)

    return {"verbs": verbs, "nouns": nouns, "rows": rows}


def debias(scores: np.ndarray, mode: str) -> np.ndarray:
    """Снятие систематического смещения фраз по всему набору сегментов."""
    if mode == "none":
        return scores
    if mode == "column":
        return scores - scores.mean(axis=0, keepdims=True)
    if mode == "sinkhorn":
        matrix = np.exp((scores - scores.max()) * 50.0)
        for _ in range(10):
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
            matrix = matrix / matrix.sum(axis=0, keepdims=True)
        return np.log(matrix + 1e-12)
    raise ValueError(mode)


def log_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores * temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def evaluate(data: dict, options: dict, prior: np.ndarray | None) -> dict:
    verbs, nouns, rows = data["verbs"], data["nouns"], data["rows"]
    verb_index = {verb: i for i, verb in enumerate(verbs)}
    noun_index = {noun: i for i, noun in enumerate(nouns)}

    pooling = options["pooling"]
    verb_scores = np.array([row[f"verb_{pooling}"] for row in rows], dtype=np.float64)
    noun_scores = np.array([row[f"noun_{pooling}"] for row in rows], dtype=np.float64)
    if options.get("verb_from_diff"):
        verb_scores = np.array([row["verb_diff"] for row in rows], dtype=np.float64)

    verb_scores = log_softmax(debias(verb_scores, options["debias"]), options["temperature"])
    noun_scores = log_softmax(debias(noun_scores, options["debias"]), options["temperature"])

    if prior is not None and options.get("gamma", 1.0) < 1.0:
        gamma = options["gamma"]
        verb_scores = gamma * verb_scores + (1 - gamma) * np.log(prior + 1e-9)[None, :]

    # Маска допустимых пар: всё, чего нет в словаре, недостижимо по построению.
    vocabulary = load_vocabulary()
    mask = np.full((len(verbs), len(nouns)), -np.inf)
    for verb, allowed in (vocabulary.pairs or {}).items():
        for noun in allowed:
            if verb in verb_index and noun in noun_index:
                mask[verb_index[verb], noun_index[noun]] = 0.0

    alpha, beta = options["alpha"], options["beta"]
    verb_hits = noun_hits = pair_hits = 0
    margins = []
    correct = []

    for index, row in enumerate(rows):
        combined = alpha * verb_scores[index][:, None] + beta * noun_scores[index][None, :] + mask
        flat = combined.ravel()
        order = np.argsort(flat)[::-1]
        best_verb, best_noun = divmod(int(order[0]), len(nouns))
        margins.append(float(flat[order[0]] - flat[order[1]]) if len(order) > 1 else 0.0)

        verb_hits += verbs[int(np.argmax(verb_scores[index]))] == row["verb"]
        noun_hits += nouns[int(np.argmax(noun_scores[index]))] == row["noun"]
        hit = verbs[best_verb] == row["verb"] and nouns[best_noun] == row["noun"]
        pair_hits += hit
        correct.append(hit)

    total = len(rows)
    return {
        **options,
        "verb": round(verb_hits / total, 3),
        "noun": round(noun_hits / total, 3),
        "pair": round(pair_hits / total, 3),
        "margin_auc": round(margin_quality(margins, correct), 3),
    }


def margin_quality(margins: list[float], correct: list[bool]) -> float:
    """Насколько margin отделяет верные ответы от неверных — это и есть польза для триажа."""
    positives = [m for m, ok in zip(margins, correct) if ok]
    negatives = [m for m, ok in zip(margins, correct) if not ok]
    if not positives or not negatives:
        return 0.5
    wins = sum(1 for p in positives for n in negatives if p > n)
    ties = sum(1 for p in positives for n in negatives if p == n)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/devset/scores.json"))
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="пересчитать скоры")
    args = parser.parse_args()

    base_url = os.environ.get("PRAXIS_CLIP_BASE_URL", "http://127.0.0.1:8101")
    if args.refresh or not args.cache.exists():
        started = time.perf_counter()
        data = collect(args.clips, args.gt, base_url, args.frames, not args.no_crop)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(data), encoding="utf-8")
        print(f"скоры посчитаны за {time.perf_counter() - started:.1f} с\n")
    data = json.loads(args.cache.read_text(encoding="utf-8"))

    counts = collections.Counter(row["verb"] for row in data["rows"])
    prior = np.array([counts.get(verb, 0) + 0.5 for verb in data["verbs"]])
    prior = prior / prior.sum()

    grid = []
    for pooling, mode, alpha, beta in itertools.product(
        ["mean", "max"], ["none", "column", "sinkhorn"], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]
    ):
        grid.append(
            {
                "pooling": pooling,
                "debias": mode,
                "alpha": alpha,
                "beta": beta,
                "temperature": 100.0,
                "gamma": 1.0,
            }
        )

    rows = [evaluate(data, options, prior) for options in grid]
    rows.sort(key=lambda row: (-row["pair"], -row["verb"], -row["noun"]))

    print(
        f"{'пул':>5} {'калибровка':>11} {'alpha':>6} {'beta':>6} "
        f"{'глагол':>8} {'объект':>8} {'пара':>7} {'margin':>8}"
    )
    for row in rows[:14]:
        print(
            f"{row['pooling']:>5} {row['debias']:>11} {row['alpha']:>6.1f} {row['beta']:>6.1f} "
            f"{row['verb']:>8.3f} {row['noun']:>8.3f} {row['pair']:>7.3f} {row['margin_auc']:>8.3f}"
        )

    best = rows[0]
    print(f"\nлучшее: {best}")


if __name__ == "__main__":
    main()
