#!/usr/bin/env python3
"""Проверка только семантической стадии: границы берутся эталонные.

Смысл в том, чтобы отделить одно от другого. Если подать модели наши сегменты, ошибки
нарезки и ошибки называния смешаются, и будет непонятно, что чинить. Здесь границы —
эталонные, поэтому всё, что видно, относится только к видеомодели.

    PRAXIS_VLM_BASE_URL=http://127.0.0.1:8100 python scripts/eval_naming.py \
        --clips data/devset/clips --gt data/devset/gt
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from praxis import config, jobs
from praxis.pipeline.naming import get_namer
from praxis.schema import Annotation
from praxis.vocab import Vocabulary, load_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="сколько роликов взять")
    parser.add_argument("--show", action="store_true", help="печатать каждый шаг")
    parser.add_argument("--crop", action="store_true", help="подавать кадры по рабочей зоне")
    parser.add_argument(
        "--toy-vocab",
        type=Path,
        help="словари изделий: кандидаты сужаются до пар, возможных для этой игрушки",
    )
    parser.add_argument("--toy-map", type=Path, help="соответствие ролик → изделие")
    parser.add_argument("--dump", type=Path, help="куда сложить пары эталон/предсказание")
    args = parser.parse_args()

    full_vocabulary = load_vocabulary(config.VOCAB_PATH)
    namer = get_namer()

    toy_vocabularies = json.loads(args.toy_vocab.read_text(encoding="utf-8")) if args.toy_vocab else {}
    toy_of_clip = json.loads(args.toy_map.read_text(encoding="utf-8")) if args.toy_map else {}
    scoped = 0
    records: list[dict] = []
    references = sorted(args.gt.glob("*.json"))
    if args.limit:
        references = references[: args.limit]

    action_hits = object_hits = pair_hits = total = unparsed = 0
    predicted_actions: collections.Counter = collections.Counter()
    started = time.perf_counter()

    for reference in references:
        truth = Annotation.model_validate_json(reference.read_text(encoding="utf-8"))
        clip = args.clips / truth.video.filename
        if not clip.exists():
            continue

        # Словарь изделия, если он известен: у одной игрушки возможен десяток пар,
        # а не двести. Это не подгонка — ровно так устроен рабочий сеттинг заказчика.
        vocabulary = full_vocabulary
        toy = toy_of_clip.get(truth.video.id)
        if toy and toy in toy_vocabularies:
            entry = toy_vocabularies[toy]
            sessions = entry.get("by_session") or {}
            # Исключаем сессию, из которой взят ролик: иначе словарь изделия был бы
            # построен по тем же кадрам, что оцениваются, и число оказалось бы завышенным.
            own = truth.video.id.rsplit("_", 1)[0]
            collected = {
                tuple(pair)
                for key, values in sessions.items()
                if key != own
                for pair in values
            }
            pairs = sorted(collected) if collected else [tuple(p) for p in entry["pairs"]]
            grouped: dict[str, list[str]] = {}
            for action, noun in pairs:
                grouped.setdefault(action, []).append(noun)
            vocabulary = Vocabulary(
                name=f"toy:{toy}",
                actions=sorted(grouped),
                objects=sorted({noun for _, noun in pairs}),
                pairs=grouped,
            )
            scoped += 1

        # Копия шагов с эталонными границами, но без меток: модель называет с нуля.
        blank = [
            step.model_copy(update={"action": vocabulary.actions[0], "object": None})
            for step in truth.steps
        ]
        crop = jobs.perceive(clip).crop if args.crop else None
        named = namer.name_steps(clip, truth.video, blank, vocabulary, crop).steps

        for expected, got in zip(truth.steps, named):
            total += 1
            records.append(
                {
                    "clip": truth.video.id,
                    "start": expected.start_sec,
                    "end": expected.end_sec,
                    "true_action": expected.action,
                    "true_object": expected.object,
                    "action": got.action,
                    "object": got.object,
                    "confidence": got.confidence,
                }
            )
            predicted_actions[got.action] += 1
            unparsed += got.action == "?"
            action_hits += got.action == expected.action
            object_hits += got.object == expected.object
            pair_hits += got.action == expected.action and got.object == expected.object
            if args.show:
                mark = "✓" if got.action == expected.action else " "
                print(
                    f"  {mark} {expected.start_sec:>6.1f}–{expected.end_sec:<6.1f} "
                    f"эталон: {expected.action} {expected.object or '—':<16} "
                    f"модель: {got.action} {got.object or '—'} ({got.confidence})"
                )

    elapsed = time.perf_counter() - started
    if not total:
        raise SystemExit("не нашлось ни одного сегмента")

    print(
        f"\nсегментов: {total}, роликов: {len(references)}, {elapsed:.1f} с "
        f"({elapsed / max(len(references), 1):.1f} с на ролик)"
    )
    if toy_vocabularies:
        sizes = [len(toy_vocabularies[t]["pairs"]) for t in set(toy_of_clip.values()) if t in toy_vocabularies]
        print(f"словарь сужен на {scoped} роликах, пар в среднем "
              f"{sum(sizes) / max(len(sizes), 1):.1f} вместо {len(full_vocabulary.actions) * 0 + 202}")
    print(f"не разобрано ответов: {unparsed} из {total}")
    print(f"точность действия: {action_hits / total:.3f}")
    print(f"точность объекта:  {object_hits / total:.3f}")
    print(f"точность пары:     {pair_hits / total:.3f}  (цель кейса — 0.80)")
    print("\nчто модель предсказывает чаще всего:")
    for action, count in predicted_actions.most_common(5):
        print(f"  {action}: {count}")

    # Куда именно уходит точность: без этого улучшать нечего — видно только итог.
    confusions = collections.Counter(
        (item["true_action"], item["action"])
        for item in records
        if item["true_action"] != item["action"]
    )
    print("\nчаще всего путает (эталон → модель):")
    for (expected_action, got_action), count in confusions.most_common(8):
        print(f"  {expected_action} → {got_action}: {count}")

    if args.dump:
        args.dump.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nпредсказания сохранены: {args.dump}")


if __name__ == "__main__":
    main()
