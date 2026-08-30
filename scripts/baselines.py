#!/usr/bin/env python3
"""Тривиальные базовые уровни — линейка, к которой прикладывается любой namer.

Без этих чисел легко радоваться цифре 0.38 и не заметить, что константа «всегда attach»
даёт 0.66. Любой вариант, не бьющий majority-verb, в пайплайн не идёт.

    python scripts/baselines.py --gt data/devset/gt
"""

from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path

from praxis.schema import Annotation


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал для доли. При n=50 полуширина около 13 п.п."""
    if not total:
        return 0.0, 0.0
    phat = hits / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    args = parser.parse_args()

    verbs: collections.Counter = collections.Counter()
    nouns: collections.Counter = collections.Counter()
    pairs: collections.Counter = collections.Counter()
    per_toy: dict[str, set] = collections.defaultdict(set)
    steps = 0

    for reference in sorted(args.gt.glob("*.json")):
        annotation = Annotation.model_validate_json(reference.read_text(encoding="utf-8"))
        toy = annotation.video.id.split("_")[0]
        for step in annotation.steps:
            steps += 1
            verbs[step.action] += 1
            nouns[step.object] += 1
            pairs[(step.action, step.object)] += 1
            per_toy[toy].add((step.action, step.object))

    if not steps:
        raise SystemExit("в каталоге нет разметки")

    top_verb, verb_hits = verbs.most_common(1)[0]
    top_noun, noun_hits = nouns.most_common(1)[0]
    top_pair, pair_hits = pairs.most_common(1)[0]

    print(
        f"сегментов: {steps}, уникальных глаголов: {len(verbs)}, объектов: {len(nouns)}, "
        f"пар: {len(pairs)}"
    )
    print(f"\nраспределение глаголов: {dict(verbs.most_common(6))}")
    print(f"распределение объектов:  {dict(nouns.most_common(6))}")

    print("\nтривиальные базовые уровни (то, что нужно побить):")
    rows = [
        (f"константа «всегда {top_verb}»", "глагол", verb_hits),
        (f"константа «всегда {top_noun}»", "объект", noun_hits),
        (f"константа «{top_pair[0]} {top_pair[1]}»", "пара", pair_hits),
    ]
    for name, axis, hits in rows:
        low, high = wilson(hits, steps)
        print(f"  {name:<34} {axis:<7} {hits / steps:.3f}  [{low:.3f}, {high:.3f}]")

    print(f"\n  случайный выбор из {len(pairs)} встреченных пар: {1 / len(pairs):.3f}")
    oracle_noun = sum(count for (verb, _), count in pairs.items() if verb == top_verb) / steps
    print(f"  идеальный объект + константный глагол:      {oracle_noun:.3f}")

    sizes = [len(values) for values in per_toy.values()]
    if sizes:
        print(
            f"\nпар на одну игрушку: в среднем {sum(sizes) / len(sizes):.1f} "
            f"(от {min(sizes)} до {max(sizes)}) — во столько раз можно сузить словарь, "
            "если знать изделие"
        )


if __name__ == "__main__":
    main()
