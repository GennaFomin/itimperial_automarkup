#!/usr/bin/env python3
"""Сравнение способов расстановки границ на одном наборе и под одной метрикой.

Спор «границы должна ставить VLM или отдельный алгоритм» решается измерением. Скрипт
берёт готовые каталоги предсказаний, посчитанные разными сегментаторами, и печатает их
рядом: качество нарезки, ошибку границ, число шагов и время на ролик.

Метки в этом сравнении намеренно не участвуют: вопрос только про время. Метрику кейса
целиком (с классом действия) считает scripts/eval.py.

    python scripts/compare_bounds.py --gt data/devset_big/gt \\
        --pred motion-dp=work/dp vlm-direct=work/direct vlm-bisect=work/bisect
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from praxis.metrics import evaluate
from praxis.schema import Annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", nargs="+", required=True, help="имя=каталог")
    args = parser.parse_args()

    references = sorted(args.gt.glob("*.json"))
    if not references:
        raise SystemExit("эталонов не нашлось")

    print(f"\nроликов в наборе: {len(references)}\n")
    header = f"{'способ':<14}{'F1@0.1':>9}{'F1@0.25':>9}{'F1@0.5':>9}{'границы':>10}"
    print(header + f"{'шагов':>8}{'эталон':>8}{'с/ролик':>9}")

    for entry in args.pred:
        name, _, directory = entry.partition("=")
        pairs = []
        for reference in references:
            candidate = Path(directory) / reference.name
            pairs.append((reference, candidate if candidate.exists() else None))

        summary = evaluate(pairs)["summary"]
        latencies = []
        predicted_steps = []
        for reference in references:
            candidate = Path(directory) / reference.name
            if not candidate.exists():
                continue
            annotation = Annotation.model_validate_json(candidate.read_text(encoding="utf-8"))
            predicted_steps.append(len(annotation.steps))
            if annotation.provenance.processing_sec:
                latencies.append(annotation.provenance.processing_sec)

        truth_steps = [
            len(Annotation.model_validate_json(item.read_text(encoding="utf-8")).steps)
            for item in references
        ]
        print(
            f"{name:<14}"
            f"{summary['f1@0.1_nolabel']:>9.3f}"
            f"{summary['f1@0.25_nolabel']:>9.3f}"
            f"{summary['f1@0.5_nolabel']:>9.3f}"
            f"{summary['boundary_mae_sec']:>9.2f}с"
            f"{statistics.fmean(predicted_steps) if predicted_steps else 0:>8.1f}"
            f"{statistics.fmean(truth_steps):>8.1f}"
            f"{statistics.fmean(latencies) if latencies else 0:>9.1f}"
        )
        low, high = summary["ci90"]["f1@0.5_nolabel"]
        missing = summary["clips_without_prediction"]
        note = f", без предсказания {missing}" if missing else ""
        print(f"{'':<14}90% для F1@0.5: {low:.3f}–{high:.3f}{note}")


if __name__ == "__main__":
    main()
