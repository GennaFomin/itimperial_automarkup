#!/usr/bin/env python3
"""Отчёт по метрикам: сравнивает каталог предсказаний с каталогом эталонов.

Считает то, что записано в целях кейса, и отдельно то, что нужно нам самим, чтобы
понимать, где именно ломается пайплайн:

* **step-F1@IoU** в двух вариантах — со сверкой меток (как оценивают нас) и без неё
  (чистое качество нарезки). Разница между ними сразу показывает, что чинить: границы
  или семантику.
* **ошибка границ** по сопоставленным сегментам: среднее и 95-й процентиль.
* **точность** отдельно по действию и по объекту — они ломаются по разным причинам.
* **edit score** и **пофреймовая точность** — стандартные метрики temporal action
  segmentation, по ним нас можно сравнить с литературой.

    python scripts/eval.py --gt devset/gt --pred out/pred
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from praxis.metrics import IOU_THRESHOLDS, evaluate

def targets(summary: dict) -> list[str]:
    """Цели кейса — печатаем рядом с фактом, чтобы не сверять руками."""
    checks = [
        ("step-F1 >= 0.75", summary["case_f1"] >= 0.75, f"{summary['case_f1']:.3f}"),
        (
            "границы <= 2 c",
            summary["boundary_mae_sec"] <= 2.0,
            f"{summary['boundary_mae_sec']:.3f}",
        ),
        (
            "действие+объект >= 0.80",
            summary["action_object_accuracy"] >= 0.80,
            f"{summary['action_object_accuracy']:.3f}",
        ),
        (
            "обработка <= 120 c",
            summary["processing_sec_max"] <= 120,
            f"{summary['processing_sec_max']:.1f}",
        ),
    ]
    return [f"  {'✓' if ok else '✗'} {name}: {value}" for name, ok, value in checks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="каталог с эталонной разметкой")
    parser.add_argument("--pred", type=Path, required=True, help="каталог с предсказаниями")
    parser.add_argument("--json", type=Path, help="куда сохранить полный отчёт")
    args = parser.parse_args()

    # Ролик без предсказания идёт в пары как None: упавший прогон — полный промах, а не
    # повод исключить ролик из знаменателя.
    pairs = []
    for gt_path in sorted(args.gt.glob("*.json")):
        pred_path = args.pred / gt_path.name
        pairs.append((gt_path, pred_path if pred_path.exists() else None))
    if not pairs:
        raise SystemExit("не нашлось ни одного эталона")

    report = evaluate(pairs)
    summary = report["summary"]

    print(f"\nроликов: {summary['clips']}, сопоставлено сегментов: {summary['matched_segments']}")
    if summary["clips_without_prediction"]:
        print(f"без предсказания: {summary['clips_without_prediction']} — считаются промахом")

    # Метрика кейса первой и отдельно: остальные числа — диагностика, а не оценка.
    low, high = summary["ci90"]["case_f1"]
    print("\nправило кейса — IoU >= 0.5 и верное действие, один-к-одному:")
    print(f"{'  step-F1':<28}{summary['case_f1']:>18.3f}")
    print(f"{'  precision':<28}{summary['case_precision']:>18.3f}")
    print(f"{'  recall':<28}{summary['case_recall']:>18.3f}")
    print(f"{'  среднее по роликам':<28}{summary['case_f1_macro']:>18.3f}   ({low:.3f}–{high:.3f})")
    print()
    print("диагностика — что именно ломается:")
    print(f"{'метрика':<28} {'действие+предмет':>18} {'только нарезка':>16}   {'90% для нарезки'}")
    for threshold in IOU_THRESHOLDS:
        low, high = summary["ci90"][f"f1@{threshold}_nolabel"]
        print(
            f"{'step-F1 @ IoU ' + str(threshold):<28}"
            f"{summary[f'f1@{threshold}']:>18.3f}{summary[f'f1@{threshold}_nolabel']:>16.3f}"
            f"   ({low:.3f}–{high:.3f})"
        )
    print()
    print(f"{'ошибка границ, среднее':<28}{summary['boundary_mae_sec']:>18.3f} с")
    print(f"{'ошибка границ, p95':<28}{summary['boundary_p95_sec']:>18.3f} с")
    print(f"{'точность действия':<28}{summary['action_accuracy']:>18.3f}")
    print(f"{'точность объекта':<28}{summary['object_accuracy']:>18.3f}")
    print(f"{'точность пары':<28}{summary['action_object_accuracy']:>18.3f}")
    print(f"{'edit score':<28}{summary['edit']:>18.3f}")
    print(f"{'пофреймовая точность':<28}{summary['frame_accuracy']:>18.3f}")
    print(f"{'обработка, среднее':<28}{summary['processing_sec_mean']:>18.3f} с")
    print("\nцели кейса:")
    print("\n".join(targets(summary)))

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nполный отчёт: {args.json}")


if __name__ == "__main__":
    main()
