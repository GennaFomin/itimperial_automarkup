#!/usr/bin/env python3
"""Сравнение вариантов детектора на фиксированных наборах, с интервалами.

Каждый вариант — именованный чекпоинт на сервисе. Наборы не используются для подбора:
порог один и тот же для всех вариантов и задаётся здесь, а не подбирается по ним.

    python scripts/eval_variants.py --variants base,motion,sigma3 \\
        --sets human=/mnt/data/praxis-pool/validation robot=data/robo_holdout
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path

import json
import urllib.error
import urllib.request

from praxis import config
from praxis.pipeline.learned import post


def get_health() -> dict:
    with urllib.request.urlopen(config.TAS_BASE_URL.rstrip("/") + "/health", timeout=30) as response:
        return json.loads(response.read())

LINE = re.compile(
    r"^(?P<method>\S+)\s+(?P<f01>[\d.]+)\s+(?P<f025>[\d.]+)\s+(?P<f05>[\d.]+)\s+"
    r"(?P<lo>[\d.]+)–(?P<hi>[\d.]+)\s+(?P<err>[\d.]+)с\s+(?P<steps>[\d.]+)"
)
BASE_DIM = 768


def parse_sweep(text: str) -> dict[str, dict]:
    rows = {}
    for line in text.splitlines():
        match = LINE.match(line.strip())
        if match:
            rows[match["method"]] = {
                "f1_05": float(match["f05"]), "lo": float(match["lo"]),
                "hi": float(match["hi"]), "steps": float(match["steps"]),
                "err": float(match["err"]),
            }
    return rows


def render(rows: list[dict]) -> str:
    lines = ["| вариант | набор | F1@0.5 | 90 % | ошибка границ | шагов |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['set']} | {r['f1_05']:.3f} | {r['lo']:.3f}–{r['hi']:.3f} "
                     f"| {r.get('err', float('nan')):.2f} с | {r['steps']:.1f} |")
    return "\n".join(lines)


def align_input_flags() -> None:
    """Чекпоинт сам говорит, что ему слать: 768 — голые признаки, +1 — полоса движения,
    +6 — разности признаков. Клиент в подпроцессе читает флаги из окружения."""
    dim = int(get_health().get("dim") or BASE_DIM)
    extra = dim - BASE_DIM
    os.environ["PRAXIS_TAS_MOTION"] = "1" if extra in (1, 7) else "0"
    os.environ["PRAXIS_TAS_DIFF"] = "1" if extra in (6, 7) else "0"


def sweep(clips: Path, gt: Path, threshold: float) -> dict:
    method = f"learned-boundaries:PRAXIS_TAS_THRESHOLD={threshold}"
    out = subprocess.run(
        [sys.executable, "scripts/sweep.py", "--clips", str(clips), "--gt", str(gt), "--methods", method],
        capture_output=True, text=True, check=False,
    ).stdout
    empty = {"f1_05": float("nan"), "lo": float("nan"), "hi": float("nan"), "steps": float("nan"), "err": float("nan")}
    return parse_sweep(out).get(method, empty)


def evaluate(variants: list[str], sets: dict[str, Path], threshold: float) -> list[dict]:
    rows = []
    for variant in variants:
        # "a+b" — усреднение нескольких чекпоинтов; base — основной. Вариант без
        # чекпоинта пропускается с пометкой, а не роняет всю таблицу: остальные строки
        # дороже, чем одна недостающая.
        names = [("" if n == "base" else n) for n in variant.split("+")]
        try:
            post("/load", {"names": names} if len(names) > 1 else {"name": names[0]}, timeout=120)
        except urllib.error.HTTPError as error:
            print(f"пропуск {variant}: {error.code}", file=sys.stderr)
            continue
        align_input_flags()
        for name, root in sets.items():
            # Валидационный пул хранит источники подпапками; отложенный роботный — плоско.
            if (root / "annotations").exists():
                pairs = [(root / "clips" / d.name, root / "annotations" / d.name)
                         for d in sorted((root / "clips").iterdir()) if d.is_dir()]
            else:
                pairs = [(root / "clips", root / "gt")]
            for clips, gt in pairs:
                label = f"{name}/{clips.name}" if len(pairs) > 1 else name
                rows.append({"variant": variant, "set": label, **sweep(clips, gt, threshold)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True, help="через запятую; base = основной чекпоинт")
    parser.add_argument("--sets", nargs="+", required=True, help="имя=путь")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    sets = {k: Path(v) for k, _, v in (s.partition("=") for s in args.sets)}
    rows = evaluate(args.variants.split(","), sets, args.threshold)
    table = render(rows)
    print(table)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{dt.date.today().isoformat()}-variants.md"
    target.write_text(f"# Варианты детектора, порог {args.threshold}\n\n{table}\n", encoding="utf-8")
    print(f"\nзаписано в {target}")


if __name__ == "__main__":
    main()
