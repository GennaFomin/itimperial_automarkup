#!/usr/bin/env python3
"""Словарь одного изделия: какие пары (действие, объект) вообще возможны для этой модели.

202 пары Assembly101 — это объединение по 101 разной игрушке. Для одного конкретного
изделия физически возможен десяток пар: у самосвала есть кузов, у экскаватора — стрела, и
перепутать их нельзя, потому что второго в кадре просто нет.

Это не подгонка под датасет, а ровно тот режим, в котором живёт заказчик: размечают
конкретный процесс с конкретными деталями, а не абстрактные «действия вообще». Поэтому в
отчёте честно показывать две колонки — полный словарь как нижняя граница и словарь изделия
как аналог рабочего сеттинга.

Словари строятся по ОБУЧАЮЩЕМУ сплиту, валидационный не используется.

    HF_TOKEN=... python scripts/toy_vocab.py --out data/toy_vocab.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "cvml-nus/assembly101"
LABEL_FPS = 30.0


def fetch(path: str) -> Path:
    return Path(hf_hub_download(REPO, path, repo_type="dataset", token=os.environ.get("HF_TOKEN")))


def action_map() -> dict[str, tuple[str, str]]:
    with fetch("annotations/coarse-annotations/actions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return {
            row["action_cls"].strip(): (row["verb_cls"].strip(), row["noun_cls"].strip())
            for row in csv.DictReader(handle)
        }


def read_split(name: str) -> list[tuple[str, str, str]]:
    """Строки сплита: имя файла разметки, код изделия, имя изделия."""
    path = fetch(f"annotations/coarse-annotations/coarse_splits/{name}")
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part for part in line.split("\t") if part.strip()]
        if len(parts) >= 4:
            entries.append((parts[0].strip(), parts[2].strip(), parts[3].strip()))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_coarse_assembly.txt", "train_coarse_disassembly.txt"],
    )
    args = parser.parse_args()

    actions = action_map()
    per_toy: dict[str, set] = collections.defaultdict(set)
    # Пары по каждой сессии отдельно: это нужно, чтобы при оценке исключать ту самую
    # сессию, из которой взят ролик, и не подглядывать в собственный ответ.
    per_session: dict[str, dict[str, list]] = collections.defaultdict(dict)
    names: dict[str, str] = {}
    sessions = 0

    for split in args.splits:
        for label_file, toy, toy_name in read_split(split):
            names[toy] = toy_name
            try:
                labels = fetch(f"annotations/coarse-annotations/coarse_labels/{label_file}")
            except Exception as error:  # noqa: BLE001 — недостающая разметка не должна ронять сбор
                print(f"пропуск {label_file}: {error}")
                continue
            sessions += 1
            session = label_file.removeprefix("disassembly_").removeprefix("assembly_").removesuffix(".txt")
            found = set()
            for line in labels.read_text(encoding="utf-8").splitlines():
                parts = [part for part in line.split("\t") if part.strip()]
                if len(parts) < 3:
                    continue
                verb, noun = actions.get(parts[2].strip(), (None, None))
                if verb:
                    per_toy[toy].add((verb, noun))
                    found.add((verb, noun))
            per_session[toy][session[-19:]] = sorted([list(pair) for pair in found])

    payload = {
        toy: {
            "name": names.get(toy, ""),
            "pairs": sorted([list(pair) for pair in pairs]),
            "by_session": per_session.get(toy, {}),
        }
        for toy, pairs in sorted(per_toy.items())
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    sizes = [len(value["pairs"]) for value in payload.values()]
    print(f"изделий: {len(payload)}, сессий обработано: {sessions}")
    print(
        f"пар на изделие: в среднем {sum(sizes) / len(sizes):.1f}, "
        f"от {min(sizes)} до {max(sizes)} (в полном словаре 202)"
    )
    print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
