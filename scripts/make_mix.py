#!/usr/bin/env python3
"""Единый обучающий корпус из нескольких пулов, с пометкой эмбодимента в имени.

Имя ролика: <эмбодимент>__<пул>__<исходное имя>. Ролики из валидации исключаются и по
имени, и по размеру файла: валидация копировалась из этих же пулов, и совпадение размера
ловит переименованный дубликат.

    python scripts/make_mix.py --out data/train_mix \\
        --validation /mnt/data/praxis-pool/validation \\
        --source human=data/train_atomic --source robot=data/robo_sim
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def forbidden_from(validation: Path | None) -> tuple[set[str], set[int]]:
    if validation is None or not validation.exists():
        return set(), set()
    names = {p.stem for p in (validation / "annotations").rglob("*.json")}
    sizes = {p.stat().st_size for p in (validation / "clips").rglob("*.mp4")}
    return names, sizes


def select(
    sources: list[tuple[str, Path]], forbidden_names: set[str], forbidden_sizes: set[int]
) -> list[tuple[str, Path, Path]]:
    chosen = []
    for embodiment, pool in sources:
        for gt in sorted((pool / "gt").glob("*.json")):
            filename = json.loads(gt.read_text(encoding="utf-8"))["video"]["filename"]
            clip = pool / "clips" / filename
            if not clip.exists():
                continue
            if gt.stem in forbidden_names or clip.stat().st_size in forbidden_sizes:
                continue
            chosen.append((f"{embodiment}__{pool.name}__{gt.stem}", clip, gt))
    return chosen


def build(
    sources: list[tuple[str, Path]], out: Path, forbidden_names: set[str], forbidden_sizes: set[int]
) -> int:
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    count = 0
    for stem, clip, gt in select(sources, forbidden_names, forbidden_sizes):
        link = out / "clips" / f"{stem}.mp4"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(clip.resolve())
        annotation = json.loads(gt.read_text(encoding="utf-8"))
        annotation["video"]["id"] = stem
        annotation["video"]["filename"] = f"{stem}.mp4"
        (out / "gt" / f"{stem}.json").write_text(
            json.dumps(annotation, ensure_ascii=False), encoding="utf-8"
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--source", action="append", required=True,
                        help="эмбодимент=путь к пулу с clips/ и gt/")
    args = parser.parse_args()
    sources = []
    for item in args.source:
        embodiment, _, path = item.partition("=")
        sources.append((embodiment, Path(path)))
    names, sizes = forbidden_from(args.validation)
    total = build(sources, args.out, names, sizes)
    print(f"собрано {total} роликов в {args.out}; исключено по валидации: {len(names)} имён")


if __name__ == "__main__":
    main()
