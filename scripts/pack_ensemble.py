#!/usr/bin/env python3
"""Собрать несколько именованных чекпоинтов детектора границ в один файл-ансамбль.

Сервис усредняет вероятности участников по кадрам (как /load с несколькими именами),
но развёртывание остаётся одним файлом: первый участник — обычный чекпоинт, остальные
лежат в поле members. Старый сервис такой файл тоже откроет — как одиночную модель.

    python scripts/pack_ensemble.py checkpoints/boundary-a.pt checkpoints/boundary-b.pt --out checkpoints/boundary.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def normalise(payload: dict) -> dict:
    """Старый двухстадийный формат без поля stages: вторая стадия звалась "second".
    Сервис делает то же самое при загрузке; здесь — чтобы участники были в одном формате."""
    weights = payload["weights"]
    if "stages" in payload:
        return {"dim": payload["dim"], "stages": int(payload["stages"]), "weights": weights}
    legacy = any(k.startswith("second.") for k in weights)
    if legacy:
        weights = {k.replace("second.", "rest.0.", 1) if k.startswith("second.") else k: v for k, v in weights.items()}
    return {"dim": payload["dim"], "stages": 2 if legacy else 4, "weights": weights}


def pack(paths: list[Path], fusion: str = "mean") -> dict:
    payloads = [normalise(torch.load(p, map_location="cpu")) for p in paths]
    dims = {int(p["dim"]) for p in payloads}
    if len(dims) != 1:
        raise SystemExit(f"разная размерность входа у участников: {sorted(dims)}")
    stages = [p["stages"] for p in payloads]
    first = payloads[0]
    return {
        "dim": first["dim"], "stages": stages[0], "weights": first["weights"],
        "members": [{"stages": s, "weights": p["weights"]} for s, p in zip(stages[1:], payloads[1:])],
        "names": [p.name for p in paths],
        "fusion": fusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="упаковка ансамбля чекпоинтов в один файл")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fusion", choices=["mean", "min"], default="mean",
                        help="mean — среднее вероятностей; min — разрез только при согласии всех участников")
    args = parser.parse_args()
    payload = pack(args.checkpoints, args.fusion)
    torch.save(payload, args.out)
    print(f"участников {1 + len(payload['members'])}, слияние {payload['fusion']}, вход {payload['dim']} → {args.out}")


if __name__ == "__main__":
    main()
