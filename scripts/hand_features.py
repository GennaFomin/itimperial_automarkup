#!/usr/bin/env python3
"""Слой разметки по рукам: поза кисти на каждый кадр и сводка на каждый шаг.

Зачем отдельно от разметки действий. Контракт кейсодателя описывает шаг четырьмя полями —
границы, действие, предмет, ключевой кадр — и трогать его нельзя. Но для переноса на робота
нужно другое: где была кисть, как она раскрывалась, что делали пальцы. Поэтому руки живут
рядом, отдельным файлом: разметка остаётся валидной по контракту, а робот получает своё.

Что складывается. На каждый кадр — 21 сустав в долях кадра (где рука в сцене) и те же 21
в метрах относительно центра кисти (какая это поза). Метрические координаты — вход для
ретаргетинга на робота: dex-retargeting и родственные ему оптимизаторы работают с
векторами между суставами, а не с пикселями, поэтому именно они переносятся на чужую
кинематику вроде Inspire RH56.

На каждый шаг — сводка: доля кадров с найденными руками, раскрытие кисти (расстояние между
кончиками большого и указательного), длина пути запястья, скорость. Этого достаточно,
чтобы отобрать шаги, пригодные для обучения, не листая видео.

    python scripts/hand_features.py --clip путь/к/ролику.mp4 --steps разметка.json \\
        --out work/hands/ролик.json
    python scripts/hand_features.py --clips папка/ --steps-dir разметка/ --out-dir work/hands/
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import tempfile
import urllib.request
from pathlib import Path

from praxis import config, media

# Кончики пальцев в разметке MediaPipe: большой, указательный, средний, безымянный, мизинец.
TIPS = (4, 8, 12, 16, 20)
WRIST = 0


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/pose",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def distance(left: list[float], right: list[float]) -> float:
    return math.dist(left[:3], right[:3])


def describe(hand: dict) -> dict:
    """Производные величины позы: раскрытие и сжатость кисти, в метрах."""
    world = hand.get("world") or []
    if len(world) < 21:
        return {}
    palm = world[WRIST]
    return {
        # Расстояние между кончиками большого и указательного — самая говорящая величина:
        # по ней видно и щипковый захват, и раскрытую ладонь.
        "aperture": round(distance(world[4], world[8]), 4),
        # Средняя удалённость кончиков от запястья: кисть сжата в кулак или раскрыта.
        "spread": round(sum(distance(world[tip], palm) for tip in TIPS) / len(TIPS), 4),
    }


def summarise(frames: list[dict], start: float, end: float) -> dict:
    """Сводка по шагу: сколько кадров с руками, как двигалось запястье, как раскрывалась кисть."""
    inside = [frame for frame in frames if start <= frame["t"] <= end]
    if not inside:
        return {"frames": 0, "hands_seen": 0.0}

    seen = [frame for frame in inside if frame["hands"]]
    path, previous = 0.0, None
    apertures = []
    for frame in seen:
        # Ведущая рука шага — самая уверенная на кадре.
        hand = max(frame["hands"], key=lambda item: item.get("score", 0.0))
        wrist = hand["landmarks"][WRIST]
        if previous is not None:
            path += distance(wrist, previous)
        previous = wrist
        if hand.get("aperture") is not None:
            apertures.append(hand["aperture"])

    span = max(end - start, 1e-3)
    return {
        "frames": len(inside),
        "hands_seen": round(len(seen) / len(inside), 3),
        # Путь запястья в долях кадра: сколько рука прошла по сцене за шаг.
        "wrist_path": round(path, 4),
        "wrist_speed": round(path / span, 4),
        "aperture_min": round(min(apertures), 4) if apertures else None,
        "aperture_max": round(max(apertures), 4) if apertures else None,
    }


def process(clip: Path, steps: list[dict], url: str, fps: float, width: int) -> dict:
    """Нарезать ролик, снять позу кистей и собрать слой."""
    with tempfile.TemporaryDirectory() as directory:
        meta = media.probe(clip)
        paths = media.window_frames(
            clip, 0.0, meta["duration_sec"], Path(directory), fps=fps,
            width=width, limit=int(meta["duration_sec"] * fps) + 8,
        )
        encoded = [base64.b64encode(path.read_bytes()).decode() for path in paths]
        times = [round(index / fps, 3) for index in range(len(paths))]
        answer = post(url, {"frames": encoded, "times": times})

    frames = answer["frames"]
    for frame in frames:
        for hand in frame["hands"]:
            hand.update(describe(hand))

    return {
        "clip": clip.stem,
        "duration_sec": meta["duration_sec"],
        "fps": fps,
        "model": answer.get("model", ""),
        "frames_with_hands": answer.get("frames_with_hands", 0),
        "frames": frames,
        "steps": [
            {
                "id": step.get("id", index),
                "start_sec": step["start_sec"],
                "end_sec": step["end_sec"],
                "action": step.get("action"),
                "object": step.get("object"),
                **summarise(frames, step["start_sec"], step["end_sec"]),
            }
            for index, step in enumerate(steps)
        ],
    }


def load_steps(path: Path | None) -> list[dict]:
    """Шаги из файла разметки. Без него слой считается по всему ролику одним куском."""
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps") or data.get("annotations") or []
    return [
        {
            "id": step.get("id", index),
            "start_sec": float(step.get("start_sec", step.get("start", 0.0))),
            "end_sec": float(step.get("end_sec", step.get("end", 0.0))),
            "action": step.get("action"),
            "object": step.get("object"),
        }
        for index, step in enumerate(steps)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--clips", type=Path, help="папка с роликами")
    parser.add_argument("--steps", type=Path, help="разметка одного ролика")
    parser.add_argument("--steps-dir", type=Path, help="папка с разметкой по имени ролика")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--url", default=config.HANDS_BASE_URL or "http://127.0.0.1:8104")
    # Восемь кадров в секунду: движение кисти на этом шаге ещё различимо, а детектор на
    # CPU успевает за ролик примерно вдвое быстрее реального времени.
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    clips = [args.clip] if args.clip else sorted((args.clips or Path()).glob("*.mp4"))
    for clip in clips:
        steps_path = args.steps or (args.steps_dir / f"{clip.stem}.json" if args.steps_dir else None)
        layer = process(clip, load_steps(steps_path), args.url, args.fps, args.width)
        out = args.out or (args.out_dir / f"{clip.stem}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(layer, ensure_ascii=False), encoding="utf-8")
        share = layer["frames_with_hands"] / max(len(layer["frames"]), 1)
        print(
            f"{clip.stem}: {len(layer['frames'])} кадров, руки на {share:.0%}, "
            f"шагов {len(layer['steps'])} → {out}"
        )


if __name__ == "__main__":
    main()
