#!/usr/bin/env python3
"""Роботный корпус с автоматическими границами подзадач.

Разметки руками нет и не будет. Границы берутся из того, что робот и так знает о себе:
смена подзадачи, а чаще — момент захвата или отпускания. Открыл/закрыл схват — это и
есть граница атомарного действия: «взял», «положил», «переставил» начинаются и
кончаются именно там.

Источники (все — каталог с <clip>.mp4 и файлом-спутником):
  --dataset gripper    <clip>.gripper.json: {"fps": .., "values": [...]} — сигнал схвата
  --dataset subtasks   <clip>.subtasks.json: {"fps": .., "labels": [...]} — подзадача на кадр
                       (LIBERO-10 subtasks и любой набор с покадровыми стадиями)
  --dataset rh20t_p    <clip>.primitives.json: [{"start": s, "end": s}, ...]

    python scripts/make_robot_devset.py --dataset subtasks --in data/raw_libero --out data/robo_libero
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def boundaries_from_gripper(
    gripper: list[float], fps: float, closed_below: float = 0.5, min_gap_sec: float = 0.4
) -> list[float]:
    """Секунды переходов открыт↔закрыт.

    Переход засчитывается, если новое состояние удерживается большинством следующих
    min_gap_sec кадров — так дребезг датчика не плодит границ, а момент берётся с
    первого кадра перехода, где действие действительно началось. После принятого
    перехода следующий возможен не раньше, чем через min_gap_sec.
    """
    if not gripper:
        return []
    states = [value < closed_below for value in gripper]
    window = max(1, int(round(min_gap_sec * fps)))
    result: list[float] = []
    last_accepted = -window
    for index in range(1, len(states)):
        if states[index] == states[index - 1] or index - last_accepted < window:
            continue
        ahead = states[index: index + window]
        if sum(1 for s in ahead if s == states[index]) * 2 > len(ahead):
            result.append(round(index / fps, 3))
            last_accepted = index
    return result


def boundaries_from_labels(labels: list[str], fps: float) -> list[float]:
    """Секунды, где покадровая подзадача меняется."""
    return [round(i / fps, 3) for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


def steps_from_boundaries(
    boundaries: list[float], duration: float, min_step_sec: float = 0.5
) -> list[dict]:
    edges = [0.0, *[b for b in boundaries if 0 < b < duration], duration]
    steps = []
    for start, end in zip(edges, edges[1:]):
        if end - start < min_step_sec:
            continue
        steps.append({
            "start_sec": round(start, 3), "end_sec": round(end, 3),
            "action": "step", "object": None,
            "keyframe_sec": round((start + end) / 2, 3), "confidence": None,
        })
    return steps


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration",
         "-of", "json", str(video)], capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    stream, fmt = data["streams"][0], data["format"]
    num, den = stream["r_frame_rate"].split("/")
    return {"width": stream["width"], "height": stream["height"],
            "fps": float(num) / float(den), "duration_sec": float(fmt["duration"])}


def write_annotation(out: Path, stem: str, meta: dict, steps: list[dict]) -> None:
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "gt" / f"{stem}.json").write_text(json.dumps({
        "video": {"id": stem, "filename": f"{stem}.mp4", **meta},
        "steps": [{"id": i, **s} for i, s in enumerate(steps)],
        "provenance": {"pipeline": "robot-auto", "app_version": "0", "vocabulary": "robot-auto"},
    }, ensure_ascii=False), encoding="utf-8")


def boundaries_for(clip: Path, dataset: str) -> list[float] | None:
    if dataset == "gripper":
        side = clip.with_suffix(".gripper.json")
        if not side.exists():
            return None
        signal = json.loads(side.read_text(encoding="utf-8"))
        return boundaries_from_gripper(signal["values"], float(signal["fps"]))
    if dataset == "subtasks":
        side = clip.with_suffix(".subtasks.json")
        if not side.exists():
            return None
        data = json.loads(side.read_text(encoding="utf-8"))
        return boundaries_from_labels(data["labels"], float(data["fps"]))
    side = clip.with_suffix(".primitives.json")
    if not side.exists():
        return None
    primitives = json.loads(side.read_text(encoding="utf-8"))
    return sorted({float(p["start"]) for p in primitives} | {float(p["end"]) for p in primitives})


def build(source: Path, out: Path, dataset: str) -> int:
    (out / "clips").mkdir(parents=True, exist_ok=True)
    count = 0
    for clip in sorted(source.glob("*.mp4")):
        boundaries = boundaries_for(clip, dataset)
        if boundaries is None:
            continue
        meta = probe(clip)
        steps = steps_from_boundaries(boundaries, meta["duration_sec"])
        if len(steps) < 2:
            continue
        link = out / "clips" / clip.name
        if not link.exists():
            link.symlink_to(clip.resolve())
        write_annotation(out, clip.stem, meta, steps)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gripper", "subtasks", "rh20t_p"], required=True)
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(f"собрано {build(args.source, args.out, args.dataset)} роликов в {args.out}")


if __name__ == "__main__":
    main()
