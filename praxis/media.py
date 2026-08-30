"""Работа с видео через ffmpeg: метаданные, кадры, плёнка превью, сигнал движения."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from praxis import config


class MediaError(RuntimeError):
    pass


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, **kwargs)
    if result.returncode != 0:
        tail = result.stderr.decode("utf-8", "replace")[-500:]
        raise MediaError(f"{cmd[0]} завершился с кодом {result.returncode}: {tail}")
    return result


def probe(path: Path) -> dict:
    """Длительность, частота кадров и разрешение первого видеопотока."""
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,codec_name:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise MediaError("в файле нет видеопотока")
    stream = data["streams"][0]

    num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    duration = float(data.get("format", {}).get("duration", 0.0))
    if duration <= 0 or fps <= 0:
        raise MediaError("не удалось определить длительность или частоту кадров")

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(fps, 3),
        "duration_sec": round(duration, 3),
        "codec": stream.get("codec_name", ""),
    }


def extract_frame(video: Path, at_sec: float, out: Path, width: int = 640) -> Path:
    """Один кадр в JPEG. Используется для ключевых кадров шагов."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(at_sec, 0):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-q:v",
            "3",
            str(out),
        ]
    )
    if not out.exists():
        raise MediaError(f"кадр на {at_sec:.3f} с не извлёкся")
    return out


def filmstrip(
    video: Path, duration_sec: float, out_dir: Path, count: int | None = None
) -> list[str]:
    """Равномерная плёнка превью для таймлайна. Возвращает имена файлов по порядку."""
    count = count or config.FILMSTRIP_COUNT
    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for index in range(count):
        at = duration_sec * (index + 0.5) / count
        name = f"strip_{index:03d}.jpg"
        extract_frame(video, at, out_dir / name, width=160)
        names.append(name)
    return names


def motion_signal(video: Path, fps: int | None = None) -> list[float]:
    """Насколько сильно меняется картинка во времени, 0..1.

    Считается по крошечным серым кадрам: это дёшево, устойчиво к шуму и даёт редактору
    полосу, на которой видно, где вообще происходит движение. Позже тот же сигнал
    станет одним из входов сегментатора.
    """
    fps = fps or config.MOTION_FPS
    width, height = 64, 36
    result = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={fps},scale={width}:{height},format=gray",
            "-f",
            "rawvideo",
            "-",
        ]
    )
    frames = np.frombuffer(result.stdout, dtype=np.uint8)
    count = frames.size // (width * height)
    if count < 2:
        return []
    frames = frames[: count * width * height].reshape(count, height * width).astype(np.float32)

    diff = np.abs(np.diff(frames, axis=0)).mean(axis=1)
    diff = np.concatenate([diff[:1], diff])  # выравниваем длину с числом кадров
    peak = float(diff.max())
    normalised = diff / peak if peak > 0 else diff
    return [round(float(value), 4) for value in normalised]
