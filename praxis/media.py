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


def extract_frame(
    video: Path,
    at_sec: float,
    out: Path,
    width: int = 640,
    crop: tuple[float, float, float, float] | None = None,
    stamp: str | None = None,
) -> Path:
    """Один кадр в JPEG. Кроп задаётся долями кадра (слева, сверху, ширина, высота).

    stamp впечатывает подпись прямо в пиксели: языковая модель связывает время с кадром
    заметно лучше, когда оно нарисовано на самом кадре, а не подписано рядом текстом.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if crop:
        left, top, span_x, span_y = crop
        filters.append(
            f"crop=iw*{span_x:.4f}:ih*{span_y:.4f}:iw*{left:.4f}:ih*{top:.4f}"
        )
    filters.append(f"scale={width}:-2")
    if stamp:
        filters.append(
            f"drawtext=text='{stamp}':x=8:y=8:fontsize=28:fontcolor=white:box=1:boxcolor=black"
        )
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
            ",".join(filters),
            "-q:v",
            "3",
            str(out),
        ]
    )
    if not out.exists():
        raise MediaError(f"кадр на {at_sec:.3f} с не извлёкся")
    return out


def window_frames(
    video: Path,
    start_sec: float,
    end_sec: float,
    out_dir: Path,
    fps: float,
    width: int = 640,
    limit: int = 48,
    crop: tuple[float, float, float, float] | None = None,
) -> list[Path]:
    """Кадры куска ролика с заданной частотой — одним вызовом ffmpeg.

    Нужно для сплошной нарезки длинного шага: там кадров выходит втрое-вчетверо больше,
    чем при выборке из пяти моментов, а покадровый вызов стоит около 320 мс на кадр —
    это запуск процесса, а не декодирование. Здесь тот же приём, что в filmstrip.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    span = max(end_sec - start_sec, 1.0 / max(fps, 0.001))
    filters = [f"fps={fps:g}"]
    if crop:
        left, top, span_x, span_y = crop
        filters.append(f"crop=iw*{span_x:.4f}:ih*{span_y:.4f}:iw*{left:.4f}:ih*{top:.4f}")
    filters.append(f"scale={width}:-2")
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{max(start_sec, 0.0):.3f}",
            "-i",
            str(video),
            "-t",
            f"{span:.3f}",
            "-vf",
            ",".join(filters),
            "-frames:v",
            str(limit),
            "-q:v",
            "3",
            str(out_dir / "w_%03d.jpg"),
        ]
    )
    return sorted(out_dir.glob("w_*.jpg"))


def filmstrip(
    video: Path, duration_sec: float, out_dir: Path, count: int | None = None
) -> list[str]:
    """Равномерная плёнка превью для таймлайна.

    Одним вызовом ffmpeg, а не покадрово: сорок запусков процесса стоили десять секунд
    на четырнадцатисекундном ролике, а скорость обработки — метрика кейса.
    """
    count = count or config.FILMSTRIP_COUNT
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("strip_*.jpg"):
        stale.unlink()

    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={count / max(duration_sec, 0.001)},scale=160:-2",
            "-frames:v",
            str(count),
            "-q:v",
            "5",
            str(out_dir / "strip_%03d.jpg"),
        ]
    )
    return sorted(path.name for path in out_dir.glob("strip_*.jpg"))


def _upload_encoder() -> list[str]:
    """Чем сжимать копию для отправки: libx264, если он есть, иначе встроенный mpeg4.

    Сборки ffmpeg без libx264 встречаются на серверах, и там перекодирование падало,
    а прогон сообщал «сервис признаков недоступен» — то есть указывал не на ту причину.
    """
    if _upload_encoder.cached is None:
        try:
            encoders = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=20, check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            encoders = ""
        if " libx264" in encoders:
            _upload_encoder.cached = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                                      "-pix_fmt", "yuv420p"]
        else:
            # mjpeg покадровый и заметно чище mpeg4 на тех же настройках. Формат
            # пикселей у него свой: yuv420p он не принимает и падает «Could not open
            # encoder», а прогон молча уходит на серые блоки.
            _upload_encoder.cached = ["-c:v", "mjpeg", "-qscale:v", "2",
                                      "-pix_fmt", "yuvj420p"]
    return _upload_encoder.cached


_upload_encoder.cached = None


def transcode_for_upload(video: Path, out: Path, height: int = 384) -> Path:
    """Уменьшенная копия ролика для отправки на GPU-машину: мегабайты вместо десятков."""
    _run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video),
            "-vf", f"scale=-2:{height}", *_upload_encoder(), "-an", str(out),
        ]
    )
    return out


def gray_frames(video: Path, fps: int | None = None, width: int = 64, height: int = 36) -> np.ndarray:
    """Кадры ролика в оттенках серого как матрица (кадры, высота, ширина).

    Один проход ffmpeg на всё: из этих кадров считаются и сигнал движения для таймлайна,
    и признаки внешнего вида для сегментатора. Разрешение намеренно крошечное — нас
    интересует, что происходит в кадре в целом, а не детали.
    """
    fps = fps or config.MOTION_FPS
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
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    count = raw.size // (width * height)
    if count < 1:
        return np.zeros((0, height, width), dtype=np.float32)
    return raw[: count * width * height].reshape(count, height, width).astype(np.float32)


def motion_from_frames(frames: np.ndarray) -> np.ndarray:
    """Насколько сильно меняется картинка от кадра к кадру, нормировано в 0..1."""
    if len(frames) < 2:
        return np.zeros(len(frames), dtype=np.float32)
    flat = frames.reshape(len(frames), -1)
    diff = np.abs(np.diff(flat, axis=0)).mean(axis=1)
    diff = np.concatenate([diff[:1], diff])  # выравниваем длину с числом кадров
    peak = float(diff.max())
    return (diff / peak if peak > 0 else diff).astype(np.float32)


def _pool(frames: np.ndarray, blocks: tuple[int, int]) -> np.ndarray:
    """Усреднение по блокам: (кадры, высота, ширина) → (кадры, блоки)."""
    if len(frames) == 0:
        return np.zeros((0, blocks[0] * blocks[1]), dtype=np.float32)
    count, height, width = frames.shape
    rows, columns = blocks
    trimmed = frames[:, : height - height % rows, : width - width % columns]
    grid = trimmed.reshape(count, rows, (height - height % rows) // rows, columns, -1)
    return grid.mean(axis=(2, 4)).reshape(count, rows * columns)


def appearance_from_frames(frames: np.ndarray, blocks: tuple[int, int] = (9, 16)) -> np.ndarray:
    """Огрублённый вид кадра: усреднение по блокам. Устойчиво к шуму и дрожанию камеры."""
    return _pool(frames, blocks)


def active_region(frames: np.ndarray, quantile: float = 0.8, margin: float = 0.1) -> tuple:
    """Прямоугольник, в котором вообще происходит движение за весь ролик.

    Камера статична, и большую часть кадра занимает неподвижный фон: стол, стены, штативы.
    Признаки, посчитанные по всему кадру, в основном описывают этот фон. Ограничиваем их
    рабочей зоной — тем местом, где руки что-то делают.
    """
    if len(frames) < 2:
        return (0, frames.shape[1], 0, frames.shape[2]) if len(frames) else (0, 1, 0, 1)

    activity = np.abs(np.diff(frames, axis=0)).mean(axis=0)
    threshold = np.quantile(activity, quantile)
    rows, columns = np.where(activity >= threshold)
    if not len(rows):
        return 0, frames.shape[1], 0, frames.shape[2]

    height, width = frames.shape[1], frames.shape[2]
    pad_y, pad_x = int(height * margin), int(width * margin)
    return (
        max(0, int(rows.min()) - pad_y),
        min(height, int(rows.max()) + pad_y + 1),
        max(0, int(columns.min()) - pad_x),
        min(width, int(columns.max()) + pad_x + 1),
    )


def motion_signal(video: Path, fps: int | None = None) -> list[float]:
    """Сигнал движения для полосы под таймлайном."""
    frames = gray_frames(video, fps)
    return [round(float(value), 4) for value in motion_from_frames(frames)]
