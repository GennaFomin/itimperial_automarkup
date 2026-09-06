#!/usr/bin/env python3
"""Демонстрационный ролик: видео плюс таймлайн нарезки под ним.

Разбиение во времени невозможно оценить по таблице метрик — его надо видеть. Под кадром
рисуются две полосы, эталонная разметка и наша, и по ним едет бегунок: сразу видно и
попадание в границы, и лишние либо пропущенные шаги.

Собирается фильтрами ffmpeg, без новых зависимостей.

    python scripts/demo_video.py --clip data/devset_big/clips/x.mp4 \\
        --pred work/pred/x.json --gt data/devset_big/gt/x.json --out demo.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WIDTH = 960
LEFT = 176
RIGHT = WIDTH - 16
BAR = 30
ROW = 62
# Цвета по кругу: соседние сегменты обязаны отличаться, иначе стык не виден.
COLORS = ["0x4F86C6", "0x5FB49C", "0xC65D7B", "0xE0A458", "0x8A6FBF", "0x4FA3A5"]


def clean(text: str) -> str:
    """Убирает из подписи то, что ffmpeg считает синтаксисом фильтра."""
    return text.replace("\\", "").replace(":", " ").replace("'", "").replace(",", " ")


def measure(text: str, size: int) -> float:
    """Ширина строки в пикселях для того же шрифта, которым рисует ffmpeg.

    PIL здесь только измеряет: рисование остаётся за ffmpeg, а подбирать кегль «на глаз»
    по числу символов нельзя — «put down» и «illiquid» одной длины занимают разное место.
    """
    from PIL import ImageFont

    key = ("font", size)
    if key not in _FONTS:
        try:
            _FONTS[key] = ImageFont.truetype(FONT, size)
        except OSError:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key].getlength(text)


def fit_label(action: str, noun: str, width: float) -> list[tuple[str, int, int]]:
    """Как уместить «действие · предмет» в полосу заданной ширины.

    Порядок уступок: сначала пробуем всё одной строкой крупно, потом двумя строками
    (действие сверху, предмет снизу — в полосе высотой 30 px помещаются), потом только
    действие, и лишь в самом конце обрезаем. Возвращает список (текст, кегль, сдвиг вниз).
    """
    room = max(width - 8, 0)
    full = f"{action} {noun}".strip()
    if not full:
        return []

    for size in (15, 13, 12):
        if measure(full, size) <= room:
            return [(full, size, (BAR - size) // 2)]

    if noun:
        for size in (13, 12, 11):
            if max(measure(action, size), measure(noun, size)) <= room:
                return [(action, size, 3), (noun, size, 3 + size + 1)]

    for size in (13, 12, 11):
        if measure(action, size) <= room:
            return [(action, size, (BAR - size) // 2)]

    # Ничего не влезло: обрезаем действие по месту, но не прячем — пустая полоса выглядит
    # как сбой именования, а не как короткий шаг.
    size = 11
    text = action
    while text and measure(text + "…", size) > room:
        text = text[:-1]
    return [(text + "…", size, (BAR - size) // 2)] if text else []


_FONTS: dict = {}


def row(steps: list[dict], duration: float, top: int, title: str) -> list[str]:
    """Полоса таймлайна. Координата y — абсолютная: в drawbox переменная h означает
    высоту самого прямоугольника, а не кадра, и выражение через неё уезжает за экран."""
    span = RIGHT - LEFT
    y = top
    parts = [
        f"drawbox=x={LEFT}:y={y}:w={span}:h={BAR}:color=0x2A2F3A@1:t=fill",
        f"drawtext=fontfile={FONT}:text='{clean(title)}':x=16:y={y + 6}:"
        f"fontsize=18:fontcolor=0xB8BEC8",
    ]
    for index, step in enumerate(steps):
        start = LEFT + span * step["start_sec"] / duration
        width = max(2.0, span * (step["end_sec"] - step["start_sec"]) / duration)
        parts.append(
            f"drawbox=x={start:.1f}:y={y}:w={width:.1f}:h={BAR}:"
            f"color={COLORS[index % len(COLORS)]}@1:t=fill"
        )
        # Ключевой кадр — доказательство шага, кейс требует его отдельно. Отмечаем
        # тонкой светлой чертой внутри сегмента.
        keyframe = step.get("keyframe_sec")
        if keyframe is not None:
            mark = LEFT + span * float(keyframe) / duration
            parts.append(
                f"drawbox=x={mark:.1f}:y={y}:w=2:h={BAR}:color=0xF2F5F9@0.9:t=fill"
            )
        # Подпись обрезается под ширину полосы, а не прячется: скрытая подпись выглядит
        # как шаг без имени, и по картинке невозможно отличить короткий шаг от сбоя
        # именования. На самую узкую полосу ставится номер шага.
        lines = fit_label(clean(step["action"] or ""), clean(step.get("object") or ""), width)
        if lines:
            for text, size, shift in lines:
                parts.append(
                    f"drawtext=fontfile={FONT}:text='{text}':x={start + 4:.1f}:y={y + shift}:"
                    f"fontsize={size}:fontcolor=white"
                )
        elif width >= 14:
            parts.append(
                f"drawtext=fontfile={FONT}:text='{index + 1}':x={start + 3:.1f}:y={y + 7}:"
                f"fontsize=13:fontcolor=white"
            )
    return parts


def caption(steps: list[dict], title: str, y: int, colour: str) -> list[str]:
    """Имя класса текущего шага целиком, крупно, включаясь на время самого шага.

    Подпись внутри полосы таймлайна обрезается по её ширине: на шаге в полсекунды от
    «pick up screwdriver» остаётся «pick…» или вовсе номер. Здесь текст живёт под роликом
    и показывается, пока идёт свой шаг, — читается любой класс независимо от длины шага.
    """
    parts = []
    for step in steps:
        label = f"{step.get('action') or '?'}"
        if step.get("object"):
            label += f" · {step['object']}"
        parts.append(
            f"drawtext=fontfile={FONT}:text='{clean(title)}  {clean(label)}':"
            f"x=16:y={y}:fontsize=20:fontcolor={colour}:"
            f"enable='between(t,{float(step['start_sec']):.3f},{float(step['end_sec']):.3f})'"
        )
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--pred", nargs="+", required=True, help="подпись=путь, можно несколько")
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    tracks = []
    for entry in args.pred:
        title, _, path = entry.partition("=")
        tracks.append((title, json.loads(Path(path).read_text(encoding="utf-8"))))
    prediction = tracks[0][1]
    duration = float(prediction["video"]["duration_sec"])
    truth = json.loads(args.gt.read_text(encoding="utf-8")) if args.gt else None

    rows = len(tracks) + (1 if truth else 0)
    # Место под строку с именем класса: по одной на дорожку, чтобы эталон и предсказание
    # читались рядом.
    caption_height = 26 * (len(tracks) + (1 if truth else 0))
    panel = 24 + rows * ROW + caption_height
    # Высота кадра после масштабирования известна заранее — значит все координаты
    # панели можно посчитать числами и не полагаться на выражения ffmpeg.
    meta = prediction["video"]
    scaled = int(round(meta["height"] * WIDTH / meta["width"])) // 2 * 2
    filters = [f"scale={WIDTH}:{scaled}", f"pad=iw:ih+{panel}:0:0:color=0x14171C"]

    top = scaled + 8
    if truth:
        filters += row(truth["steps"], duration, top, "эталон")
        top += ROW
    for title, annotation in tracks:
        filters += row(annotation["steps"], duration, top, title)
        top += ROW

    # Строки с именами классов — под полосами таймлайна.
    caption_top = top + 6
    if truth:
        filters += caption(truth["steps"], "эталон:", caption_top, "0x9FB4C7")
        caption_top += 26
    for title, annotation in tracks:
        filters += caption(annotation["steps"], f"{title}:", caption_top, "0xF2F5F9")
        caption_top += 26

    span = RIGHT - LEFT
    if args.title:
        filters.append(
            f"drawtext=fontfile={FONT}:text='{clean(args.title)}':x=16:y={scaled + panel - 18}:"
            f"fontsize=16:fontcolor=0x8A93A0"
        )

    # Бегунок — отдельный источник поверх картинки: в drawbox переменная t означает
    # толщину рамки, а не время, и анимировать им положение нельзя. У overlay время есть.
    head_height = rows * ROW
    graph = (
        f"[0:v]{','.join(filters)}[bg];"
        f"color=c=0xFFD166:s=3x{head_height}:d={duration:.3f}[head];"
        f"[bg][head]overlay=x='{LEFT}+{span}*t/{duration:.3f}':y={scaled + 4}"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(args.clip),
            "-filter_complex", graph,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", str(args.out),
        ],
        check=True,
    )
    print(f"готово: {args.out}")


if __name__ == "__main__":
    main()
