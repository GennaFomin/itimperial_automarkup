#!/usr/bin/env python3
"""Ролик с наложенной позой кистей и полосой признаков под ним.

Числа в слое рук проверить по таблице нельзя: «раскрытие 0.03 м» ничего не говорит о том,
поймал ли детектор ту руку и ту фазу. Поэтому рядом с разметкой действий, у которой есть
demo_video.py, у слоя рук должен быть свой рендер.

Что рисуется. Скелет кисти поверх кадра (двадцать одна точка, связи как в MediaPipe),
подпись стороны и уверенности. Снизу — полоса шагов эталона или предсказания и две
кривые: раскрытие кисти и скорость запястья, с бегунком по времени. Видно сразу и провалы
детекции, и то, совпадают ли всплески скорости с границами шагов.

    python scripts/hand_video.py --clip ролик.mp4 --hands work/hands/ролик.json \\
        --out demo.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from praxis import media

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# Связи кисти как в MediaPipe: большой, указательный, средний, безымянный, мизинец и ладонь.
BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
SIDE_COLOR = {"Left": (90, 200, 255), "Right": (255, 170, 60)}
PANEL = 178  # полосы, кривые и строка с именем текущего класса
STEP_COLORS = ["#4F86C6", "#5FB49C", "#C65D7B", "#E0A458", "#8A6FBF", "#4FA3A5"]


def font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def draw_hands(image: Image.Image, hands: list[dict]) -> None:
    """Скелет каждой кисти поверх кадра."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for order, hand in enumerate(hands):
        points = [(p[0] * width, p[1] * height) for p in hand["landmarks"]]
        colour = SIDE_COLOR.get(hand.get("side", ""), (200, 200, 200))
        for start, end in BONES:
            if start < len(points) and end < len(points):
                draw.line([points[start], points[end]], fill=colour, width=max(2, width // 400))
        for index, point in enumerate(points):
            radius = max(3, width // 260) if index in (0, 4, 8) else max(2, width // 400)
            draw.ellipse(
                [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
                fill=colour,
            )
        if points:
            label = f"{hand.get('side', '?')} {hand.get('score', 0):.2f}"
            if hand.get("aperture") is not None:
                label += f"  раскрытие {hand['aperture'] * 100:.1f} см"
            # Подписи разводятся по вертикали: две кисти рядом писали поверх друг друга.
            draw.text(
                (points[0][0] + 8, points[0][1] + 8 + order * 22), label, fill=colour, font=font(16)
            )


def curve(draw: ImageDraw.ImageDraw, values: list[float | None], box: tuple[int, int, int, int],
          colour: str, title: str) -> None:
    """Кривая признака внутри прямоугольника, с подписью и приведением к своему размаху."""
    left, top, right, bottom = box
    draw.rectangle(box, outline="#333333")
    draw.text((left + 4, top + 2), title, fill=colour, font=font(12))
    real = [value for value in values if value is not None]
    if len(real) < 2:
        return
    low, high = min(real), max(real)
    span = max(high - low, 1e-6)
    points = []
    for index, value in enumerate(values):
        if value is None:
            continue
        x = left + (right - left) * index / max(len(values) - 1, 1)
        y = bottom - (bottom - top - 14) * (value - low) / span
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=colour, width=2)
    draw.text((right - 62, bottom - 14), f"{low:.3f}…{high:.3f}", fill="#888888", font=font(11))


def fit_label(action: str, noun: str, width: float, height: float) -> list[tuple[str, int, int]]:
    """Как уместить «действие · предмет» в полосу: кегль подбирается по замеру шрифта.

    Обрезка по числу символов не годится — «put down» и «illiquid» одной длины занимают
    разное место. Порядок уступок: одна строка крупно, две строки (действие и предмет),
    только действие, и лишь в конце обрезка с многоточием.
    """
    room = max(width - 6, 0)
    full = f"{action} {noun}".strip()
    if not full:
        return []
    for size in (13, 12, 11):
        if font(size).getlength(full) <= room:
            return [(full, size, int((height - size) // 2))]
    if noun and height >= 26:
        for size in (11, 10):
            if max(font(size).getlength(action), font(size).getlength(noun)) <= room:
                return [(action, size, 2), (noun, size, 3 + size)]
    for size in (11, 10):
        if font(size).getlength(action) <= room:
            return [(action, size, int((height - size) // 2))]
    size, text = 10, action
    while text and font(size).getlength(text + "…") > room:
        text = text[:-1]
    return [(text + "…", size, int((height - size) // 2))] if text else []


def steps_bar(draw: ImageDraw.ImageDraw, steps: list[dict], duration: float,
              box: tuple[int, int, int, int]) -> None:
    """Полоса шагов: цвет по кругу, чтобы соседние отличались, подпись вписана в полосу."""
    left, top, right, bottom = box
    draw.rectangle(box, outline="#333333")
    for index, step in enumerate(steps):
        x0 = left + (right - left) * step["start_sec"] / max(duration, 1e-6)
        x1 = left + (right - left) * step["end_sec"] / max(duration, 1e-6)
        draw.rectangle([x0, top + 1, max(x1, x0 + 2), bottom - 1], fill=STEP_COLORS[index % len(STEP_COLORS)])
        for text, size, shift in fit_label(
            step.get("action") or "", step.get("object") or "", x1 - x0, bottom - top
        ):
            draw.text((x0 + 3, top + shift), text, fill="white", font=font(size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--hands", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()

    layer = json.loads(args.hands.read_text(encoding="utf-8"))
    frames_data = layer["frames"]
    fps = layer.get("fps", 8.0)
    duration = layer.get("duration_sec") or (len(frames_data) / fps)
    steps = layer.get("steps", [])

    # Признаки ведущей руки по кадрам: их же рисуем кривыми под роликом.
    apertures: list[float | None] = []
    speeds: list[float | None] = []
    previous = None
    for frame in frames_data:
        hand = max(frame["hands"], key=lambda item: item.get("score", 0.0)) if frame["hands"] else None
        apertures.append(hand.get("aperture") if hand else None)
        wrist = hand["landmarks"][0] if hand else None
        if wrist and previous:
            speeds.append(((wrist[0] - previous[0]) ** 2 + (wrist[1] - previous[1]) ** 2) ** 0.5 * fps)
        else:
            speeds.append(None)
        previous = wrist

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Кадры режутся тем же вызовом, что и в hand_features: свой fps-фильтр давал сетку,
        # сдвинутую на кадр, и скелет уезжал с руки на быстром движении. Один способ
        # нарезки на оба скрипта — единственный способ этого не допустить.
        sources = media.window_frames(
            args.clip, 0.0, duration, root, fps=fps, width=args.width,
            limit=len(frames_data) + 8,
        )
        for index, source in enumerate(sources):
            if index >= len(frames_data):
                break
            image = Image.open(source).convert("RGB")
            draw_hands(image, frames_data[index]["hands"])

            canvas = Image.new("RGB", (image.width, image.height + PANEL), (18, 18, 18))
            canvas.paste(image, (0, 0))
            draw = ImageDraw.Draw(canvas)
            base = image.height
            margin = 10
            steps_bar(draw, steps, duration, (margin, base + 6, image.width - margin, base + 36))
            half = (image.width - 3 * margin) // 2
            curve(draw, apertures, (margin, base + 42, margin + half, base + PANEL - 34),
                  "#5FB49C", "раскрытие кисти, м")
            curve(draw, speeds, (2 * margin + half, base + 42, image.width - margin, base + PANEL - 34),
                  "#E0A458", "скорость запястья")

            # Имя класса текущего шага крупно: в полосе таймлайна оно обрезается по её
            # ширине, и на коротком шаге от «pick up screwdriver» остаётся «pick…».
            moment = frames_data[index]["t"]
            current = next(
                (s for s in steps if s["start_sec"] <= moment <= s["end_sec"]), None
            )
            if current:
                label = current.get("action") or "?"
                if current.get("object"):
                    label += f" · {current['object']}"
            else:
                label = "— вне шага —"
            draw.text((margin, base + PANEL - 26), label, fill="#F2F5F9", font=font(20))

            x = margin + (image.width - 2 * margin) * moment / max(duration, 1e-6)
            draw.line([(x, base + 6), (x, base + PANEL - 6)], fill="#ffffff", width=1)
            draw.text((image.width - 96, 8), f"{moment:6.2f} с", fill="white", font=font(18))
            canvas.save(root / f"out_{index:05d}.jpg", quality=92)

        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(root / "out_%05d.jpg"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(args.out)], check=True,
        )
    print(f"готово: {args.out}")


if __name__ == "__main__":
    main()
