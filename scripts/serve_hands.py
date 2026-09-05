#!/usr/bin/env python3
"""Руки в кадре: оставить рабочую зону, остальное закрасить.

Зачем. Предмет модель называет хуже всего: на валидации он верен в 21 ответе из 142, а в 60
случаях назван родовым словом — «part», «toy car» — вместо детали, с которой на самом деле
работают. Модель отвечает по общему впечатлению от сцены, а не по тому, что в руках: на
столе лежит собранная игрушка, она заметнее мелкой детали и перетягивает ответ.

Закраска отвечает на это прямо: всё, что дальше заданного радиуса от кистей, заливается
белым. Разрешение при этом не растёт — в отличие от кропа, — но со сцены исчезают
отвлекающие предметы, и «то, что в руках» остаётся единственным кандидатом.

Почему закраска, а не кроп. Кадры шага уходят в модель видеотрактом, и она читает движение
из пар соседних кадров. Кроп, прыгающий за руками, добавил бы к этому движению своё
собственное и испортил бы ровно тот признак, ради которого видеотракт и заведён. Закраска
оставляет геометрию сцены неподвижной: двигается только граница видимой области.

Почему не 100DOH. Детектор рук-и-предметов из «Understanding Human Hands in Contact at
Internet Scale» даёт больше — он отмечает и предмет в контакте, — но собран на Faster R-CNN
2020 года с собственными CUDA-ядрами под torch 1.x и на нынешнем стеке не собирается.
MediaPipe отвечает на нужную часть вопроса: где кисть. Предмет попадает в область вместе с
рукой, если задать радиус с запасом.

Модель на CPU и весит 8 МБ: карта не нужна, память у VLM не отнимается.

    /venv/main/bin/python scripts/serve_hands.py --port 8104 \\
        --model /workspace/models/hand_landmarker.task
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import time

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image, ImageDraw, ImageFilter
from pydantic import BaseModel


class Segment(BaseModel):
    id: int
    frames: list[str]  # JPEG в base64, по порядку


class FocusRequest(BaseModel):
    segments: list[Segment]
    # Радиус видимой области вокруг кисти, долями её собственного размера. Предмет,
    # инструмент и опора должны попасть внутрь вместе с рукой.
    margin: float = 1.5
    # Нижняя граница радиуса долей кадра: у мелкой кисти своего размера не хватает.
    min_radius: float = 0.12
    # Размытие границы области. Резкий край сам по себе становится заметным объектом.
    feather: float = 0.02
    # Что делать с кадром:
    #   "white"  — всё вне области залить белым;
    #   "circle" — ничего не убирать, обвести область красным овалом;
    #   "dim"    — притушить всё вне области, оставив различимым.
    # Обводка — визуальный промптинг: модель притягивается к обведённой области, но сцена
    # остаётся целой. Это важно, потому что прошлый замер кропа по предмету дал 0.209
    # против 0.358 без него — контекст нужен, чтобы предмет вообще опознать.
    mode: str = "white"
    # Насколько притушить фон в режиме "dim": ноль — до черноты, единица — не трогать.
    dim: float = 0.35
    # Качество JPEG на обратном пути: кадры возвращаются перерисованными.
    quality: int = 90


app = FastAPI(title="Praxis hands")
state: dict = {}


def load(model_path: str) -> None:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    started = time.perf_counter()
    print(f"загружаю детектор рук {model_path}…", flush=True)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        num_hands=2,
        # Порог ниже стандартного: на смазанном кадре сборки уверенность падает, а нам
        # достаточно приблизительного места кисти — по нему только строится область.
        min_hand_detection_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE,
    )
    state["detector"] = vision.HandLandmarker.create_from_options(options)
    state["model_id"] = f"mediapipe hand_landmarker ({os.path.basename(model_path)})"
    print(f"готово за {time.perf_counter() - started:.1f} с", flush=True)


def hands_on(image: Image.Image) -> list[tuple[float, float, float]]:
    """Круги вокруг кистей в долях кадра: (центр x, центр y, радиус).

    Детектор работает на увеличенной копии, если кадр мелкий: кисть занимает около полутора
    процентов площади, и на 640 пикселях ширины детектор её теряет. Возвращаются доли, так
    что к исходному кадру они применяются без пересчёта.
    """
    import mediapipe as mp

    probe = image
    if image.width < 1000:
        scale = 1000 / image.width
        probe = image.resize((1000, int(image.height * scale)), Image.BILINEAR)

    answer = state["detector"].detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(probe.convert("RGB")))
    )
    circles = []
    for hand in answer.hand_landmarks:
        xs = [point.x for point in hand]
        ys = [point.y for point in hand]
        center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        # Радиус в долях ширины: по высоте кадр обычно короче, и доля там крупнее.
        radius = max(max(xs) - min(xs), (max(ys) - min(ys)) * image.height / image.width) / 2
        circles.append((center_x, center_y, radius))
    return circles


def regions(
    image: Image.Image,
    circles: list[tuple[float, float, float]],
    margin: float,
    min_radius: float,
) -> list[tuple[float, float, float, float]]:
    """Области вокруг кистей в пикселях кадра: (слева, сверху, справа, снизу)."""
    boxes = []
    for center_x, center_y, radius in circles:
        span = max(radius * (1 + margin), min_radius) * image.width
        x, y = center_x * image.width, center_y * image.height
        boxes.append((x - span, y - span, x + span, y + span))
    return boxes


def paint(image: Image.Image, circles: list[tuple[float, float, float]], request) -> Image.Image:
    """Отметить рабочую зону выбранным способом. Без кистей кадр возвращается как есть."""
    if not circles:
        return image

    boxes = regions(image, circles, request.margin, request.min_radius)
    image = image.convert("RGB")

    if request.mode == "circle":
        # Одна фигура на кадр, а не по одной на кисть: предмет находится между руками, и
        # два овала указывали бы на руки по отдельности, а не на то, с чем работают.
        left = min(box[0] for box in boxes)
        top = min(box[1] for box in boxes)
        right = max(box[2] for box in boxes)
        bottom = max(box[3] for box in boxes)
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        draw.ellipse(
            (left, top, right, bottom),
            outline=(255, 0, 0),
            width=max(3, round(image.width * 0.006)),
        )
        return marked

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.ellipse(box, fill=255)
    if request.feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(request.feather * image.width))

    if request.mode == "dim":
        background = Image.eval(image, lambda value: round(value * request.dim))
    else:
        background = Image.new("RGB", image.size, (255, 255, 255))
    return Image.composite(image, background, mask)


@app.get("/health")
def health() -> dict:
    return {"ready": "detector" in state, "model": state.get("model_id")}


@app.post("/focus")
def focus(request: FocusRequest) -> dict:
    """Кадры с закрашенным фоном — в том же порядке и количестве, что пришли.

    Кадр без найденных кистей наследует область предыдущего кадра шага, а если рук не было
    ни разу — возвращается нетронутым. Пустая закраска хуже, чем никакой: лучше показать
    модели всю сцену, чем белый лист.
    """
    started = time.perf_counter()
    results, painted, seen = [], 0, 0

    for segment in request.segments:
        frames, last = [], []
        for encoded in segment.frames:
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            circles = hands_on(image)
            seen += bool(circles)
            if circles:
                last = circles
            painted += bool(last)
            shown = paint(image, last, request)
            buffer = io.BytesIO()
            shown.save(buffer, format="JPEG", quality=request.quality)
            frames.append(base64.b64encode(buffer.getvalue()).decode())
        results.append({"id": segment.id, "frames": frames})

    total = sum(len(segment.frames) for segment in request.segments)
    return {
        "results": results,
        "frames": total,
        "frames_with_hands": seen,
        "frames_painted": painted,
        "model": state["model_id"],
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=os.getenv("PRAXIS_HANDS_MODEL", "/workspace/models/hand_landmarker.task")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8104)
    args = parser.parse_args()

    load(args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
