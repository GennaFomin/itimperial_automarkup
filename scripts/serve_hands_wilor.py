#!/usr/bin/env python3
"""Поза кистей моделью WiLoR: тот же контракт /pose, что у сервиса на MediaPipe.

Зачем менять детектор. MediaPipe на домене кейса находит хотя бы одну руку почти всегда,
но вторую теряет: на assembly101-fine две кисти видны лишь в 37 % кадров, хотя работают
почти всегда двумя. Вторая рука обычно держит предмет снизу и частично перекрыта первой —
именно тот случай, под который MediaPipe не рассчитан.

WiLoR (CVPR 2025) устроен иначе: сначала свёрточный детектор находит кисти, потом
трансформер восстанавливает позу. На тех же кадрах две руки находятся в 67 % случаев.
Кроме того он отдаёт параметры MANO — стандартный вход для переноса на робота, тогда как
MediaPipe даёт только точки.

Цена: 71 мс на кадр на карте против 40 мс на процессоре, плюс отдельное окружение с
Python 3.10 (в PyPI-версии chumpy сломан setup.py, а на 3.12 он не собирается вовсе).

    CUDA_VISIBLE_DEVICES=1 /venv/wilor/bin/python scripts/serve_hands_wilor.py --port 8106
"""

from __future__ import annotations

import argparse
import base64
import io
import time

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel


class PoseRequest(BaseModel):
    frames: list[str]  # JPEG в base64, по порядку
    times: list[float] = []  # момент каждого кадра в секундах ролика
    # Сколько кистей оставлять на кадре. Двух хватает: в кадре один человек, а лишние
    # срабатывания WiLoR даёт примерно на 7 % кадров — обычно на смазанных участках.
    max_hands: int = 2


app = FastAPI(title="Praxis hands (WiLoR)")
state: dict = {}


def load(device: str, half: bool) -> None:
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    started = time.perf_counter()
    print(f"загружаю WiLoR на {device}…", flush=True)
    state["pipe"] = WiLorHandPose3dEstimationPipeline(
        device=device, dtype=torch.float16 if half else torch.float32
    )
    state["model_id"] = "wilor-mini (WiLoR, CVPR 2025)"
    print(f"готово за {time.perf_counter() - started:.1f} с", flush=True)


def pose_of(image: Image.Image, limit: int) -> list[dict]:
    """Кисти на кадре в том же виде, что отдаёт сервис на MediaPipe.

    Точки приводятся к долям кадра, метрические — как есть: WiLoR отдаёт их в метрах
    относительно кисти, ровно как MediaPipe, так что клиент менять не нужно.
    """
    frame = np.array(image.convert("RGB"))
    width, height = image.size
    found = state["pipe"].predict(frame)

    # Лишние срабатывания отсекаются по площади рамки: настоящая кисть крупнее случайного
    # отклика на смазанном фоне. Уверенности WiLoR наружу не отдаёт, поэтому размер —
    # единственная доступная мера, и в score кладётся именно она.
    ordered = sorted(
        found,
        key=lambda item: (item["hand_bbox"][2] - item["hand_bbox"][0])
        * (item["hand_bbox"][3] - item["hand_bbox"][1]),
        reverse=True,
    )[:limit]

    hands = []
    for item in ordered:
        preds = item["wilor_preds"]
        points = np.asarray(preds["pred_keypoints_2d"]).reshape(-1, 2)
        metric = np.asarray(preds["pred_keypoints_3d"]).reshape(-1, 3)
        box = item["hand_bbox"]
        area = (box[2] - box[0]) * (box[3] - box[1]) / max(width * height, 1)
        hands.append(
            {
                "side": "Right" if item.get("is_right") else "Left",
                # Не уверенность модели, а доля кадра, занятая рамкой: сравнивать кисти
                # между собой она позволяет, выдавать за вероятность — нет.
                "score": round(float(min(area * 20, 1.0)), 3),
                "landmarks": [
                    [round(float(x) / width, 4), round(float(y) / height, 4), 0.0]
                    for x, y in points
                ],
                "world": [[round(float(v), 4) for v in point] for point in metric],
                "bbox": [round(float(v), 1) for v in box],
            }
        )
    return hands


@app.get("/health")
def health() -> dict:
    return {"ready": "pipe" in state, "model": state.get("model_id")}


@app.post("/pose")
def pose(request: PoseRequest) -> dict:
    started = time.perf_counter()
    frames = []
    for index, encoded in enumerate(request.frames):
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        moment = request.times[index] if index < len(request.times) else float(index)
        frames.append({"t": round(moment, 3), "hands": pose_of(image, request.max_hands)})
    return {
        "frames": frames,
        "frames_with_hands": sum(1 for frame in frames if frame["hands"]),
        "model": state["model_id"],
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8106)
    parser.add_argument("--full-precision", action="store_true")
    args = parser.parse_args()

    load(args.device, not args.full_precision)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
