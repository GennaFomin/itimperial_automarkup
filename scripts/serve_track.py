#!/usr/bin/env python3
"""Отслеживание предмета в руках через SAM2.

Зачем. Языковая модель называет предмет по общему впечатлению от сцены и часто выбирает
самый вероятный по языку, а не тот, что в руках. Детектор с открытым словарём эту задачу
не решил: на смазанной съёмке он находит «руку» и «еду». SAM2 отвечает на третий вопрос —
не «что это», а «какая связная область движется вместе с руками», — и на него можно
ответить без всякого словаря.

Как устроено. На середине сегмента SAM2 предлагает маски всех объектов, каждая
протягивается по кадрам сегмента, и выбирается та, что смещается сильнее прочих: именно
её человек и перемещает. Дальше её рамка отдаётся наружу — по ней можно вырезать предмет
крупным планом и спросить название уже у него, а не у всего кадра.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 HF_HOME=~/praxis/hf \\
        ~/praxis/venv/bin/python serve_track.py --port 8103
"""

from __future__ import annotations

import argparse
import base64
import io
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel


class Segment(BaseModel):
    id: int
    frames: list[str]  # JPEG в base64, по порядку


class TrackRequest(BaseModel):
    segments: list[Segment]
    grid: int = 3  # сетка точек-подсказок по кадру
    min_area: float = 0.005  # доля кадра, ниже которой маска считается мусором
    max_area: float = 0.5


app = FastAPI(title="Praxis SAM2 tracker")
state: dict = {}


def load(model_id: str, device: str) -> None:
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    started = time.perf_counter()
    print(f"загружаю {model_id} на {device}…", flush=True)
    state["processor"] = Sam2VideoProcessor.from_pretrained(model_id)
    state["model"] = Sam2VideoModel.from_pretrained(model_id).to(device).eval()
    state["device"] = device
    state["model_id"] = model_id
    print(f"готово за {time.perf_counter() - started:.1f} с", flush=True)


def centre_of(mask: np.ndarray) -> tuple[float, float]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return 0.0, 0.0
    return float(columns.mean()), float(rows.mean())


@app.get("/health")
def health() -> dict:
    return {"ready": "model" in state, "model": state.get("model_id")}


@app.post("/track")
@torch.inference_mode()
def track(request: TrackRequest) -> dict:
    model, processor, device = state["model"], state["processor"], state["device"]
    started = time.perf_counter()
    results = []

    for segment in request.segments:
        frames = [
            Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB")
            for frame in segment.frames
        ]
        if len(frames) < 2:
            results.append({"id": segment.id, "box": None, "shift": 0.0})
            continue

        width, height = frames[0].size
        # Подсказки сеткой: без детектора мы не знаем, где предмет, поэтому предлагаем
        # SAM2 равномерные точки и смотрим, какая из полученных областей поедет.
        points = [
            [[[float(width * (x + 0.5) / request.grid), float(height * (y + 0.5) / request.grid)]]]
            for y in range(request.grid)
            for x in range(request.grid)
        ]

        session = processor.init_video_session(video=frames, inference_device=device)
        # Все точки-подсказки добавляются одним вызовом: сессия ведёт общий учёт объектов,
        # и по одному их регистрировать нельзя — протяжка не находит опорную память.
        flat_points = [[[point[0][0][0], point[0][0][1]]] for point in points]
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=list(range(len(points))),
            input_points=[flat_points],
            input_labels=[[[1]] * len(points)],
        )

        tracks: dict[int, list] = {index: [] for index in range(len(points))}
        for output in model.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=0
        ):
            masks = processor.post_process_masks(
                [output.pred_masks], original_sizes=[[height, width]], binarize=True
            )[0]
            for index in range(min(len(points), len(masks))):
                tracks[index].append(masks[index].squeeze().cpu().numpy() > 0)

        best_index, best_shift, best_mask = None, 0.0, None
        for index, masks in tracks.items():
            if len(masks) < 2:
                continue
            areas = [float(mask.mean()) for mask in masks]
            if not (request.min_area <= np.mean(areas) <= request.max_area):
                continue
            first, last = centre_of(masks[0]), centre_of(masks[-1])
            shift = float(np.hypot(last[0] - first[0], last[1] - first[1])) / width
            if shift > best_shift:
                best_index, best_shift, best_mask = index, shift, masks[len(masks) // 2]

        box = None
        if best_mask is not None and best_mask.any():
            rows, columns = np.nonzero(best_mask)
            box = [
                float(columns.min() / width),
                float(rows.min() / height),
                float((columns.max() - columns.min()) / width),
                float((rows.max() - rows.min()) / height),
            ]

        results.append({"id": segment.id, "box": box, "shift": round(best_shift, 4)})

    return {
        "results": results,
        "model": state["model_id"],
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8103)
    args = parser.parse_args()

    load(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
