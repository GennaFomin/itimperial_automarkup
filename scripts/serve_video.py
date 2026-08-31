#!/usr/bin/env python3
"""Признаки движения: видеоэнкодер вместо покадровых картинок.

Зачем отдельный сервис. Для границ действий покадровые признаки не годятся принципиально:
они описывают, ЧТО в кадре, а не что происходит. Кадр «рука держит крышку» и кадр «рука
закручивает крышку» почти одинаковы, а действия разные. Видеоэнкодер кодирует окно кадров
вместе с движением внутри него — это тот класс признаков, на котором построены работы по
temporal action segmentation.

Ролик режется скользящим окном (по умолчанию 16 кадров с шагом 4), каждое окно кодируется
в один вектор, на выходе — последовательность векторов с шагом в четверть секунды.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HOME=~/praxis/hf \\
        ~/praxis/venv/bin/python serve_video.py --port 8102
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class EmbedRequest(BaseModel):
    video: str  # mp4 в base64
    fps: float = 16.0  # с какой частотой декодировать кадры
    window: int = 16  # сколько кадров в одном окне
    stride: int = 4  # на сколько кадров сдвигать окно


app = FastAPI(title="Praxis video features")
state: dict = {}

# Что чем считать. Литература по temporal action segmentation стоит на признаках,
# обученных на Kinetics (I3D и его наследники) — их и берём для сравнения, плюс
# современные self-supervised видеоэнкодеры.
BACKENDS = {
    "vjepa2": ("hf-video", "facebook/vjepa2-vitl-fpc64-256"),
    "videomae": ("hf-video", "MCG-NJU/videomae-large"),
    "videomae-kinetics": ("hf-video", "MCG-NJU/videomae-base-finetuned-kinetics"),
    "timesformer": ("hf-video", "facebook/timesformer-base-finetuned-k400"),
    "dinov2": ("hf-image", "facebook/dinov2-large"),
    "swin3d": ("torchvision", "swin3d_b"),
    "mvit": ("torchvision", "mvit_v2_s"),
    "r2plus1d": ("torchvision", "r2plus1d_18"),
}


def load(name: str, device: str) -> None:
    kind, model_id = BACKENDS.get(name, ("hf-video", name))
    started = time.perf_counter()
    print(f"загружаю {model_id} ({kind}) на {device}…", flush=True)

    if kind == "torchvision":
        import torchvision.models.video as tv

        builder = getattr(tv, model_id)
        model = builder(weights="DEFAULT").to(device).eval()
        # Классификатор нам не нужен — берём предпоследний слой.
        for attribute in ("head", "fc"):
            if hasattr(model, attribute):
                setattr(model, attribute, torch.nn.Identity())
        state["model"] = model
        state["mean"] = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 3, 1, 1, 1)
        state["std"] = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1, 1)
    elif kind == "hf-image":
        from transformers import AutoImageProcessor, AutoModel

        state["processor"] = AutoImageProcessor.from_pretrained(model_id)
        state["model"] = (
            AutoModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
        )
    else:
        from transformers import AutoModel, AutoVideoProcessor

        state["processor"] = AutoVideoProcessor.from_pretrained(model_id)
        state["model"] = (
            AutoModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
        )

    state["kind"] = kind
    state["device"] = device
    state["model_id"] = f"{name} ({model_id})"
    print(f"готово за {time.perf_counter() - started:.1f} с", flush=True)


def encode_windows(windows: list, device: str) -> torch.Tensor:
    """Один вектор на окно кадров. Разные семейства моделей вызываются по-разному."""
    kind = state["kind"]
    model = state["model"]

    if kind == "torchvision":
        batch = torch.tensor(np.stack(windows), device=device).float() / 255.0
        batch = batch.permute(0, 4, 1, 2, 3)  # (окна, каналы, кадры, высота, ширина)
        batch = (batch - state["mean"]) / state["std"]
        output = model(batch)
        return output.float() if output.ndim == 2 else output.flatten(1).float()

    processor = state["processor"]
    if kind == "hf-image":
        # Покадровая модель: усредняем векторы кадров окна.
        images = [frame for window in windows for frame in window]
        inputs = processor(images=images, return_tensors="pt").to(device, dtype=torch.float16)
        hidden = model(**inputs).last_hidden_state[:, 0]
        return hidden.reshape(len(windows), -1, hidden.shape[-1]).mean(dim=1).float()

    inputs = processor(windows, return_tensors="pt").to(device, dtype=torch.float16)
    output = model.get_vision_features(**inputs) if hasattr(model, "get_vision_features") else (
        model(**inputs).last_hidden_state
    )
    return output.mean(dim=1).float()


def decode(path: Path, fps: float, size: int = 256) -> np.ndarray:
    """Кадры ролика как массив (кадры, высота, ширина, 3)."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps},scale={size}:{size}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    count = raw.size // (size * size * 3)
    return raw[: count * size * size * 3].reshape(count, size, size, 3)


@app.get("/health")
def health() -> dict:
    return {"ready": "model" in state, "model": state.get("model_id")}


@app.post("/embed")
@torch.inference_mode()
def embed(request: EmbedRequest) -> dict:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "clip.mp4"
        source.write_bytes(base64.b64decode(request.video))
        frames = decode(source, request.fps)

    if len(frames) < request.window:
        return {"embeddings": "", "count": 0, "dim": 0}

    device = state["device"]
    starts = list(range(0, len(frames) - request.window + 1, request.stride))

    vectors = []
    for chunk_start in range(0, len(starts), 8):
        windows = [
            list(frames[start : start + request.window])
            for start in starts[chunk_start : chunk_start + 8]
        ]
        pooled = encode_windows(windows, device)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        vectors.append(pooled.cpu().numpy().astype(np.float16))

    matrix = np.concatenate(vectors)
    # Частота полученной последовательности: одно окно на stride кадров исходной сетки.
    return {
        "embeddings": base64.b64encode(matrix.tobytes()).decode(),
        "count": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "fps": request.fps / request.stride,
        "offset_sec": request.window / (2 * request.fps),
        "model": state["model_id"],
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="vjepa2", help=f"одно из: {', '.join(BACKENDS)} или путь на HF"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    args = parser.parse_args()

    load(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
