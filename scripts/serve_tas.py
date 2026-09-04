#!/usr/bin/env python3
"""Обучаемый детектор границ действий — MS-TCN, переделанный под открытый словарь.

Зачем так, а не как в статьях. MS-TCN, ASFormer и прочие обучаются предсказывать класс
каждого кадра, и их веса намертво привязаны к таксономии обучающего набора: модель с
50 Salads знает свои семнадцать классов и никакие другие. У нас списка классов нет
вовсе — значит классифицирующая голова непригодна.

Но сама граница класс-агностична. Поэтому здесь та же многостадийная свёрточная
архитектура с растущей дилатацией, что и в MS-TCN, но выход один: вероятность того, что
в этом моменте одно действие сменилось другим. Такую голову можно обучить на любом
размеченном наборе и применять к любому домену — она учится тому, как выглядит смена
действия, а не тому, какие действия бывают.

Обучающая цель размазывается по нескольким кадрам вокруг эталонной границы: точную
позицию человек и сам ставит с разбросом в полсекунды, требовать от модели точности выше
разметки бессмысленно.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 \\
        ~/praxis/venv/bin/python serve_tas.py --port 8104
"""

from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class Sample(BaseModel):
    features: str  # float16 матрица (кадры, признаки) в base64
    frames: int
    dim: int
    boundaries: list[float] | None = None  # позиции границ в кадрах сетки признаков


class TrainRequest(BaseModel):
    samples: list[Sample]
    epochs: int = 60
    tolerance: int = 2  # на сколько кадров размазать цель вокруг границы


class PredictRequest(BaseModel):
    samples: list[Sample]


class DilatedBlock(nn.Module):
    """Остаточный блок с дилатацией — основной кирпич MS-TCN."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv(x))
        return x + self.drop(self.mix(y))


class BoundaryStage(nn.Module):
    """Одна стадия: сжать вход, десять блоков с дилатацией 1,2,4,…, выдать логит на кадр."""

    def __init__(self, in_channels: int, channels: int = 64, layers: int = 10) -> None:
        super().__init__()
        self.entry = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.ModuleList([DilatedBlock(channels, 2**i) for i in range(layers)])
        self.exit = nn.Conv1d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.entry(x)
        for block in self.blocks:
            y = block(y)
        return self.exit(y)


class BoundaryNet(nn.Module):
    """Две стадии: первая предсказывает, вторая исправляет — как в MS-TCN."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.first = BoundaryStage(dim)
        self.second = BoundaryStage(dim + 1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        one = self.first(x)
        two = self.second(torch.cat([x, torch.sigmoid(one)], dim=1))
        return [one, two]


app = FastAPI(title="Praxis boundary head")
state: dict = {}


def decode(sample: Sample) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(sample.features), dtype=np.float16)
    return raw.reshape(sample.frames, sample.dim).astype(np.float32)


def target_of(sample: Sample, tolerance: int) -> np.ndarray:
    """Цель: единицы вокруг каждой границы, размазанные на tolerance кадров."""
    target = np.zeros(sample.frames, dtype=np.float32)
    for position in sample.boundaries or []:
        centre = int(round(position))
        for offset in range(-tolerance, tolerance + 1):
            index = centre + offset
            if 0 <= index < sample.frames:
                target[index] = max(target[index], 1.0 - abs(offset) / (tolerance + 1))
    return target


@app.get("/health")
def health() -> dict:
    return {"ready": "model" in state, "model": state.get("name", "boundary-net")}


@app.post("/train")
def train(request: TrainRequest) -> dict:
    device = state["device"]
    started = time.perf_counter()
    data = [(decode(s), target_of(s, request.tolerance)) for s in request.samples]
    dim = data[0][0].shape[1]

    model = BoundaryNet(dim).to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-4)
    # Границ на порядок меньше, чем обычных кадров: без веса модель научится молчать.
    positives = sum(float(t.sum()) for _, t in data)
    negatives = sum(len(t) for _, t in data) - positives
    weight = torch.tensor(max(1.0, negatives / max(positives, 1.0)), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)

    history = []
    for epoch in range(request.epochs):
        total = 0.0
        for features, target in data:
            x = torch.tensor(features.T, device=device).unsqueeze(0)
            y = torch.tensor(target, device=device).view(1, 1, -1)
            optimiser.zero_grad()
            loss = sum(loss_fn(stage, y) for stage in model(x))
            loss.backward()
            optimiser.step()
            total += float(loss)
        history.append(round(total / len(data), 4))

    state["model"] = model.eval()
    state["dim"] = dim
    # Веса сразу на диск: обучение идёт по HTTP, и клиент может отвалиться вместе с
    # ноутбуком. Модель в памяти сервиса переживёт разрыв соединения, но не перезапуск
    # самого сервиса — а переобучать полчаса из-за этого незачем.
    path = Path(state["checkpoint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dim": dim, "weights": model.state_dict()}, path)

    return {
        "saved_to": str(path),
        "trained_on": len(data),
        "dim": dim,
        "loss_first": history[0],
        "loss_last": history[-1],
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }


@app.post("/predict")
@torch.inference_mode()
def predict(request: PredictRequest) -> dict:
    if "model" not in state:
        return {"error": "модель не обучена"}
    device = state["device"]
    scores = []
    for sample in request.samples:
        features = decode(sample)
        x = torch.tensor(features.T, device=device).unsqueeze(0)
        probability = torch.sigmoid(state["model"](x)[-1]).squeeze().cpu().numpy()
        scores.append([round(float(value), 4) for value in np.atleast_1d(probability)])
    return {"scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8104)
    parser.add_argument("--checkpoint", default="checkpoints/boundary.pt")
    args = parser.parse_args()
    state["device"] = args.device
    state["checkpoint"] = args.checkpoint
    # Если веса уже есть на диске — поднимаемся сразу обученными.
    saved = Path(args.checkpoint)
    if saved.exists():
        payload = torch.load(saved, map_location=args.device)
        model = BoundaryNet(payload["dim"]).to(args.device)
        model.load_state_dict(payload["weights"])
        state["model"] = model.eval()
        state["dim"] = payload["dim"]
        print(f"подняты сохранённые веса из {saved}", flush=True)
    print(f"детектор границ готов на {args.device}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
