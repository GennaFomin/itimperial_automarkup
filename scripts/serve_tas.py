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
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def checkpoint_path(base: str, name: str) -> Path:
    """Именованный вариант лежит рядом с основным чекпоинтом и никогда его не трогает."""
    path = Path(base)
    if not name:
        return path
    return path.with_name(f"{path.stem}-{name}{path.suffix}")


class Sample(BaseModel):
    features: str  # float16 матрица (кадры, признаки) в base64
    frames: int
    dim: int
    boundaries: list[float] | None = None  # позиции границ в кадрах сетки признаков


class TrainRequest(BaseModel):
    samples: list[Sample]
    epochs: int = 60
    # Вес сглаживающей потери. 0.15 — значение из статьи, там же показано, что 0.05
    # слишком слабо, а 0.25 уже размывает границы.
    smoothing: float = 0.15
    stages: int = 4
    tolerance: int = 2  # на сколько кадров размазать цель вокруг границы
    name: str = ""          # имя варианта: чекпоинт boundary-<name>.pt
    activate: bool = True   # подменять ли живую модель обученной
    # Временная аугментация C2F-TCN: стохастический max-pool по времени — та же
    # последовательность на другой скорости. Окно base_window с p=0.5, иначе
    # равномерно из [base_window*pool_low, base_window*pool_high].
    augment: bool = False
    base_window: int = 5
    pool_low: float = 0.5
    pool_high: float = 2.0


class PredictRequest(BaseModel):
    samples: list[Sample]
    tta: bool = False  # усреднить по временным масштабам 1..4 (C2F-TCN TTA)


class LoadRequest(BaseModel):
    name: str = ""
    # Несколько чекпоинтов сразу — усреднение вероятностей. Модели, обученные на
    # разных корпусах или с разной σ, ошибаются в разных местах; среднее надёжнее любой.
    names: list[str] = []


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
    """MS-TCN по статье Farha & Gall, CVPR 2019: четыре стадии по десять слоёв.

    От статьи отличается ровно одним — головой. Там на каждой стадии softmax по C
    классам действий, здесь один логит: вероятность, что в этом кадре действие
    сменилось. Причина в требовании кейса: списка классов нет и не будет, а
    классифицирующая голова привязана к таксономии обучающего набора. Граница же
    класс-агностична, поэтому голову можно обучить на любой разметке и применить к
    чужому домену. Всё остальное — число стадий, слоёв и фильтров, удвоение
    дилатации, остаточные связи, потеря на каждой стадии — как в статье.

    Каждая следующая стадия получает предсказание предыдущей и исправляет его: в
    статье это подаётся как основной приём против пересегментации.
    """

    def __init__(self, dim: int, stages: int = 4) -> None:
        super().__init__()
        self.first = BoundaryStage(dim)
        self.rest = nn.ModuleList([BoundaryStage(dim + 1) for _ in range(stages - 1)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = [self.first(x)]
        for stage in self.rest:
            out.append(stage(torch.cat([x, torch.sigmoid(out[-1])], dim=1)))
        return out


def temporal_pool(features: torch.Tensor, target: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Стохастический max-pool по времени (C2F-TCN). Цель пулится максимумом, поэтому
    граница не теряется, а лишь переезжает на новую сетку."""
    if window <= 1:
        return features, target
    length = (features.shape[0] // window) * window
    if length == 0:
        return features, target
    pooled = features[:length].view(-1, window, features.shape[1]).amax(dim=1)
    squeezed = target[:length].view(-1, window).amax(dim=1)
    return pooled, squeezed


def smoothing_loss(logits: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    """Усечённая среднеквадратичная ошибка по логарифмам вероятностей соседних кадров.

    Формулы 8–10 статьи. Она штрафует резкие скачки предсказания между соседними
    кадрами и введена авторами именно против пересегментации — нашей измеренной
    слабости: детектор ставил пятнадцать границ там, где их девять.

    У нас два исхода вместо C классов, поэтому распределение — (p, 1-p).
    """
    probability = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    both = torch.cat([probability, 1 - probability], dim=1).log()
    delta = (both[:, :, 1:] - both[:, :, :-1]).abs()
    return (delta.clamp(max=tau) ** 2).mean()


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


def load_checkpoint(path: Path, device: str) -> None:
    """Число стадий берём из чекпоинта. Файлы, записанные до перехода на четыре стадии,
    его не содержат: там вторая стадия называлась "second", по этому имени старый формат
    опознаётся и переименовывается — сервис не должен терять рабочую модель при обновлении."""
    payload = torch.load(path, map_location=device)
    weights = payload["weights"]
    if "stages" in payload:
        stages = int(payload["stages"])
    else:
        stages = 2 if any(k.startswith("second.") for k in weights) else 4
        if stages == 2:
            weights = {
                k.replace("second.", "rest.0.", 1) if k.startswith("second.") else k: v
                for k, v in weights.items()
            }
    model = BoundaryNet(payload["dim"], stages=stages).to(device)
    model.load_state_dict(weights)
    state["model"] = model.eval()
    state["dim"] = payload["dim"]
    state["stages"] = stages
    state["loaded"] = str(path)
    # Упакованный ансамбль: остальные участники лежат в том же файле (pack_ensemble.py),
    # чтобы развёртывание оставалось одним чекпоинтом.
    members = []
    for member in payload.get("members", []):
        net = BoundaryNet(payload["dim"], stages=int(member["stages"])).to(device)
        net.load_state_dict(member["weights"])
        members.append(net.eval())
    state["ensemble"] = members
    # Как сливать участников: "mean" — среднее вероятностей, "min" — согласие всех
    # (разрез только там, где каждый участник видит смену). Задаётся при упаковке.
    state["fusion"] = str(payload.get("fusion", "mean"))


@app.post("/load")
def load(request: LoadRequest) -> dict:
    names = request.names or [request.name]
    paths = [checkpoint_path(state["checkpoint"], n) for n in names]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise HTTPException(404, f"нет чекпоинта {missing}")
    load_checkpoint(paths[0], state["device"])
    extra = list(state["ensemble"])
    for path in paths[1:]:
        payload = torch.load(path, map_location=state["device"])
        if payload["dim"] != state["dim"]:
            raise HTTPException(400, f"{path}: размерность {payload['dim']} против {state['dim']}")
        model = BoundaryNet(payload["dim"], stages=int(payload.get("stages", 4))).to(state["device"])
        model.load_state_dict(payload["weights"])
        extra.append(model.eval())
    state["ensemble"] = extra
    state["loaded"] = "+".join(str(p) for p in paths)
    return {"loaded": state["loaded"], "dim": state["dim"], "stages": state["stages"], "models": 1 + len(extra)}


@app.get("/health")
def health() -> dict:
    return {"ready": "model" in state, "model": "boundary-net", "dim": state.get("dim"),
            "stages": state.get("stages"), "checkpoint": state.get("loaded"),
            "members": 1 + len(state.get("ensemble", [])) if "model" in state else 0,
            "fusion": state.get("fusion", "mean")}


@app.post("/train")
def train(request: TrainRequest) -> dict:
    device = state["device"]
    started = time.perf_counter()
    data = [(decode(s), target_of(s, request.tolerance)) for s in request.samples]
    dim = data[0][0].shape[1]

    model = BoundaryNet(dim, stages=request.stages).to(device).train()
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
            if request.augment:
                window = request.base_window if random.random() < 0.5 else random.randint(
                    max(1, int(request.base_window * request.pool_low)),
                    max(1, int(request.base_window * request.pool_high)),
                )
                f_t, t_t = temporal_pool(torch.tensor(features), torch.tensor(target), window)
                x = f_t.T.to(device).unsqueeze(0)
                y = t_t.to(device).view(1, 1, -1)
            else:
                x = torch.tensor(features.T, device=device).unsqueeze(0)
                y = torch.tensor(target, device=device).view(1, 1, -1)
            optimiser.zero_grad()
            # Потеря считается на каждой стадии — в статье это и делает многостадийность
            # осмысленной: без надзора на промежуточных выходах они ничему не учатся.
            loss = sum(
                loss_fn(stage, y) + request.smoothing * smoothing_loss(stage)
                for stage in model(x)
            )
            loss.backward()
            optimiser.step()
            total += float(loss)
        history.append(round(total / len(data), 4))

    # Веса сразу на диск: обучение идёт по HTTP, и клиент может отвалиться вместе с
    # ноутбуком. Именованный вариант пишется рядом и живую модель не трогает, если его
    # об этом не просили: эксперимент не должен подменять то, чем пользуется демо.
    path = checkpoint_path(state["checkpoint"], request.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dim": dim, "stages": request.stages, "weights": model.state_dict()}, path)
    if request.activate:
        state["ensemble"] = []
        state["model"] = model.eval()
        state["dim"] = dim
        state["stages"] = request.stages
        state["loaded"] = str(path)

    return {
        "saved_to": str(path),
        "trained_on": len(data),
        "dim": dim,
        "stages": request.stages,
        "loss_first": history[0],
        "loss_last": history[-1],
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }


@app.post("/predict")
@torch.inference_mode()
def predict(request: PredictRequest) -> dict:
    if "model" not in state:
        raise HTTPException(503, "модель ещё не обучена")
    for sample in request.samples:
        if sample.dim != state["dim"]:
            raise HTTPException(400, f"детектор ждёт {state['dim']} признаков, получено {sample.dim}")
    device, model = state["device"], state["model"]
    scores = []
    means = []
    for sample in request.samples:
        features = torch.tensor(decode(sample))
        windows = (1, 2, 3, 4) if request.tta else (1,)
        outputs = []
        spreads = []
        for window in windows:
            pooled, _ = temporal_pool(features, torch.zeros(len(features)), window)
            with torch.no_grad():
                members = [model, *state.get("ensemble", [])]
                stacked = torch.stack([
                    torch.sigmoid(m(pooled.T.to(device).unsqueeze(0))[-1])[0, 0].cpu() for m in members
                ])
                score = stacked.min(0).values if state.get("fusion") == "min" else stacked.mean(0)
                spread = stacked.mean(0)  # среднее по участникам — для поиска плато переходов
            score = score.repeat_interleave(window)[: len(features)]
            spread = spread.repeat_interleave(window)[: len(features)]
            if len(score) < len(features):
                score = torch.nn.functional.pad(score, (0, len(features) - len(score)), value=float(score[-1]))
                spread = torch.nn.functional.pad(spread, (0, len(features) - len(spread)), value=float(spread[-1]))
            outputs.append(score)
            spreads.append(spread)
        scores.append(torch.stack(outputs).mean(0).tolist())
        means.append(torch.stack(spreads).mean(0).tolist())
    # "mean" — среднее вероятностей участников независимо от правила слияния: по нему
    # ищутся плато переходов (пауза с движением), по "scores" — разрезы.
    return {"scores": scores, "mean": means}


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
        load_checkpoint(saved, args.device)
        print(f"подняты сохранённые веса из {saved}, стадий {state['stages']}", flush=True)
    print(f"детектор границ готов на {args.device}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
