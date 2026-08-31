#!/usr/bin/env python3
"""Классификатор шагов по закрытому словарю на SigLIP2.

Словарь закрыт: 202 допустимые пары (действие, объект). Значит это задача классификации,
а не свободной генерации — и решать её моделью «опиши, что видишь» расточительно и
неточно. Здесь мы один раз считаем текстовые эмбеддинги всех пар, а дальше на каждый
сегмент нужно лишь закодировать несколько кадров и взять ближайшую пару.

Три следствия: это быстро (кадры кодируются пачкой, текст посчитан заранее), это никогда
не выдаёт значение вне словаря, и уверенность получается настоящей — из распределения по
классам, а не из самооценки языковой модели.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 HF_HOME=~/praxis/hf \\
        ~/praxis/venv/bin/python serve_clip.py --port 8101
"""

from __future__ import annotations

import argparse
import base64
import io
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

# Домен вписан в каждый шаблон: в абляциях это даёт больше, чем вся временная механика,
# и стоит ноль. Ансамбль формулировок устойчивее любой одной.
VERB_TEMPLATES = [
    "a person {gerund} a part in toy vehicle assembly",
    "hands {gerund} a plastic component of a toy vehicle",
    "someone is {gerund} a piece while assembling a toy vehicle",
    "a video of {gerund} during toy vehicle assembly",
]

NOUN_TEMPLATES = [
    "the {noun} of a toy vehicle",
    "a photo of the {noun}, a toy vehicle part",
    "hands holding the {noun} of a toy model",
    "close-up of a plastic {noun} in toy vehicle assembly",
]

# Ансамбль шаблонов: усреднение по нескольким формулировкам заметно устойчивее одной.
TEMPLATES = [
    "a photo of a person {gerund} a {noun}",
    "a person is {gerund} the {noun} of a toy vehicle",
    "hands {gerund} a {noun}",
    "close-up of {gerund} a {noun} during assembly",
]


class Segment(BaseModel):
    id: int
    frames: list[str]


class Request(BaseModel):
    segments: list[Segment]
    pairs: list[list[str]]  # [[действие, объект], ...]
    top_k: int = 3
    # pair — одна фраза на всю пару; factored — отдельные классификаторы глагола и
    # объекта, связанные матрицей допустимых пар. Второе устойчивее к тому, что одна
    # неудачная формулировка перетягивает на себя всю сцену.
    mode: str = "factored"
    verb_weight: float = 1.0
    noun_weight: float = 1.0


app = FastAPI(title="Praxis SigLIP")
state: dict = {}


def gerund(verb: str) -> str:
    """attach → attaching, remove → removing, attempt to attach → attempting to attach."""
    head, _, tail = verb.partition(" ")
    if head.endswith("e") and not head.endswith("ee"):
        head = head[:-1]
    head = head + "ing"
    return f"{head} {tail}".strip()


def load(model_id: str, device: str) -> None:
    from transformers import AutoModel, AutoProcessor

    started = time.perf_counter()
    print(f"загружаю {model_id} на {device}…", flush=True)
    state["processor"] = AutoProcessor.from_pretrained(model_id)
    state["model"] = AutoModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
    state["device"] = device
    state["model_id"] = model_id
    state["text_cache"] = {}
    print(f"готово за {time.perf_counter() - started:.1f} с", flush=True)


def _tensor(output) -> torch.Tensor:
    """В transformers 5.x get_*_features отдаёт объект, а не тензор."""
    for attribute in ("pooler_output", "last_hidden_state"):
        value = getattr(output, attribute, None)
        if isinstance(value, torch.Tensor):
            return value
    return output


@torch.inference_mode()
def text_embeddings(pairs: list[tuple[str, str]]) -> torch.Tensor:
    """Эмбеддинги всех пар словаря. Считаются один раз на словарь и кэшируются.

    Каждая пара описывается ансамблем формулировок, их эмбеддинги усредняются: одна
    формулировка слишком зависит от случайных слов, ансамбль заметно устойчивее.
    """
    key = hash(tuple(pairs))
    if key in state["text_cache"]:
        return state["text_cache"][key]

    model, processor, device = state["model"], state["processor"], state["device"]
    phrases = [
        template.format(gerund=gerund(action), noun=noun or "part")
        for action, noun in pairs
        for template in TEMPLATES
    ]

    chunks = []
    for start in range(0, len(phrases), 256):
        inputs = processor(
            text=phrases[start : start + 256],
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(device)
        features = _tensor(model.get_text_features(**inputs)).float()
        chunks.append(features / features.norm(dim=-1, keepdim=True))

    stacked = torch.cat(chunks).reshape(len(pairs), len(TEMPLATES), -1).mean(dim=1)
    matrix = stacked / stacked.norm(dim=-1, keepdim=True)
    state["text_cache"][key] = matrix
    return matrix


@torch.inference_mode()
def phrase_embeddings(phrases: list[str], key: str) -> torch.Tensor:
    """Эмбеддинги произвольного набора фраз с кэшем по ключу."""
    if key in state["text_cache"]:
        return state["text_cache"][key]

    model, processor, device = state["model"], state["processor"], state["device"]
    chunks = []
    for start in range(0, len(phrases), 256):
        inputs = processor(
            text=phrases[start : start + 256],
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(device)
        features = _tensor(model.get_text_features(**inputs)).float()
        chunks.append(features / features.norm(dim=-1, keepdim=True))

    matrix = torch.cat(chunks)
    state["text_cache"][key] = matrix
    return matrix


def factored_embeddings(pairs: list[tuple[str, str]]) -> tuple:
    """Отдельные наборы фраз для глаголов и для объектов плюс их индексы в парах."""
    verbs = sorted({action for action, _ in pairs})
    nouns = sorted({noun for _, noun in pairs if noun})

    verb_matrix = phrase_embeddings(
        [template.format(gerund=gerund(verb)) for verb in verbs for template in VERB_TEMPLATES],
        f"verbs:{hash(tuple(verbs))}",
    ).reshape(len(verbs), len(VERB_TEMPLATES), -1).mean(dim=1)
    noun_matrix = phrase_embeddings(
        [template.format(noun=noun) for noun in nouns for template in NOUN_TEMPLATES],
        f"nouns:{hash(tuple(nouns))}",
    ).reshape(len(nouns), len(NOUN_TEMPLATES), -1).mean(dim=1)

    verb_matrix = verb_matrix / verb_matrix.norm(dim=-1, keepdim=True)
    noun_matrix = noun_matrix / noun_matrix.norm(dim=-1, keepdim=True)
    verb_index = {verb: i for i, verb in enumerate(verbs)}
    noun_index = {noun: i for i, noun in enumerate(nouns)}
    return verb_matrix, noun_matrix, verb_index, noun_index


@torch.inference_mode()
def image_embedding(frames: list[Image.Image]) -> torch.Tensor:
    """Один вектор на сегмент: кадры кодируются пачкой и усредняются."""
    model, processor, device = state["model"], state["processor"], state["device"]
    inputs = processor(images=frames, return_tensors="pt").to(device)
    features = _tensor(model.get_image_features(**inputs)).float()
    features = features / features.norm(dim=-1, keepdim=True)
    pooled = features.mean(dim=0)
    return pooled / pooled.norm()


@app.get("/health")
def health() -> dict:
    return {"ready": "model" in state, "model": state.get("model_id")}


@app.post("/classify")
def classify(request: Request) -> dict:
    started = time.perf_counter()
    pairs = [(pair[0], pair[1] if len(pair) > 1 else "") for pair in request.pairs]
    texts = text_embeddings(pairs)

    factored = factored_embeddings(pairs) if request.mode == "factored" else None

    results = []
    for segment in request.segments:
        frames = [
            Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB")
            for frame in segment.frames
        ]
        vector = image_embedding(frames)

        if factored is not None:
            verb_matrix, noun_matrix, verb_index, noun_index = factored
            verb_scores = torch.log_softmax((verb_matrix @ vector) * 100.0, dim=0)
            noun_scores = torch.log_softmax((noun_matrix @ vector) * 100.0, dim=0)
            scores = torch.tensor(
                [
                    request.verb_weight * float(verb_scores[verb_index[action]])
                    + request.noun_weight * float(noun_scores[noun_index[noun]])
                    if noun in noun_index
                    else -1e9
                    for action, noun in pairs
                ],
                device=vector.device,
            )
            probabilities = torch.softmax(scores, dim=0)
            # Отдельные победители по глаголу и по объекту — для диагностики: сразу видно,
            # какая из двух половин задачи не работает.
            verbs_sorted = sorted(verb_index, key=lambda v: -float(verb_scores[verb_index[v]]))
            nouns_sorted = sorted(noun_index, key=lambda n: -float(noun_scores[noun_index[n]]))
            parts = {"best_verb": verbs_sorted[0], "best_noun": nouns_sorted[0]}
        else:
            parts = {}
            scores = texts @ vector
            # Температура подобрана так, чтобы распределение не было ни плоским, ни
            # вырожденным.
            probabilities = torch.softmax(scores * 100.0, dim=0)

        order = torch.argsort(probabilities, descending=True)[: request.top_k]
        best = int(order[0])
        results.append(
            {
                "id": segment.id,
                "action": pairs[best][0],
                "object": pairs[best][1] or None,
                "confidence": round(float(probabilities[best]), 3),
                **parts,
                "top": [
                    {
                        "action": pairs[int(index)][0],
                        "object": pairs[int(index)][1] or None,
                        "score": round(float(probabilities[int(index)]), 3),
                    }
                    for index in order
                ],
            }
        )

    return {
        "results": results,
        "model": state["model_id"],
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


class EmbedRequest(BaseModel):
    """Ролик целиком: декодировать и кодировать кадры дешевле здесь, чем гнать их по сети."""

    video: str  # mp4 в base64
    fps: float = 10.0
    max_frames: int = 400


@app.post("/embed")
def embed(request: EmbedRequest) -> dict:
    """Покадровые эмбеддинги визуального энкодера.

    Это замена разнице соседних пикселей: сегментатору нужны признаки, в которых «человек
    держит крышку» и «человек закручивает винт» далеки друг от друга, а два кадра одного
    действия — близки. Разница яркостей такого не даёт в принципе, эмбеддинги дают.
    """
    import numpy as np

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "clip.mp4"
        source.write_bytes(base64.b64decode(request.video))
        pattern = Path(directory) / "f_%04d.jpg"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(source),
                "-vf", f"fps={request.fps},scale=384:-2",
                "-frames:v", str(request.max_frames), "-q:v", "4", str(pattern),
            ],
            check=True,
        )
        paths = sorted(Path(directory).glob("f_*.jpg"))
        if not paths:
            return {"embeddings": "", "count": 0, "dim": 0}

        model, processor, device = state["model"], state["processor"], state["device"]
        chunks = []
        for start in range(0, len(paths), 64):
            images = [Image.open(path).convert("RGB") for path in paths[start : start + 64]]
            with torch.inference_mode():
                inputs = processor(images=images, return_tensors="pt").to(device)
                features = _tensor(model.get_image_features(**inputs)).float()
                features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.cpu().numpy().astype(np.float16))

    matrix = np.concatenate(chunks)
    return {
        "embeddings": base64.b64encode(matrix.tobytes()).decode(),
        "count": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "fps": request.fps,
        "model": state["model_id"],
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


class ScoreRequest(BaseModel):
    """Сырые скоры без всякой калибровки: она делается на клиенте по всему батчу."""

    segments: list[Segment]
    verbs: list[str]
    nouns: list[str]


@app.post("/scores")
def scores(request: ScoreRequest) -> dict:
    """Матрицы сходства кадров с фразами глаголов и объектов.

    Калибровка здесь намеренно не делается: она трансдуктивная — вычитание среднего по
    столбцу имеет смысл только когда виден весь набор сегментов, а сервер видит по одному
    ролику. Поэтому наружу отдаются сырые числа, а вся арифметика живёт в клиенте, где
    можно перебирать варианты без повторного прогона модели.

    Возвращаются два варианта агрегации по кадрам: среднее и максимум. Максимум устойчивее,
    когда деталь видна только в части кадров сегмента.
    """
    started = time.perf_counter()
    verb_matrix = phrase_embeddings(
        [t.format(gerund=gerund(v)) for v in request.verbs for t in VERB_TEMPLATES],
        f"verbs:{hash(tuple(request.verbs))}",
    ).reshape(len(request.verbs), len(VERB_TEMPLATES), -1).mean(dim=1)
    noun_matrix = phrase_embeddings(
        [t.format(noun=n) for n in request.nouns for t in NOUN_TEMPLATES],
        f"nouns:{hash(tuple(request.nouns))}",
    ).reshape(len(request.nouns), len(NOUN_TEMPLATES), -1).mean(dim=1)
    verb_matrix = verb_matrix / verb_matrix.norm(dim=-1, keepdim=True)
    noun_matrix = noun_matrix / noun_matrix.norm(dim=-1, keepdim=True)

    model, processor, device = state["model"], state["processor"], state["device"]
    results = []
    for segment in request.segments:
        frames = [
            Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB")
            for frame in segment.frames
        ]
        with torch.inference_mode():
            inputs = processor(images=frames, return_tensors="pt").to(device)
            features = _tensor(model.get_image_features(**inputs)).float()
            features = features / features.norm(dim=-1, keepdim=True)

            per_frame_verb = features @ verb_matrix.T  # (кадры, глаголы)
            per_frame_noun = features @ noun_matrix.T
            pooled = features.mean(dim=0)
            pooled = pooled / pooled.norm()

        results.append(
            {
                "id": segment.id,
                "verb_mean": (verb_matrix @ pooled).tolist(),
                "verb_max": per_frame_verb.max(dim=0).values.tolist(),
                "noun_mean": (noun_matrix @ pooled).tolist(),
                "noun_max": per_frame_noun.max(dim=0).values.tolist(),
                "verb_first_last": (verb_matrix @ (features[-1] - features[0])).tolist(),
            }
        )

    return {
        "results": results,
        "model": state["model_id"],
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    args = parser.parse_args()

    load(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
