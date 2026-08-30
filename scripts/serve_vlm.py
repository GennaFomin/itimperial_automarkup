#!/usr/bin/env python3
"""Сервис именования шагов: крутится на GPU-машине, отвечает по HTTP.

Намеренно самодостаточный — не импортирует praxis, чтобы жить в изолированном venv на
DL5 рядом с весами. Получает уже нарезанные сегменты в виде кадров и возвращает для
каждого пару (действие, объект) из закрытого словаря. Границы модель не двигает: их
ставит физика, а языковая модель делает то, что умеет хорошо, — называет увиденное.

    CUDA_VISIBLE_DEVICES=2 HF_HOME=~/praxis/hf ~/praxis/venv/bin/python serve_vlm.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

PROMPT = """Ты размечаешь видео сборки и разборки игрушечных машин.

На кадрах — один фрагмент видео. Определи, какое действие выполняет человек и с каким
объектом он это делает.

Допустимые действия: {actions}
Допустимые объекты: {objects}

Ответь строго одним объектом JSON без пояснений:
{{"action": "<одно из допустимых действий>", "object": "<один из допустимых объектов>", "confidence": <число от 0 до 1>}}"""


class Segment(BaseModel):
    id: int
    frames: list[str]  # JPEG в base64


class Request(BaseModel):
    segments: list[Segment]
    actions: list[str]
    objects: list[str]
    pairs: dict[str, list[str]] | None = None


app = FastAPI(title="Praxis VLM")
state: dict = {}


def load(model_id: str, device: str) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    started = time.perf_counter()
    print(f"загружаю {model_id} на {device}…", flush=True)
    state["processor"] = AutoProcessor.from_pretrained(model_id)
    state["model"] = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device
    ).eval()
    state["model_id"] = model_id
    print(f"готово за {time.perf_counter() - started:.1f} с", flush=True)


def decode(frame: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB")


def closest(value: str, allowed: list[str]) -> str | None:
    """Притягиваем ответ модели к словарю: точное совпадение, затем подстрока."""
    if not value:
        return None
    lowered = value.strip().lower()
    exact = {item.lower(): item for item in allowed}
    if lowered in exact:
        return exact[lowered]
    contained = [item for item in allowed if item.lower() in lowered or lowered in item.lower()]
    return min(contained, key=len) if contained else None


def parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


@app.get("/health")
def health() -> dict:
    return {"ready": "model" in state, "model": state.get("model_id")}


@app.post("/annotate")
def annotate(request: Request) -> dict:
    model, processor = state["model"], state["processor"]
    prompt = PROMPT.format(actions=", ".join(request.actions), objects=", ".join(request.objects))
    started = time.perf_counter()
    results = []

    for segment in request.segments:
        images = [decode(frame) for frame in segment.frames]
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": prompt}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        text = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]

        answer = parse(text)
        action = closest(str(answer.get("action", "")), request.actions)
        objects = (
            request.pairs.get(action, request.objects)
            if (request.pairs and action)
            else request.objects
        )
        obj = closest(str(answer.get("object", "")), objects)

        # Уверенность модели корректируем на то, попали ли мы в словарь без натяжек.
        reported = answer.get("confidence")
        confidence = float(reported) if isinstance(reported, (int, float)) else 0.5
        if action is None:
            confidence = min(confidence, 0.2)
        if obj is None:
            confidence = min(confidence, 0.35)

        results.append(
            {
                "id": segment.id,
                "action": action,
                "object": obj,
                "confidence": round(max(0.0, min(confidence, 1.0)), 3),
                "raw": text.strip()[:200],
            }
        )

    return {
        "results": results,
        "model": state["model_id"],
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    load(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
