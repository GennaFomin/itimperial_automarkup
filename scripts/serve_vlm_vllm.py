#!/usr/bin/env python3
"""Тот же сервис именования, но на движке vLLM вместо model.generate из transformers.

Зачем. Именование — самая дорогая стадия: 3.9 с на шаг, и на ролике из двенадцати шагов
это сорок семь секунд из ста двадцати, отпущенных кейсом. Причём шаги уходят в модель по
одному, и между запросами карта простаивает: у каждого прохода свой разогрев, а батча нет.

vLLM отвечает ровно на это — непрерывная пакетизация и paged attention. Все шаги ролика
подаются одним вызовом, движок сам складывает их в батч и держит карту занятой.

Промпт, разбор ответа и сборка результата берутся из serve_vlm.py импортом, а не копией:
сравнение должно мерить движок, а не разошедшиеся тексты.

Живёт в отдельном окружении: у vLLM свой torch, и ставить его в основной venv нельзя.

    /venv/vllm/bin/python scripts/serve_vlm_vllm.py --port 8100 \\
        --model Qwen/Qwen3-VL-8B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_vlm as base  # noqa: E402  — тот же промпт и тот же разбор ответа

app = FastAPI(title="Praxis VLM (vLLM)")
state: dict = {}


def load(model_id: str, memory: float, max_len: int, quantization: str | None,
         kv_dtype: str) -> None:
    from transformers import AutoProcessor
    from vllm import LLM

    started = time.perf_counter()
    print(f"загружаю {model_id} движком vLLM…", flush=True)
    state["processor"] = AutoProcessor.from_pretrained(model_id)
    state["llm"] = LLM(
        model=model_id,
        dtype="bfloat16",
        # Веса в fp8 вместо bf16: на Ampere нативного fp8 нет, и ядра Marlin разворачивают
        # их обратно в fp16 на лету. Считается в той же точности, экономится память —
        # модель ужимается вдвое, и освободившееся уходит под KV-кэш, ради которого vLLM
        # и берётся: при bf16 кэша хватало на 1.2 запроса, то есть пакетизации не было.
        quantization=quantization,
        kv_cache_dtype=kv_dtype,
        gpu_memory_utilization=memory,
        max_model_len=max_len,
        # Один ролик на запрос: кадры шага склеены в одно видео.
        limit_mm_per_prompt={"video": 1, "image": base.MAX_IMAGES},
        enforce_eager=False,
    )
    state["model_id"] = model_id
    print(f"готово за {time.perf_counter() - started:.1f} с", flush=True)


def video_item(frames: list, indices: list[int], fps: float) -> tuple:
    """Кадры шага и метаданные ролика в том виде, в каком их ждёт vLLM.

    Поля те же, что мы заполняли для процессора transformers: по ним считаются отметки
    времени в промпте. `do_sample_frames: False` обязателен — иначе процессор проредит
    поданное до своих двух кадров в секунду, и половина кадров до модели не доедет.
    """
    # Кадры приводятся к одному размеру: контекстные режутся фильтром «длинная сторона»,
    # а сплошная нарезка — «ширина», и на ролике другого соотношения сторон размеры
    # расходятся. Процессор transformers это молча сглаживал, а видео обязано быть
    # однородным. За образец берётся самый частый размер — это кадры внутри шага.
    sizes = [frame.size for frame in frames]
    target = max(set(sizes), key=sizes.count)
    array = np.stack(
        [
            np.array((frame if frame.size == target else frame.resize(target)).convert("RGB"))
            for frame in frames
        ]
    )
    metadata = {
        "fps": fps,
        "duration": (indices[-1] + 1) / fps if indices else len(frames) / fps,
        "total_num_frames": indices[-1] + 1 if indices else len(frames),
        "frames_indices": indices or list(range(len(frames))),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return array, metadata


@app.get("/health")
def health() -> dict:
    return {"ready": "llm" in state, "model": state.get("model_id"), "engine": "vllm"}


@app.post("/annotate")
def annotate(request: base.Request) -> dict:
    """Все шаги ролика одним вызовом: движок сам разложит их по батчам."""
    from vllm import SamplingParams

    processor = state["processor"]
    started = time.perf_counter()

    def build_prompt(segment) -> str:
        # Та же функция, что и у сервиса на transformers: промпт обязан совпадать,
        # иначе сравнение движков превращается в сравнение текстов.
        return base.open_prompt(request, segment)

    inputs = []
    for segment in request.segments:
        frames = [base.decode(frame) for frame in segment.frames]
        kind = "video" if request.video_mode else "image"
        messages = [
            {
                "role": "user",
                "content": [{"type": kind}, {"type": "text", "text": build_prompt(segment)}],
            }
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if request.video_mode:
            array, metadata = video_item(
                frames, segment.frame_indices or [], segment.fps or 2.0
            )
            payload = {"video": [(array, metadata)]}
        else:
            payload = {"image": frames}
        inputs.append({"prompt": text, "multi_modal_data": payload})

    sampling = SamplingParams(temperature=0.0, max_tokens=base.MAX_NEW_TOKENS)
    outputs = state["llm"].generate(inputs, sampling, use_tqdm=False)
    answers = [output.outputs[0].text for output in outputs]

    results = [
        base._apply(segment, base.parse(text), request, text)
        for segment, text in zip(request.segments, answers)
    ]
    return {
        "results": results,
        "model": state["model_id"] + " (vllm)",
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("PRAXIS_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    # Доля памяти карты под движок. Ниже 0.9, потому что рядом живут признаки и границы.
    parser.add_argument("--memory", type=float, default=float(os.getenv("PRAXIS_VLLM_MEMORY", "0.72")))
    parser.add_argument("--max-len", type=int, default=int(os.getenv("PRAXIS_VLLM_MAX_LEN", "16384")))
    parser.add_argument("--quantization", default=os.getenv("PRAXIS_VLLM_QUANT") or None)
    parser.add_argument("--kv-cache-dtype", default=os.getenv("PRAXIS_VLLM_KV", "auto"))
    args = parser.parse_args()

    load(args.model, args.memory, args.max_len, args.quantization, args.kv_cache_dtype)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
