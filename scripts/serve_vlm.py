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

DEFAULT_DOMAIN = "человек выполняет руками последовательность бытовых действий с предметами"

PROMPT = """Кадры идут по порядку и показывают один фрагмент видео: {domain}.

Твоя задача — назвать действие, которое человек **выполнил за этот фрагмент**, и предмет,
над которым он работал. Не описывай каждое мгновение: важен итог фрагмента.

Сравни первый и последний кадр и спроси себя, что изменилось: где оказался предмет, что
открылось или закрылось, что взято в руки или отложено. Действие выбирай по этому
изменению, а не по тому, что человек держит что-то в руках — держит он почти всегда.
{context}
Допустимые действия: {actions}
Допустимые предметы: {objects}

Ответь ровно двумя строками. Первая — короткое наблюдение: что изменилось между первым и
последним кадром. Вторая — только JSON:
{{"action": "<действие из списка>", "object": "<предмет из списка>", "confidence": <0..1>,
 "alternatives": [["<второе по вероятности действие>", "<предмет>"], ["<третье>", "<предмет>"]]}}"""


# Калибровка уверенности по измеренной точности, а не по самооценке модели.
CONFIDENCE_BOTH = 0.4
CONFIDENCE_ACTION_ONLY = 0.25
CONFIDENCE_NONE = 0.1

# Сколько шагов кодировать за один проход. Упирается в память карты, а не в скорость.
BATCH_SIZE = 4

# Сколько кандидатов оценивать за один проход при скоринге.
SCORE_BATCH = 8


JOINT_PROMPT = """Кадры показывают один ролик, разбитый на {count} последовательных шагов: {domain}.

Кадры идут по порядку. Для каждого шага дано по несколько кадров, шаги перечислены ниже с
указанием, какие кадры к какому относятся:
{layout}

Назови, что человек делает на каждом шаге и с каким предметом. Смотри на ролик как на
последовательность: если предмет взяли, дальше его несут и куда-то кладут; если дверцу
открыли, позже её закроют. Соседние шаги обычно разные — если два подряд получаются
одинаковыми, скорее всего один из них назван неверно.

Допустимые действия: {actions}
Допустимые предметы: {objects}

Ответь одним JSON-массивом длиной ровно {count}, по объекту на шаг, и больше ничем:
[{{"step": 1, "action": "...", "object": "...", "alternatives": [["...", "..."]]}}, ...]"""


class Segment(BaseModel):
    id: int
    frames: list[str]  # JPEG в base64
    # Короткий список гипотез для режима скоринга: [[действие, предмет], ...].
    candidates: list[list[str]] | None = None
    # Что модель сказала про соседние шаги на первом проходе.
    previous: str | None = None
    following: str | None = None


class Request(BaseModel):
    # Подписывать ли кадры их позицией во времени: проверяется замером, поэтому флаг.
    frame_labels: bool = True
    segments: list[Segment]
    actions: list[str]
    objects: list[str]
    pairs: dict[str, list[str]] | None = None
    # Одна фраза про домен: без неё модель не знает, кухня это или сборочный стол.
    domain: str | None = None
    # Разбирать все шаги ролика одним запросом вместо пошагового. Идея была в том, чтобы
    # модель видела последовательность целиком, но измерение показало обратное: при десятке
    # кадров в одном запросе она теряет соответствие кадров шагам. Пара 0.149 против 0.224
    # у пошагового разбора на кухонном наборе. Оставлено выключенным, но не выброшено:
    # на длинных роликах с малым числом шагов может сработать иначе.
    joint: bool = False
    # generate — модель называет свободным текстом; score — оценивает готовые гипотезы.
    mode: str = "generate"


def frame_label(index: int, total: int) -> str:
    """Подпись кадра.

    Половина словаря EPIC — это пары, различимые только направлением времени: взять и
    положить, открыть и закрыть. Пачка картинок без подписей не сообщает модели, где
    начало шага, а где конец, и на таких парах она угадывает. Подпись стоит десяток
    токенов и делает порядок явным.
    """
    if total == 1:
        return "Кадр шага:"
    if index == 0:
        return f"Кадр 1 из {total}, начало шага:"
    if index == total - 1:
        return f"Кадр {total} из {total}, конец шага:"
    return f"Кадр {index + 1} из {total}:"


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
    matches = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


@app.get("/health")
def health() -> dict:
    return {"ready": "model" in state, "model": state.get("model_id")}


@torch.inference_mode()
def _logprobs(prompts: list[str], targets: list[str], images_per_prompt: list) -> list[float]:
    """Средний логарифм правдоподобия каждого ответа при своём промпте."""
    model, processor = state["model"], state["processor"]
    processor.tokenizer.padding_side = "left"

    full = [prompt + target for prompt, target in zip(prompts, targets)]
    flat_images = [image for group in images_per_prompt for image in group]
    kwargs = {"text": full, "padding": True, "return_tensors": "pt"}
    if flat_images:
        kwargs["images"] = flat_images
    inputs = processor(**kwargs).to(model.device)
    logits = model(**inputs).logits.float().log_softmax(dim=-1)

    scores = []
    for index, target in enumerate(targets):
        target_ids = processor.tokenizer(target, add_special_tokens=False).input_ids
        if not target_ids:
            scores.append(-1e9)
            continue
        length = len(target_ids)
        total = sum(
            float(logits[index, -length - 1 + step, token])
            for step, token in enumerate(target_ids)
        )
        scores.append(total / length)
    return scores


def _score_candidates(images: list, candidates: list[tuple[str, str]], domain: str) -> list[float]:
    """Насколько правдоподобен каждый вариант ответа с точки зрения модели.

    Это другой режим работы, чем генерация. Мы не спрашиваем «что здесь происходит» и не
    надеемся, что модель вспомнит нужное слово из списка в шестьдесят позиций. Вместо этого
    для каждого кандидата считается, насколько вероятен именно такой ответ при этих кадрах,
    и берётся лучший. Модель переходит из режима «вспомни» в режим «сравни картинку с
    гипотезой», а он ей даётся заметно лучше.

    Обязательная часть — вычитание языкового приора: то же правдоподобие считается БЕЗ
    кадров, и разность показывает, сколько к ответу добавила именно картинка. Без этой
    поправки побеждают частые в языке словосочетания, а не то, что видно.
    """
    question = f"На кадрах: {domain}. Одним коротким ответом назови действие и предмет."
    prompt_with = state["processor"].apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": question}],
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_without = state["processor"].apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": question}]}],
        add_generation_prompt=True,
        tokenize=False,
    )

    answers = [f"{action} {noun}".strip() for action, noun in candidates]
    scores: list[float] = []
    for start in range(0, len(answers), SCORE_BATCH):
        chunk = answers[start : start + SCORE_BATCH]
        grounded = _logprobs([prompt_with] * len(chunk), chunk, [images] * len(chunk))
        prior = _logprobs([prompt_without] * len(chunk), chunk, [])
        scores.extend(g - p for g, p in zip(grounded, prior))
    return scores


def _joint(request: Request) -> list[dict]:
    """Все шаги ролика одним запросом: модель видит последовательность целиком."""
    model, processor = state["model"], state["processor"]

    images, layout, index = [], [], 1
    for number, segment in enumerate(request.segments, start=1):
        frames = [decode(frame) for frame in segment.frames]
        layout.append(f"  шаг {number}: кадры {index}-{index + len(frames) - 1}")
        index += len(frames)
        images.extend(frames)

    prompt = JOINT_PROMPT.format(
        count=len(request.segments),
        domain=request.domain or DEFAULT_DOMAIN,
        layout="\n".join(layout),
        actions=", ".join(request.actions),
        objects=", ".join(request.objects),
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image} for image in images]
            + [{"type": "text", "text": prompt}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=64 * len(request.segments) + 96, do_sample=False
        )
    text = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]

    match = re.search(r"\[.*\]", text, re.DOTALL)
    parsed = []
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = []

    answers = []
    for number in range(len(request.segments)):
        item = parsed[number] if number < len(parsed) and isinstance(parsed[number], dict) else {}
        answers.append(item)
    return answers


@app.post("/annotate")
def annotate(request: Request) -> dict:
    model, processor = state["model"], state["processor"]
    def build_prompt(segment) -> str:
        # Соседние шаги — это контекст, без которого «открыл» и «закрыл» неразличимы:
        # у них одинаковые кадры и разный смысл, который виден только из последовательности.
        lines = []
        if segment.previous:
            lines.append(f"Предыдущий шаг ролика: {segment.previous}.")
        if segment.following:
            lines.append(f"Следующий шаг ролика: {segment.following}.")
        if lines:
            lines.append(
                "Учти порядок: взятый предмет потом куда-то кладут, открытое потом закрывают,"
                " и два соседних шага обычно разные."
            )
        return PROMPT.format(
            domain=request.domain or DEFAULT_DOMAIN,
            actions=", ".join(request.actions),
            objects=", ".join(request.objects),
            context=("\n" + "\n".join(lines) + "\n") if lines else "",
        )
    started = time.perf_counter()

    # Все шаги ролика уезжают в модель одной пачкой. Раньше они шли по очереди, и между
    # запросами GPU простаивал: на ролике из семи шагов это давало более ста секунд при
    # лимите кейса в сто двадцать.
    prompt = PROMPT.format(
        domain=request.domain or DEFAULT_DOMAIN,
        actions=", ".join(request.actions),
        objects=", ".join(request.objects),
        context="",
    )

    if request.mode == "score":
        results = []
        for segment in request.segments:
            images = [decode(frame) for frame in segment.frames]
            pairs = [
                (pair[0], pair[1] if len(pair) > 1 else "")
                for pair in (segment.candidates or [])
            ] or [(action, "") for action in request.actions]
            scored = _score_candidates(images, pairs, request.domain or DEFAULT_DOMAIN)
            order = sorted(range(len(pairs)), key=lambda i: -scored[i])
            best = pairs[order[0]]
            margin = scored[order[0]] - scored[order[1]] if len(order) > 1 else 1.0
            results.append(
                {
                    "id": segment.id,
                    "action": best[0],
                    "object": best[1] or None,
                    "alternatives": [
                        {"action": pairs[i][0], "object": pairs[i][1] or None} for i in order[1:4]
                    ],
                    # Уверенность из отрыва лидера: это настоящая мера, а не самооценка.
                    "confidence": round(float(min(0.95, max(0.05, 0.5 + margin))), 3),
                    "raw": f"скоринг {len(pairs)} гипотез",
                }
            )
        return {
            "results": results,
            "model": state["model_id"],
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }

    if request.joint and len(request.segments) > 1:
        parsed_answers = _joint(request)
        results = []
        for segment, answer in zip(request.segments, parsed_answers):
            results.append(_apply(segment, answer, request, ""))
        return {
            "results": results,
            "model": state["model_id"],
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }

    batch_size = max(1, BATCH_SIZE)
    texts, image_groups = [], []
    for segment in request.segments:
        images = [decode(frame) for frame in segment.frames]
        content = []
        for index, image in enumerate(images):
            if request.frame_labels:
                content.append({"type": "text", "text": frame_label(index, len(images))})
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": build_prompt(segment)})
        messages = [{"role": "user", "content": content}]
        texts.append(
            processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        )
        image_groups.append(images)

    processor.tokenizer.padding_side = "left"
    answers: list[str] = []
    for start in range(0, len(texts), batch_size):
        chunk_texts = texts[start : start + batch_size]
        chunk_images = [image for group in image_groups[start : start + batch_size] for image in group]
        inputs = processor(
            text=chunk_texts, images=chunk_images, padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        answers.extend(
            processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        )

    results = [
        _apply(segment, parse(text), request, text)
        for segment, text in zip(request.segments, answers)
    ]
    return {
        "results": results,
        "model": state["model_id"],
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


def _apply(segment, answer: dict, request: Request, text: str) -> dict:
    """Притягиваем ответ модели к словарю и считаем уверенность."""
    action = closest(str(answer.get("action", "")), request.actions)
    objects = (
        request.pairs.get(action, request.objects)
        if (request.pairs and action)
        else request.objects
    )
    obj = closest(str(answer.get("object", "")), objects)

    # Самооценку модели наружу не отдаём: измерено, что она ставит 0.95 и там, где
    # ошибается, — с такой «уверенностью» триаж в редакторе перестаёт работать.
    confidence = CONFIDENCE_BOTH if (action and obj) else CONFIDENCE_ACTION_ONLY
    if action is None:
        confidence = CONFIDENCE_NONE

    alternatives = []
    for candidate in answer.get("alternatives") or []:
        if not isinstance(candidate, (list, tuple)) or len(candidate) < 2:
            continue
        alternative_action = closest(str(candidate[0]), request.actions)
        if not alternative_action or alternative_action == action:
            continue
        allowed = (
            request.pairs.get(alternative_action, request.objects)
            if request.pairs
            else request.objects
        )
        alternatives.append(
            {"action": alternative_action, "object": closest(str(candidate[1]), allowed)}
        )

    return {
        "id": segment.id,
        "action": action,
        "object": obj,
        "alternatives": alternatives[:3],
        "confidence": round(max(0.0, min(confidence, 1.0)), 3),
        "raw": text.strip()[:200],
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
