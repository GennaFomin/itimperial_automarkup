"""Стадия именования: как называется то, что уже нарезано.

Границы к этому моменту стоят, и языковая модель их не двигает — она отвечает только на
вопрос «что здесь делают и с чем». Ответ притягивается к закрытому словарю, а если сервис
недоступен или отвечает мусором, шаги остаются без меток: пустая разметка не выдаётся
никогда, и весь ролик не разваливается из-за одной неудачной стадии.
"""

from __future__ import annotations

import base64
import json
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from praxis import config, media
from praxis.schema import Step, VideoMeta
from praxis.vocab import Vocabulary


@dataclass
class NamingResult:
    steps: list[Step]
    models: dict[str, str] = field(default_factory=dict)
    # Подсказки для редактора: {id шага: [{action, object}, ...]}. В экспорт не идут —
    # это средство ускорить проверку, а не часть разметки.
    alternatives: dict[int, list[dict]] = field(default_factory=dict)


class Namer(Protocol):
    name: str

    def name_steps(
        self,
        video_path: Path,
        meta: VideoMeta,
        steps: list[Step],
        vocabulary: Vocabulary,
        crop: tuple[float, float, float, float] | None = None,
    ) -> NamingResult: ...


class NullNamer:
    """Семантики нет: шаги уходят в редактор без меток, человек проставит их сам."""

    name = "none"

    def name_steps(
        self,
        video_path: Path,
        meta: VideoMeta,
        steps: list[Step],
        vocabulary: Vocabulary,
        crop: tuple[float, float, float, float] | None = None,
    ) -> NamingResult:
        return NamingResult(steps=steps, models={"namer": "none"})


class HttpNamer:
    """Общая часть клиентов: нарезка кадров сегмента и запрос к сервису на GPU-машине."""

    name = "http"

    def __init__(
        self,
        base_url: str,
        frames_per_step: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.frames_per_step = frames_per_step or config.VLM_FRAMES
        self.timeout = timeout or config.VLM_TIMEOUT

    def _track(
        self,
        video_path: Path,
        steps: list[Step],
        fallback: tuple[float, float, float, float] | None,
    ) -> dict[int, tuple[float, float, float, float]]:
        """Рамка движущегося предмета на каждый шаг. При отказе трекера — общая зона."""
        try:
            answer = self._post(
                "/track",
                {
                    "segments": [
                        {"id": step.id, "frames": self._frames(video_path, step, None, width=512)}
                        for step in steps
                    ],
                    "grid": config.TRACK_GRID,
                },
                base_url=config.TRACK_BASE_URL,
            )
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return {}

        boxes: dict[int, tuple[float, float, float, float]] = {}
        for item in answer.get("results", []):
            box = item.get("box")
            if not box or item.get("shift", 0.0) < config.TRACK_MIN_SHIFT:
                continue
            left, top, width, height = box
            # Расширяем рамку: предмет полезнее видеть вместе с руками и опорой.
            margin = config.TRACK_MARGIN
            left = max(0.0, left - width * margin)
            top = max(0.0, top - height * margin)
            width = min(1.0 - left, width * (1 + 2 * margin))
            height = min(1.0 - top, height * (1 + 2 * margin))
            if width > 0.05 and height > 0.05:
                boxes[item["id"]] = (left, top, width, height)
        return boxes

    @staticmethod
    def _shortlist(named: dict, vocabulary: Vocabulary) -> list[tuple[str, str]]:
        """Гипотезы для оценки: ответ модели, её альтернативы и перекрёстные комбинации."""
        actions, objects = [], []
        if named.get("action"):
            actions.append(named["action"])
        if named.get("object"):
            objects.append(named["object"])
        for alternative in named.get("alternatives") or []:
            if alternative.get("action") and alternative["action"] not in actions:
                actions.append(alternative["action"])
            if alternative.get("object") and alternative["object"] not in objects:
                objects.append(alternative["object"])

        candidates: list[tuple[str, str]] = []
        for action in actions[:3]:
            for noun in objects[:3]:
                if vocabulary.is_valid_pair(action, noun) and (action, noun) not in candidates:
                    candidates.append((action, noun))
        return candidates

    def _objects_first(self, frames: dict, vocabulary, base: dict) -> dict:
        """Первая ступень: найти предмет, чтобы сузить выбор действия.

        Замер показал, что глагол — узкое место: из 139 шагов пара верна в 51, а глагол
        неверен в 65. Модель подставляет самые частые действия («walk», «put») по общему
        впечатлению от сцены. Но словарь задаёт закрытый список сочетаний, и знание
        предмета сокращает список действий в разы — тогда глагол выбирается из того, что
        с этим предметом вообще делают.
        """
        answer = self._post(
            "/annotate",
            {
                "segments": [{"id": key, "frames": value} for key, value in frames.items()],
                **base,
                "stage": "object",
            },
        )
        # Список действий под найденный предмет НЕ сужаем. Замер: сужение дало предмет
        # 0.626 против 0.540, но пару 0.317 против 0.367 — когда предмет угадан неверно,
        # глагол выбирается из неправильного подсписка и ошибки перемножаются.
        return {
            item["id"]: {"hint_object": item["object"]}
            for item in answer.get("results", [])
            if item.get("object")
        }

    def _frames(
        self,
        video_path: Path,
        step: Step,
        crop: tuple[float, float, float, float] | None = None,
        width: int | None = None,
    ) -> list[str]:
        """Кадры, равномерно разбросанные внутри шага, плюс его ключевой кадр.

        При включённом контексте добавляются два кадра снаружи — до начала и после конца.
        Они нужны для глаголов состояния: «взял» и «положил» дают почти одинаковую
        середину действия и различаются только тем, где предмет был до и оказался после.
        """
        return self._shot(video_path, step, crop, width)[0]

    def _shot(
        self,
        video_path: Path,
        step: Step,
        crop: tuple[float, float, float, float] | None = None,
        width: int | None = None,
    ) -> tuple[list[str], list[int], float]:
        """То же, что _frames, но вместе с местами кадров на временной сетке и её частотой.

        Частота ноль означает, что сетки нет: у выборки из пяти моментов кадры неравномерны
        по времени, и сервис подставит номинальную частоту.
        """
        frames, indices, grid_fps = self._cut(video_path, step, crop, width)
        # Закраска фона — последним шагом, уже над отобранными кадрами: детектор рук стоит
        # миллисекунды на кадр, и тратить их на кадры, которые всё равно выбросит склейка,
        # незачем.
        if config.HANDS_BASE_URL and frames:
            frames = self._focus(frames)
        return frames, indices, grid_fps

    def _cut(
        self,
        video_path: Path,
        step: Step,
        crop: tuple[float, float, float, float] | None = None,
        width: int | None = None,
    ) -> tuple[list[str], list[int], float]:
        """Нарезка кадров шага: сплошная на длинном шаге, выборка моментов на коротком."""
        span = step.end_sec - step.start_sec
        # Сплошная нарезка только на длинных шагах: на коротких она измеренно вредит,
        # см. VLM_DENSE_MIN_SEC.
        if config.VLM_VIDEO_MODE and config.VLM_VIDEO_FPS > 0 and span >= config.VLM_DENSE_MIN_SEC:
            frames, indices, grid_fps = self._window(video_path, step, crop, width)
            if config.VLM_SELECT and len(frames) > 2 and grid_fps > 0:
                # Абсолютное время каждого кадра: места на сетке считаются от самого раннего
                # показанного кадра, а признаки видеоэнкодера живут во времени ролика.
                origin = max(0.0, step.start_sec - config.CONTEXT_FRAMES)
                times = [origin + place / grid_fps for place in indices]
                frames, indices = self._select(frames, indices, times, video_path)
            return frames, indices, grid_fps
        offsets = [
            step.start_sec + span * (index + 0.5) / self.frames_per_step
            for index in range(self.frames_per_step)
        ]
        if step.keyframe_sec is not None:
            offsets.append(step.keyframe_sec)
        if config.CONTEXT_FRAMES > 0:
            offsets.append(max(0.0, step.start_sec - config.CONTEXT_FRAMES))
            offsets.append(step.end_sec + config.CONTEXT_FRAMES)

        encoded: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            for index, at in enumerate(sorted(set(round(value, 2) for value in offsets))):
                path = Path(directory) / f"{index}.jpg"
                try:
                    media.extract_frame(
                        video_path, at, path, width=width or config.VLM_FRAME_WIDTH, crop=crop
                    )
                except media.MediaError:
                    # Контекстный кадр может выйти за конец ролика — это не повод падать,
                    # шаг просто останется без одной подсказки.
                    continue
                encoded.append(base64.b64encode(path.read_bytes()).decode())
        return encoded, list(range(len(encoded))), 0.0

    @staticmethod
    def _window_fps(step: Step) -> float:
        """Частота сплошной нарезки внутри шага.

        Считается по длине самого шага, а не окна с контекстом: кадр, потраченный на
        соседнее действие, стоит дороже, чем добытый им контекст. На длинном шаге частота
        снижается ровно настолько, чтобы шаг уложился в потолок — иначе ffmpeg обрежет
        лишнее с конца, и шаг потеряет свою концовку, по которой и видно, что изменилось.
        """
        span = max(step.end_sec - step.start_sec, 0.001)
        inner = max(config.VLM_VIDEO_MAX_FRAMES - 2, 1)  # два места отданы контексту
        return max(
            min(config.VLM_VIDEO_FPS, inner / span), config.VLM_VIDEO_MIN_FRAMES / span
        )

    def _window(
        self,
        video_path: Path,
        step: Step,
        crop: tuple[float, float, float, float] | None = None,
        width: int | None = None,
    ) -> tuple[list[str], list[int], float]:
        """Сплошная последовательность кадров шага плюс по кадру снаружи с каждой стороны.

        Контекст остаётся ровно двумя кадрами: он подсказывает направление времени, но
        занимать собой половину последовательности не должен.

        Возвращает кадры вместе с их местами на общей временной сетке шага и частотой этой
        сетки. Места нужны потому, что дальше часть кадров отсеивается как повторы, и без
        них отметки времени в промпте разъехались бы: сервис считает время как номер на
        сетке, делённый на частоту.
        """
        encoded: list[str] = []
        width = width or config.VLM_FRAME_WIDTH
        grid_fps = self._window_fps(step)
        context = config.CONTEXT_FRAMES if config.CONTEXT_FRAMES > 0 else 0.0
        offset = round(context * grid_fps)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                paths = media.window_frames(
                    video_path,
                    step.start_sec,
                    step.end_sec,
                    root,
                    fps=grid_fps,
                    width=width,
                    limit=max(config.VLM_VIDEO_MAX_FRAMES - 2, 1),
                    crop=crop,
                )
            except media.MediaError:
                return [], [], grid_fps
            indices = [offset + position for position in range(len(paths))]
            if context > 0:
                before = self._context_frame(
                    video_path, max(0.0, step.start_sec - context),
                    root / "a_before.jpg", crop, width,
                )
                after = self._context_frame(
                    video_path, step.end_sec + context, root / "z_after.jpg", crop, width,
                )
                if before:
                    paths, indices = [before] + paths, [0] + indices
                if after:
                    paths = paths + [after]
                    indices = indices + [round((step.end_sec - step.start_sec + 2 * context) * grid_fps)]
            for path in paths:
                encoded.append(base64.b64encode(path.read_bytes()).decode())
        return encoded, indices, grid_fps

    @staticmethod
    def _context_frame(
        video_path: Path,
        at_sec: float,
        out: Path,
        crop: tuple[float, float, float, float] | None,
        width: int,
    ) -> Path | None:
        """Кадр снаружи шага. Может не существовать — ролик кончился, это не ошибка."""
        try:
            return media.extract_frame(video_path, at_sec, out, width=width, crop=crop)
        except media.MediaError:
            return None

    @staticmethod
    def _video_vectors(video_path: Path, times: list[float]) -> tuple[np.ndarray, float] | None:
        """Признаки видеоэнкодера в моменты кадров — те же, по которым ставятся границы.

        Считать заново не нужно: признаки всего ролика уже посчитаны на стадии нарезки и
        лежат в кэше на диске. Здесь берётся ближайшее окно к каждому кадру; сетка окон
        мельче нашей нарезки (четыре вектора в секунду против двух кадров), так что
        промаха по времени не возникает.

        Вместе с векторами возвращается порог «здесь ничего не происходит», посчитанный по
        самому ролику. Абсолютная константа не работает: окна перекрываются на три
        четверти, соседние векторы похожи по построению, и уровень этой похожести зависит
        от того, насколько подвижен ролик целиком.
        """
        from praxis import jobs  # локально: jobs импортирует этот модуль

        features = jobs.video_features(video_path)
        if not features or features.get("count", 0) < 2:
            return None
        matrix = features["matrix"]
        fps = float(features.get("fps") or 0.0)
        if fps <= 0:
            return None
        matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-6, None)
        neighbours = np.sum(matrix[1:] * matrix[:-1], axis=1)
        threshold = float(np.quantile(neighbours, config.VLM_SELECT_QUANTILE))

        offset = float(features.get("offset_sec") or 0.0)
        rows = [
            int(min(max(round((moment - offset) * fps), 0), len(matrix) - 1)) for moment in times
        ]
        return matrix[rows], threshold

    def _select(
        self,
        frames: list[str],
        indices: list[int],
        times: list[float],
        video_path: Path,
    ) -> tuple[list[str], list[int]]:
        """Выбросить из плотной нарезки кадры, повторяющие предыдущий.

        Сравнение идёт с последним ОСТАВЛЕННЫМ кадром, а не с непосредственно предыдущим:
        иначе долгая пауза проходила бы целиком, покадрово набирая малые изменения.

        Выбрасывается именно застой, а не похожесть как таковая. Это важно: модель читает
        движение из пар соседних кадров внутри временного патча, а соседние кадры похожи по
        определению. Жёсткий отбор «оставить восемь самых разных» на замере убивал весь
        выигрыш (0.118 против 0.529) — он возвращал разреженность, ради устранения которой
        нарезка и уплотняется. Потолок здесь — страховка от переполнения памяти, а не цель.

        Первый и последний кадр остаются всегда: по ним и определяется, что изменилось.
        """
        answer = self._video_vectors(video_path, times)
        if answer is None:
            return frames, indices
        vectors, similarity = answer
        if len(vectors) != len(frames):
            return frames, indices

        keep = [0]
        for index in range(1, len(frames)):
            if float(vectors[index] @ vectors[keep[-1]]) < similarity:
                keep.append(index)
        if keep[-1] != len(frames) - 1:
            keep.append(len(frames) - 1)

        limit = max(config.VLM_SELECT_MAX, 2)
        if len(keep) > limit:
            walk = vectors[keep]
            # Путь в пространстве признаков: сколько сцена изменилась к этому кадру. Кадры
            # берутся равными шагами по нему, а не по времени, — на паузе не тратятся.
            steps = 1.0 - np.sum(walk[1:] * walk[:-1], axis=1)
            path = np.concatenate([[0.0], np.cumsum(np.clip(steps, 0.0, None))])
            if path[-1] <= 0:
                chosen = np.linspace(0, len(keep) - 1, limit).round().astype(int)
            else:
                chosen = np.clip(
                    np.searchsorted(path, np.linspace(0, path[-1], limit)), 0, len(keep) - 1
                )
            keep = sorted(set(keep[position] for position in chosen) | {keep[0], keep[-1]})

        return [frames[i] for i in keep], [indices[i] for i in keep]

    def _focus(self, frames: list[str]) -> list[str]:
        """Закрасить всё, что дальше радиуса от кистей. Отказ сервиса — не повод падать.

        Предмет — самое слабое место именования: он верен в 21 ответе из 142, а в 60 назван
        родовым словом вместо детали. Модель отвечает по общему впечатлению от сцены, и
        лежащая на столе собранная игрушка перетягивает ответ у мелкой детали в руке.

        Закраска, а не кроп: кадры уходят видеотрактом, и модель читает движение из пар
        соседних кадров. Кроп, прыгающий за руками, добавил бы своё движение к настоящему.
        """
        try:
            answer = self._post(
                "/focus",
                {
                    "segments": [{"id": 0, "frames": frames}],
                    "margin": config.HANDS_MARGIN,
                    "min_radius": config.HANDS_MIN_RADIUS,
                    "feather": config.HANDS_FEATHER,
                    "mode": config.HANDS_MODE,
                },
                base_url=config.HANDS_BASE_URL,
            )
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return frames
        results = answer.get("results") or []
        painted = results[0].get("frames") if results else None
        # Число кадров обязано совпасть: иначе места на временной сетке разъедутся.
        return painted if painted and len(painted) == len(frames) else frames

    def _post(self, path: str, payload: dict, base_url: str | None = None) -> dict:
        request = urllib.request.Request(
            (base_url.rstrip("/") if base_url else self.base_url) + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())


class RemoteVlmNamer(HttpNamer):
    """Клиент к генеративной видеомодели (scripts/serve_vlm.py)."""

    name = "vlm"

    def name_steps(
        self,
        video_path: Path,
        meta: VideoMeta,
        steps: list[Step],
        vocabulary: Vocabulary,
        crop: tuple[float, float, float, float] | None = None,
    ) -> NamingResult:
        if not steps:
            return NamingResult(steps=steps, models={"namer": "vlm", "namer_status": "нет шагов"})

        # Область предмета от трекера, если он поднят: кадры для каждого шага режутся по
        # своей рамке, а не по общей рабочей зоне. Языковая модель тогда смотрит на предмет
        # крупно, а не ищет его на общем плане.
        boxes = self._track(video_path, steps, crop) if config.TRACK_BASE_URL else {}

        # Кадры режутся один раз: их извлечение стоит дороже, чем сам запрос, а ступеней
        # может быть две.
        # Вместе с кадрами запоминаются их места на временной сетке шага: после склейки
        # повторов последовательность неравномерна, и без мест отметки времени в промпте
        # оказались бы неверными — шаг с выброшенной серединой выглядел бы вдвое короче.
        shots = {
            step.id: self._shot(video_path, step, boxes.get(step.id, crop)) for step in steps
        }
        frames = {step_id: shot[0] for step_id, shot in shots.items()}
        base = {
            "actions": vocabulary.actions,
            "objects": vocabulary.objects,
            "pairs": vocabulary.pairs,
            "frame_labels": config.VLM_FRAME_LABELS,
            "video_mode": config.VLM_VIDEO_MODE,
            # Про овал модели надо сказать словами: иначе она принимает его за предмет
            # сцены и начинает описывать саму отметку.
            "marked": bool(config.HANDS_BASE_URL) and config.HANDS_MODE == "circle",
            # Три флага ниже пропали при переписывании истории и без них сервис молча
            # притягивал ответ к словарю Assembly101 на любом домене.
            "open_vocabulary": config.OPEN_VOCABULARY,
            "language": config.LANGUAGE,
            "context_frames": config.CONTEXT_FRAMES > 0,
            "domain": config.DOMAIN or vocabulary.description or None,
        }

        try:
            hints = self._objects_first(frames, vocabulary, base) if config.VLM_TWO_STAGE else {}
            payload = {
                "segments": [
                    {
                        "id": step.id,
                        "frames": shots[step.id][0],
                        # Частота и места кадров у каждого шага свои: по ним сервис строит
                        # отметки времени внутри промпта.
                        **(
                            {
                                "frame_indices": shots[step.id][1],
                                "fps": round(shots[step.id][2], 3),
                            }
                            if shots[step.id][2] > 0
                            else {}
                        ),
                        **hints.get(step.id, {}),
                    }
                    for step in steps
                ],
                **base,
            }
            answer = self._post("/annotate", payload)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as error:
            # Сервис недоступен — отдаём нарезку без меток, а не падаем.
            return NamingResult(
                steps=steps, models={"namer": "vlm", "namer_status": f"недоступен: {error}"}
            )

        by_id = {item["id"]: item for item in answer.get("results", [])}

        # Второй проход с контекстом: модель переспрашивается, зная, что она же сказала про
        # соседние шаги. «Открыл» и «закрыл» на отдельных кадрах неразличимы — их
        # различает только порядок, и его надо дать модели явно.
        if config.VLM_CONTEXT and len(steps) > 1 and by_id:
            def label(step_id: int) -> str | None:
                item = by_id.get(step_id)
                if not item or not item.get("action"):
                    return None
                return f"{item['action']} {item.get('object') or ''}".strip()

            ordered = sorted(steps, key=lambda step: step.start_sec)
            segments = []
            for index, step in enumerate(ordered):
                segments.append(
                    {
                        "id": step.id,
                        "frames": self._frames(video_path, step, crop),
                        "previous": label(ordered[index - 1].id) if index else None,
                        "following": (
                            label(ordered[index + 1].id) if index + 1 < len(ordered) else None
                        ),
                    }
                )
            try:
                second = self._post(
                    "/annotate",
                    {
                        "segments": segments,
                        "actions": vocabulary.actions,
                        "objects": vocabulary.objects,
                        "pairs": vocabulary.pairs,
                        "domain": config.DOMAIN or vocabulary.description or None,
                    },
                )
                for item in second.get("results", []):
                    if item.get("action"):
                        by_id[item["id"]] = item
            except (urllib.error.URLError, OSError, TimeoutError, ValueError):
                pass  # остаёмся с первым проходом

        # Ещё один проход: модель не придумывает ответ, а оценивает короткий список гипотез.
        # Свободная генерация заставляет её вспоминать слово из длинного списка; оценка
        # правдоподобия превращает задачу в «сравни картинку с гипотезой», а это ей даётся
        # заметно лучше. Список собирается из её же первого ответа и альтернатив, включая
        # перекрёстные комбинации — так проверяются обе оси, и действие, и предмет.
        if config.VLM_RESCORE and by_id:
            segments = []
            for step in steps:
                named = by_id.get(step.id)
                if not named:
                    continue
                candidates = self._shortlist(named, vocabulary)
                if len(candidates) > 1:
                    segments.append(
                        {
                            "id": step.id,
                            "frames": self._frames(video_path, step, crop),
                            "candidates": [list(pair) for pair in candidates],
                        }
                    )
            if segments:
                try:
                    rescored = self._post(
                        "/annotate",
                        {
                            "segments": segments,
                            "actions": vocabulary.actions,
                            "objects": vocabulary.objects,
                            "pairs": vocabulary.pairs,
                            "domain": config.DOMAIN or vocabulary.description or None,
                            "mode": "score",
                        },
                    )
                    for item in rescored.get("results", []):
                        by_id[item["id"]] = item
                except (urllib.error.URLError, OSError, TimeoutError, ValueError):
                    pass  # остаёмся с результатом свободной генерации

        for step in steps:
            named = by_id.get(step.id)
            if not named:
                continue
            # При открытом словаре сверять ответ не с чем: модель отвечает своими словами,
            # и отбрасывать всё, чего нет в списке, значило бы отбросить каждый ответ.
            # При закрытом словаре проверка остаётся — выдуманная метка не проходит.
            action = named.get("action")
            if action and (config.OPEN_VOCABULARY or vocabulary.has_action(action)):
                step.action = action
            obj = named.get("object")
            step.object = (
                obj if obj and (config.OPEN_VOCABULARY or vocabulary.has_object(obj)) else None
            )
            if named.get("confidence") is not None:
                step.confidence = round(float(named["confidence"]), 3)

        return NamingResult(
            steps=steps,
            alternatives={
                step.id: by_id[step.id].get("alternatives", [])
                for step in steps
                if step.id in by_id and by_id[step.id].get("alternatives")
            },
            models={
                "namer": "vlm",
                "vlm": answer.get("model", config.VLM_MODEL),
                "vlm_sec": str(answer.get("elapsed_sec", "")),
            },
        )



class ClipNamer(HttpNamer):
    """Клиент к классификатору по закрытому словарю (scripts/serve_clip.py).

    Отдаёт не свободный текст, а распределение по 202 допустимым парам: значение вне
    словаря невозможно по построению, а уверенность берётся из самого распределения.
    """

    name = "siglip"

    def name_steps(
        self,
        video_path: Path,
        meta: VideoMeta,
        steps: list[Step],
        vocabulary: Vocabulary,
        crop: tuple[float, float, float, float] | None = None,
    ) -> NamingResult:
        if not steps:
            return NamingResult(steps=steps, models={"namer": self.name, "namer_status": "нет шагов"})

        pairs = (
            [[action, obj] for action, objects in vocabulary.pairs.items() for obj in objects]
            if vocabulary.pairs
            else [[action, ""] for action in vocabulary.actions]
        )
        try:
            answer = self._post(
                "/classify",
                {
                    "segments": [
                        {"id": step.id, "frames": self._frames(video_path, step, crop)}
                        for step in steps
                    ],
                    "pairs": pairs,
                    "mode": config.CLIP_MODE,
                    "verb_weight": config.CLIP_VERB_WEIGHT,
                    "noun_weight": config.CLIP_NOUN_WEIGHT,
                },
            )
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as error:
            return NamingResult(
                steps=steps,
                models={"namer": self.name, "namer_status": f"недоступен: {error}"},
            )

        by_id = {item["id"]: item for item in answer.get("results", [])}

        # Второй проход с контекстом: модель переспрашивается, зная, что она же сказала про
        # соседние шаги. «Открыл» и «закрыл» на отдельных кадрах неразличимы — их
        # различает только порядок, и его надо дать модели явно.
        if config.VLM_CONTEXT and len(steps) > 1 and by_id:
            def label(step_id: int) -> str | None:
                item = by_id.get(step_id)
                if not item or not item.get("action"):
                    return None
                return f"{item['action']} {item.get('object') or ''}".strip()

            ordered = sorted(steps, key=lambda step: step.start_sec)
            segments = []
            for index, step in enumerate(ordered):
                segments.append(
                    {
                        "id": step.id,
                        "frames": self._frames(video_path, step, crop),
                        "previous": label(ordered[index - 1].id) if index else None,
                        "following": (
                            label(ordered[index + 1].id) if index + 1 < len(ordered) else None
                        ),
                    }
                )
            try:
                second = self._post(
                    "/annotate",
                    {
                        "segments": segments,
                        "actions": vocabulary.actions,
                        "objects": vocabulary.objects,
                        "pairs": vocabulary.pairs,
                        "domain": config.DOMAIN or vocabulary.description or None,
                    },
                )
                for item in second.get("results", []):
                    if item.get("action"):
                        by_id[item["id"]] = item
            except (urllib.error.URLError, OSError, TimeoutError, ValueError):
                pass  # остаёмся с первым проходом

        # Ещё один проход: модель не придумывает ответ, а оценивает короткий список гипотез.
        # Свободная генерация заставляет её вспоминать слово из длинного списка; оценка
        # правдоподобия превращает задачу в «сравни картинку с гипотезой», а это ей даётся
        # заметно лучше. Список собирается из её же первого ответа и альтернатив, включая
        # перекрёстные комбинации — так проверяются обе оси, и действие, и предмет.
        if config.VLM_RESCORE and by_id:
            segments = []
            for step in steps:
                named = by_id.get(step.id)
                if not named:
                    continue
                candidates = self._shortlist(named, vocabulary)
                if len(candidates) > 1:
                    segments.append(
                        {
                            "id": step.id,
                            "frames": self._frames(video_path, step, crop),
                            "candidates": [list(pair) for pair in candidates],
                        }
                    )
            if segments:
                try:
                    rescored = self._post(
                        "/annotate",
                        {
                            "segments": segments,
                            "actions": vocabulary.actions,
                            "objects": vocabulary.objects,
                            "pairs": vocabulary.pairs,
                            "domain": config.DOMAIN or vocabulary.description or None,
                            "mode": "score",
                        },
                    )
                    for item in rescored.get("results", []):
                        by_id[item["id"]] = item
                except (urllib.error.URLError, OSError, TimeoutError, ValueError):
                    pass  # остаёмся с результатом свободной генерации

        for step in steps:
            named = by_id.get(step.id)
            if not named:
                continue
            # При открытом словаре сверять ответ не с чем: модель отвечает своими словами,
            # и отбрасывать всё, чего нет в списке, значило бы отбросить каждый ответ.
            # При закрытом словаре проверка остаётся — выдуманная метка не проходит.
            action = named.get("action")
            if action and (config.OPEN_VOCABULARY or vocabulary.has_action(action)):
                step.action = action
            obj = named.get("object")
            step.object = (
                obj if obj and (config.OPEN_VOCABULARY or vocabulary.has_object(obj)) else None
            )
            if named.get("confidence") is not None:
                step.confidence = round(float(named["confidence"]), 3)

        return NamingResult(
            steps=steps,
            alternatives={
                step.id: [
                    {"action": item["action"], "object": item["object"]}
                    for item in by_id[step.id].get("top", [])[1:4]
                ]
                for step in steps
                if step.id in by_id
            },
            models={
                "namer": self.name,
                "classifier": answer.get("model", ""),
                "classifier_sec": str(answer.get("elapsed_sec", "")),
            },
        )


def merge_adjacent(steps: list[Step], labelled: bool = True) -> list[Step]:
    """Склеивает соседние шаги с одинаковой меткой.

    Сегментатор работает по картинке и не знает, что два соседних куска — это одно и то же
    действие: он видит, что кадры изменились, и режет. Понять, что «place tray» и следующий
    «place tray» — один шаг, можно только после того, как метки проставлены. Поэтому склейка
    живёт здесь, а не в сегментаторе.
    """
    # Без настоящих меток склеивать нельзя: у всех шагов стоит одна и та же заглушка,
    # и склейка схлопнула бы весь ролик в один шаг, уничтожив заодно и нарезку.
    if not labelled:
        return steps

    ordered = sorted(steps, key=lambda step: step.start_sec)
    merged: list[Step] = []
    for step in ordered:
        previous = merged[-1] if merged else None
        same_label = (
            previous is not None
            and previous.action == step.action
            and previous.object == step.object
            and previous.level == step.level
            and abs(previous.end_sec - step.start_sec) < 0.05
        )
        if not same_label:
            merged.append(step.model_copy())
            continue

        # Ключевой кадр берём у более длинного куска: он представительнее.
        longer = previous if previous.duration_sec >= step.duration_sec else step
        previous.end_sec = step.end_sec
        previous.keyframe_sec = longer.keyframe_sec
        previous.confidence = max(
            previous.confidence or 0.0, step.confidence or 0.0
        ) or None

    for index, step in enumerate(merged):
        step.id = index
    return merged


def get_namer() -> Namer:
    """Какой источник семантики использовать. Пустые адреса — работаем без неё."""
    choice = config.NAMER
    if choice in {"auto", "vlm"} and config.VLM_BASE_URL:
        return RemoteVlmNamer(config.VLM_BASE_URL)
    if choice in {"auto", "siglip"} and config.CLIP_BASE_URL:
        return ClipNamer(config.CLIP_BASE_URL)
    return NullNamer()
