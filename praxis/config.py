"""Настройки. Всё, что может отличаться между ноутбуком, DL5 и площадкой, живёт здесь."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORK_DIR = Path(os.getenv("PRAXIS_WORK_DIR", ROOT / "work"))
# Собранный фронт. В докере пакет ставится в site-packages, поэтому путь задаётся явно.
WEB_DIST = Path(os.getenv("PRAXIS_WEB_DIST", ROOT / "web" / "dist"))
DB_PATH = WORK_DIR / "praxis.db"
VOCAB_PATH = os.getenv("PRAXIS_VOCAB") or None

# Требования кейса к входному видео.
MAX_DURATION_SEC = float(os.getenv("PRAXIS_MAX_DURATION", "30"))
MIN_HEIGHT = int(os.getenv("PRAXIS_MIN_HEIGHT", "720"))
ALLOWED_SUFFIXES = {".mp4", ".mov"}

# Какой сегментатор поднимать: stub пока пайплайна нет, дальше — реальный.
PIPELINE = os.getenv("PRAXIS_PIPELINE", "motion-dp")

# Куда ходить за VLM. Пусто — значит семантика недоступна и работает запасной путь.
VLM_BASE_URL = os.getenv("PRAXIS_VLM_BASE_URL", "")
VLM_MODEL = os.getenv("PRAXIS_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
CLIP_BASE_URL = os.getenv("PRAXIS_CLIP_BASE_URL", "")
# auto — генеративная модель, если она поднята, иначе классификатор, иначе без семантики.
# Порядок именно такой по измерению: на таксономии Assembly101 классификатор на SigLIP2
# показал 0.06 по глаголу против 0.40 у генеративной модели. На другой таксономии стоит
# перемерить: PRAXIS_NAMER=siglip переключает вручную.
NAMER = os.getenv("PRAXIS_NAMER", "auto")
CLIP_MODE = os.getenv("PRAXIS_CLIP_MODE", "factored")
CLIP_VERB_WEIGHT = float(os.getenv("PRAXIS_CLIP_VERB_WEIGHT", "1.0"))
CLIP_NOUN_WEIGHT = float(os.getenv("PRAXIS_CLIP_NOUN_WEIGHT", "1.0"))
# Одна фраза, описывающая снимаемый процесс. Модель без неё не знает, что за домен.
DOMAIN = os.getenv("PRAXIS_DOMAIN", "")
VLM_FRAMES = int(os.getenv("PRAXIS_VLM_FRAMES", "8"))
VLM_FRAME_WIDTH = int(os.getenv("PRAXIS_VLM_FRAME_WIDTH", "640"))
VLM_TIMEOUT = float(os.getenv("PRAXIS_VLM_TIMEOUT", "120"))

# Писать ли в экспорт отметку о проверке шага человеком. В требованиях кейса такого
# поля нет, но заказчику важно знать, что подтверждено живым человеком, а что нет.
# Гасится одной переменной, если приёмка окажется строгой к лишним полям.
EXPORT_VERIFIED = os.getenv("PRAXIS_EXPORT_VERIFIED", "1").lower() not in {"0", "false", "no"}

# Плёнка кадров под таймлайном и разрешение сигнала движения.
FILMSTRIP_COUNT = 40
MOTION_FPS = 10

# Чем описывать кадр для сегментатора:
#   video — окна кадров через видеоэнкодер: единственные признаки, которые видят движение,
#           а не только содержимое кадра (нужен PRAXIS_VIDEO_BASE_URL),
#   embed — покадровые эмбеддинги визуального энкодера (нужен PRAXIS_CLIP_BASE_URL),
#   gray  — усреднённые серые блоки, запасной вариант без GPU.
FEATURES = os.getenv("PRAXIS_FEATURES", "video")
VIDEO_BASE_URL = os.getenv("PRAXIS_VIDEO_BASE_URL", "")
# Какую модель поднимать в сервисе признаков. Сравнение на двадцати роликах (одна сетка
# параметров, F1 без учёта меток при IoU 0.1/0.25/0.5):
#   timesformer  0.808 / 0.808 / 0.744, границы 1.48 с  ← выбрана
#   swin3d       0.820 / 0.820 / 0.725, границы 1.67 с
#   vjepa2       0.818 / 0.818 / 0.707, границы 1.88 с
#   videomae     0.814 / 0.814 / 0.594, границы 1.74 с
#   серые пиксели 0.805 / 0.785 / 0.715
# Закономерность: признаки, обученные распознавать действия на Kinetics, обходят
# самообучаемые. Ровно поэтому работы по temporal action segmentation стоят на I3D.
VIDEO_MODEL = os.getenv("PRAXIS_VIDEO_MODEL", "timesformer")
VIDEO_WINDOW = int(os.getenv("PRAXIS_VIDEO_WINDOW", "16"))
VIDEO_STRIDE = int(os.getenv("PRAXIS_VIDEO_STRIDE", "4"))
VIDEO_FPS = float(os.getenv("PRAXIS_VIDEO_FPS", "16"))
# Сколько главных компонент оставлять от признаков. Ноль — не сжимать.
# Сжатие главными компонентами нужно самообучаемым признакам (V-JEPA, VideoMAE), где
# соседние окна похожи на 0.997. Признакам, обученным на Kinetics, оно только вредит.
COMPONENTS = int(os.getenv("PRAXIS_COMPONENTS", "0"))

# Ручки сегментатора. Штраф за отрезок — главная: он задаёт гранулярность и защищает от
# пересегментации. Значения подобраны scripts/tune.py на признаках V-JEPA 2
# (F1@0.1 и F1@0.25 = 0.818, F1@0.5 = 0.707 без учёта меток на двадцати роликах).
# Минимальная длина шага намеренно оставлена скромной: подгонять её под крупные шаги
# одного набора рискованно, гранулярность скрытого набора неизвестна.
SEGMENT_PENALTY = float(os.getenv("PRAXIS_SEGMENT_PENALTY", "0.2"))
BOUNDARY_WEIGHT = float(os.getenv("PRAXIS_BOUNDARY_WEIGHT", "0.1"))
MAX_SEGMENTS = int(os.getenv("PRAXIS_MAX_SEGMENTS", "8"))
MIN_SEGMENT_SEC = float(os.getenv("PRAXIS_MIN_SEGMENT_SEC", "1.5"))
# Ниже какой доли среднего движения по ролику отрезок считается паузой, а не шагом.
# Ноль отключает пропуски и возвращает сплошное покрытие таймлайна.
IDLE_RATIO = float(os.getenv("PRAXIS_IDLE_RATIO", "0.45"))
MERGE_GAIN = float(os.getenv("PRAXIS_MERGE_GAIN", "0"))
