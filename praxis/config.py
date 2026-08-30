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
# auto — классификатор, если он поднят, иначе генеративная модель, иначе без семантики.
NAMER = os.getenv("PRAXIS_NAMER", "auto")
CLIP_MODE = os.getenv("PRAXIS_CLIP_MODE", "factored")
CLIP_VERB_WEIGHT = float(os.getenv("PRAXIS_CLIP_VERB_WEIGHT", "1.0"))
CLIP_NOUN_WEIGHT = float(os.getenv("PRAXIS_CLIP_NOUN_WEIGHT", "1.0"))
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

# Ручки сегментатора. Штраф за отрезок — главная: он задаёт гранулярность и защищает от
# пересегментации. Значения подобраны scripts/tune.py по двадцати роликам Assembly101
# (F1@0.25 = 0.785, F1@0.5 = 0.715 без учёта меток). Минимальная длина шага намеренно
# оставлена скромной: на этом наборе выгоднее 3.5 с, но это подгонка под его крупные шаги,
# а гранулярность скрытого набора нам пока неизвестна.
SEGMENT_PENALTY = float(os.getenv("PRAXIS_SEGMENT_PENALTY", "0.3"))
BOUNDARY_WEIGHT = float(os.getenv("PRAXIS_BOUNDARY_WEIGHT", "0.2"))
MAX_SEGMENTS = int(os.getenv("PRAXIS_MAX_SEGMENTS", "8"))
MIN_SEGMENT_SEC = float(os.getenv("PRAXIS_MIN_SEGMENT_SEC", "1.5"))
MERGE_GAIN = float(os.getenv("PRAXIS_MERGE_GAIN", "0"))
