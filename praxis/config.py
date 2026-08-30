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
PIPELINE = os.getenv("PRAXIS_PIPELINE", "stub")

# Куда ходить за VLM. Пусто — значит семантика недоступна и работает запасной путь.
VLM_BASE_URL = os.getenv("PRAXIS_VLM_BASE_URL", "")
VLM_MODEL = os.getenv("PRAXIS_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

# Плёнка кадров под таймлайном и разрешение сигнала движения.
FILMSTRIP_COUNT = 40
MOTION_FPS = 10
