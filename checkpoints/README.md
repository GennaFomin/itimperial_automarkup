# Веса детектора границ

`boundary.pt` — упакованный ансамбль из двух MS-TCN (2 стадии, вход 768 = TimeSformer-K400,
окно 16 кадров при 32 fps = 0.5 с, шаг 4 кадра = 8 Гц, без каналов движения и разностей).
Сервис читает поле `members` и усредняет вероятности границ по кадрам; для старого
сервиса файл выглядит как одиночная модель (первый участник). Участники лежат в
`members/`, собрать заново:

    python scripts/pack_ensemble.py checkpoints/members/boundary-fine604-win05-s2.pt \
        checkpoints/members/boundary-fine604-robot60-win05-s2.pt --out checkpoints/boundary.pt

| участник | корпус | σ |
| --- | --- | --- |
| fine604-win05-s2 | train_atomic ∪ asm_extra — 554 ролика Assembly101 fine-grained | 2 кадра (±0.25 с) |
| fine604-robot60-win05-s2 | те же 554 + 60 эпизодов LIBERO с подзадачами | 2 кадра (±0.25 с) |

Замеры (F1@0.5, декодер приложения: пики выше 0.7, интервал 0.5 с) — в
`experiments/results/2026-09-05-boundary-final.md`: 85 атомарных роликов 0.498 (одиночная
модель в main до этого — 0.422), EPIC 0.535 (0.413), робот 0.733 (0.096).

Запуск сервиса:

    python scripts/serve_tas.py --port 8104 --checkpoint checkpoints/boundary.pt

`/health` должен показать `"dim": 768, "stages": 2, "members": 2`. Клиент извлекает
признаки с `PRAXIS_VIDEO_FPS=32`, `PRAXIS_VIDEO_WINDOW=16`, `PRAXIS_VIDEO_STRIDE=4` и шлёт
768 признаков (`PRAXIS_TAS_MOTION=0`, `PRAXIS_TAS_DIFF=0`); порог `PRAXIS_TAS_THRESHOLD=0.7`,
интервал `PRAXIS_MIN_SEGMENT_SEC=0.5` — всё это умолчания `praxis/config.py`.
