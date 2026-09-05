# Веса детектора границ

`boundary.pt` — ансамбль-консенсус из двух MS-TCN (2 стадии, вход 768 = TimeSformer-K400,
окно 16 кадров при 32 fps = 0.5 с, шаг 4 кадра = 8 Гц, без каналов движения и разностей).
Сервис читает поля `members` и `fusion`: при `fusion: min` разрез ставится только там, где
каждый участник видит смену действия (минимум вероятностей по кадрам). Для старого
сервиса файл выглядит как одиночная модель (первый участник). Собрать заново:

    python scripts/pack_ensemble.py checkpoints/members/boundary-fine604-win05-s2.pt \
        checkpoints/members/boundary-atomic203-win05-s2.pt --fusion min --out checkpoints/boundary.pt

| участник | корпус | σ |
| --- | --- | --- |
| fine604-win05-s2 | train_atomic ∪ asm_extra — 554 ролика Assembly101 fine-grained | 2 кадра (±0.25 с) |
| atomic203-win05-s2 | train_atomic — 203 ролика Assembly101 fine-grained | 2 кадра (±0.25 с) |

Замеры (F1@0.5, декодер приложения: пики выше 0.5, интервал 0.5 с) — в
`experiments/results/2026-09-05-boundary-final.md`: 85 атомарных роликов 0.503 (одиночная
модель в main до этого — 0.422), EPIC 0.516 (0.413), роботный держ-аут 0.483 (0.096).
Согласие двух моделей делает результат почти нечувствительным к порогу (0.497–0.503 на
85 роликах при порогах 0.3–0.7) — это и есть причина выбора для неизвестного домена.

Запуск сервиса:

    python scripts/serve_tas.py --port 8104 --checkpoint checkpoints/boundary.pt

`/health` должен показать `"dim": 768, "stages": 2, "members": 2, "fusion": "min"`. Клиент
извлекает признаки с `PRAXIS_VIDEO_FPS=32`, `PRAXIS_VIDEO_WINDOW=16`, `PRAXIS_VIDEO_STRIDE=4`
и шлёт 768 признаков (`PRAXIS_TAS_MOTION=0`, `PRAXIS_TAS_DIFF=0`); порог
`PRAXIS_TAS_THRESHOLD=0.5`, интервал `PRAXIS_MIN_SEGMENT_SEC=0.5` — умолчания `praxis/config.py`.

`gapness.pt` — опциональный детектор пауз-переходов: та же MS-TCN, обученная на кадры вне
размеченных шагов Assembly101 fine (`train_boundaries.py --target gaps`). По умолчанию не
используется (см. `experiments/results/2026-09-05-boundary-final.md`, раздел про паузы);
подключается вторым сервисом: `python scripts/serve_tas.py --port 8105 --checkpoint
checkpoints/gapness.pt` и `PRAXIS_GAP_BASE_URL=http://127.0.0.1:8105`.
