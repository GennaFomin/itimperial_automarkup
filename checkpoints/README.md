# Веса детектора границ

`boundary.pt` — обученная голова детектора границ действий (MS-TCN, 2 стадии, вход
768 признаков TimeSformer-K400, обучена на 203 роликах Assembly101 fine-grained). Это
модель, которая стоит на демо и даёт F1@0.5 = 0.41 без сверки меток на 85 атомарных
роликах; ошибка границ ≈ 0.4 с.

Как поднять сервис (на машине с GPU, из корня репозитория):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/serve_tas.py --port 8104 --checkpoint checkpoints/boundary.pt
curl http://127.0.0.1:8104/health   # {"ready":true,...,"dim":768,"stages":2}
```

Приложение указывает на него через `PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104` (см.
`.env.example`, `docs/PIPELINE.md`). Файл 1.7 МБ, формат `torch.save`: словарь с
`dim`, `weights` и, для новых версий, `stages`.
