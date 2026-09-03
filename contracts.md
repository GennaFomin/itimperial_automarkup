# Контракты сервиса разметки действий по видео

`schema_version: 1.0` · статус: черновик к согласованию · владелец: — · дата: —

Документ фиксирует границы между ML-пайплайном, бэкендом и UI. Всё, что не описано здесь, считается неопределённым — не додумывать, а поднимать вопрос.

---

## 0. Карта контрактов

| № | Контракт | Кто отдаёт | Кто потребляет |
|---|---|---|---|
| 1 | Job API | бэкенд | UI |
| 2 | Prediction | ML-пайплайн | бэкенд → UI |
| 3 | Review | UI | бэкенд |
| 4 | Export | бэкенд | внешний потребитель |
| 5 | Vocabulary | бэкенд | UI, ML |

Отдельные контракты нужны потому, что prediction и review живут разными жизнями: прогноз неизменяем и используется для метрик, правка — производная от него. Смешивать их в одном документе нельзя.

---

## 1. Общие правила

**Время.** Только целые миллисекунды от начала видео (`int`). Не кадры — fps плавает и отличается между контейнером и декодером. Не float — сравнения и округления разъезжаются между Python и JS.

**Сегменты.** Внутри одного `prediction` сегменты не пересекаются и отсортированы по `start_ms`. Дыры между сегментами разрешены — это idle-участки, их не нужно ничем заполнять. Сплошное покрытие не требуем: иначе правка одной границы каскадно двигает соседа, и это усложняет и UI, и метрики.

**Неизвестное.** `unknown` — это значение в словаре, а не `null` и не пропущенное поле. Поля `action.value` и `object.value` всегда присутствуют и всегда строки. Уверенность передаётся отдельно.

**Confidence по полям, а не на сегмент.** Модель может уверенно найти границы и не понять объект. Один общий `confidence` не даёт UI подсветить конкретное место ошибки.

**Идентификаторы.** `segment.id` стабилен внутри одного прогона и не переносится между прогонами. Review всегда привязан к конкретному `prediction_id`, а не к «видео вообще».

**Версионирование.** Любой ответ содержит `schema_version`. Ломающее изменение — новый мажор, добавление опционального поля — минор. UI игнорирует незнакомые поля, а не падает.

**Незнакомое значение словаря.** Если UI встретил `action.value`, которого нет в его копии словаря, — показывает как есть, помечает предупреждением, не падает и не заменяет на `unknown`.

---

## 2. Job API

Все ответы `application/json; charset=utf-8`.

### POST /api/v1/jobs

Запуск обработки. `multipart/form-data` с файлом либо JSON со ссылкой.

```json
{
  "video_url": "https://storage/…/clip.mp4",
  "options": {
    "vocab_version": "1.0",
    "max_segments": 200
  }
}
```

Ответ `202 Accepted`:

```json
{ "job_id": "j_a81f3c", "status": "queued", "created_at": "2026-01-01T10:00:00Z" }
```

Лимиты (согласовать числами до старта): максимальная длительность ролика, максимальный размер файла, допустимые контейнеры и кодеки. Нарушение — `422` с кодом ошибки, а не `500`.

### GET /api/v1/jobs/{job_id}

```json
{
  "job_id": "j_a81f3c",
  "status": "running",
  "stage": "recognize",
  "progress": 0.62,
  "started_at": "2026-01-01T10:00:04Z",
  "finished_at": null,
  "error": null
}
```

`status`: `queued` · `running` · `done` · `done_with_errors` · `failed` · `cancelled`
`stage`: `decode` · `proposals` · `recognize` · `keyframe` · `validate`

`progress` — число `0…1`, монотонно не убывает. Если честного прогресса нет, отдаём прогресс по стадиям, но не откатываем назад.

**Поллинг:** интервал 2 с, бэкофф до 10 с после первой минуты. Таймаут на стороне UI — 10 мин, дальше показываем «долго» и кнопку отмены, но не убиваем job.

### GET /api/v1/jobs/{job_id}/prediction

Отдаёт документ из раздела 3. Пока `status` не `done` / `done_with_errors` — `409 Conflict`.

### POST /api/v1/jobs/{job_id}/review

Принимает документ из раздела 4. Ответ `200` с сохранённым `review_id` и `saved_at`.

### GET /api/v1/jobs/{job_id}/export?format=json|csv

Раздел 5. По умолчанию экспортируется review, если он есть, иначе prediction. Явно управляется параметром `source=review|prediction`.

### POST /api/v1/jobs/{job_id}/cancel

`200`, статус переходит в `cancelled`. Идемпотентно.

### GET /api/v1/vocab

Раздел 6.

### Ошибки

Единый формат для всех 4xx/5xx:

```json
{
  "error": {
    "code": "VIDEO_TOO_LONG",
    "message": "Ролик длиннее 5 минут",
    "details": { "duration_ms": 412000, "limit_ms": 300000 }
  }
}
```

Коды: `VIDEO_TOO_LONG` · `UNSUPPORTED_FORMAT` · `DECODE_FAILED` · `MODEL_TIMEOUT` · `JOB_NOT_FOUND` · `NOT_READY` · `INTERNAL`.
UI показывает `message`, логирует `code`. Тексты для пользователя живут на бэкенде — не дублировать их в UI.

---

## 3. Prediction

Неизменяемый документ. После записи не редактируется никогда — на нём считаются метрики.

```json
{
  "schema_version": "1.0",
  "prediction_id": "p_5f2a91",
  "job_id": "j_a81f3c",
  "model_version": "pipeline-0.3",
  "vocab_version": "1.0",
  "created_at": "2026-01-01T10:01:26Z",
  "video": {
    "duration_ms": 11400,
    "fps": 30.0,
    "width": 1920,
    "height": 1080
  },
  "segments": [
    {
      "id": "seg_001",
      "start_ms": 2000,
      "end_ms": 5100,
      "boundary_confidence": 0.92,
      "action": { "value": "move", "confidence": 0.87 },
      "object": { "value": "part", "confidence": 0.61 },
      "keyframe_ms": 3800,
      "keyframe_confidence": 0.74
    }
  ],
  "stats": {
    "latency_ms": 84000,
    "cost_usd": 0.12,
    "frames_decoded": 342
  },
  "errors": []
}
```

### Инварианты (проверяются валидатором в конце пайплайна)

1. `0 ≤ start_ms < end_ms ≤ video.duration_ms`
2. `start_ms ≤ keyframe_ms ≤ end_ms`
3. Сегменты отсортированы по `start_ms` и не пересекаются
4. `id` уникальны внутри документа
5. Все `confidence` в диапазоне `0…1`
6. `action.value` и `object.value` присутствуют в словаре версии `vocab_version`
7. `segments` может быть пустым массивом — это валидный результат, а не ошибка

### Частичный результат

Если стадия упала, но что-то посчитано — отдаём что есть, `status: done_with_errors`, и заполняем `errors`:

```json
"errors": [
  { "stage": "keyframe", "code": "MODEL_TIMEOUT", "message": "keyframe не посчитан для 3 сегментов", "segment_ids": ["seg_007","seg_008","seg_009"] }
]
```

Сегменты без keyframe отдаются с `keyframe_ms: null`. UI обязан это отрисовать (сегмент есть, кадр не выбран) и не показывать спиннер бесконечно.

---

## 4. Review

UI присылает **полный итоговый массив сегментов**, а не список операций. Диффом с prediction бэкенд получает телеметрию правок.

```json
{
  "schema_version": "1.0",
  "prediction_id": "p_5f2a91",
  "reviewer": "user_12",
  "submitted_at": "2026-01-01T10:07:03Z",
  "segments": [
    {
      "id": "seg_001",
      "origin": "model",
      "start_ms": 2100,
      "end_ms": 5100,
      "action": "move",
      "object": "part",
      "keyframe_ms": 3800
    },
    {
      "id": "seg_new_1",
      "origin": "human",
      "start_ms": 5100,
      "end_ms": 6400,
      "action": "place",
      "object": "part",
      "keyframe_ms": 5900
    }
  ],
  "time_spent_ms": 128000
}
```

`origin`: `model` — сегмент пришёл из прогноза (возможно, отредактирован), `human` — создан руками.
Удалённые сегменты просто отсутствуют в массиве.
Confidence в review нет — человек не выдаёт вероятностей.

### Что бэкенд считает диффом

`boundaries_edited` · `actions_changed` · `objects_changed` · `keyframes_moved` · `segments_added` · `segments_deleted` · `segments_untouched`

Это и есть база для метрики «≥ 3× быстрее разметки с нуля»: `time_spent_ms` review против замеренного времени ручной разметки того же ролика.

---

## 5. Export

### JSON

Плоский документ: `video`, `segments` (в формате review), `model_version`, `vocab_version`, `exported_at`, `source: review|prediction`. Без внутренних id джобов и без confidence, если `source=review`.

### CSV

UTF-8 с BOM, разделитель `,`, перевод строки `\n`, заголовок обязателен.

```
video_id,segment_id,start_ms,end_ms,action,object,keyframe_ms,confidence_action,confidence_object,origin
clip_01,seg_001,2000,5100,move,part,3800,0.87,0.61,model
```

Для `source=review` колонки confidence пустые. Порядок колонок фиксирован и не меняется без бампа версии.

Целевая метрика: **100% валидный JSON/CSV** — экспорт всегда проходит через тот же валидатор схемы, что и пайплайн.

---

## 6. Vocabulary

`GET /api/v1/vocab` — единственный источник правды по действиям и объектам. UI не хардкодит списки, иначе новый класс нельзя добавить без релиза фронта.

```json
{
  "version": "1.0",
  "actions": [
    { "id": "pick",    "label_ru": "Взять",       "color": "#FF5A1F" },
    { "id": "move",    "label_ru": "Переместить", "color": "#3DDCC8" },
    { "id": "place",   "label_ru": "Положить",    "color": "#1B2430" },
    { "id": "unknown", "label_ru": "Неизвестно",  "color": "#9AA3AD" }
  ],
  "objects": [
    { "id": "part",    "label_ru": "Деталь" },
    { "id": "tray",    "label_ru": "Лоток" },
    { "id": "tool",    "label_ru": "Инструмент" },
    { "id": "unknown", "label_ru": "Неизвестно" }
  ]
}
```

Цвета для таймлайна отдаёт словарь, а не подбирает UI — иначе цвета разъедутся между таймлайном и экспортом-скриншотом.

---

## 7. Метрики: где что считается

| Метрика | Считается из | Целевое |
|---|---|---|
| Step-level F1 | prediction ↔ ground truth, one-to-one matching, temporal IoU ≥ 0.5 + верный класс действия | ≥ 0.75 |
| Средняя ошибка границ | \|start_pred − start_gt\| и \|end_pred − end_gt\| по совпавшим шагам | ≤ 2 с |
| Точность action / object | по совпавшим шагам | ≥ 80% |
| Ускорение разметки | `review.time_spent_ms` vs baseline ручной разметки | ≥ 3× |
| Latency | `prediction.stats.latency_ms` | ≤ 2 мин |
| Валидность экспорта | доля прогонов, прошедших валидатор схемы | 100% |

Ground truth хранится отдельно от review. Review — это исправленный прогноз, он смещён в сторону модели и не годится как честный эталон.

---

## 8. Порядок работ

Первые два часа обе стороны работают против фикстуры, а не друг против друга:

```
/contracts
  prediction.schema.json
  review.schema.json
  vocab.schema.json
/fixtures
  prediction_ok.json
  prediction_with_errors.json
  prediction_empty.json
  review_example.json
  vocab.json
```

- UI рисует таймлайн по `fixtures/prediction_ok.json` — не ждёт модель
- ML дописывает валидацию схемы последним шагом пайплайна — падает на своей стороне, а не на демо
- Бэкенд поднимает моковые эндпоинты, отдающие фикстуры, за 20 минут

Обязательные сценарии для UI до подключения реального бэкенда: пустой результат, `done_with_errors` с пропущенными keyframe, ролик длиннее лимита, job упал, поллинг дольше 10 минут.

---

## 9. Реализация и отклонения

Контракт реализован слоем `/api/v1` поверх пайплайна praxis. Что именно совпало, что
отличается и почему — в [docs/CONTRACT_V1.md](docs/CONTRACT_V1.md). Коротко:

- **Уверенности по полям частично `null`.** Пайплайн производит одно число на шаг, и
  это уверенность именования пары «действие + объект». Оно уходит в `action.confidence`;
  `object.confidence`, `boundary_confidence` и `keyframe_confidence` приходят `null`,
  потому что таких величин пайплайн не измеряет. Заполнять их константой не стали:
  выдуманное число неотличимо от измеренного и портит доверие ко всей разметке.
  Ответ `prediction` несёт блок `capabilities`, который называет это машиночитаемо.
- **Стадии `proposals` и `keyframe` не возникают:** сегментация и именование слиты в
  одну стадию `recognize`.
- **`done_with_errors`** означает прогон, прошедший не в полную силу (сервис признаков
  или именования был недоступен); причина — в `errors[]` с кодом `DEGRADED`.
- **Расширения** (все опциональные): `verified_ids[]` и `mode` в теле правки,
  `capabilities`, `problems[]`, `id_map` в ответах, эндпоинты `GET /jobs`,
  `GET /limits`, `GET /stats`, `POST /jobs/{id}/activity`, `GET /jobs/{id}/media`,
  `GET /jobs/{id}/frame?ms=`, коды `VIDEO_TOO_SMALL`, `INVALID_REVIEW`, `DEGRADED`.
- **Не поддержано:** приём по ссылке (`video_url`), `options.vocab_version`,
  `options.max_segments`, `stats.frames_decoded`.

---

## 10. Открытые вопросы

| # | Вопрос | Решение по умолчанию | Кто решает |
|---|---|---|---|
| 1 | Лимит длительности ролика | 5 мин | — |
| 2 | Хранение видео — своё или ссылка | ссылка на внешнее хранилище | — |
| 3 | Авторизация | без неё на хакатоне, заголовок-заглушка | — |
| 4 | Несколько review на один prediction | разрешены, последний считается актуальным | — |
| 5 | keyframe как изображение в экспорте | не отдаём, только `keyframe_ms` | — |
| 6 | Пересечения сегментов в review | запрещены, UI валидирует до отправки | — |
| 7 | Кто наполняет ground truth | — | — |
