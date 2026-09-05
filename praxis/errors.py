"""Коды ошибок контракта и их конверт.

Контракт (§2) требует машиночитаемый код рядом с сообщением: UI показывает текст,
а логирует и ветвится по коду. Внутри praxis ошибки живут как `HTTPException` с
русской строкой в `detail` — этого достаточно для собственного редактора, но не для
внешнего потребителя, который обязан отличать «ролик длиннее лимита» от «сервис лёг».

Коды не выводятся разбором текста сообщения: текст меняется при первой же правке
формулировки, и такой разбор ломается молча. Вместо этого место, где ошибка
возникает, само называет свой код.
"""

from __future__ import annotations

from fastapi import HTTPException

# Коды из contracts.md §2 плюс те, что понадобились сверх него.
VIDEO_TOO_LONG = "VIDEO_TOO_LONG"
UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
DECODE_FAILED = "DECODE_FAILED"
MODEL_TIMEOUT = "MODEL_TIMEOUT"
JOB_NOT_FOUND = "JOB_NOT_FOUND"
NOT_READY = "NOT_READY"
INTERNAL = "INTERNAL"

# Расширения: кейс требует минимум 720p, а невалидную правку надо отличать от
# любой другой ошибки в теле запроса.
VIDEO_TOO_SMALL = "VIDEO_TOO_SMALL"
INVALID_REVIEW = "INVALID_REVIEW"
# Прогон прошёл не в полную силу: сервис признаков или именования был недоступен.
DEGRADED = "DEGRADED"
# Настройки прогона в запросе на создание задания вне допустимого.
INVALID_OPTIONS = "INVALID_OPTIONS"


class ContractError(HTTPException):
    """Ошибка, уже знающая свой код контракта.

    Наследуется от HTTPException, чтобы FastAPI обрабатывал её штатным путём, а
    обработчик в api.py просто доставал готовые поля вместо угадывания.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.message = message
        self.details = details or {}

    def envelope(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class UploadRejected(ValueError):
    """Ролик не прошёл требования кейса.

    Остаётся ValueError, потому что существующий обработчик `POST /api/videos`
    ловит именно его и отдаёт `str(error)`. Текст сообщения не меняется — старый
    эндпоинт продолжает отвечать ровно как раньше, а новый берёт из исключения код.
    """

    def __init__(self, code: str, message: str, **details) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def __str__(self) -> str:
        return self.message


def classify_failure(error_text: str | None) -> str:
    """Код для упавшего прогона.

    Статус `failed` в praxis не хранит причину машиночитаемо — только текст со
    стектрейсом. Разбор здесь неизбежен, но он ограничен ровно двумя случаями,
    которые контракт называет отдельными кодами; всё остальное честно INTERNAL,
    а не подгоняется под красивый код.
    """
    if not error_text:
        return INTERNAL
    head = error_text.splitlines()[0].lower()
    if "превысил" in head or "timeout" in head:
        return MODEL_TIMEOUT
    if "ffmpeg" in head or "ffprobe" in head or "mediaerror" in head or "декодир" in head:
        return DECODE_FAILED
    return INTERNAL


def first_line(error_text: str | None) -> str | None:
    """Сообщение без стектрейса: praxis кладёт в `error` traceback следом за текстом."""
    if not error_text:
        return None
    return error_text.splitlines()[0]
