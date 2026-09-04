"""Словарь действий и объектов.

Кейс запрещает действия за пределами заданного списка, поэтому словарь — конфиг, а не
константа в коде: когда организаторы дадут свою таксономию, меняется один YAML.
"""

from __future__ import annotations

from praxis import config

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from praxis.schema import Annotation

DEFAULT_VOCAB = Path(__file__).parent / "data" / "vocab_assembly101_coarse.yaml"


class Vocabulary(BaseModel):
    name: str
    version: int = 1
    description: str = ""
    actions: list[str] = Field(min_length=1)
    objects: list[str] = Field(default_factory=list)
    # Необязательный белый список: какие объекты допустимы для каждого действия.
    pairs: dict[str, list[str]] | None = None
    # Необязательные подписи для интерфейса: «значение словаря → как показать человеку».
    # Пусто по умолчанию, и тогда подписью служит само значение. Машинного перевода нет
    # намеренно: при открытой лексике модель отвечает на языке PRAXIS_LANGUAGE, и её
    # ответ уже является той строкой, которую человек должен увидеть.
    labels: dict[str, str] = Field(default_factory=dict)

    def has_action(self, action: str) -> bool:
        return action in self.actions

    def has_object(self, obj: str) -> bool:
        return obj in self.objects

    def is_valid_pair(self, action: str, obj: str | None) -> bool:
        if not self.has_action(action):
            return False
        if obj is None:
            # Объект ещё не определён — это незаполненное поле, а не нарушение словаря.
            # Пайплайн отдаёт пустой объект, пока до сегмента не дошла семантическая стадия.
            return True
        if not self.has_object(obj):
            return False
        if self.pairs is None:
            return True
        return obj in self.pairs.get(action, [])

    def objects_for(self, action: str) -> list[str]:
        """Что предлагать в выпадающем списке редактора для выбранного действия."""
        if self.pairs is None:
            return self.objects
        return self.pairs.get(action, [])


def load_vocabulary(path: Path | str | None = None) -> Vocabulary:
    path = Path(path) if path else DEFAULT_VOCAB
    return Vocabulary(**yaml.safe_load(path.read_text(encoding="utf-8")))


def check_annotation(annotation: Annotation, vocabulary: Vocabulary) -> list[str]:
    """Мягкая проверка по словарю: возвращает список проблем, пустой список — всё чисто.

    Не исключение, потому что редактору нужно показать проблемы, а не упасть.
    """
    problems: list[str] = []
    for step in annotation.steps:
        if not vocabulary.has_action(step.action):
            problems.append(f"шаг {step.id}: действие «{step.action}» вне словаря")
        elif config.OPEN_VOCABULARY:
            pass  # словаря нет — сверять не с чем, это не ошибка разметки
        elif not vocabulary.is_valid_pair(step.action, step.object):
            problems.append(f"шаг {step.id}: пара «{step.action}» + «{step.object}» вне словаря")
    return problems
