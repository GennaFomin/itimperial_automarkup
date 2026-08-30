"""Стадия именования: применение ответа модели и поведение при отказе сервиса."""

from __future__ import annotations

from pathlib import Path

from praxis.pipeline.naming import NullNamer, RemoteVlmNamer, get_namer
from praxis.schema import Source, Step, VideoMeta
from praxis.vocab import load_vocabulary

VIDEO = VideoMeta(id="v", filename="clip.mp4", duration_sec=20.0, fps=30.0, width=1280, height=720)


def steps() -> list[Step]:
    return [
        Step(id=0, start_sec=0.0, end_sec=8.0, action="inspect", object=None, confidence=0.3),
        Step(id=1, start_sec=8.0, end_sec=20.0, action="inspect", object=None, confidence=0.3),
    ]


class FakeVlm(RemoteVlmNamer):
    """Подменяем сеть и извлечение кадров: проверяем разбор ответа, а не ffmpeg."""

    def __init__(self, results: list[dict]) -> None:
        super().__init__("http://example.invalid")
        self.results = results

    def _frames(self, video_path: Path, step: Step, crop=None) -> list[str]:
        return ["ZmFrZQ=="]

    def _post(self, path: str, payload: dict) -> dict:
        self.payload = payload
        return {"results": self.results, "model": "qwen3-vl", "elapsed_sec": 1.2}


def test_null_namer_leaves_steps_untouched():
    original = steps()
    result = NullNamer().name_steps(Path("clip.mp4"), VIDEO, original, load_vocabulary())
    assert result.steps == original
    assert result.models["namer"] == "none"


def test_applies_labels_from_vocabulary():
    vocabulary = load_vocabulary()
    namer = FakeVlm(
        [
            {"id": 0, "action": "attach", "object": "wheel", "confidence": 0.82},
            {"id": 1, "action": "screw", "object": "chassis", "confidence": 0.61},
        ]
    )
    result = namer.name_steps(Path("clip.mp4"), VIDEO, steps(), vocabulary)

    assert [(s.action, s.object) for s in result.steps] == [
        ("attach", "wheel"),
        ("screw", "chassis"),
    ]
    assert result.steps[0].confidence == 0.82
    assert result.models["vlm"] == "qwen3-vl"
    # Словарь уезжает на сервис целиком: модель не должна выдумывать значения.
    assert namer.payload["actions"] == vocabulary.actions
    assert namer.payload["pairs"]["attach"]


def test_rejects_values_outside_vocabulary():
    namer = FakeVlm([{"id": 0, "action": "танцует", "object": "банан", "confidence": 0.9}])
    result = namer.name_steps(Path("clip.mp4"), VIDEO, steps(), load_vocabulary())

    assert result.steps[0].action == "inspect", "выдуманное действие не должно применяться"
    assert result.steps[0].object is None, "выдуманный объект отбрасывается"


def test_survives_unreachable_service():
    """Отказ сервиса не должен ронять разметку: шаги остаются, статус попадает в провенанс."""
    namer = RemoteVlmNamer("http://127.0.0.1:9/annotate", frames_per_step=1, timeout=1)
    namer._frames = lambda video_path, step, crop=None: ["ZmFrZQ=="]  # noqa: SLF001

    original = steps()
    result = namer.name_steps(Path("clip.mp4"), VIDEO, original, load_vocabulary())

    assert result.steps == original
    assert "недоступен" in result.models["namer_status"]


def test_namer_choice_follows_configuration(monkeypatch):
    from praxis import config

    monkeypatch.setattr(config, "VLM_BASE_URL", "")
    assert isinstance(get_namer(), NullNamer)

    monkeypatch.setattr(config, "VLM_BASE_URL", "http://dl5:8100")
    assert isinstance(get_namer(), RemoteVlmNamer)


def test_steps_keep_source_after_naming():
    """Метка от модели — по-прежнему автоматическая разметка, а не правка человека."""
    namer = FakeVlm([{"id": 0, "action": "attach", "object": "wheel", "confidence": 0.7}])
    result = namer.name_steps(Path("clip.mp4"), VIDEO, steps(), load_vocabulary())
    assert result.steps[0].source is Source.auto
