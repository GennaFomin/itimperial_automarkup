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


def test_merges_adjacent_steps_with_the_same_label():
    """Три подряд «place tray» — это один шаг, разрезанный по смене картинки."""
    from praxis.pipeline.naming import merge_adjacent

    parts = [
        Step(id=0, start_sec=0.0, end_sec=1.6, action="place", object="tray", confidence=0.4),
        Step(id=1, start_sec=1.6, end_sec=3.7, action="place", object="tray", confidence=0.4),
        Step(id=2, start_sec=3.7, end_sec=5.2, action="place", object="tray", confidence=0.3),
        Step(id=3, start_sec=5.2, end_sec=7.4, action="put-down", object="tray", confidence=0.4),
    ]
    merged = merge_adjacent(parts)

    assert [(s.action, s.object) for s in merged] == [("place", "tray"), ("put-down", "tray")]
    assert merged[0].start_sec == 0.0 and merged[0].end_sec == 5.2
    assert [s.id for s in merged] == [0, 1], "идентификаторы должны стать подряд идущими"


def test_keeps_different_labels_apart():
    from praxis.pipeline.naming import merge_adjacent

    parts = [
        Step(id=0, start_sec=0.0, end_sec=2.0, action="open", object="drawer"),
        Step(id=1, start_sec=2.0, end_sec=4.0, action="take", object="spoon"),
    ]
    assert len(merge_adjacent(parts)) == 2


def test_does_not_merge_without_real_labels():
    """Без именования у всех шагов одна заглушка — склейка схлопнула бы весь ролик."""
    from praxis.pipeline.naming import merge_adjacent

    parts = [
        Step(id=0, start_sec=0.0, end_sec=4.0, action="inspect", object=None),
        Step(id=1, start_sec=4.0, end_sec=9.0, action="inspect", object=None),
        Step(id=2, start_sec=9.0, end_sec=14.0, action="inspect", object=None),
    ]
    assert len(merge_adjacent(parts, labelled=False)) == 3
    assert len(merge_adjacent(parts, labelled=True)) == 1


def test_namer_request_carries_every_mode_flag(monkeypatch, tmp_path):
    """Флаги режима обязаны доезжать до сервиса: без open_vocabulary он молча притягивает
    ответ к словарю, без context_frames подписи «до/после» не ставятся. Однажды эти
    строки потерялись при переписывании истории, и целый день замеров ушёл впустую."""
    from praxis import config
    from praxis.pipeline import naming

    sent: dict = {}

    class Fake(naming.RemoteVlmNamer):
        def _frames(self, *args, **kwargs):
            return ["x"]

        def _post(self, path, payload, base_url=None):
            sent.update(payload)
            return {"results": []}

    monkeypatch.setattr(config, "OPEN_VOCABULARY", True)
    monkeypatch.setattr(config, "LANGUAGE", "en")
    monkeypatch.setattr(config, "CONTEXT_FRAMES", 0.6)
    monkeypatch.setattr(config, "TRACK_BASE_URL", "")
    monkeypatch.setattr(config, "VLM_CONTEXT", False)
    monkeypatch.setattr(config, "VLM_TWO_STAGE", False)
    monkeypatch.setattr(config, "VLM_RESCORE", 0)

    from praxis.schema import Step, VideoMeta
    from praxis.vocab import load_vocabulary

    meta = VideoMeta(id="v", filename="v.mp4", duration_sec=10.0, fps=30.0, width=1280, height=720)
    steps = [Step(id=0, start_sec=0.0, end_sec=5.0, action="attach", object=None,
                  keyframe_sec=2.5, confidence=None)]
    Fake("http://127.0.0.1:1").name_steps(tmp_path / "v.mp4", meta, steps, load_vocabulary(), None)

    assert sent.get("open_vocabulary") is True
    assert sent.get("language") == "en"
    assert sent.get("context_frames") is True
