"""Конверсия между внутренней моделью и контрактом — без HTTP и без базы.

Отдельный файл от `test_api_v1.py` намеренно: сквозные тесты идут на заглушке
сегментатора, а она отдаёт круглые секунды, на которых ошибки округления просто
не проявляются. Здесь времена задаются руками — такие, какие даёт реальный
пайплайн (`motion-dp` кладёт границы без округления, длительность приходит из
ffprobe).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from praxis import contract_v1
from praxis.schema import Annotation, Level, Provenance, Source, Step, VideoMeta, diff_steps


def make_annotation(steps: list[Step], duration_sec: float = 20.0) -> Annotation:
    return Annotation(
        video=VideoMeta(
            id="v1", filename="clip.mp4", duration_sec=duration_sec, fps=30.0, width=1280, height=720
        ),
        steps=steps,
        provenance=Provenance(
            app_version="0.1.0",
            pipeline="tsm-kernel",
            vocabulary="test",
            created_at=datetime(2026, 9, 3, tzinfo=UTC),
        ),
    )


def step(step_id: int, start: float, end: float, **kwargs) -> Step:
    return Step(
        id=step_id,
        start_sec=start,
        end_sec=end,
        action=kwargs.get("action", "attach"),
        object=kwargs.get("object", "wheel"),
        keyframe_sec=kwargs.get("keyframe", (start + end) / 2),
        confidence=kwargs.get("confidence", 0.8),
        source=kwargs.get("source", Source.auto),
        verified=kwargs.get("verified", False),
    )


def as_review(prediction: dict) -> list[dict]:
    """Прогноз в том виде, в каком его вернёт редактор, ничего не изменив."""
    return [
        {
            "id": segment["id"],
            "origin": "model",
            "start_ms": segment["start_ms"],
            "end_ms": segment["end_ms"],
            "action": segment["action"]["value"],
            "object": segment["object"]["value"],
            "keyframe_ms": segment["keyframe_ms"],
        }
        for segment in prediction["segments"]
    ]


# Границы, которые не ложатся на целые миллисекунды. Ровно такие приходят из
# motion-dp и из длительности ffprobe.
RAGGED = [
    step(0, 0.0, 5.10049, keyframe=2.55012),
    step(1, 5.10049, 11.00071, keyframe=8.0004),
    step(2, 11.00071, 19.99983, keyframe=15.4),
]


def test_round_trip_on_ragged_times_is_not_an_edit():
    """Прогноз, вернувшийся без изменений, не должен выглядеть правкой человека.

    Наружу время уходит целыми миллисекундами, внутри оно в секундах с плавающей
    точкой. Без защиты `5.10049 → 5100 → 5.100` записалось бы как сдвиг границы —
    и телеметрия правок завысила бы работу человека ровно в той величине, на
    которой строится KPI «в три раза быстрее».
    """
    base = make_annotation(RAGGED, duration_sec=19.99983)
    prediction = contract_v1.to_prediction(base, {"id": "v1"}, base.model_dump_json())

    restored, problems, id_map = contract_v1.annotation_from_review(
        as_review(prediction), base=base, prediction=base, verified_ids=set()
    )

    assert not problems and not id_map
    changes = diff_steps(base.steps, restored.steps)
    assert changes == {
        "boundary": 0,
        "action": 0,
        "object": 0,
        "keyframe": 0,
        "added": 0,
        "removed": 0,
    }, f"округление выдумало правки: {changes}"


def test_real_edits_survive_the_round_trip_guard():
    """Защита от округления не должна проглатывать настоящую правку.

    Допуск в половину миллисекунды обязан отличать артефакт конверсии от сдвига
    границы человеком, иначе он молча отменял бы работу.
    """
    base = make_annotation(RAGGED, duration_sec=19.99983)
    prediction = contract_v1.to_prediction(base, {"id": "v1"}, base.model_dump_json())

    segments = as_review(prediction)
    segments[1]["start_ms"] += 250  # человек сдвинул границу на четверть секунды
    segments[1]["action"] = "detach"

    restored, _, _ = contract_v1.annotation_from_review(
        segments, base=base, prediction=base, verified_ids=set()
    )
    changes = diff_steps(base.steps, restored.steps)
    assert changes["boundary"] == 1
    assert changes["action"] == 1
    assert restored.steps[1].source is Source.edited


def test_end_is_clamped_to_duration():
    """Округление конца не должно выносить сегмент за длительность ролика.

    `round(19.9996 * 1000)` даёт 20000 при длительности 19999 мс — это нарушает
    инвариант, который проверяют и валидатор, и редактор.
    """
    base = make_annotation([step(0, 0.0, 19.9996)], duration_sec=19.9996)
    prediction = contract_v1.to_prediction(base, {"id": "v1"}, base.model_dump_json())
    duration_ms = prediction["video"]["duration_ms"]
    assert prediction["segments"][0]["end_ms"] <= duration_ms


def test_unknown_object_survives_both_directions():
    """Контракт знает «unknown» как значение, praxis — как отсутствие объекта."""
    base = make_annotation([step(0, 0.0, 5.0, object=None)])
    prediction = contract_v1.to_prediction(base, {"id": "v1"}, base.model_dump_json())
    assert prediction["segments"][0]["object"]["value"] == "unknown"

    restored, _, _ = contract_v1.annotation_from_review(
        as_review(prediction), base=base, prediction=base, verified_ids=set()
    )
    assert restored.steps[0].object is None


def test_human_segment_gets_an_id_and_manual_source():
    base = make_annotation([step(0, 0.0, 5.0)])
    segments = [
        {
            "id": "seg_000",
            "origin": "model",
            "start_ms": 0,
            "end_ms": 5000,
            "action": "attach",
            "object": "wheel",
            "keyframe_ms": 2500,
        },
        {
            "id": "seg_new_zz",
            "origin": "human",
            "start_ms": 6000,
            "end_ms": 8000,
            "action": "проверить",
            "object": "деталь",
            "keyframe_ms": 7000,
        },
    ]
    restored, _, id_map = contract_v1.annotation_from_review(
        segments, base=base, prediction=base, verified_ids={"seg_new_zz"}
    )
    assert id_map == {"seg_new_zz": "seg_001"}
    created = restored.steps[1]
    assert created.id == 1
    assert created.source is Source.manual
    assert created.verified is True
    # Человек не выдаёт вероятностей (§4).
    assert created.confidence is None


def test_deleted_segments_simply_disappear():
    base = make_annotation([step(0, 0.0, 5.0), step(1, 5.0, 10.0)])
    prediction = contract_v1.to_prediction(base, {"id": "v1"}, base.model_dump_json())
    segments = as_review(prediction)[:1]

    restored, _, _ = contract_v1.annotation_from_review(
        segments, base=base, prediction=base, verified_ids=set()
    )
    assert len(restored.steps) == 1
    assert diff_steps(base.steps, restored.steps)["removed"] == 1


def test_provenance_never_comes_from_the_client():
    """Происхождение описывает прогон модели и не может приходить из браузера."""
    base = make_annotation([step(0, 0.0, 5.0)])
    prediction = contract_v1.to_prediction(base, {"id": "v1"}, base.model_dump_json())
    restored, _, _ = contract_v1.annotation_from_review(
        as_review(prediction), base=base, prediction=base, verified_ids=set()
    )
    assert restored.provenance == base.provenance


def test_fine_steps_are_dropped_when_their_parent_moves():
    """Подшаг нельзя оставить вложенным во что-то, чего больше нет.

    Клампить его границы под новые родительские означало бы придумать данные,
    поэтому он отбрасывается, и об этом сообщается явно.
    """
    parent = step(0, 0.0, 10.0)
    child = Step(
        id=1,
        level=Level.fine,
        parent_id=0,
        start_sec=1.0,
        end_sec=3.0,
        action="attach",
        object="wheel",
        keyframe_sec=2.0,
    )
    base = make_annotation([parent, child])

    moved = [
        {
            "id": "seg_000",
            "origin": "model",
            "start_ms": 0,
            "end_ms": 4000,
            "action": "attach",
            "object": "wheel",
            "keyframe_ms": 2000,
        }
    ]
    restored, problems, _ = contract_v1.annotation_from_review(
        moved, base=base, prediction=base, verified_ids=set()
    )
    assert len(restored.steps) == 1
    assert any("подшаг" in problem for problem in problems)


def test_fine_steps_survive_when_the_parent_is_untouched():
    parent = step(0, 0.0, 10.0)
    child = Step(
        id=1,
        level=Level.fine,
        parent_id=0,
        start_sec=1.0,
        end_sec=3.0,
        action="attach",
        object="wheel",
        keyframe_sec=2.0,
    )
    base = make_annotation([parent, child])
    prediction = contract_v1.to_prediction(base, {"id": "v1"}, base.model_dump_json())
    # Контракт плоский: наружу уходит только верхний уровень.
    assert len(prediction["segments"]) == 1

    restored, problems, _ = contract_v1.annotation_from_review(
        as_review(prediction), base=base, prediction=base, verified_ids=set()
    )
    assert not problems
    assert len(restored.steps) == 2


def test_segment_ids_round_trip():
    assert contract_v1.segment_id(7) == "seg_007"
    assert contract_v1.step_id("seg_007") == 7
    assert contract_v1.step_id("seg_new_a1") is None
    assert contract_v1.step_id("") is None


def test_colors_are_stable_and_distinct():
    assert contract_v1.color_for("attach") == contract_v1.color_for("attach")
    assert contract_v1.color_for("attach") != contract_v1.color_for("detach")
    assert contract_v1.color_for("unknown") == "#9aa3ad"
    assert contract_v1.color_for("взять").startswith("#")


@pytest.mark.parametrize("value", [0.0, 0.001, 5.4321, 19.9999])
def test_milliseconds_are_integers(value):
    base = make_annotation([step(0, 0.0, max(value, 0.5))], duration_sec=20.0)
    prediction = contract_v1.to_prediction(base, {"id": "v1"}, base.model_dump_json())
    segment = prediction["segments"][0]
    assert isinstance(segment["start_ms"], int)
    assert isinstance(segment["end_ms"], int)
