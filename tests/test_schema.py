import pytest
from pydantic import ValidationError

from praxis.schema import (
    Annotation,
    Level,
    Provenance,
    Source,
    Step,
    VideoMeta,
    steps_from_csv,
    to_csv,
)
from praxis.vocab import check_annotation, load_vocabulary

VIDEO = VideoMeta(
    id="v1", filename="clip.mp4", duration_sec=20.0, fps=30.0, width=1920, height=1080
)
PROV = Provenance(app_version="0.1.0", pipeline="stub", vocabulary="assembly101-coarse")


def annotation(*steps: Step) -> Annotation:
    return Annotation(video=VIDEO, steps=list(steps), provenance=PROV)


def step(step_id: int, start: float, end: float, **kwargs) -> Step:
    kwargs.setdefault("action", "attach")
    kwargs.setdefault("object", "wheel")
    return Step(id=step_id, start_sec=start, end_sec=end, **kwargs)


def test_json_roundtrip():
    original = annotation(
        step(0, 0.0, 4.5, keyframe_sec=2.0, confidence=0.9),
        step(1, 4.5, 12.25, action="screw", object="chassis"),
    )
    restored = Annotation.model_validate_json(original.model_dump_json())
    assert restored == original


def test_csv_roundtrip_preserves_steps():
    original = annotation(
        step(0, 0.0, 4.5, keyframe_sec=2.0, confidence=0.9),
        step(1, 4.5, 12.25, action="screw", object="chassis", source=Source.edited),
        step(2, 4.5, 8.0, level=Level.fine, parent_id=1, action="attach", object="nut"),
    )
    restored = steps_from_csv(to_csv(original))
    assert restored == original.steps
    assert Annotation(video=VIDEO, steps=restored, provenance=PROV) == original


def test_csv_header_is_stable():
    header = to_csv(annotation(step(0, 0.0, 1.0))).splitlines()[0]
    assert header.startswith("video_id,filename,step_id,level,parent_id,start_sec,end_sec")


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda: step(0, 5.0, 5.0), id="нулевая длительность"),
        pytest.param(lambda: step(0, 5.0, 1.0), id="конец раньше начала"),
        pytest.param(lambda: step(0, 0.0, 5.0, keyframe_sec=9.0), id="ключевой кадр вне шага"),
        pytest.param(lambda: step(0, 0.0, 5.0, confidence=1.4), id="уверенность вне 0..1"),
        pytest.param(lambda: step(0, 0.0, 5.0, level=Level.fine), id="подшаг без родителя"),
        pytest.param(lambda: step(0, 0.0, 5.0, parent_id=1), id="крупный шаг с родителем"),
    ],
)
def test_step_rejects_broken_values(bad):
    with pytest.raises(ValidationError):
        bad()


def test_rejects_overlapping_steps_on_same_level():
    with pytest.raises(ValidationError, match="пересекаются"):
        annotation(step(0, 0.0, 5.0), step(1, 4.0, 8.0))


def test_allows_touching_steps():
    assert len(annotation(step(0, 0.0, 5.0), step(1, 5.0, 8.0)).steps) == 2


def test_rejects_step_beyond_video_duration():
    with pytest.raises(ValidationError, match="длительность видео"):
        annotation(step(0, 0.0, 25.0))


def test_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="не уникальны"):
        annotation(step(0, 0.0, 5.0), step(0, 5.0, 8.0))


def test_rejects_substep_outside_parent():
    with pytest.raises(ValidationError, match="не вложен"):
        annotation(
            step(0, 0.0, 5.0),
            step(1, 4.0, 6.0, level=Level.fine, parent_id=0),
        )


def test_rejects_substep_with_unknown_parent():
    with pytest.raises(ValidationError, match="родитель"):
        annotation(step(1, 0.0, 5.0, level=Level.fine, parent_id=7))


def test_vocabulary_accepts_known_pairs_and_reports_unknown():
    vocab = load_vocabulary()
    assert vocab.is_valid_pair("attach", "wheel")
    assert not vocab.is_valid_pair("attach", "banana")
    assert not vocab.is_valid_pair("fly", "wheel")
    assert "wheel" in vocab.objects_for("attach")

    ok = annotation(step(0, 0.0, 5.0, action="attach", object="wheel"))
    assert check_annotation(ok, vocab) == []

    bad = annotation(step(0, 0.0, 5.0, action="attach", object="banana"))
    assert len(check_annotation(bad, vocab)) == 1


def test_vocabulary_has_expected_assembly101_shape():
    vocab = load_vocabulary()
    assert len(vocab.actions) == 11
    assert len(vocab.objects) == 61
    assert sum(len(v) for v in vocab.pairs.values()) == 202


def test_verified_flag_survives_roundtrip_and_can_be_hidden():
    from praxis.schema import to_json

    original = annotation(
        step(0, 0.0, 5.0, verified=True),
        step(1, 5.0, 12.0, action="screw", object="chassis"),
    )

    assert steps_from_csv(to_csv(original)) == original.steps
    assert '"verified": true' in to_json(original)

    # Приёмка может не ждать лишних полей — тогда отметка гасится одним флагом.
    hidden_csv = to_csv(original, include_verified=False)
    assert "verified" not in hidden_csv.splitlines()[0]
    assert "verified" not in to_json(original, include_verified=False)

    # Старый CSV без колонки читается по-прежнему.
    assert steps_from_csv(hidden_csv)[0].verified is False


def test_missing_object_is_not_a_vocabulary_violation():
    """Пустой объект — незаполненное поле, а не нарушение словаря.

    Сегментатор отдаёт шаги без объекта, пока до них не дошла семантическая стадия;
    подсвечивать это как ошибку словаря было бы неверно.
    """
    vocab = load_vocabulary()
    assert vocab.is_valid_pair("attach", None)
    assert check_annotation(annotation(step(0, 0.0, 5.0, object=None)), vocab) == []
