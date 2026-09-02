"""Проверка харнесса метрик: он должен быть строгим там, где кейс строгий."""

from __future__ import annotations

import importlib.util
import sys
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "praxis_eval", Path(__file__).resolve().parent.parent / "scripts" / "eval.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
# Без регистрации в sys.modules ломается @dataclass: он ищет модуль класса по имени.
sys.modules["praxis_eval"] = evaluation
SPEC.loader.exec_module(evaluation)


def annotation(steps: list[tuple[float, float, str, str]], duration: float = 20.0) -> dict:
    return {
        "video": {
            "id": "clip",
            "filename": "clip.mp4",
            "duration_sec": duration,
            "fps": 30.0,
            "width": 1280,
            "height": 720,
        },
        "steps": [
            {
                "id": index,
                "level": "coarse",
                "parent_id": None,
                "start_sec": start,
                "end_sec": end,
                "action": action,
                "object": obj,
                "keyframe_sec": (start + end) / 2,
                "confidence": None,
                "source": "manual",
            }
            for index, (start, end, action, obj) in enumerate(steps)
        ],
        "provenance": {
            "app_version": "test",
            "pipeline": "test",
            "vocabulary": "assembly101-coarse",
            "models": {},
            "backend": None,
            "processing_sec": 3.0,
            "created_at": "2026-08-31T00:00:00Z",
        },
    }


TRUTH = [
    (0.0, 5.0, "attach", "wheel"),
    (5.0, 12.0, "screw", "chassis"),
    (12.0, 20.0, "attach", "cabin"),
]


@pytest.fixture
def write(tmp_path):
    def _write(name: str, payload: dict) -> Path:
        directory = tmp_path / name
        directory.mkdir(exist_ok=True)
        path = directory / "clip.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write


def test_identical_annotation_scores_perfectly(write):
    gt = write("gt", annotation(TRUTH))
    pred = write("pred", annotation(TRUTH))
    summary = evaluation.evaluate([(gt, pred)])["summary"]

    assert summary["f1@0.5"] == 1.0
    assert summary["boundary_mae_sec"] == 0.0
    assert summary["action_object_accuracy"] == 1.0
    assert summary["edit"] == 1.0
    assert summary["frame_accuracy"] == 1.0


def test_shifted_boundaries_degrade_predictably(write):
    shifted = [(start + 1.0, end + 1.0, action, obj) for start, end, action, obj in TRUTH[:-1]]
    shifted.append((TRUTH[-1][0] + 1.0, TRUTH[-1][1], TRUTH[-1][2], TRUTH[-1][3]))

    gt = write("gt", annotation(TRUTH))
    pred = write("pred", annotation(shifted))
    summary = evaluation.evaluate([(gt, pred)])["summary"]

    assert summary["boundary_mae_sec"] == pytest.approx(0.833, abs=0.01)
    assert summary["f1@0.1"] == 1.0, "при мягком пороге сегменты всё ещё сопоставляются"
    assert summary["action_object_accuracy"] == 1.0, "метки не менялись"


def test_wrong_labels_separate_segmentation_from_semantics(write):
    relabelled = [(start, end, "detach", "roof") for start, end, _, _ in TRUTH]
    gt = write("gt", annotation(TRUTH))
    pred = write("pred", annotation(relabelled))
    summary = evaluation.evaluate([(gt, pred)])["summary"]

    assert summary["f1@0.5"] == 0.0, "со сверкой меток всё должно упасть"
    assert summary["f1@0.5_nolabel"] == 1.0, "нарезка при этом идеальна"
    assert summary["action_accuracy"] == 0.0


def test_oversegmentation_is_punished(write):
    split = [(0.0, 2.5, "attach", "wheel"), (2.5, 5.0, "attach", "wheel")] + TRUTH[1:]
    gt = write("gt", annotation(TRUTH))
    pred = write("pred", annotation(split))
    summary = evaluation.evaluate([(gt, pred)])["summary"]

    assert summary["f1@0.5"] < 1.0, "лишний разрез обязан снижать F1"
    assert summary["f1@0.5_nolabel"] < 1.0


def test_case_rule_matches_on_action_only(write):
    """Правило кейса: совпадение при IoU >= 0.5 И верном действии. Предмет в матчинг не входит."""
    wrong_object = [(start, end, action, "roof") for start, end, action, _ in TRUTH]
    gt = write("gt", annotation(TRUTH))
    pred = write("pred", annotation(wrong_object))
    summary = evaluation.evaluate([(gt, pred)])["summary"]

    assert summary["case_f1"] == 1.0, "действия верны, предмет в правило кейса не входит"
    assert summary["f1@0.5"] == 0.0, "наш строгий вариант требует ещё и предмет"
    assert summary["f1@0.5_nolabel"] == 1.0


def test_case_rule_requires_correct_action(write):
    """Верные границы при неверном глаголе — промах и в step-F1 тоже, а не только в семантике."""
    wrong_action = [(start, end, "detach", obj) for start, end, _, obj in TRUTH]
    gt = write("gt", annotation(TRUTH))
    pred = write("pred", annotation(wrong_action))
    summary = evaluation.evaluate([(gt, pred)])["summary"]

    assert summary["case_f1"] == 0.0
    assert summary["f1@0.5_nolabel"] == 1.0, "нарезка при этом идеальна"


def test_precision_and_recall_are_reported_separately(write):
    """Кейс требует precision и recall явно: лишние шаги и пропущенные бьют по разным числам."""
    split = [(0.0, 2.4, "attach", "wheel"), (2.6, 5.0, "attach", "wheel")] + list(TRUTH[1:])
    gt = write("gt", annotation(TRUTH))
    pred = write("pred", annotation(split))
    summary = evaluation.evaluate([(gt, pred)])["summary"]

    assert summary["case_precision"] < summary["case_recall"], "лишний разрез бьёт по precision"
    assert 0.0 < summary["case_f1"] < 1.0


def test_missing_prediction_counts_as_total_miss(write):
    """Ролик, на котором пайплайн упал, обязан попасть в знаменатель, а не исчезнуть."""
    gt = write("gt", annotation(TRUTH))
    pred = write("pred", annotation(TRUTH))

    both = evaluation.evaluate([(gt, pred)])["summary"]
    with_failure = evaluation.evaluate([(gt, pred), (gt, None)])["summary"]

    assert both["case_f1"] == 1.0
    assert with_failure["clips"] == 2, "упавший ролик считается роликом"
    assert with_failure["case_recall"] == pytest.approx(0.5), "его шаги — пропуски"
    assert with_failure["case_f1"] < both["case_f1"]
