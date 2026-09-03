"""Проверка разбиения на синтетике, где правильный ответ известен заранее."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from praxis.pipeline.segment import boundary_scores, pick_keyframe, segment

FPS = 10.0
rng = np.random.default_rng(0)


def clip_with_regimes(changes: list[int], length: int = 200, noise: float = 0.05):
    """Ролик из нескольких «режимов» с затиханием движения ровно на границах."""
    features = np.zeros((length, 4))
    edges = [0, *changes, length]
    for index, (start, end) in enumerate(zip(edges, edges[1:])):
        features[start:end] = index * 1.5
    features += rng.normal(0, noise, features.shape)

    motion = np.full(length, 0.6)
    motion += rng.normal(0, 0.02, length)
    for change in changes:
        motion[change - 2 : change + 2] = 0.03
    return features, motion


def test_recovers_known_boundaries():
    features, motion = clip_with_regimes([60, 130])
    bounds = segment(features, motion, fps=FPS)

    assert len(bounds) == 3, f"ожидали три шага, получили {bounds}"
    assert abs(bounds[0][1] - 60) <= 3
    assert abs(bounds[1][1] - 130) <= 3
    assert bounds[0][0] == 0 and bounds[-1][1] == len(features)


def test_uniform_clip_is_not_split():
    features = rng.normal(0, 0.05, (150, 4))
    motion = np.full(150, 0.5)
    assert len(segment(features, motion, fps=FPS)) == 1, "однородный ролик резать не за что"


def test_penalty_controls_granularity():
    features, motion = clip_with_regimes([40, 80, 120, 160])
    coarse = segment(features, motion, fps=FPS, penalty=1.0)
    fine = segment(features, motion, fps=FPS, penalty=0.001)
    assert len(coarse) < len(fine), "штраф обязан управлять числом шагов"
    assert len(fine) <= 8


def test_respects_segment_limits():
    features, motion = clip_with_regimes([40, 80, 120, 160])
    bounds = segment(features, motion, fps=FPS, penalty=0.001, max_segments=3)
    assert len(bounds) <= 3
    assert all(end - start >= 8 for start, end in bounds), "отрезки короче 0.8 с не допускаются"


def test_boundary_scores_peak_at_calm_moments():
    motion = np.full(100, 0.8)
    motion[50] = 0.0
    scores = boundary_scores(motion)
    assert scores[50] == pytest.approx(scores.max())


def test_keyframe_avoids_smeared_motion():
    features = np.tile(np.arange(3.0), (40, 1))
    motion = np.full(40, 0.9)
    motion[30] = 0.0
    assert pick_keyframe(features, motion, 0, 40) == 30


def test_short_clip_returns_single_segment():
    features = rng.normal(0, 1, (3, 4))
    assert segment(features, np.ones(3), fps=FPS) == [(0, 3)]


def test_learned_segmenter_falls_back_to_kernel_and_says_so(monkeypatch):
    """Без сервиса детектора нарезку делает ядровой change-point, а не один шаг на весь
    ролик: одиночный шаг выглядел бы как результат, а не как отказ. Причина уходит в
    статус, из которого jobs делает предупреждение."""
    import numpy as np

    from praxis import config
    from praxis.pipeline.base import Perception, get_segmenter
    from praxis.schema import VideoMeta
    from praxis.vocab import load_vocabulary

    monkeypatch.setattr(config, "TAS_BASE_URL", "")
    monkeypatch.setattr(config, "MIN_SEGMENT_SEC", 0.5)
    monkeypatch.setattr(config, "IDLE_RATIO", 0.0)
    rng = np.random.default_rng(0)
    first, second = rng.normal(size=64), rng.normal(size=64)
    features = np.vstack([first + 0.05 * rng.normal(size=(40, 64)),
                          second + 0.05 * rng.normal(size=(40, 64))]).astype(np.float32)
    perception = Perception(fps=8.0, motion=np.ones(80), appearance=features)
    meta = VideoMeta(id="v", filename="v.mp4", duration_sec=10.0, fps=30.0, width=1280, height=720)

    result = get_segmenter("learned-boundaries").run(Path("v.mp4"), meta, load_vocabulary(), perception)

    assert len(result.steps) >= 2, "откат обязан резать, а не отдавать один шаг"
    assert result.models["segmenter"] == "tsm-kernel"
    assert "детектор границ" in result.models["segmenter_status"]
