"""Проверка разбиения на синтетике, где правильный ответ известен заранее."""

from __future__ import annotations

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
