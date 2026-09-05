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


def test_learned_segmenter_skips_the_detector_on_degraded_features(monkeypatch):
    """Признаки серых блоков имеют другую размерность, и детектор отвечает на них 500.
    Звать его в таком прогоне нельзя — сразу ядровой, с причиной в статусе."""
    import numpy as np

    from praxis import config
    from praxis.pipeline.base import Perception, get_segmenter
    from praxis.schema import VideoMeta
    from praxis.vocab import load_vocabulary

    monkeypatch.setattr(config, "TAS_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(config, "MIN_SEGMENT_SEC", 0.5)
    monkeypatch.setattr(config, "IDLE_RATIO", 0.0)
    rng = np.random.default_rng(1)
    features = np.vstack([rng.normal(size=(40, 64)), 5 + rng.normal(size=(40, 64))]).astype(np.float32)
    perception = Perception(fps=8.0, motion=np.ones(80), appearance=features,
                            degraded=("сервис признаков недоступен: нарезка на серых блоках",))
    meta = VideoMeta(id="v", filename="v.mp4", duration_sec=10.0, fps=30.0, width=1280, height=720)

    result = get_segmenter("learned-boundaries").run(Path("v.mp4"), meta, load_vocabulary(), perception)

    assert result.models["segmenter"] == "tsm-kernel"
    assert "признаки просели" in result.models["segmenter_status"]


def test_stack_motion_appends_one_channel():
    import numpy as np

    from praxis.pipeline.learned import stack_motion

    appearance = np.zeros((7, 768), dtype=np.float32)
    motion = np.linspace(0, 1, 7)
    stacked = stack_motion(appearance, motion)
    assert stacked.shape == (7, 769) and stacked.dtype == np.float32
    assert np.allclose(stacked[:, -1], motion)


def test_stack_motion_tolerates_length_mismatch():
    """Полоса движения на кадр короче или длиннее — обрезаем/дополняем, а не падаем."""
    import numpy as np

    from praxis.pipeline.learned import stack_motion

    appearance = np.zeros((7, 768), dtype=np.float32)
    assert stack_motion(appearance, np.ones(9)).shape == (7, 769)
    assert stack_motion(appearance, np.ones(5)).shape == (7, 769)


def test_learned_segmenter_falls_back_on_dimension_mismatch(monkeypatch):
    """Сервис ждёт другую размерность (чекпоинт без канала движения) → 400 → ядровой."""
    import urllib.error

    import numpy as np

    from praxis import config
    from praxis.pipeline import learned
    from praxis.pipeline.base import Perception, get_segmenter
    from praxis.schema import VideoMeta
    from praxis.vocab import load_vocabulary

    monkeypatch.setattr(config, "TAS_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(config, "MIN_SEGMENT_SEC", 0.5)
    monkeypatch.setattr(config, "IDLE_RATIO", 0.0)

    def refuse(path, payload, timeout=None):
        import io
        raise urllib.error.HTTPError(path, 400, "Bad Request", {}, io.BytesIO(
            '{"detail": "детектор ждёт 768 признаков, получено 769"}'.encode()))

    monkeypatch.setattr(learned, "post", refuse)
    rng = np.random.default_rng(2)
    features = np.vstack([rng.normal(size=(40, 64)), 5 + rng.normal(size=(40, 64))]).astype(np.float32)
    perception = Perception(fps=8.0, motion=np.ones(80), appearance=features)
    meta = VideoMeta(id="v", filename="v.mp4", duration_sec=10.0, fps=30.0, width=1280, height=720)

    result = get_segmenter("learned-boundaries").run(Path("v.mp4"), meta, load_vocabulary(), perception)
    assert result.models["segmenter"] == "tsm-kernel"
    assert "768" in result.models["segmenter_status"]


def test_difference_channels_add_two_per_lag():
    import numpy as np

    from praxis.pipeline.learned import difference_channels

    matrix = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    out = difference_channels(matrix, lags=(1, 2, 4))
    assert out.shape == (20, 8 + 6)
    assert np.all(out[:, 8] >= 0)
    assert np.all(np.abs(out[:, 9]) <= 1.0 + 1e-6)


def test_difference_channels_flag_a_jump():
    import numpy as np

    from praxis.pipeline.learned import difference_channels

    matrix = np.vstack([np.zeros((10, 4)), np.ones((10, 4))]).astype(np.float32)
    out = difference_channels(matrix, lags=(1,))
    assert out[10, 4] > out[5, 4] and out[10, 4] > out[15, 4]


def test_prominence_ignores_ripples_on_a_plateau():
    """Плато 0.8 с рябью — не граница: без выраженности каждый бугорок стал бы пиком.
    Одиночный настоящий подъём над низким фоном остаётся."""
    import numpy as np

    from praxis.pipeline.learned import peaks_above

    plateau = np.full(40, 0.8, dtype=np.float32)
    plateau[10] += 0.02
    plateau[30] += 0.02
    assert len(peaks_above(plateau, level=0.5, minimum=3)) >= 2
    assert peaks_above(plateau, level=0.5, minimum=3, prominence=0.2) == []

    lone = np.full(40, 0.1, dtype=np.float32)
    lone[20] = 0.9
    assert peaks_above(lone, level=0.5, minimum=3, prominence=0.2) == [20]
