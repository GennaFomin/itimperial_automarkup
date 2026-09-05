from pathlib import Path

import numpy as np

from praxis import config, jobs, media


def test_motion_band_is_resampled_and_cached(tmp_path, monkeypatch):
    """Полоса движения приводится к сетке признаков и считается один раз на ролик."""
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    monkeypatch.setattr(config, "MOTION_FPS", 10)
    monkeypatch.setattr(config, "FEATURE_CACHE", True)
    calls = {"gray": 0}

    def fake_gray(video, fps=None, width=64, height=36):
        calls["gray"] += 1
        return np.zeros((100, height, width), dtype=np.uint8)  # 10 с при 10 fps

    monkeypatch.setattr(media, "gray_frames", fake_gray)
    monkeypatch.setattr(media, "motion_from_frames", lambda frames: np.linspace(0, 1, len(frames)))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)

    band = jobs.motion_band(clip, fps=8.0, count=80)
    again = jobs.motion_band(clip, fps=8.0, count=80)

    assert band.shape == (80,) and band.dtype == np.float32
    assert 0.0 <= band.min() and band.max() <= 1.0
    assert np.allclose(band, again)
    assert calls["gray"] == 1, "второй вызов обязан прийти из кэша"
    assert list((tmp_path / "motion").glob("*.npy")), "кэш должен лечь на диск"
