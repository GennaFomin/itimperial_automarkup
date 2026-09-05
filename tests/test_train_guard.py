import pytest

from scripts.train_boundaries import check_corpus, gap_frames


def test_a_few_missing_clips_are_tolerated():
    check_corpus(554, 551)
    check_corpus(20, 17)


def test_a_dropped_feature_service_aborts_training():
    with pytest.raises(SystemExit, match="корпус неполный"):
        check_corpus(554, 320)


def test_gap_frames_mark_only_the_pauses():
    # сетка 4 Гц со смещением 0.25: кадры в 0.25, 0.5, ..., 4.75; шаги 1–2 и 3–4
    frames = gap_frames([(1.0, 2.0), (3.0, 4.0)], 5.0, 0.25, 4.0, 19)
    times = [0.25 + i / 4 for i in range(19)]
    assert all(not (1.0 <= times[int(f)] < 2.0 or 3.0 <= times[int(f)] < 4.0) for f in frames)
    assert len(frames) == 19 - 8
