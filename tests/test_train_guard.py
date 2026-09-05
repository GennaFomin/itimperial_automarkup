import pytest

from scripts.train_boundaries import check_corpus


def test_a_few_missing_clips_are_tolerated():
    check_corpus(554, 551)
    check_corpus(20, 17)


def test_a_dropped_feature_service_aborts_training():
    with pytest.raises(SystemExit, match="корпус неполный"):
        check_corpus(554, 320)
