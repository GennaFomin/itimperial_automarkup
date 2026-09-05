import numpy as np
import pytest

from scripts.ensemble_scores import average


def dump(names, scores):
    return {
        "names": np.array(names),
        "scores": np.array(scores, dtype=object),
        "fps": np.array([4.0] * len(names)),
        "offsets": np.array([0.5] * len(names)),
        "truths": np.array([[(0.0, 1.0)]] * len(names), dtype=object),
        "durations": np.array([3.0] * len(names)),
    }


def test_average_matches_by_name_and_keeps_first_order():
    a = dump(["x", "y"], [np.array([0.2, 0.4]), np.array([1.0, 0.0, 0.0])])
    b = dump(["y", "x"], [np.array([0.0, 1.0, 0.0]), np.array([0.6, 0.0])])
    merged = average([a, b])
    assert list(merged["names"]) == ["x", "y"]
    assert np.allclose(merged["scores"][0], [0.4, 0.2])
    assert np.allclose(merged["scores"][1], [0.5, 0.5, 0.0])


def test_average_rejects_missing_clip_and_mismatched_grid():
    a = dump(["x"], [np.array([0.2, 0.4])])
    with pytest.raises(SystemExit):
        average([a, dump(["z"], [np.array([0.2, 0.4])])])
    with pytest.raises(SystemExit):
        average([a, dump(["x"], [np.array([0.2, 0.4, 0.6])])])
