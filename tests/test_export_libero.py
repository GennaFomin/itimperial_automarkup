def test_episodes_are_grouped_and_ordered():
    from scripts.export_libero import group_episodes

    rows = [
        {"episode_index": 1, "frame_index": 1, "subtask": "b", "img": b"1"},
        {"episode_index": 0, "frame_index": 0, "subtask": "a", "img": b"0"},
        {"episode_index": 1, "frame_index": 0, "subtask": "a", "img": b"1"},
    ]
    episodes = group_episodes(rows)
    assert list(episodes) == [0, 1]
    assert [r["frame_index"] for r in episodes[1]] == [0, 1]
    assert [r["subtask"] for r in episodes[1]] == ["a", "b"]


def test_holdout_is_the_tail_by_episode_number():
    from scripts.export_libero import split_holdout

    train, hold = split_holdout([0, 1, 2, 3, 4], holdout=2)
    assert train == [0, 1, 2] and hold == [3, 4]
