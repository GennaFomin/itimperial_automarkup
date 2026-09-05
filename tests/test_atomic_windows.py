def act(start, end, action="pick up"):
    return {"start": start, "end": end, "verb": action, "object": "toy"}


def test_clean_mode_rejects_windows_with_short_steps():
    from scripts.make_atomic_devset import windows

    seq = [act(0, 2), act(2, 2.5), act(2.5, 5), act(5, 8), act(8, 11), act(11, 14)]
    loose = windows(seq, span=20, min_steps=3, max_steps=8)
    clean = windows(seq, span=20, min_steps=3, max_steps=8, min_step_sec=1.5)
    assert loose[0][0]["start"] == 0, "без фильтра окно начинается с первого шага"
    # С фильтром окно с полусекундным шагом отброшено, а следующее чистое — найдено.
    assert clean[0][0]["start"] == 2.5
    assert all(a["end"] - a["start"] >= 1.5 for a in clean[0])


def test_clean_mode_rejects_micro_verbs():
    from scripts.make_atomic_devset import windows

    seq = [act(0, 3), act(3, 6, "attempt to pick up"), act(6, 9), act(9, 12)]
    assert windows(seq, 20, 3, 8, drop_verbs=("attempt",)) == []
    assert len(windows(seq, 20, 3, 8)) == 1
