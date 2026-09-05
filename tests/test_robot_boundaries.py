def test_gripper_transitions_become_boundaries():
    from scripts.make_robot_devset import boundaries_from_gripper

    fps = 10.0
    gripper = [1.0] * 10 + [0.0] * 15 + [1.0] * 10
    assert boundaries_from_gripper(gripper, fps) == [1.0, 2.5]


def test_gripper_chatter_is_debounced():
    from scripts.make_robot_devset import boundaries_from_gripper

    fps = 10.0
    gripper = [1.0] * 10 + [0.0, 1.0, 0.0, 1.0] + [0.0] * 20
    assert boundaries_from_gripper(gripper, fps, min_gap_sec=0.5) == [1.0]


def test_steps_between_boundaries_have_keyframes_inside():
    from scripts.make_robot_devset import steps_from_boundaries

    steps = steps_from_boundaries([1.0, 2.5], duration=4.0)
    assert [(s["start_sec"], s["end_sec"]) for s in steps] == [(0.0, 1.0), (1.0, 2.5), (2.5, 4.0)]
    assert all(s["start_sec"] <= s["keyframe_sec"] <= s["end_sec"] for s in steps)


def test_subtask_runs_become_boundaries():
    from scripts.make_robot_devset import boundaries_from_labels

    labels = ["reach"] * 10 + ["grasp"] * 5 + ["place"] * 15
    assert boundaries_from_labels(labels, fps=10.0) == [1.0, 1.5]


def test_written_annotation_validates_against_the_schema(tmp_path):
    """Эталон роботного корпуса обязан проходить схему Praxis — иначе обучение падает
    на первом же файле, как и случилось."""
    from praxis.schema import Annotation
    from scripts.make_robot_devset import write_annotation, steps_from_boundaries

    meta = {"duration_sec": 4.0, "fps": 10.0, "width": 256, "height": 256}
    write_annotation(tmp_path, "ep", meta, steps_from_boundaries([1.0, 2.5], 4.0))
    Annotation.model_validate_json((tmp_path / "gt" / "ep.json").read_text(encoding="utf-8"))

