def test_parse_sweep_reads_f1_and_interval():
    from scripts.eval_variants import parse_sweep

    text = (
        "метод                               F1@0.1  F1@0.25  F1@0.5   90% для F1@0.5  границы  шагов  с/ролик\n"
        "learned-boundaries:PRAXIS_TAS_THRESHOLD=0.7   0.732    0.680   0.411      0.385–0.438    0.40с    9.1      1.6\n"
    )
    row = parse_sweep(text)["learned-boundaries:PRAXIS_TAS_THRESHOLD=0.7"]
    assert row == {"f1_05": 0.411, "lo": 0.385, "hi": 0.438, "steps": 9.1, "err": 0.40}


def test_render_makes_a_markdown_table():
    from scripts.eval_variants import render

    table = render([{"variant": "base", "set": "human", "f1_05": 0.411, "lo": 0.385, "hi": 0.438, "steps": 9.1, "err": 0.4}])
    assert "| base | human | 0.411 | 0.385–0.438 | 0.40 с | 9.1 |" in table


def test_motion_flag_follows_the_loaded_dimension(monkeypatch):
    """base на 768 каналах и варианты на 775: клиент обязан слать то, что ждёт чекпоинт."""
    import os

    from scripts import eval_variants

    monkeypatch.setattr(eval_variants, "get_health", lambda: {"dim": 768})
    eval_variants.align_input_flags()
    assert os.environ["PRAXIS_TAS_MOTION"] == "0" and os.environ["PRAXIS_TAS_DIFF"] == "0"
    monkeypatch.setattr(eval_variants, "get_health", lambda: {"dim": 775})
    eval_variants.align_input_flags()
    assert os.environ["PRAXIS_TAS_MOTION"] == "1" and os.environ["PRAXIS_TAS_DIFF"] == "1"
    monkeypatch.setattr(eval_variants, "get_health", lambda: {"dim": 769})
    eval_variants.align_input_flags()
    assert os.environ["PRAXIS_TAS_MOTION"] == "1" and os.environ["PRAXIS_TAS_DIFF"] == "0"


def test_missing_variant_is_skipped_not_fatal(monkeypatch, tmp_path):
    import urllib.error

    from scripts import eval_variants

    def fake_post(path, payload, timeout=None):
        if payload.get("name") == "ghost":
            raise urllib.error.HTTPError(path, 404, "нет чекпоинта", {}, None)
        return {}

    monkeypatch.setattr(eval_variants, "post", fake_post)
    monkeypatch.setattr(eval_variants, "align_input_flags", lambda: None)
    monkeypatch.setattr(eval_variants, "sweep", lambda clips, gt, thr: {"f1_05": 0.5, "lo": 0.4, "hi": 0.6, "steps": 5.0, "err": 0.3})
    root = tmp_path / "set"; (root / "clips").mkdir(parents=True); (root / "gt").mkdir()
    rows = eval_variants.evaluate(["ghost", "base"], {"s": root}, 0.5)
    assert [r["variant"] for r in rows] == ["base"]

