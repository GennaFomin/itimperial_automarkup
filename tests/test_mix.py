import json
from pathlib import Path


def make_pool(root: Path, name: str, stems: list[str], size: int = 10) -> Path:
    pool = root / name
    (pool / "clips").mkdir(parents=True)
    (pool / "gt").mkdir()
    for stem in stems:
        (pool / "clips" / f"{stem}.mp4").write_bytes(b"v" * size)
        (pool / "gt" / f"{stem}.json").write_text(json.dumps({
            "video": {"id": stem, "filename": f"{stem}.mp4", "duration_sec": 10.0,
                      "fps": 30.0, "width": 1280, "height": 720},
            "steps": [], "provenance": {"pipeline": "gt", "app_version": "0"},
        }), encoding="utf-8")
    return pool


def test_select_prefixes_embodiment_and_source(tmp_path):
    from scripts.make_mix import select

    pool = make_pool(tmp_path, "asm", ["a", "b"])
    chosen = select([("human", pool)], forbidden_names=set(), forbidden_sizes=set())
    assert [c[0] for c in chosen] == ["human__asm__a", "human__asm__b"]


def test_select_drops_validation_by_name_and_by_size(tmp_path):
    from scripts.make_mix import select

    pool = make_pool(tmp_path, "asm", ["a", "b", "c"], size=10)
    (pool / "clips" / "c.mp4").write_bytes(b"v" * 77)
    chosen = select([("human", pool)], forbidden_names={"a"}, forbidden_sizes={77})
    assert [c[0] for c in chosen] == ["human__asm__b"]


def test_build_rewrites_annotation_identity(tmp_path):
    from scripts.make_mix import build

    pool = make_pool(tmp_path, "asm", ["a"])
    out = tmp_path / "mix"
    build([("robot", pool)], out, forbidden_names=set(), forbidden_sizes=set())
    gt = json.loads((out / "gt" / "robot__asm__a.json").read_text(encoding="utf-8"))
    assert gt["video"]["filename"] == "robot__asm__a.mp4"
    assert (out / "clips" / "robot__asm__a.mp4").resolve() == (pool / "clips" / "a.mp4").resolve()
