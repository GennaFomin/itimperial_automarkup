"""Метаданные видео: поворот телефонных роликов и предел размера кадра."""

from __future__ import annotations

import subprocess

import pytest

from praxis import media
from tests.test_api import make_video


def test_probe_swaps_dimensions_for_rotated_streams(tmp_path):
    source = make_video(tmp_path / "landscape.mp4", seconds=2)
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-c", "copy",
         "-metadata:s:v:0", "rotate=90", str(rotated)],
        check=True,
    )
    tagged = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream_tags=rotate:stream_side_data=rotation", "-of", "json", str(rotated)],
        capture_output=True, text=True, check=True,
    ).stdout
    if "rotat" not in tagged:
        pytest.skip("эта сборка ffmpeg не сохраняет пометку о повороте")

    assert media.probe(source)["width"] == 1280
    meta = media.probe(rotated)
    assert (meta["width"], meta["height"]) == (720, 1280)


def test_rotation_reads_side_data_and_legacy_tag():
    assert media._rotation({"side_data_list": [{"rotation": -90}]}) == 270
    assert media._rotation({"tags": {"rotate": "90"}}) == 90
    assert media._rotation({"tags": {"rotate": "junk"}}) == 0
    assert media._rotation({}) == 0
