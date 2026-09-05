import pytest

torch = pytest.importorskip("torch")  # на ноутбуке torch нет: тест идёт на DL5

from scripts import serve_tas  # noqa: E402
from scripts.pack_ensemble import pack
from scripts.serve_tas import BoundaryNet, load_checkpoint


def save(path, dim, stages, seed):
    torch.manual_seed(seed)
    model = BoundaryNet(dim, stages=stages)
    torch.save({"dim": dim, "stages": stages, "weights": model.state_dict()}, path)
    return model.eval()  # без eval dropout делает выход случайным


def test_packed_ensemble_loads_all_members_and_averages(tmp_path):
    dim = 8
    a = save(tmp_path / "boundary-a.pt", dim, 2, 1)
    b = save(tmp_path / "boundary-b.pt", dim, 4, 2)
    packed = pack([tmp_path / "boundary-a.pt", tmp_path / "boundary-b.pt"])
    torch.save(packed, tmp_path / "boundary.pt")

    load_checkpoint(tmp_path / "boundary.pt", "cpu")
    assert serve_tas.state["dim"] == dim and serve_tas.state["stages"] == 2
    assert len(serve_tas.state["ensemble"]) == 1

    features = torch.randn(1, dim, 30)
    with torch.no_grad():
        expected = torch.stack([torch.sigmoid(m(features)[-1])[0, 0] for m in (a, b)]).mean(0)
        members = [serve_tas.state["model"], *serve_tas.state["ensemble"]]
        got = torch.stack([torch.sigmoid(m(features)[-1])[0, 0] for m in members]).mean(0)
    assert torch.allclose(got, expected)


def test_plain_checkpoint_clears_previous_members(tmp_path):
    save(tmp_path / "boundary-a.pt", 8, 2, 1)
    save(tmp_path / "boundary-b.pt", 8, 2, 2)
    torch.save(pack([tmp_path / "boundary-a.pt", tmp_path / "boundary-b.pt"]), tmp_path / "boundary.pt")
    load_checkpoint(tmp_path / "boundary.pt", "cpu")
    assert len(serve_tas.state["ensemble"]) == 1
    load_checkpoint(tmp_path / "boundary-a.pt", "cpu")
    assert serve_tas.state["ensemble"] == []


def test_pack_rejects_mixed_input_dims(tmp_path):
    save(tmp_path / "boundary-a.pt", 8, 2, 1)
    save(tmp_path / "boundary-b.pt", 9, 2, 2)
    with pytest.raises(SystemExit):
        pack([tmp_path / "boundary-a.pt", tmp_path / "boundary-b.pt"])


def test_legacy_two_stage_member_is_renamed_and_loads(tmp_path):
    torch.manual_seed(3)
    legacy = BoundaryNet(8, stages=2).eval()
    weights = {k.replace("rest.0.", "second.", 1) if k.startswith("rest.0.") else k: v for k, v in legacy.state_dict().items()}
    torch.save({"dim": 8, "weights": weights}, tmp_path / "boundary.pt")  # без поля stages, как старые файлы
    save(tmp_path / "boundary-b.pt", 8, 2, 4)
    torch.save(pack([tmp_path / "boundary.pt", tmp_path / "boundary-b.pt"]), tmp_path / "boundary-packed.pt")
    load_checkpoint(tmp_path / "boundary-packed.pt", "cpu")
    assert serve_tas.state["stages"] == 2 and len(serve_tas.state["ensemble"]) == 1
    features = torch.randn(1, 8, 20)
    with torch.no_grad():
        assert torch.allclose(serve_tas.state["model"](features)[-1], legacy(features)[-1])


def test_min_fusion_cuts_only_where_every_member_agrees(tmp_path):
    a = save(tmp_path / "boundary-a.pt", 8, 2, 5)
    b = save(tmp_path / "boundary-b.pt", 8, 2, 6)
    torch.save(pack([tmp_path / "boundary-a.pt", tmp_path / "boundary-b.pt"], fusion="min"), tmp_path / "boundary.pt")
    load_checkpoint(tmp_path / "boundary.pt", "cpu")
    assert serve_tas.state["fusion"] == "min"
    features = torch.randn(1, 8, 25)
    with torch.no_grad():
        expected = torch.stack([torch.sigmoid(m(features)[-1])[0, 0] for m in (a, b)]).min(0).values
        members = [serve_tas.state["model"], *serve_tas.state["ensemble"]]
        stacked = torch.stack([torch.sigmoid(m(features)[-1])[0, 0] for m in members])
        got = stacked.min(0).values if serve_tas.state["fusion"] == "min" else stacked.mean(0)
    assert torch.allclose(got, expected)
