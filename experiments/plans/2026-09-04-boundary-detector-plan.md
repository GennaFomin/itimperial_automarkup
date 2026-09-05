# Boundary Detector on a Mixed Human + Robot Corpus — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise class-agnostic step-F1@0.5 of the boundary detector from 0.411 toward 0.75 on 5–30 s tabletop clips where the agent may be a human or a robot, using only open datasets and simulation.

**Architecture:** The MS-TCN boundary head stays. We change what it eats (a 769th motion channel), what it is trained on (a mixed human+robot corpus with automatic boundaries and a leak-checked build), how it is trained (target width σ, focal loss, named checkpoints that never overwrite the live model) and how it is judged (three fixed sets, mandatory cross-embodiment measurement, CIs). Only a variant that wins on both labelled sets is promoted.

**Tech Stack:** Python 3.10 on the GPU box (DL5) / 3.12 locally, PyTorch (service only, on DL5), FastAPI, numpy, ffmpeg with libx264, pytest. Services on DL5: features `:8102`, boundary head `:8104`.

**Spec:** `experiments/plans/2026-09-04-boundary-detector-design.md`

## Global Constraints

- Commits: English, no `Co-Authored-By: Claude*` and no `Claude-Session:` trailers — ever (standing instruction from the repo owner).
- Branch: `feature/naming-quality`. Never push to `main`.
- Do not modify `web/`, `praxis/api.py`, `praxis/api_v1.py`, `praxis/contract_v1.py` — the colleague's area.
- The live checkpoint `~/praxis/checkpoints/boundary.pt` on DL5 is never overwritten by an experiment; variants write `boundary-<name>.pt`.
- `praxis-pool/validation` is never used to pick thresholds, σ or anything else.
- The service runs on DL5; the laptop has no torch. Service code is edited locally in `scripts/serve_tas.py`, copied with `scp`, and restarted with **two separate ssh calls** (stop, then start) — see `docs/PIPELINE.md`, section "Перезапуск сервиса".
- ffmpeg on DL5 is `~/bin/ffmpeg` (libx264); every remote shell must `export PATH="$HOME/bin:$PATH"`.

---

### Task 1: Cached motion band for training

**Files:**
- Modify: `praxis/jobs.py` (add `motion_band()` after `_video_features`)
- Modify: `scripts/train_boundaries.py:44-55`
- Test: `tests/test_motion_band.py`

**Interfaces:**
- Consumes: `media.gray_frames(video, fps=None, width=64, height=36) -> np.ndarray`, `media.motion_from_frames(frames) -> np.ndarray` (one value per gray frame at `config.MOTION_FPS`), `_resample(values, source_fps, target_fps, count) -> np.ndarray`, `_feature_cache_path(source) -> Path`.
- Produces: `jobs.motion_band(source: Path, fps: float, count: int) -> np.ndarray` of shape `(count,)`, float32, values in the feature grid; cached at `config.WORK_DIR / "motion" / f"{key}.npy"` where `key` is the stem of `_feature_cache_path(source)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_motion_band.py
from pathlib import Path

import numpy as np

from praxis import config, jobs, media


def test_motion_band_is_resampled_and_cached(tmp_path, monkeypatch):
    """Полоса движения приводится к сетке признаков и считается один раз на ролик."""
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    monkeypatch.setattr(config, "MOTION_FPS", 10)
    calls = {"gray": 0}

    def fake_gray(video, fps=None, width=64, height=36):
        calls["gray"] += 1
        return np.zeros((100, height, width), dtype=np.uint8)  # 10 с при 10 fps

    monkeypatch.setattr(media, "gray_frames", fake_gray)
    monkeypatch.setattr(media, "motion_from_frames", lambda frames: np.linspace(0, 1, len(frames)))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)

    band = jobs.motion_band(clip, fps=8.0, count=80)
    again = jobs.motion_band(clip, fps=8.0, count=80)

    assert band.shape == (80,) and band.dtype == np.float32
    assert 0.0 <= band.min() and band.max() <= 1.0
    assert np.allclose(band, again)
    assert calls["gray"] == 1, "второй вызов обязан прийти из кэша"
    assert list((tmp_path / "motion").glob("*.npy")), "кэш должен лечь на диск"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_motion_band.py -v`
Expected: FAIL with `AttributeError: module 'praxis.jobs' has no attribute 'motion_band'`

- [ ] **Step 3: Write minimal implementation**

Insert into `praxis/jobs.py` right after the end of `_video_features` (before `_feature_cache_path`):

```python
def motion_band(source: Path, fps: float, count: int) -> np.ndarray:
    """Полоса движения на сетке признаков — 769-й канал детектора границ.

    Считается из серых кадров, это самое дорогое в обучении (декодирование всего
    ролика), поэтому кладётся в кэш рядом с признаками и под тем же ключом.
    """
    cache = config.WORK_DIR / "motion" / f"{_feature_cache_path(source).stem}.npy"
    if config.FEATURE_CACHE and cache.exists():
        band = np.load(cache)
        if len(band) == count:
            return band.astype(np.float32)
    frames = media.gray_frames(source)
    raw = media.motion_from_frames(frames)
    band = _resample(np.asarray(raw, dtype=np.float32), float(config.MOTION_FPS), fps, count)
    peak = float(band.max()) if len(band) else 0.0
    band = (band / peak).astype(np.float32) if peak > 0 else band.astype(np.float32)
    if config.FEATURE_CACHE:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, band)
    return band
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_motion_band.py -v`
Expected: PASS

- [ ] **Step 5: Use it in the training script**

In `scripts/train_boundaries.py` replace the `perception = jobs.Perception(...)` block (the one that sets `motion=np.ones(...)`) with:

```python
        matrix = video["matrix"]
        perception = jobs.Perception(
            fps=float(video["fps"]),
            motion=jobs.motion_band(source, float(video["fps"]), len(matrix)),
            appearance=matrix,
            offset=float(video.get("offset_sec", 0.0)),
        )
```

- [ ] **Step 6: Run the whole suite and commit**

Run: `~/.venvs/praxis/bin/python -m pytest -q`
Expected: all pass (previously 102 + 1 new)

```bash
git add praxis/jobs.py scripts/train_boundaries.py tests/test_motion_band.py
git commit -m "Cache the motion band on the feature grid for boundary training"
```

---

### Task 2: Motion as the 769th input channel, client side

**Files:**
- Modify: `praxis/config.py` (add `TAS_MOTION`)
- Modify: `praxis/pipeline/learned.py:29-36, 84-90`
- Modify: `scripts/train_boundaries.py` (build the stacked matrix)
- Test: `tests/test_segment.py` (append)

**Interfaces:**
- Produces: `learned.stack_motion(appearance: np.ndarray, motion: np.ndarray) -> np.ndarray` of shape `(T, D+1)` float32; `config.TAS_MOTION: bool` from `PRAXIS_TAS_MOTION` (default `"1"`).
- The training script sends `encode(stack_motion(...))` when `config.TAS_MOTION` else `encode(appearance)`. `LearnedSegmenter.run` does the same at inference.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_segment.py`:

```python
def test_stack_motion_appends_one_channel():
    import numpy as np

    from praxis.pipeline.learned import stack_motion

    appearance = np.zeros((7, 768), dtype=np.float32)
    motion = np.linspace(0, 1, 7)
    stacked = stack_motion(appearance, motion)
    assert stacked.shape == (7, 769) and stacked.dtype == np.float32
    assert np.allclose(stacked[:, -1], motion)


def test_stack_motion_tolerates_length_mismatch():
    """Полоса движения на кадр короче или длиннее — обрезаем/дополняем, а не падаем."""
    import numpy as np

    from praxis.pipeline.learned import stack_motion

    appearance = np.zeros((7, 768), dtype=np.float32)
    assert stack_motion(appearance, np.ones(9)).shape == (7, 769)
    assert stack_motion(appearance, np.ones(5)).shape == (7, 769)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_segment.py -k stack_motion -v`
Expected: FAIL with `ImportError: cannot import name 'stack_motion'`

- [ ] **Step 3: Implement**

In `praxis/config.py`, next to `TAS_THRESHOLD`:

```python
# Полоса движения как 769-й канал детектора. Одинакова для руки человека и
# манипулятора — граница это событие захвата, а не внешний вид.
TAS_MOTION = os.getenv("PRAXIS_TAS_MOTION", "1").lower() not in {"0", "false", "no"}
```

In `praxis/pipeline/learned.py`, after `encode`:

```python
def stack_motion(appearance: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Признаки энкодера плюс полоса движения последним каналом."""
    count = len(appearance)
    band = np.asarray(motion, dtype=np.float32).ravel()
    if len(band) >= count:
        band = band[:count]
    else:
        band = np.pad(band, (0, count - len(band)), mode="edge") if len(band) else np.zeros(count, np.float32)
    return np.concatenate([appearance.astype(np.float32), band[:, None]], axis=1)
```

In `LearnedSegmenter.run`, replace `answer = post("/predict", {"samples": [encode(perception.appearance)]})` with:

```python
            matrix = (
                stack_motion(perception.appearance, perception.motion)
                if config.TAS_MOTION else perception.appearance
            )
            answer = post("/predict", {"samples": [encode(matrix)]})
```

In `scripts/train_boundaries.py`, replace `samples.append({**encode(perception.appearance), "boundaries": boundaries})` with:

```python
        matrix = (
            stack_motion(perception.appearance, perception.motion)
            if config.TAS_MOTION else perception.appearance
        )
        samples.append({**encode(matrix), "boundaries": boundaries})
```

and extend the import: `from praxis.pipeline.learned import encode, post, stack_motion`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_segment.py -v`
Expected: PASS (all, including the two fallback tests already there)

- [ ] **Step 5: Commit**

```bash
git add praxis/config.py praxis/pipeline/learned.py scripts/train_boundaries.py tests/test_segment.py
git commit -m "Feed the motion band to the boundary detector as a 769th channel"
```

---

### Task 3: Service — dimension check, named checkpoints, `/load`

**Files:**
- Modify: `scripts/serve_tas.py` (`TrainRequest`, `train`, `predict`, `health`, new `/load`)
- Modify: `praxis/pipeline/learned.py:84-90` (treat HTTP 400 as a clear fallback)
- Modify: `scripts/train_boundaries.py` (`--name`, `--no-activate`, `--focal` later in Task 4)
- Create: `scripts/tas_load.py`
- Test: `tests/test_segment.py` (append), `tests/test_tas_paths.py`

**Interfaces:**
- `TrainRequest` gains `name: str = ""`, `activate: bool = True`.
- `checkpoint_path(base: str, name: str) -> Path`: `base` when `name` is empty, else `<dir>/<stem>-<name><suffix>`. Pure function, importable without torch (keep it above the torch imports or guard imports — see Step 3).
- `POST /load {"name": str}` → loads `checkpoint_path(state["checkpoint"], name)` into the live model, returns `{"loaded": path, "dim": int, "stages": int}`; 404 if missing.
- `GET /health` → `{"ready": bool, "model": "boundary-net", "dim": int|null, "stages": int|null, "checkpoint": str|null}`.
- `POST /predict` → 400 with `{"detail": "детектор ждёт D признаков, получено M"}` when `sample.dim != state["dim"]`.
- `scripts/tas_load.py --name <name>` posts `/load`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tas_paths.py
"""serve_tas живёт на GPU-машине и тянет torch; сюда вынесено то, что проверяется без него."""
import importlib.util
import sys
import types
from pathlib import Path


def load_serve_tas():
    # torch/fastapi отсутствуют локально — подставляем пустые модули, нам нужна одна функция.
    for name in ("torch", "torch.nn", "fastapi", "uvicorn"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].Module = object
    sys.modules["fastapi"].FastAPI = lambda *a, **k: types.SimpleNamespace(
        get=lambda *a, **k: (lambda f: f), post=lambda *a, **k: (lambda f: f))
    sys.modules["fastapi"].HTTPException = Exception
    spec = importlib.util.spec_from_file_location("serve_tas", Path("scripts/serve_tas.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_path_keeps_base_when_unnamed():
    st = load_serve_tas()
    assert st.checkpoint_path("checkpoints/boundary.pt", "") == Path("checkpoints/boundary.pt")


def test_checkpoint_path_names_variants_beside_the_base():
    st = load_serve_tas()
    assert st.checkpoint_path("checkpoints/boundary.pt", "sigma3-focal") == Path(
        "checkpoints/boundary-sigma3-focal.pt"
    )
```

Append to `tests/test_segment.py`:

```python
def test_learned_segmenter_falls_back_on_dimension_mismatch(monkeypatch):
    """Сервис ждёт другую размерность (чекпоинт без канала движения) → 400 → ядровой."""
    import urllib.error

    import numpy as np

    from praxis import config
    from praxis.pipeline import learned
    from praxis.pipeline.base import Perception, get_segmenter
    from praxis.schema import VideoMeta
    from praxis.vocab import load_vocabulary

    monkeypatch.setattr(config, "TAS_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(config, "MIN_SEGMENT_SEC", 0.5)
    monkeypatch.setattr(config, "IDLE_RATIO", 0.0)

    def refuse(path, payload, timeout=None):
        raise urllib.error.HTTPError(path, 400, "детектор ждёт 768 признаков, получено 769", {}, None)

    monkeypatch.setattr(learned, "post", refuse)
    rng = np.random.default_rng(2)
    features = np.vstack([rng.normal(size=(40, 64)), 5 + rng.normal(size=(40, 64))]).astype(np.float32)
    perception = Perception(fps=8.0, motion=np.ones(80), appearance=features)
    meta = VideoMeta(id="v", filename="v.mp4", duration_sec=10.0, fps=30.0, width=1280, height=720)

    result = get_segmenter("learned-boundaries").run(Path("v.mp4"), meta, load_vocabulary(), perception)
    assert result.models["segmenter"] == "tsm-kernel"
    assert "768" in result.models["segmenter_status"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_tas_paths.py tests/test_segment.py -k "checkpoint_path or dimension_mismatch" -v`
Expected: `test_tas_paths` FAIL (`AttributeError: checkpoint_path`); `dimension_mismatch` FAIL because the current except-clause swallows the message text differently — check the assertion text.

- [ ] **Step 3: Implement in `scripts/serve_tas.py`**

Add near the top, before any class that needs torch (a pure helper):

```python
def checkpoint_path(base: str, name: str) -> Path:
    """Именованный вариант лежит рядом с основным чекпоинтом и никогда его не трогает."""
    path = Path(base)
    if not name:
        return path
    return path.with_name(f"{path.stem}-{name}{path.suffix}")
```

Extend `TrainRequest`:

```python
    name: str = ""          # имя варианта: чекпоинт boundary-<name>.pt
    activate: bool = True   # подменять ли живую модель обученной
```

In `train()`, replace the block from `state["model"] = model.eval()` through `torch.save(...)` with:

```python
    path = checkpoint_path(state["checkpoint"], request.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dim": dim, "stages": request.stages, "weights": model.state_dict()}, path)
    if request.activate:
        state["model"] = model.eval()
        state["dim"] = dim
        state["stages"] = request.stages
        state["loaded"] = str(path)
```

Add the loader helper (reuse the same remapping logic that `main()` has, and make `main()` call it):

```python
def load_checkpoint(path: Path, device: str) -> None:
    payload = torch.load(path, map_location=device)
    weights = payload["weights"]
    if "stages" in payload:
        stages = int(payload["stages"])
    else:
        stages = 2 if any(k.startswith("second.") for k in weights) else 4
        if stages == 2:
            weights = {
                k.replace("second.", "rest.0.", 1) if k.startswith("second.") else k: v
                for k, v in weights.items()
            }
    model = BoundaryNet(payload["dim"], stages=stages).to(device)
    model.load_state_dict(weights)
    state["model"] = model.eval()
    state["dim"] = payload["dim"]
    state["stages"] = stages
    state["loaded"] = str(path)
```

Replace the checkpoint-loading block inside `main()` with:

```python
    saved = Path(args.checkpoint)
    if saved.exists():
        load_checkpoint(saved, args.device)
        print(f"подняты сохранённые веса из {saved}, стадий {state['stages']}", flush=True)
```

Add the endpoints:

```python
class LoadRequest(BaseModel):
    name: str = ""


@app.post("/load")
def load(request: LoadRequest) -> dict:
    path = checkpoint_path(state["checkpoint"], request.name)
    if not path.exists():
        raise HTTPException(404, f"нет чекпоинта {path}")
    load_checkpoint(path, state["device"])
    return {"loaded": state["loaded"], "dim": state["dim"], "stages": state["stages"]}
```

Update `health()` to return `{"ready": "model" in state, "model": "boundary-net", "dim": state.get("dim"), "stages": state.get("stages"), "checkpoint": state.get("loaded")}`.

In `predict()`, before decoding:

```python
    for sample in request.samples:
        if "dim" in state and sample.dim != state["dim"]:
            raise HTTPException(400, f"детектор ждёт {state['dim']} признаков, получено {sample.dim}")
```

Make sure `from fastapi import FastAPI, HTTPException` is imported.

- [ ] **Step 4: Client — surface the 400 text**

In `praxis/pipeline/learned.py` `run()`, change the except clause so an `HTTPError` carries its body:

```python
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:200] if hasattr(error, "read") else str(error)
            return self._fallback(video_path, meta, vocabulary, perception, f"недоступен: {error.code} {detail}")
        except (urllib.error.URLError, OSError, TimeoutError, ValueError, KeyError) as error:
            return self._fallback(video_path, meta, vocabulary, perception, f"недоступен: {error}")
```

(`HTTPError` is a subclass of `URLError`, so its clause must come first.)

- [ ] **Step 5: Training script flags and the loader script**

In `scripts/train_boundaries.py` add arguments and pass them through:

```python
    parser.add_argument("--name", default="", help="имя варианта: чекпоинт boundary-<name>.pt")
    parser.add_argument("--no-activate", action="store_true", help="не подменять живую модель")
```

and in the `post("/train", {...})` payload add `"name": args.name, "activate": not args.no_activate`.

Create `scripts/tas_load.py`:

```python
#!/usr/bin/env python3
"""Переключить живую модель детектора границ на именованный чекпоинт.

    PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104 python scripts/tas_load.py --name sigma3
    python scripts/tas_load.py            # обратно на основной boundary.pt
"""
import argparse

from praxis.pipeline.learned import post


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    print(post("/load", {"name": args.name}, timeout=120))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests, deploy the service, smoke it**

Run: `~/.venvs/praxis/bin/python -m pytest -q` → all pass.

Deploy (three separate commands):

```bash
scp scripts/serve_tas.py dl5:~/praxis/
ssh -n dl5 'pgrep -f "serve_ta[s].py" | xargs -r kill; until ! pgrep -f "serve_ta[s].py" >/dev/null; do sleep 1; done; echo stopped'
ssh -n dl5 'cd ~/praxis && setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 ~/praxis/venv/bin/python serve_tas.py --port 8104 > ~/praxis/logs/tas.log 2>&1 < /dev/null & disown'
```

Smoke: `curl -s http://127.0.0.1:8104/health` (via the tunnel) must show `"dim":768,"stages":2,"checkpoint":".../boundary.pt"`.

- [ ] **Step 7: Commit**

```bash
git add scripts/serve_tas.py scripts/train_boundaries.py scripts/tas_load.py praxis/pipeline/learned.py tests/test_tas_paths.py tests/test_segment.py
git commit -m "Named checkpoints, /load and a dimension check on the boundary service"
```

---

### Task 4: Focal loss as a training option

**Files:**
- Modify: `scripts/serve_tas.py` (`TrainRequest.focal`, `focal_loss()`, use in `train`)
- Modify: `scripts/train_boundaries.py` (`--focal`)
- Test: `tests/test_tas_paths.py` (append; the loss is testable with numpy-free logic? No — it needs torch. Verify on DL5 instead, Step 3.)

**Interfaces:**
- `TrainRequest.focal: bool = False`, `TrainRequest.gamma: float = 2.0`.
- `focal_loss(logits, target, alpha, gamma) -> torch.Tensor` where `alpha` is the positive weight already computed as `weight`.

- [ ] **Step 1: Implement in `scripts/serve_tas.py`**

```python
def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor, gamma: float) -> torch.Tensor:
    """Focal loss (Lin et al., 2017) для бинарной цели: лёгкие кадры гасятся, редкие
    границы — нет. Альтернатива взвешенной BCE, у которой один вес на все положительные."""
    probability = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probability * target + (1 - probability) * (1 - target)
    weight = alpha * target + (1 - target)
    return (weight * (1 - p_t) ** gamma * ce).mean()
```

In `TrainRequest` add `focal: bool = False` and `gamma: float = 2.0`. In `train()`, replace the `loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)` line and the per-stage term with:

```python
    if request.focal:
        loss_fn = lambda logits, y: focal_loss(logits, y, weight, request.gamma)  # noqa: E731
    else:
        bce = nn.BCEWithLogitsLoss(pos_weight=weight)
        loss_fn = bce
```

(The per-stage expression `loss_fn(stage, y) + request.smoothing * smoothing_loss(stage)` stays as is.)

- [ ] **Step 2: Training script flag**

In `scripts/train_boundaries.py`: `parser.add_argument("--focal", action="store_true")` and `"focal": args.focal` in the payload.

- [ ] **Step 3: Verify on DL5 with a tiny run**

```bash
scp scripts/serve_tas.py dl5:~/praxis/   # then the two-step restart from Task 3, Step 6
ssh -n dl5 'cd ~/praxis/app && export PATH="$HOME/bin:$PATH" PYTHONPATH="$PWD" PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104 PRAXIS_FEATURES=video PRAXIS_VIDEO_BASE_URL=http://127.0.0.1:8102 PRAXIS_NAMER=none PRAXIS_VOCAB=$PWD/data/vocab_atomic.yaml PRAXIS_WORK_DIR=$HOME/praxis/work && ~/praxis/venv/bin/python scripts/train_boundaries.py --train data/train_big --epochs 2 --limit 20 --focal --name smoke --no-activate'
```

Expected: prints `готово за … с: потеря a -> b` with `b < a`; `~/praxis/checkpoints/boundary-smoke.pt` exists; `/health` still reports the previous checkpoint (not activated).

- [ ] **Step 4: Commit**

```bash
git add scripts/serve_tas.py scripts/train_boundaries.py
git commit -m "Focal loss as an option for the boundary head"
```

---

### Task 5: Mixed-corpus builder with a leak check

**Files:**
- Create: `scripts/make_mix.py`
- Test: `tests/test_mix.py`

**Interfaces:**
- CLI: `python scripts/make_mix.py --out data/train_mix --validation /path/praxis-pool/validation --source human=data/train_atomic --source robot=data/robo_sim ...`
- Pure function `select(sources: list[tuple[str, Path]], forbidden_names: set[str], forbidden_sizes: set[int]) -> list[tuple[str, Path, Path]]` returning `(new_stem, clip_path, gt_path)` with `new_stem = f"{embodiment}__{source_dir.name}__{gt.stem}"`.
- Output: symlinks in `out/clips/<new_stem>.mp4`, rewritten annotations in `out/gt/<new_stem>.json` with `video.id` and `video.filename` updated.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mix.py
import json
from pathlib import Path

import pytest


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
    (pool / "clips" / "c.mp4").write_bytes(b"v" * 77)  # уникальный размер
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_mix.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.make_mix'` (add `scripts/__init__.py` if importing scripts as a package is not already possible — check `tests/` for an existing precedent first; if none, add an empty `scripts/__init__.py`).

- [ ] **Step 3: Implement `scripts/make_mix.py`**

```python
#!/usr/bin/env python3
"""Единый обучающий корпус из нескольких пулов, с пометкой эмбодимента в имени.

Имя ролика: <эмбодимент>__<пул>__<исходное имя>. Ролики из валидации исключаются и по
имени, и по размеру файла: валидация копировалась из этих же пулов, и совпадение размера
ловит переименованный дубликат.

    python scripts/make_mix.py --out data/train_mix \\
        --validation /mnt/data/praxis-pool/validation \\
        --source human=data/train_atomic --source robot=data/robo_sim
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def forbidden_from(validation: Path | None) -> tuple[set[str], set[int]]:
    if validation is None or not validation.exists():
        return set(), set()
    names = {p.stem for p in (validation / "annotations").rglob("*.json")}
    sizes = {p.stat().st_size for p in (validation / "clips").rglob("*.mp4")}
    return names, sizes


def select(
    sources: list[tuple[str, Path]], forbidden_names: set[str], forbidden_sizes: set[int]
) -> list[tuple[str, Path, Path]]:
    chosen = []
    for embodiment, pool in sources:
        for gt in sorted((pool / "gt").glob("*.json")):
            filename = json.loads(gt.read_text(encoding="utf-8"))["video"]["filename"]
            clip = pool / "clips" / filename
            if not clip.exists():
                continue
            if gt.stem in forbidden_names or clip.stat().st_size in forbidden_sizes:
                continue
            chosen.append((f"{embodiment}__{pool.name}__{gt.stem}", clip, gt))
    return chosen


def build(
    sources: list[tuple[str, Path]], out: Path, forbidden_names: set[str], forbidden_sizes: set[int]
) -> int:
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    count = 0
    for stem, clip, gt in select(sources, forbidden_names, forbidden_sizes):
        link = out / "clips" / f"{stem}.mp4"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(clip.resolve())
        annotation = json.loads(gt.read_text(encoding="utf-8"))
        annotation["video"]["id"] = stem
        annotation["video"]["filename"] = f"{stem}.mp4"
        (out / "gt" / f"{stem}.json").write_text(
            json.dumps(annotation, ensure_ascii=False), encoding="utf-8"
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--source", action="append", required=True,
                        help="эмбодимент=путь к пулу с clips/ и gt/")
    args = parser.parse_args()
    sources = []
    for item in args.source:
        embodiment, _, path = item.partition("=")
        sources.append((embodiment, Path(path)))
    names, sizes = forbidden_from(args.validation)
    total = build(sources, args.out, names, sizes)
    print(f"собрано {total} роликов в {args.out}; исключено по валидации: {len(names)} имён")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_mix.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/make_mix.py tests/test_mix.py scripts/__init__.py
git commit -m "Mixed-corpus builder that tags embodiment and cannot leak validation"
```

---

### Task 6: Robot corpus from simulation with automatic boundaries

**Files:**
- Create: `scripts/make_robot_devset.py`
- Test: `tests/test_robot_boundaries.py`

**Interfaces:**
- Pure function `boundaries_from_gripper(gripper: list[float], fps: float, closed_below: float = 0.5, min_gap_sec: float = 0.4) -> list[float]` — seconds at which the gripper state crosses `closed_below` (open→closed or closed→open), debounced by `min_gap_sec`. Used for any dataset that stores a gripper signal per frame (DROID, BridgeData V2, RH20T, LIBERO, ManiSkill demos).
- Pure function `steps_from_boundaries(boundaries: list[float], duration: float, min_step_sec: float = 0.5) -> list[dict]` — Praxis steps `{start_sec, end_sec, action: "step", object: None, keyframe_sec}` between consecutive boundaries.
- The dataset-specific loader is chosen by `--dataset`: implement `rh20t_p` (primitive annotations → boundaries directly) **and** `gripper` (generic: expects a directory of `*.mp4` with sibling `*.gripper.json` = list of per-frame gripper values and `fps`) — the research report decides which one is populated first; the generic one lets any demo dataset be converted with a tiny exporter.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_robot_boundaries.py
def test_gripper_transitions_become_boundaries():
    from scripts.make_robot_devset import boundaries_from_gripper

    fps = 10.0
    gripper = [1.0] * 10 + [0.0] * 15 + [1.0] * 10  # открыт → закрыт на 1.0 с → открыт на 2.5 с
    assert boundaries_from_gripper(gripper, fps) == [1.0, 2.5]


def test_gripper_chatter_is_debounced():
    from scripts.make_robot_devset import boundaries_from_gripper

    fps = 10.0
    gripper = [1.0] * 10 + [0.0, 1.0, 0.0, 1.0] + [0.0] * 20  # дребезг в 4 кадра
    assert boundaries_from_gripper(gripper, fps, min_gap_sec=0.5) == [1.0]


def test_steps_between_boundaries_have_keyframes_inside():
    from scripts.make_robot_devset import steps_from_boundaries

    steps = steps_from_boundaries([1.0, 2.5], duration=4.0)
    assert [(s["start_sec"], s["end_sec"]) for s in steps] == [(0.0, 1.0), (1.0, 2.5), (2.5, 4.0)]
    assert all(s["start_sec"] <= s["keyframe_sec"] <= s["end_sec"] for s in steps)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_robot_boundaries.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `scripts/make_robot_devset.py`**

```python
#!/usr/bin/env python3
"""Роботный корпус с автоматическими границами подзадач.

Разметки руками нет и не будет. Границы берутся из того, что робот и так знает о себе:
смена стадии, а чаще — момент захвата или отпускания. Открыл/закрыл схват — это и есть
граница атомарного действия: «взял», «положил», «переставил» начинаются и кончаются
именно там.

Два источника:
  --dataset gripper   каталог с <clip>.mp4 и <clip>.gripper.json ({"fps": .., "values": [...]})
  --dataset rh20t_p   RH20T-P: примитивы с таймкодами → границы напрямую

    python scripts/make_robot_devset.py --dataset gripper --in data/raw_robot --out data/robo_sim
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def boundaries_from_gripper(
    gripper: list[float], fps: float, closed_below: float = 0.5, min_gap_sec: float = 0.4
) -> list[float]:
    """Секунды переходов открыт↔закрыт, дребезг короче min_gap_sec отбрасывается."""
    result: list[float] = []
    previous = gripper[0] < closed_below if gripper else False
    for index, value in enumerate(gripper[1:], 1):
        closed = value < closed_below
        if closed != previous:
            moment = index / fps
            if not result or moment - result[-1] >= min_gap_sec:
                result.append(round(moment, 3))
            else:
                result.pop()  # дребезг: пара переходов внутри окна взаимно уничтожается
            previous = closed
    return result


def steps_from_boundaries(
    boundaries: list[float], duration: float, min_step_sec: float = 0.5
) -> list[dict]:
    edges = [0.0, *[b for b in boundaries if 0 < b < duration], duration]
    steps = []
    for start, end in zip(edges, edges[1:]):
        if end - start < min_step_sec:
            continue
        steps.append({
            "start_sec": round(start, 3), "end_sec": round(end, 3),
            "action": "step", "object": None,
            "keyframe_sec": round((start + end) / 2, 3), "confidence": None,
        })
    return steps


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration",
         "-of", "json", str(video)], capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    stream, fmt = data["streams"][0], data["format"]
    num, den = stream["r_frame_rate"].split("/")
    return {"width": stream["width"], "height": stream["height"],
            "fps": float(num) / float(den), "duration_sec": float(fmt["duration"])}


def write_annotation(out: Path, stem: str, meta: dict, steps: list[dict]) -> None:
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "gt" / f"{stem}.json").write_text(json.dumps({
        "video": {"id": stem, "filename": f"{stem}.mp4", **meta},
        "steps": [{"id": i, **s} for i, s in enumerate(steps)],
        "provenance": {"pipeline": "robot-gripper", "app_version": "0"},
    }, ensure_ascii=False), encoding="utf-8")


def from_gripper_dir(source: Path, out: Path) -> int:
    (out / "clips").mkdir(parents=True, exist_ok=True)
    count = 0
    for clip in sorted(source.glob("*.mp4")):
        side = clip.with_suffix(".gripper.json")
        if not side.exists():
            continue
        signal = json.loads(side.read_text(encoding="utf-8"))
        meta = probe(clip)
        boundaries = boundaries_from_gripper(signal["values"], float(signal["fps"]))
        steps = steps_from_boundaries(boundaries, meta["duration_sec"])
        if len(steps) < 2:
            continue
        link = out / "clips" / clip.name
        if not link.exists():
            link.symlink_to(clip.resolve())
        write_annotation(out, clip.stem, meta, steps)
        count += 1
    return count


def from_rh20t_p(source: Path, out: Path) -> int:
    """RH20T-P: рядом с каждым роликом json со списком примитивов {start, end} в секундах."""
    (out / "clips").mkdir(parents=True, exist_ok=True)
    count = 0
    for clip in sorted(source.glob("*.mp4")):
        side = clip.with_suffix(".primitives.json")
        if not side.exists():
            continue
        primitives = json.loads(side.read_text(encoding="utf-8"))
        moments = sorted({float(p["start"]) for p in primitives} | {float(p["end"]) for p in primitives})
        meta = probe(clip)
        steps = steps_from_boundaries(moments, meta["duration_sec"])
        if len(steps) < 2:
            continue
        link = out / "clips" / clip.name
        if not link.exists():
            link.symlink_to(clip.resolve())
        write_annotation(out, clip.stem, meta, steps)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gripper", "rh20t_p"], required=True)
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    builder = from_gripper_dir if args.dataset == "gripper" else from_rh20t_p
    print(f"собрано {builder(args.source, args.out)} роликов в {args.out}")


if __name__ == "__main__":
    main()
```

The exporter that turns a chosen demo dataset into `<clip>.mp4 + <clip>.gripper.json` is dataset-specific and lives in `scripts/export_<dataset>.py`; it is written when the research report names the dataset that downloads within the day (RH20T-P annotations, DROID/Bridge episodes, or ManiSkill/RoboTwin renders). Its contract is exactly the sidecar format above, nothing else.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_robot_boundaries.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/make_robot_devset.py tests/test_robot_boundaries.py
git commit -m "Robot corpus with boundaries derived from gripper events or primitive annotations"
```

---

### Task 7: Evaluation protocol — variants, three sets, cross-embodiment

**Files:**
- Create: `scripts/eval_variants.py`
- Test: `tests/test_eval_variants.py`

**Interfaces:**
- `parse_sweep(text: str) -> dict[str, dict]` — parses `sweep.py` output lines `<method>  F1@0.1 F1@0.25 F1@0.5  lo–hi  err  steps  s/clip` into `{method: {"f1_05": float, "lo": float, "hi": float, "steps": float}}`.
- `render(rows: list[dict]) -> str` — Markdown table with columns `вариант | набор | F1@0.5 | 90 % | шагов`.
- CLI: `python scripts/eval_variants.py --variants base,sigma1,sigma3,focal --sets human=/path/validation robot=/path/robo_holdout --threshold 0.7`. For each variant: `POST /load {"name": variant}` (empty name for `base`), then for each set run `sweep.py` per source directory and collect. Prints the table and writes `experiments/results/<date>-variants.md`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_variants.py
def test_parse_sweep_reads_f1_and_interval():
    from scripts.eval_variants import parse_sweep

    text = (
        "метод                               F1@0.1  F1@0.25  F1@0.5   90% для F1@0.5  границы  шагов  с/ролик\n"
        "learned-boundaries:PRAXIS_TAS_THRESHOLD=0.7   0.732    0.680   0.411      0.385–0.438    0.40с    9.1      1.6\n"
    )
    parsed = parse_sweep(text)
    row = parsed["learned-boundaries:PRAXIS_TAS_THRESHOLD=0.7"]
    assert row == {"f1_05": 0.411, "lo": 0.385, "hi": 0.438, "steps": 9.1}


def test_render_makes_a_markdown_table():
    from scripts.eval_variants import render

    table = render([{"variant": "base", "set": "human", "f1_05": 0.411, "lo": 0.385, "hi": 0.438, "steps": 9.1}])
    assert "| base | human | 0.411 | 0.385–0.438 | 9.1 |" in table
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_eval_variants.py -v` → FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `scripts/eval_variants.py`**

```python
#!/usr/bin/env python3
"""Сравнение вариантов детектора на фиксированных наборах, с интервалами.

Каждый вариант — именованный чекпоинт на сервисе. Наборы не используются для подбора:
порог один и тот же для всех вариантов и задаётся здесь, а не подбирается по ним.

    python scripts/eval_variants.py --variants base,sigma1,sigma3,focal \\
        --sets human=/mnt/data/praxis-pool/validation robot=data/robo_holdout
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

from praxis.pipeline.learned import post

LINE = re.compile(
    r"^(?P<method>\S+)\s+(?P<f01>[\d.]+)\s+(?P<f025>[\d.]+)\s+(?P<f05>[\d.]+)\s+"
    r"(?P<lo>[\d.]+)–(?P<hi>[\d.]+)\s+[\d.]+с\s+(?P<steps>[\d.]+)"
)


def parse_sweep(text: str) -> dict[str, dict]:
    rows = {}
    for line in text.splitlines():
        match = LINE.match(line.strip())
        if match:
            rows[match["method"]] = {
                "f1_05": float(match["f05"]), "lo": float(match["lo"]),
                "hi": float(match["hi"]), "steps": float(match["steps"]),
            }
    return rows


def render(rows: list[dict]) -> str:
    lines = ["| вариант | набор | F1@0.5 | 90 % | шагов |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['set']} | {r['f1_05']:.3f} | {r['lo']:.3f}–{r['hi']:.3f} | {r['steps']:.1f} |")
    return "\n".join(lines)


def sweep(clips: Path, gt: Path, threshold: float) -> dict:
    method = f"learned-boundaries:PRAXIS_TAS_THRESHOLD={threshold}"
    out = subprocess.run(
        [sys.executable, "scripts/sweep.py", "--clips", str(clips), "--gt", str(gt), "--methods", method],
        capture_output=True, text=True, check=False,
    ).stdout
    return parse_sweep(out).get(method, {"f1_05": float("nan"), "lo": float("nan"), "hi": float("nan"), "steps": float("nan")})


def evaluate(variants: list[str], sets: dict[str, Path], threshold: float) -> list[dict]:
    rows = []
    for variant in variants:
        post("/load", {"name": "" if variant == "base" else variant}, timeout=120)
        for name, root in sets.items():
            # Валидационный пул хранит источники подпапками; отложенный роботный — плоско.
            pairs = (
                [(root / "clips" / d.name, root / "annotations" / d.name) for d in sorted((root / "clips").iterdir()) if d.is_dir()]
                if (root / "annotations").exists() else [(root / "clips", root / "gt")]
            )
            for clips, gt in pairs:
                rows.append({"variant": variant, "set": f"{name}/{clips.name}" if len(pairs) > 1 else name,
                             **sweep(clips, gt, threshold)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True, help="через запятую; base = основной чекпоинт")
    parser.add_argument("--sets", nargs="+", required=True, help="имя=путь")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    sets = {k: Path(v) for k, _, v in (s.partition("=") for s in args.sets)}
    rows = evaluate(args.variants.split(","), sets, args.threshold)
    table = render(rows)
    print(table)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{dt.date.today().isoformat()}-variants.md"
    target.write_text(f"# Варианты детектора, порог {args.threshold}\n\n{table}\n", encoding="utf-8")
    print(f"\nзаписано в {target}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/praxis/bin/python -m pytest tests/test_eval_variants.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_variants.py tests/test_eval_variants.py
git commit -m "Evaluate named detector variants on fixed sets with intervals"
```

---

### Task 8: Run the experiments on DL5

**Files:**
- Create: `scripts/run_variants.sh` (executed on DL5 from `~/praxis/app`)

**Interfaces:**
- Consumes Tasks 1–7. Produces named checkpoints `boundary-<variant>.pt` and `experiments/results/<date>-variants.md` (copied back to the laptop).

- [ ] **Step 1: Write the driver**

```bash
#!/usr/bin/env bash
# Все варианты детектора на одном корпусе, без подмены живой модели.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH" PYTHONPATH="$PWD"
export PRAXIS_FEATURES=video PRAXIS_VIDEO_BASE_URL=http://127.0.0.1:8102
export PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104 PRAXIS_NAMER=none
export PRAXIS_VOCAB="$PWD/data/vocab_atomic.yaml" PRAXIS_WORK_DIR="$HOME/praxis/work"
export PRAXIS_MIN_SEGMENT_SEC=0.5 PRAXIS_IDLE_RATIO=0
PY="$HOME/praxis/venv/bin/python"
CORPUS=${CORPUS:-data/train_mix}

train() {  # имя, доп. аргументы
  local name=$1; shift
  echo "### $(date +%H:%M) $name ###"
  $PY scripts/train_boundaries.py --train "$CORPUS" --epochs 60 --name "$name" --no-activate "$@"
}

PRAXIS_TAS_MOTION=0 train nomotion --tolerance 2
PRAXIS_TAS_MOTION=1 train motion   --tolerance 2
PRAXIS_TAS_MOTION=1 train sigma1   --tolerance 1
PRAXIS_TAS_MOTION=1 train sigma3   --tolerance 3
PRAXIS_TAS_MOTION=1 train focal    --tolerance 2 --focal

echo "### $(date +%H:%M) оценка ###"
$PY scripts/eval_variants.py --variants base,nomotion,motion,sigma1,sigma3,focal \
    --sets human="$HOME/praxis/praxis-pool/validation" robot=data/robo_holdout --threshold 0.7
```

Note: `base` (the live 2-stage checkpoint) expects 768 channels; `eval_variants.py` must run it with `PRAXIS_TAS_MOTION=0`. Add to `evaluate()` in Task 7: read `/health` after `/load` and set `os.environ["PRAXIS_TAS_MOTION"] = "1" if health["dim"] == 769 else "0"` before calling `sweep()` (the subprocess inherits it). Add this line to `scripts/eval_variants.py` and a test asserting the environment switch (monkeypatch `post` to return `{"dim": 769}` for `/health`).

- [ ] **Step 2: Cross-embodiment runs**

After the mixed-corpus run, repeat with `CORPUS=data/train_human` and `CORPUS=data/train_robot` (both built with `make_mix.py` from a single source each) and names `human-only`, `robot-only`; evaluate all three on both sets. Record in the same results file.

- [ ] **Step 3: Launch and watch**

```bash
scp scripts/*.py scripts/run_variants.sh dl5:~/praxis/app/scripts/ && rsync -az praxis dl5:~/praxis/app/
ssh -n dl5 'cd ~/praxis/app && setsid nohup bash scripts/run_variants.sh > ~/praxis/logs/variants.log 2>&1 < /dev/null & disown'
```

Poll `~/praxis/logs/variants.log` in the background; do not block the session.

- [ ] **Step 4: Commit the driver and the results**

```bash
scp dl5:~/praxis/app/experiments/results/*-variants.md experiments/results/
git add scripts/run_variants.sh experiments/results/
git commit -m "Variant experiments on the mixed corpus with cross-embodiment checks"
```

---

### Task 9: Promote the winner, verify visually, document

**Files:**
- Create: `scripts/tas_promote.sh`
- Modify: `docs/DECISION.md` (results section), `docs/MVP.md` (numbers), `docs/PIPELINE.md` (motion channel, `/load`, variants)

- [ ] **Step 1: Promotion script**

```bash
#!/usr/bin/env bash
# Сделать именованный вариант основным чекпоинтом, сохранив прежний с датой.
set -eu
name=$1
cd "$HOME/praxis/checkpoints"
cp boundary.pt "boundary.$(date +%Y%m%d-%H%M).bak"
cp "boundary-$name.pt" boundary.pt
curl -s -X POST http://127.0.0.1:8104/load -H 'Content-Type: application/json' -d '{"name": ""}'
echo; echo "основной чекпоинт: $name"
```

Promote only if the variant beats `base` on **both** `human` and `robot` sets by more than the width of its interval on each.

- [ ] **Step 2: Visual check on the filmed clips**

On DL5 the app already runs with the live model (`run.sh`); re-upload the three filmed clips through `/api/v1/jobs` (they are in `~/praxis/clips/`), then render side-by-side locally as before:

```bash
python scripts/demo_video.py --clip /mnt/data/praxis-pool/test/filmed/test_bottle.mp4 \
    --pred "было=<old json>" "стало=<new json>" --out /mnt/data/praxis-pool/test/side_by_side_v2.mp4
```

Look at a frame with the image viewer; labels must be readable, steps must match the actions visually.

- [ ] **Step 3: Documentation**

Update `docs/DECISION.md` with a table `вариант | human F1@0.5 | robot F1@0.5` and a paragraph naming what changed and why; update `docs/MVP.md` numbers; add to `docs/PIPELINE.md` the motion channel (`PRAXIS_TAS_MOTION`), `/load`, named checkpoints and `tas_promote.sh`.

- [ ] **Step 4: Commit and push the branch**

```bash
git add scripts/tas_promote.sh docs/DECISION.md docs/MVP.md docs/PIPELINE.md
git commit -m "Promote the winning boundary variant and record the numbers"
git push team feature/naming-quality
```

---

## Dropped from the spec, with reason

**ASRF refinement.** ASRF's gain comes from re-labelling each boundary-delimited segment by majority vote of a *class* branch. Our head has no classes, so the refinement has nothing to vote on; its boundary branch (Gaussian targets, peak picking) is what we already run. Nothing to add.

## Self-review

- Spec §1 corpus → Tasks 5, 6. Spec §2 items 1–3 → Tasks 1–4; item 4 dropped with reason. Spec §3 → Tasks 7, 8. Spec §4 → Task 9. Spec §5 tests → Tasks 1, 2, 3, 5, 6, 7.
- Names used consistently: `motion_band`, `stack_motion`, `checkpoint_path`, `load_checkpoint`, `focal_loss`, `select`/`build`, `boundaries_from_gripper`, `steps_from_boundaries`, `parse_sweep`, `render`, `evaluate`.
- Dataset-specific exporter for the robot corpus is the only piece whose target is decided by the research report; its contract (sidecar `*.gripper.json`) is fixed here so the rest of the plan does not wait for it.

---

## Revision after the research digests (2026-09-04, evening)

Evidence in `experiments/research/2026-09-04-boundary-sota.md` changes three things. Tasks below **replace** Task 4 and refine Tasks 6 and 8; everything else stands.

### Task 4 (replaced): temporal feature augmentation, difference channels, test-time augmentation

**Why:** focal loss is dropped — ASRF measured −11 F1@50 with it and no boundary paper uses it. The two best-verified levers for small feature-level corpora are C2F-TCN's stochastic temporal max-pooling (+2…11 F1@50 on MS-TCN) and DDM-Net's feature-difference inputs (+8.5 F1@0.05).

**Files:**
- Modify: `praxis/pipeline/learned.py` (add `difference_channels`)
- Modify: `scripts/serve_tas.py` (`TrainRequest.augment`, `temporal_pool`, TTA in `/predict`)
- Modify: `scripts/train_boundaries.py` (`--augment`, `--diff`), `praxis/config.py` (`TAS_DIFF`)
- Test: `tests/test_segment.py` (append)

**Interfaces:**
- `learned.difference_channels(matrix: np.ndarray, lags: tuple[int, ...] = (1, 2, 4)) -> np.ndarray` — returns `(T, D + 2*len(lags))`: for each lag k, `|f_t − f_{t−k}|` averaged over D as one channel, and cosine similarity between `f_t` and `f_{t−k}` as one channel (both padded at the start by repeating the first value). Cheap: 6 extra channels, no new extraction.
- `config.TAS_DIFF: bool` from `PRAXIS_TAS_DIFF` (default `"1"`). Applied after `stack_motion` in both training and inference: `matrix = difference_channels(matrix)`.
- Service: `TrainRequest.augment: bool = False`, `pool_low: float = 0.5`, `pool_high: float = 2.0`, `base_window: int = 5` (steps, ≈0.6 s at 8 fps). `temporal_pool(features: Tensor[T, D], target: Tensor[T], window: int) -> tuple[Tensor, Tensor]` — max-pool features and targets over non-overlapping windows of `window` steps (targets: max within the window keeps the boundary). With `augment`, each epoch draws `window = base_window` with p = 0.5, else uniform integer in `[base_window*pool_low, base_window*pool_high]`, and trains on the pooled sequence (boundary positions scale accordingly, handled by pooling the target).
- `PredictRequest.tta: bool = False` — when true, average sigmoid outputs over windows `{1, 2, 3, 4}` (window 1 = no pooling; pooled outputs are upsampled back by repetition before averaging).

- [ ] **Step 1: Failing tests** (append to `tests/test_segment.py`)

```python
def test_difference_channels_add_two_per_lag():
    import numpy as np

    from praxis.pipeline.learned import difference_channels

    matrix = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    out = difference_channels(matrix, lags=(1, 2, 4))
    assert out.shape == (20, 8 + 6)
    assert np.all(out[:, 8] >= 0)                    # |Δ| неотрицательна
    assert np.all(np.abs(out[:, 9]) <= 1.0 + 1e-6)   # косинус в [-1, 1]


def test_difference_channels_flag_a_jump():
    import numpy as np

    from praxis.pipeline.learned import difference_channels

    matrix = np.vstack([np.zeros((10, 4)), np.ones((10, 4))]).astype(np.float32)
    out = difference_channels(matrix, lags=(1,))
    assert out[10, 4] > out[5, 4] and out[10, 4] > out[15, 4]
```

- [ ] **Step 2: Implement `difference_channels` in `praxis/pipeline/learned.py`**

```python
def difference_channels(matrix: np.ndarray, lags: tuple[int, ...] = (1, 2, 4)) -> np.ndarray:
    """Разности признаков во времени как дополнительные каналы (идея DDM-Net).

    Граница — это изменение, и модели проще увидеть готовую разность, чем вычислить её
    самой из двух соседних векторов: в DDM-Net это дало +8.5 F1 против сырых признаков,
    тогда как оптический поток — меньше одного пункта.
    """
    matrix = matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1) + 1e-6
    extra = []
    for lag in lags:
        shifted = np.vstack([np.repeat(matrix[:1], lag, axis=0), matrix[:-lag]])
        extra.append(np.abs(matrix - shifted).mean(axis=1))
        shifted_norms = np.vstack([np.repeat(norms[:1], lag), norms[:-lag]]).ravel()
        extra.append((matrix * shifted).sum(axis=1) / (norms * shifted_norms))
    return np.concatenate([matrix, np.stack(extra, axis=1).astype(np.float32)], axis=1)
```

Wire it: in `config.py` add `TAS_DIFF = os.getenv("PRAXIS_TAS_DIFF", "1").lower() not in {"0", "false", "no"}`; in `LearnedSegmenter.run` and in `scripts/train_boundaries.py`, after the motion stacking: `if config.TAS_DIFF: matrix = difference_channels(matrix)`.

- [ ] **Step 3: Service — augmentation and TTA (`scripts/serve_tas.py`)**

```python
def temporal_pool(features: torch.Tensor, target: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Стохастический max-pool по времени (C2F-TCN): та же последовательность на другой
    скорости. Цель пулится максимумом, поэтому граница не теряется."""
    if window <= 1:
        return features, target
    length = (features.shape[0] // window) * window
    f = features[:length].view(-1, window, features.shape[1]).amax(dim=1)
    t = target[:length].view(-1, window).amax(dim=1)
    return f, t
```

In `train()`, inside the epoch loop before building `x`/`y`:

```python
            if request.augment:
                window = request.base_window if random.random() < 0.5 else random.randint(
                    max(1, int(request.base_window * request.pool_low)),
                    max(1, int(request.base_window * request.pool_high)))
                f_t, t_t = temporal_pool(torch.tensor(features), torch.tensor(target), window)
                x = f_t.T.to(device).unsqueeze(0)
                y = t_t.to(device).view(1, 1, -1)
            else:
                x = torch.tensor(features.T, device=device).unsqueeze(0)
                y = torch.tensor(target, device=device).view(1, 1, -1)
```

(`import random` at the top.) In `predict()`, when `request.tta`:

```python
        outputs = []
        for window in (1, 2, 3, 4):
            f_t, _ = temporal_pool(torch.tensor(features), torch.zeros(len(features)), window)
            with torch.no_grad():
                score = torch.sigmoid(model(f_t.T.to(device).unsqueeze(0))[-1])[0, 0].cpu()
            outputs.append(score.repeat_interleave(window)[: len(features)])
        scores = torch.stack([o if len(o) == len(features) else torch.nn.functional.pad(o, (0, len(features) - len(o))) for o in outputs]).mean(0)
```

- [ ] **Step 4: Flags** — `train_boundaries.py`: `--augment` → `"augment": True`; `eval_variants.py`/`LearnedSegmenter`: `config.TAS_TTA` (`PRAXIS_TAS_TTA`, default `"0"`) → `"tta": True` in the `/predict` payload.

- [ ] **Step 5: Tests pass, deploy, commit**

```bash
git add praxis/pipeline/learned.py praxis/config.py scripts/serve_tas.py scripts/train_boundaries.py tests/test_segment.py
git commit -m "Difference channels, temporal augmentation and test-time pooling for the boundary head"
```

### Task 6 (refined): the concrete robot source is LIBERO-10 subtasks

`KeWangRobotics/libero_10_subtasks` on Hugging Face: 500 episodes, per-frame `subtask` string, 256×256 at 10 fps, Apache-2.0, direct download — no simulator. Exporter `scripts/export_libero.py` writes `<episode>.mp4` (ffmpeg from the frame sequence at 10 fps, scaled to 720 px height with libx264) and `<episode>.subtasks.json` = list of `{"start": sec, "end": sec, "name": str}` built from runs of equal `subtask` values. `make_robot_devset.py` gains `--dataset subtasks` reading that sidecar (identical to the `rh20t_p` branch: boundaries = starts and ends of runs). Hold-out: last 60 episodes by name.

Second robot source when time allows: `unitreerobotics/G1_*Inspire*` LeRobot sets — hand joint angles in `observation.state` → `boundaries_from_gripper` on the mean finger closure.

### Task 8 (refined): thresholds and variants

- Threshold is swept on the **training corpus** (`data/train_mix`, a 10 % slice held out from training, `data/train_mix_dev`), over `{0.35, 0.5, 0.7, 0.85}`; the chosen value is then fixed for validation. `run_variants.sh` gets a `DEV=data/train_mix_dev` sweep before `eval_variants.py`.
- Variants: `nomotion`, `motion`, `motion+diff`, `motion+diff+augment`, `sigma1`, `sigma3`; evaluation with and without `PRAXIS_TAS_TTA=1`.

### Task 10 (new, after Task 9 if time remains): feature backbone A/B

`scripts/serve_video.py` already serves `videomae` (`MCG-NJU/videomae-large`). Re-extract the corpus and validation features with `PRAXIS_VIDEO_MODEL=videomae` (cache key includes the model name), retrain the best variant, compare on both sets. Evidence for a gain is mixed (TWLV-I: −10 on Breakfast, +7 on GTEA), so this is an A/B, not an assumption; VideoMAEv2 weights are CC-BY-NC.
