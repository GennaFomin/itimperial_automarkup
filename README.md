<div align="center">

<h1>
  <br>
  🅿🆁🅰🆇🅸🆂
  <br>
</h1>

<h3>Automatic action annotation for short manipulation videos</h3>

<p><i>Upload a clip, get it split into steps with an action, an object and a keyframe each,<br>correct the draft in a timeline editor, export JSON or CSV.</i></p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-services-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="React" src="https://img.shields.io/badge/React-19-20232A?style=for-the-badge&logo=react&logoColor=61DAFB">
  <img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-5.8-3178C6?style=for-the-badge&logo=typescript&logoColor=white">
</p>

<p>
  <img alt="Docker" src="https://img.shields.io/badge/Docker-compose-2496ED?style=for-the-badge&logo=docker&logoColor=white">
  <img alt="Pytest" src="https://img.shields.io/badge/Pytest-129%20tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white">
  <img alt="Ruff" src="https://img.shields.io/badge/Ruff-linted-FCC21B?style=for-the-badge&logo=ruff&logoColor=black">
  <img alt="Qwen3-VL" src="https://img.shields.io/badge/Qwen3--VL-8B-6C3FC5?style=for-the-badge">
  <img alt="TimeSformer" src="https://img.shields.io/badge/TimeSformer-K400-111F68?style=for-the-badge">
</p>

<p>
  <a href="#tldr">TL;DR</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#boundary-detector">Boundary Detector</a> •
  <a href="#metrics">Metrics</a> •
  <a href="#features">Features</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#quality">Quality</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#documentation">Documentation</a>
</p>

</div>

---

<a id="tldr"></a>
## 🚀 TL;DR

Praxis is a **video annotation assistant** for the AI Product Hack 2026 case by IT Imperial: short clips of a person or a robot manipulating objects have to be turned into a list of steps — *picked up the bottle, rotated it, put it down* — that can later feed robotics datasets. The app produces the whole draft on its own; a reviewer **checks and corrects** instead of annotating from scratch.

The pipeline has one central design decision, earned the hard way: **the language model never touches time.** Step boundaries come from a class-free, learned **boundary detector** on video features (a consensus of two MS-TCN heads over TimeSformer features at 8 Hz); the vision-language model only *names* segments that are already cut, with no class list, in open vocabulary. Every fallback the system takes — a service down, a clip too short — is reported in `warnings`, never hidden behind a green status.

> **Why a separate detector?** The case scores steps one-to-one at IoU ≥ 0.5 with a boundary error under 2 s. Asking a VLM for timecodes over-segments and costs 25–60 s per clip; the detector places boundaries in 0.4 s and infers the number of steps itself. The comparison, with numbers, is in [experiments/research/BOUNDARIES.md](experiments/research/BOUNDARIES.md).

---

<a id="quickstart"></a>
## ⚡ Quickstart

### 1. Docker

```bash
docker compose up --build       # then open http://localhost:8000
```

### 2. Native Python

Needs `ffmpeg` with libx264, Python 3.12 and Node 20+.

```bash
uv venv ~/.venvs/praxis --python 3.12
uv pip install --python ~/.venvs/praxis/bin/python -e ".[dev]"
cd web && npm install && npm run build && cd ..
~/.venvs/praxis/bin/python -m uvicorn praxis.api:app --port 8000
```

### 3. Model services (one GPU box, reached over HTTP)

| service | port | command |
| --- | --- | --- |
| video features, TimeSformer-K400 | 8102 | `python scripts/serve_video.py --port 8102 --model timesformer` |
| boundary detector, shipped ensemble | 8104 | `python scripts/serve_tas.py --port 8104 --checkpoint checkpoints/boundary.pt` |
| naming, Qwen3-VL-8B | 8100 | `python scripts/serve_vlm.py --port 8100 --model Qwen/Qwen3-VL-8B-Instruct` |

Point the app at them with `PRAXIS_VIDEO_BASE_URL`, `PRAXIS_TAS_BASE_URL`, `PRAXIS_VLM_BASE_URL` (see `.env.example` for the SSH-tunnel variant and [docs/PIPELINE.md](docs/PIPELINE.md) for every setting). `/health` of the detector must report `"dim": 768, "members": 2, "fusion": "min"`. Without the services the app still runs — boundaries fall back to kernel change-point, features to grey blocks, steps arrive unlabelled — and says so in `warnings`.

### 4. One clip end to end

```bash
curl -F "file=@examples/clip.mp4" http://localhost:8000/api/videos   # 202 + job id
curl http://localhost:8000/api/videos/<id>                           # status, stage, progress, warnings
curl http://localhost:8000/api/videos/<id>/export.json > annotation.json
```

---

<a id="architecture"></a>
## 🧭 Architecture

```mermaid
flowchart LR
    subgraph APP["⚙️ Praxis app — FastAPI + React"]
        UP["📤 upload<br/>MP4 / MOV, ≤ 2 min"]
        FF["🎞️ ffmpeg<br/>32 fps frames, motion band"]
        VAL["✅ validation<br/>schema + invariants"]
        ED["🖥️ timeline editor<br/>drag, split, merge, undo"]
        EX["📦 export<br/>JSON / CSV, case contract"]
    end

    subgraph GPU["🧠 Model services"]
        ENC["TimeSformer-K400<br/>0.5 s window, 8 Hz"]
        DET["Boundary detector<br/>2 × MS-TCN, class-free"]
        VLM["Qwen3-VL-8B<br/>open-vocabulary naming"]
    end

    UP --> FF --> ENC --> DET --> VLM --> VAL --> ED --> EX
    FF -.->|frames per step| VLM
```

| stage | what runs | why this one |
| --- | --- | --- |
| frames | ffmpeg, 32 fps, 256 px | one pass gives both the encoder input and the motion band for the timeline |
| features | TimeSformer-K400, 16-frame window (0.5 s), stride 4 (8 Hz), cached on disk | Kinetics-supervised features beat self-supervised ones on temporal tasks; the short window is what makes sub-second steps visible |
| boundaries | consensus of two MS-TCN heads (minimum of their per-frame probabilities), peaks above 0.6 at least 0.5 s apart; steps exist only where the motion band shows activity, the rest is pause | class-agnostic by construction — trained on timecodes only — so it transfers to any domain |
| labels | Qwen3-VL-8B, 5 frames + before/after context, open vocabulary | no class list exists in advance, so the answer is generated, not picked |
| validation | pydantic schema and invariants | no overlaps, keyframe inside its step, everything inside the duration |
| export | the case owner's contract, integer milliseconds | `schema_version`, `model_version`, latency, cost, artifacts |

Granularity stays a knob, not a rewrite: `PRAXIS_TAS_THRESHOLD` (peak height), `PRAXIS_TAS_PEAK_GAP_SEC` (peak interval), `PRAXIS_MIN_SEGMENT_SEC` (minimum step), `PRAXIS_IDLE_MODE` with `PRAXIS_ACTIVITY_LEVEL` / `PRAXIS_ACTIVITY_LOW` (where a step starts and ends) are all runtime settings.

---

<a id="boundary-detector"></a>
## 🎯 Boundary Detector

The detector is the part of the pipeline the case metric depends on most, and the part that was chosen by measurement rather than preference. The full evidence is in [experiments/](experiments/); this section is the summary.

### What ships

| | |
| --- | --- |
| **Model** | MS-TCN (Farha & Gall, 2019): 2 stages × 10 dilated residual layers, 64 filters, per-stage loss, smoothing loss. One logit per frame — *did the action change here* — instead of a class softmax, so no taxonomy is ever learned. |
| **Input** | TimeSformer-K400 features, 768-d, 16-frame window at 32 fps (0.5 s), stride 4 (8 Hz). No motion or difference channels: measured, no gain. |
| **Ensemble** | two checkpoints packed into one file, [checkpoints/boundary.pt](checkpoints/README.md), fused by *consensus*: the per-frame minimum of their probabilities, so a cut needs both models to see a change. Trained on 554 and 203 Assembly101 fine-grained clips respectively. |
| **Decoder** | local maxima of the fused probability above 0.6, at least 0.5 s apart. Steps are the *active* regions of the motion band (above the clip's noise floor plus 0.15 of the way to its mean, with hysteresis), split by those cuts; everything else is a pause. Steps shorter than 0.75 s are dropped. |
| **Training signal** | step start and end timecodes only, from public datasets (Assembly101, LIBERO). No manual annotation, no class labels. |

### How it was chosen

1. **Class-free head over a frozen video encoder** — the case has no class list, and a boundary is the same event in any domain ([research](experiments/research/BOUNDARIES.md)).
2. **Architecture ablation is flat on the target regime.** 2 vs 4 stages, boundary width σ, motion channels, C2F augmentation, test-time augmentation, three seeds and three times more data all land within ±0.01 on 85 atomic clips whose recording sessions never appear in training ([ranking](experiments/results/2026-09-05-ranking85.md), [stages](experiments/results/2026-09-05-stages-variants.md)).
3. **What is *not* flat: the feature time resolution.** Halving the encoder window to 0.5 s at 8 Hz sharpens the peaks enough for a 0.5 s interval between cuts, which is where the short steps live: +0.05 for the same model and corpus ([final report](experiments/results/2026-09-05-boundary-final.md)).
4. **Consensus instead of averaging.** The minimum of two models' probabilities keeps only the changes both see: on the atomic set it scores 0.497–0.503 at *any* threshold from 0.3 to 0.7, which is what makes it safe for an unknown domain, and on hand-filmed desk clips outside the training data it finds 8–9 steps per 30 s clip where an averaged robot-aware pair found 3–6 ([final report](experiments/results/2026-09-05-boundary-final.md)). A member trained with robot episodes suppresses the robot's micro-motions but also the real actions in unfamiliar human clips, so it is not shipped ([cross-embodiment](experiments/results/2026-09-05-cross-embodiment.md)).
5. **Pauses come from the motion band, not from the detector.** A step exists only where the frame-difference band shows activity above the clip's own noise floor; the detector's cuts split active regions from inside. On three hand-filmed desk clips annotated with pauses (50–73 % of the time idle, [experiments/filmed](experiments/filmed/)) this lifts step-F1 from 0.18 to 0.62, identically on 720p originals and 256p copies. Assembly101 and EPIC annotate pauses the other way round — their gaps are *transitions with motion* — so with pauses on they drop to 0.32 / 0.25; the setting is a switch (`PRAXIS_IDLE_MODE=none` restores 0.47 / 0.57). A learned pause detector trained on those transitions (`train_boundaries.py --target gaps`) separates them well but smears into the actions' edges and does not beat stillness yet; it ships as an optional second service ([final report](experiments/results/2026-09-05-boundary-final.md)).
6. **Decoding was swept offline** over dumped probabilities — threshold, interval, prominence, penalised DP, activity levels — jointly on the three sets, so every number above uses the decoder the application actually runs ([decoders](experiments/results/2026-09-05-decoders-base.md)).

### Reproduce a number

```bash
# probabilities of the served checkpoint on a labelled set, once
python scripts/dump_scores.py --clips data/pool_atomic/clips --gt data/pool_atomic/gt --out work/scores/final_85.npz
python scripts/decode_sweep.py work/scores/final_85.npz            # best decoder offline
python scripts/boundary_report.py work/scores/final_85.npz         # product metrics under the app decoder
python scripts/ensemble_scores.py a_85.npz b_85.npz --out ab_85.npz  # any ensemble without re-running clips
```

---

<a id="metrics"></a>
## 📊 Metrics

Three labelled validation sets, all from public datasets with their own ground truth. Recording sessions of the atomic set never appear in any training corpus.

| set | source | clips | steps per clip | median step | regime |
| --- | --- | --- | --- | --- | --- |
| **atomic** | Assembly101 fine-grained | 85 | 9.0 | 0.87 s | the case: pick up, rotate, move, put down |
| **mid** | EPIC-Kitchens-100 | 16 | 4.2 | 1.4 s | egocentric kitchen actions |
| **robot** | LIBERO hold-out | 10 | 2.0 | 6 s | simulated manipulator, subtask boundaries |

Boundary detector, application decoder, no labels involved:

| metric | filmed | atomic | mid | what it means |
| --- | --- | --- | --- | --- |
| step-F1 @ IoU 0.5 — the case rule, shipped decoder with pauses | **0.617** | 0.315 | 0.249 | one-to-one matching of steps at IoU ≥ 0.5; atomic and mid annotate pauses as moving transitions, see below |
| step-F1 @ IoU 0.5, contiguous decoder (`PRAXIS_IDLE_MODE=none`) | 0.179 | **0.474** | **0.568** | the same without pauses; the single model shipped before scored — / 0.422 / 0.413 |
| boundary precision, ±2 s | — | **0.90** | 0.72 | share of our cuts within 2 s of a true boundary — the case's boundary tolerance; independent of the pause mode |
| boundary precision, ±1 s | — | 0.83 | 0.67 | the same at half the tolerance |
| boundary error, matched cuts | — | 0.27 s | 0.34 s | mean distance of matched cuts to the truth |

**filmed** — three 30-second desk clips we filmed ourselves and annotated frame by frame with pauses (29 actions, [experiments/filmed](experiments/filmed/)): the closest proxy for the case owner's hidden set, and the only set whose pauses mean stillness. In Assembly101 and EPIC the unannotated gaps are transitions full of motion, which is why the pause mode costs them step-F1 while the cuts themselves stay as precise as before. On the robot hold-out the consensus scores 0.483 without pauses.

Numbers come from the dumped probabilities of `checkpoints/boundary.pt` (`scripts/boundary_report.py`, `scripts/score_table.py`) and the offline pause sweep; every table behind them is in [experiments/results/2026-09-05-boundary-final.md](experiments/results/2026-09-05-boundary-final.md).

Product constraints from the case, measured end to end on the app:

| | value | case target |
| --- | --- | --- |
| processing per 20 s clip, with naming | ~40 s | ≤ 120 s |
| boundary error, matched cuts | 0.27 s | ≤ 2 s |
| valid JSON / CSV export | 100% | 100% |
| clips returned without a step | 0 (whole clip is one step when nothing is found) | — |

Naming (open vocabulary, ground-truth boundaries, scored against every admissible label): action 0.558, object 0.660, pair 0.426 — the history and the rejected variants are in [experiments/HISTORY.md](experiments/HISTORY.md).

---

<a id="features"></a>
## ✨ Features

- **Timeline editor** — drag boundaries with magnetic snapping to detected events, split and merge steps, undo/redo, keyframe filmstrip, hot keys, autosave; every edit and the time spent are logged, which is the evidence behind the "×3 faster than manual" claim rather than a declaration.
- **Open-vocabulary naming** with confidence — the VLM answers in the language of `PRAXIS_LANGUAGE` and reports its own confidence, or none; low-confidence steps are highlighted for review.
- **Honest degradation** — every fallback (kernel change-point instead of the detector, grey features, unnamed steps) lands in `warnings` on the job; nothing pretends to be fine.
- **Two API surfaces** — the app's own `/api/videos` and the case owner's `/api/v1` contract with export in integer milliseconds, `schema_version`, latency and cost ([docs/CONTRACT_V1.md](docs/CONTRACT_V1.md)).
- **One-knob granularity** — atomic actions by default, coarse steps with pauses by three environment variables.

---

<a id="tech-stack"></a>
## 🛠️ Tech Stack

| Layer | Tooling |
| --- | --- |
| **Boundaries** | MS-TCN ensemble (PyTorch) over TimeSformer-K400 features · peak decoder · kernel change-point fallback |
| **Naming** | Qwen3-VL-8B-Instruct via vLLM · context frames · self-reported confidence |
| **Backend** | FastAPI · pydantic v2 contract · SQLite · background jobs · ffmpeg |
| **Frontend** | React 19 + TypeScript + Vite · canvas timeline |
| **Evaluation** | dumped-probability harness (`dump_scores`, `decode_sweep`, `score_table`, `ensemble_scores`, `boundary_report`) · bootstrap intervals · case-rule scorer |
| **Data** | builders for Assembly101 windows, LIBERO episodes, Charades/EPIC subsets — public ground truth only |
| **Quality** | 129 pytest tests · Ruff |
| **DevOps** | Dockerfile · docker-compose · GPU services behind HTTP with health checks |

---

<a id="quality"></a>
## 🔬 Quality

**129 tests**, all green:

```bash
pytest -q
```

| suite | what it locks down |
| --- | --- |
| `test_schema.py`, `test_contract_v1.py` | the annotation contract: round trips, invariants, export formats |
| `test_api.py`, `test_api_v1.py` | upload → job → annotation → export over both API surfaces |
| `test_segment.py`, `test_motion_band.py` | the learned segmenter: feature stacking, fallback, dimension checks, prominence |
| `test_pack_ensemble.py`, `test_ensemble_scores.py` | packed ensembles load and average exactly as the service does |
| `test_train_guard.py`, `test_eval_variants.py`, `test_eval.py` | training refuses an incomplete corpus; evaluation parsing; metric arithmetic |
| `test_mix.py`, `test_robot_boundaries.py`, `test_export_libero.py`, `test_atomic_windows.py` | dataset builders and their provenance |
| `test_naming.py` | naming request flags and open-vocabulary answers |

The trainer aborts when more than 2% of a corpus has no features — a feature service that dies mid-run used to shrink a corpus silently and produce a model that looked ordinary and was not.

---

<a id="experiments"></a>
## 🧪 Experiments

Everything measured on the way to the shipped detector lives in [experiments/](experiments/README.md): validation pools and their reports, the architecture and feature ablations with every table, the reading list and the papers, the design and the implementation plan, and the scripts that ran the experiment queue on the GPU box.

| question | answer | where |
| --- | --- | --- |
| VLM or a separate detector for boundaries? | detector: better at IoU 0.5, 60× faster | [BOUNDARIES.md](experiments/research/BOUNDARIES.md) |
| Which encoder? | frozen encoders are within the interval of each other; TimeSformer is fastest | [HISTORY.md](experiments/HISTORY.md) |
| Which architecture? | MS-TCN, 2 stages, plain features — everything else is within ±0.01 | [ranking85](experiments/results/2026-09-05-ranking85.md), [stages](experiments/results/2026-09-05-stages-variants.md) |
| Human vs robot data? | step-length regime dominates embodiment; mix the regimes in the members | [cross-embodiment](experiments/results/2026-09-05-cross-embodiment.md) |
| Where is the headroom? | feature time resolution and boundary localisation, not data volume or decoding | [boundary-final](experiments/results/2026-09-05-boundary-final.md) |

---

## 📂 Project Structure

```
praxis/
├── praxis/               # annotation contract, API, storage, jobs, media
│   └── pipeline/         # segmenters: learned (default), kernel change-point, baselines
├── scripts/              # model services, dataset builders, training, evaluation harness
├── checkpoints/          # boundary.pt — packed ensemble; members/ — its four checkpoints
├── web/                  # React timeline editor
├── docs/                 # product docs: pipeline guide, decision, case, contract, MVP, QA
├── experiments/          # research: results, reports, reading, plans, GPU-box queue scripts
├── tests/                # 129 pytest tests
├── Dockerfile
└── docker-compose.yml
```

---

<a id="documentation"></a>
## 📚 Documentation

- **Run it** — [docs/PIPELINE.md](docs/PIPELINE.md) — services, settings, restart procedure.
- **Decision** — [docs/DECISION.md](docs/DECISION.md) — the algorithm and why, with the numbers.
- **Case** — [docs/CASE.md](docs/CASE.md) — requirements, contract and the metric rule · [docs/CONTRACT_V1.md](docs/CONTRACT_V1.md) — the `/api/v1` layer · [docs/MVP.md](docs/MVP.md) — readiness checklist · [docs/QA.md](docs/QA.md).
- **Weights** — [checkpoints/README.md](checkpoints/README.md) — what is in the ensemble and how to rebuild it.
- **Research** — [experiments/README.md](experiments/README.md) — index of everything measured.

---

<div align="center">
  <sub>Built for the AI Product Hack 2026 case by IT Imperial · 2026</sub>
</div>
