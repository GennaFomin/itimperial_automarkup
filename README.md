# Praxis — automatic action annotation for video

Praxis takes a short clip, splits it into steps on its own, works out the action, the
object and the keyframe for each one, and hands a reviewer a finished draft in a timeline
editor. From there a specialist **checks and corrects** rather than annotating from
scratch, and exports the result as JSON or CSV.

Built for the AI Product Hack 2026 case by IT Imperial: annotate short manipulation clips
so the resulting steps can be reused as a data layer for robotics.

---

## Quick start

### Docker

```bash
docker compose up --build       # then open http://localhost:8000
```

### Local

Needs `ffmpeg`, Python 3.12 and Node 20+.

```bash
uv venv ~/.venvs/praxis --python 3.12
uv pip install --python ~/.venvs/praxis/bin/python -e ".[dev]"
cd web && npm install && npm run build && cd ..
~/.venvs/praxis/bin/python -m uvicorn praxis.api:app --port 8000
```

Three model services run on a GPU box and are reached over HTTP: video features
(TimeSformer, port 8102), the boundary detector (port 8104) and the naming model
(Qwen3-VL-8B, port 8100); see [docs/PIPELINE.md](docs/PIPELINE.md) for the exact commands
and `.env.example` for the tunnel. The default segmenter is the learned boundary detector
— the choice and the evidence are in [docs/DECISION.md](docs/DECISION.md). Without the
services the app still runs: boundaries fall back to kernel change-point, features to
grey blocks, steps arrive unlabelled, and every such fallback is listed in `warnings`
rather than hidden behind a successful status.

### One clip end to end

```bash
curl -F "file=@examples/clip.mp4" http://localhost:8000/api/videos   # 202 + job_id
curl http://localhost:8000/api/videos/<id>                           # status, stage, progress
curl http://localhost:8000/api/videos/<id>/export.json > annotation.json
```

---

## How it works

```
frames → features → boundaries → labels → validation → editor → export
```

The central design decision, and the one that took the longest to earn:
**the language model never touches time.** It only names segments that have already been
cut.

| stage | what runs | why this one |
| --- | --- | --- |
| frames | ffmpeg at 16 fps | one pass gives both the motion band and the encoder input |
| features | TimeSformer-K400, 16-frame window, stride 4 | Kinetics-supervised features beat self-supervised ones on temporal tasks; cached on disk |
| boundaries | penalised kernel change-point, exact DP | infers the step count itself from one penalty knob |
| labels | Qwen3-VL-8B, open vocabulary, 5 frames + before/after context | no class list exists in advance, so the answer is generated, not picked |
| validation | pydantic schema and invariants | no overlaps, keyframe inside its step, everything inside the duration |
| export | the case owner's contract, integer milliseconds | `schema_version`, `model_version`, latency, cost, artifacts |

Granularity is a single knob, not a rewrite:

```bash
# atomic actions — picked up, rotated, moved, put down (default)
PRAXIS_TSM_PENALTY=2  PRAXIS_MIN_SEGMENT_SEC=0.5  PRAXIS_IDLE_RATIO=0

# coarse steps — 2-3 actions of several seconds with pauses between them
PRAXIS_TSM_PENALTY=8  PRAXIS_MIN_SEGMENT_SEC=1.5  PRAXIS_IDLE_RATIO=0.3
```

---

## How we got here

The path mattered more than any single result, so it is worth writing down.

**We started from an end-to-end skeleton, not from a model.** Upload, background job,
timeline editor, JSON/CSV export and a metrics harness existed before the segmenter did —
the segmenter was a stub returning three equal thirds. That order is deliberate:
integration on the last night kills more hackathon projects than weak models do.

**The first real segmenter was dynamic programming over motion.** It beat the stub, and
for two days it looked like the answer. It was not.

**Naming failed, and the baselines were what revealed it.** The first naming stage scored
0.400 on the verb while the constant "always attach" scored 0.660 — the model was worse
than a fixed guess, and without a trivial baseline that is invisible. Four different
approaches were measured; none beat the constant.

**The dead end turned out to be the taxonomy, not the code.** Rather than tuning prompts,
we rebuilt the same measurement on EPIC-KITCHENS. There the same pipeline beat the
constants several times over — 0.418 against 0.194 on the action. Assembly101's parts are
near-identical grey plastic pieces; the pipeline was fine, the data was unnameable.

**Then the validation itself turned out to be broken — three times over.** Naming was
being tuned on 67 segments where the noise floor was wider than every difference we were
chasing, so several "negative results" were simply unmeasured. The sets did not match the
case profile. And in the set we built ourselves, a greedy filter discarded 299 of 438
labels and then penalised the model for answering with a discarded one. Bootstrap
intervals, a case-profile set and multi-label scoring came out of that, and the naming
number doubled — not because the model improved, but because we started measuring
correctly.

**The case owner's deck moved the target.** The scoring rule is one-to-one matching at
IoU ≥ 0.5 **and a correct action class**. Our harness computed two quantities and neither
was that rule. So segmentation and naming are one metric, not two, and it decomposes
almost exactly into a product — which is how we learned that the verb, not time, gates
everything.

**The comparison with a language model was run properly, and it lost.** Two VLM
segmenters were implemented, including the binary-search idea the team proposed. Both
over-segment and cost several times more. One VLM trick did help — stamping the time into
the pixels — and it drew the VLM level with DP at the strict threshold, but not ahead.

**A uniform split beat our method, and that was the most useful failure.** On the
household set, cutting the clip into three equal pieces scored 0.627 against 0.429 for
our segmenter. That was a property of the benchmark: 2.2 steps of six seconds in a
twenty-second clip *are* roughly thirds. On atomic annotation the ranking inverted
completely and the uniform split collapsed from 0.627 to 0.288. The lesson is that a
benchmark can flatter a trivial method, and only a second benchmark exposes it.

**The winner came from the papers, not from tuning.** Penalised kernel change-point
detection — Arlot, Celisse and Harchaoui's formulation, implemented as described in
Truong, Oudre and Vayatis, applied to video by Perochon and Oudre — beat everything we
had written ourselves, and it decides the number of steps on its own instead of being
told. The exact mapping from our code to the equations is in
[docs/papers/](docs/papers/).

---

## What is measured

Three validation sets, all built from public datasets with their own ground truth, so no
manual annotation was needed:

| set | source | clips | steps per clip | median step | what it is for |
| --- | --- | --- | --- | --- | --- |
| `pool_val` | Charades test split | 90 | 2.2 | 6.0 s | case profile: static camera, one person, gaps between steps |
| `pool_atomic` | Assembly101 fine-grained | 85 | 9.0 | 0.87 s | atomic manipulation: pick up, rotate, move, put down |
| `train_atomic` | Assembly101 fine-grained | 204 | 9.9 | 0.97 s | training the boundary head |

### Boundary methods, one set and one metric

Labels deliberately take no part here: the question is only about time. Household set, 90
clips, 90% bootstrap intervals over clips.

| method | F1@0.1 | F1@0.25 | F1@0.5 | 90% CI | steps | s/clip |
| --- | --- | --- | --- | --- | --- | --- |
| **kernel change-point** | 0.767 | 0.746 | **0.507** | 0.459–0.553 | 3.3 | **0.4** |
| VLM asked directly | 0.718 | 0.699 | 0.541 | 0.478–0.602 | 3.9 | 25.3 |
| VLM + time stamped in pixels | 0.735 | 0.705 | 0.567 | 0.498–0.639 | 3.9 | 22.3 |
| motion DP (our earlier method) | 0.788 | 0.767 | 0.429 | 0.380–0.477 | 2.5 | 0.4 |
| uniform split | 0.826 | 0.806 | 0.627 | 0.585–0.669 | 3.0 | 0.8 |
| **VLM by binary search** | 0.559 | 0.509 | **0.198** | 0.143–0.255 | 6.0 | 58.6 |

On atomic granularity — the regime the case actually describes — the ranking inverts:

| method | F1@0.1 | F1@0.25 | F1@0.5 | steps (truth 9.9) |
| --- | --- | --- | --- | --- |
| feature-difference peaks | 0.726 | 0.708 | **0.450** | 14.0 |
| **kernel change-point** | 0.710 | 0.677 | 0.401 | 14.6 |
| uniform split into 10 | **0.795** | **0.733** | 0.288 | 10.0 |
| motion DP | 0.684 | 0.589 | 0.231 | 14.0 |
| motion minima | 0.603 | 0.483 | 0.142 | 13.9 |

Peaks score higher at the strict threshold but were *given* the number of steps; the
kernel method infers it. That is why the kernel method ships.

### Feature encoders, same head

| encoder | F1@0.1 | F1@0.25 | F1@0.5 | 90% CI | s/clip |
| --- | --- | --- | --- | --- | --- |
| VideoMAE | 0.785 | 0.750 | 0.536 | 0.484–0.588 | 10.2 |
| V-JEPA 2 | 0.776 | 0.759 | 0.509 | 0.460–0.559 | 12.8 |
| **TimeSformer-K400** | 0.767 | 0.746 | 0.507 | 0.459–0.553 | **9.6** |
| DINOv2 | 0.764 | 0.739 | 0.507 | 0.459–0.556 | 12.8 |

The intervals overlap completely: **the encoder barely matters here.** That refines the
literature rather than contradicting it — EAST lifted Assembly101 by *fine-tuning* the
encoder, not by swapping a frozen one. TimeSformer stays because it is the fastest.

### Naming, ground-truth boundaries, open vocabulary

90 clips, 197 segments, scored against every admissible label.

| variant | action | object | pair |
| --- | --- | --- | --- |
| baseline, 5 frames | 0.523 | 0.660 | 0.386 |
| **+ before/after context frames** | **0.558** | 0.660 | **0.426** |
| 8 frames instead of 5 | 0.142 | 0.406 | 0.107 |

Context frames are the only naming lever that worked. A paired test gives +0.036 with a
90% interval of −0.010 to +0.081 — the direction is right, significance is not
established, and it is enabled because it costs nothing in time. More frames actively
hurt, by a factor of four.

### End to end, under the case rule

| | value | case target |
| --- | --- | --- |
| step-F1 (one-to-one, IoU ≥ 0.5, correct action) | 0.134 | ≥ 0.75 |
| segmentation alone at IoU 0.5 | 0.495 | — |
| action accuracy on matched steps | 0.269 | ≥ 0.80 |
| boundary error, mean | 1.27 s | ≤ 2 s ✅ |
| processing per clip | ~7 s | ≤ 120 s ✅ |
| valid JSON / CSV | 100% | 100% ✅ |

The metric decomposes into a product: 0.495 × 0.269 ≈ 0.134. **Naming is the bottleneck,
not boundaries** — a point of verb accuracy is worth roughly twice a point of
segmentation.

For scale: on Assembly101 the *trained* state of the art is 0.328 and the common
ASFormer and MS-TCN++ reach 0.214 and 0.206, while we run with no training at all. The
ceiling with perfect naming on that set is 0.566, so the 0.75 target only makes sense in
the case's own domain — a handful of verbs and a generic object.

---

## Tried and rejected, with numbers

Everything below is measured and switched off, with the figures recorded next to the flag
in code. The case owner's own rule applies: a new dependency without a measurable gain is
not an improvement.

| idea | result |
| --- | --- |
| VLM boundaries by binary search | F1@0.5 0.198 against 0.612, six steps instead of 2.6, 5× slower |
| VLM boundaries asked directly | over-segments 4.0 against 2.5, 2× slower |
| Recursive self-similarity parsing (UBoCo) | 0.251 at best |
| TW-FINCH clustering | 0.367 |
| Motion minima only | 0.142 |
| Encoder swap (VideoMAE, V-JEPA 2, DINOv2) | within the interval |
| 8 Hz feature grid instead of 4 Hz | 63% more expensive, no gain |
| Edge trimming by activity | 0.507 → 0.348 on generous annotation |
| Two-stage naming, three variants | object up to 0.626, pair down to 0.288 |
| Neighbour-step context in the prompt | pair 0.209 against 0.224 |
| Frame position labels as text | 0.194 against 0.224 |
| Likelihood scoring instead of generation | 0.209 against 0.224, twice the time |
| Joint parsing of the whole clip | 0.149 against 0.224 |
| SAM2 crop around the tracked object | 0.090 against 0.224 on the pair |
| Grounding DINO for the object | right object in top-5 in 17% of cases |
| SigLIP2 classifier | does not beat a constant |
| Block motion map | F1@0.5 0.605 against 0.715 |
| Crop to the active region | 0.642 |
| Temporal feature smoothing | 0.700 |

---

## Repository layout

```
praxis/            annotation contract, API, storage, pipeline stages
  pipeline/        segmenters: similarity, baselines, clustering, learned, motion DP
scripts/           model services, dataset builders, evaluation and sweep harnesses
web/               React timeline editor
docs/              case knowledge base, analysis, pipeline guide, papers
tests/             56 tests
```

## Documentation

- [docs/PIPELINE.md](docs/PIPELINE.md) — what to run and which settings matter
- [docs/CASE.md](docs/CASE.md) — the case requirements, contract and metric rule
- [docs/BOUNDARIES.md](docs/BOUNDARIES.md) — the full boundary analysis with sources
- [docs/READING.md](docs/READING.md) — what to read, and what to take from each paper
- [docs/papers/](docs/papers/) — the papers themselves
