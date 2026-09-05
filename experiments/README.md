# Experiments — how the boundary detector was chosen

Everything that led to the shipped detector: the case analysis, the validation pools,
every architecture and feature variant that was measured, the numbers, the scripts that
produced them, and the plans they followed. Nothing here is needed to run the product;
it is the evidence behind the defaults in `praxis/config.py` and the weights in
`checkpoints/`.

## Read this first

| document | what it answers |
| --- | --- |
| [results/2026-09-05-boundary-final.md](results/2026-09-05-boundary-final.md) | the decision day: protocol, ceilings, error analysis, every table, the final ensemble |
| [../docs/DECISION.md](../docs/DECISION.md) | the decision itself and why, in one page (product doc) |
| [research/BOUNDARIES.md](research/BOUNDARIES.md) | why boundaries come from a learned detector on video features, not from the VLM |
| [reports/VALIDATION_REPORT.md](reports/VALIDATION_REPORT.md) · [reports/POOL_REPORT.md](reports/POOL_REPORT.md) | the validation pool: sources, granularity, what the team should expect from the metric |
| [HISTORY.md](HISTORY.md) | the pre-detector history: how the pipeline got here, naming experiments, everything tried and rejected |

## Results, in the order they were obtained

| file | measured |
| --- | --- |
| [results/2026-09-05-decoders-base.md](results/2026-09-05-decoders-base.md) | peak picking vs penalised DP over dumped probabilities: decoding is saturated at 4 Hz |
| [results/2026-09-05-cross-embodiment.md](results/2026-09-05-cross-embodiment.md) | human vs robot corpora: step-length regime dominates, specialists beat mixtures |
| [results/2026-09-05-stages-variants.md](results/2026-09-05-stages-variants.md) | 2 vs 4 stages, σ, motion channels, seed ensembles, on six labelled sets |
| [results/2026-09-05-ranking85.md](results/2026-09-05-ranking85.md) | all sixteen checkpoints ranked on the 85 atomic clips, best decoder each |
| [results/2026-09-05-boundary-final.md](results/2026-09-05-boundary-final.md) | corpus size, mixed-regime models, ensembles, feature resolution, product metrics |

Headline numbers (step-F1 at IoU 0.5, one-to-one, no labels):

| set | single model in `main` before | shipped ensemble |
| --- | --- | --- |
| 85 atomic clips (Assembly101 fine, sessions unseen in training) | 0.422 | **0.498** |
| EPIC-100, mid granularity (16 clips) | 0.413 | **0.535** |
| LIBERO robot hold-out (10 episodes) | 0.096 | **0.733** |

## Reading and sources

- [research/2026-09-04-boundary-sota.md](research/2026-09-04-boundary-sota.md) — what current papers on action segmentation and generic boundary detection actually report, and what carries over to a class-free setting.
- [research/READING.md](research/READING.md) — annotated reading list; the papers themselves are in [research/papers/](research/papers/).

## Plans

- [plans/2026-09-04-boundary-detector-design.md](plans/2026-09-04-boundary-detector-design.md) — the design that was approved before the experiments.
- [plans/2026-09-04-boundary-detector-plan.md](plans/2026-09-04-boundary-detector-plan.md) — the task-by-task implementation plan that was executed.

## Scripts

Reproduction lives in two places. The measurement tools are product scripts in
[`../scripts/`](../scripts/): `dump_scores.py` (probabilities of a served checkpoint on a
labelled set), `decode_sweep.py` (best decoder offline), `score_table.py` (models × sets
under one decoder), `ensemble_scores.py` (average dumps = the served ensemble),
`boundary_report.py` (product metrics), `eval_variants.py` (through the application
pipeline), `train_boundaries.py` and `pack_ensemble.py`.

The orchestration that ran on the GPU box is here:

- [dl5/](dl5/) — the experiment queue (`run_stages.sh`, `run_clean.sh`, `rank85_dl5.sh`), its supervisor and repair scripts, and `deploy_final.sh`; see [dl5/README.md](dl5/README.md).
- [scripts/](scripts/) — laptop-side drivers: `run_variants.sh`, `rank85.sh`, `compare_encoders.sh`, `compare_features.sh`.

Data builders (`make_mix.py`, `make_robot_devset.py`, `export_libero.py`,
`make_atomic_devset.py`) are in `../scripts/` as well; the corpora they built are not in
the repository (Assembly101 is gated, LIBERO is large).
