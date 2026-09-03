# Papers

Every paper this project actually leaned on, with a note on what was taken from it.
The order is by how much it shaped the current pipeline, not by date.

## The method we ship

Our boundary segmenter (`praxis/pipeline/similarity.py`, `PRAXIS_PIPELINE=tsm-kernel`) is
**penalised kernel change-point detection solved by exact dynamic programming**. There is
no single paper that is "the tsm-kernel paper" — it is a named method assembled from three
sources, and here is exactly which part comes from where.

**`kernel-changepoint-arlot2019.pdf`** — Arlot, Celisse, Harchaoui, *A Kernel Multiple
Change-point Algorithm via Model Selection*, JMLR.
[arXiv:1202.3878](https://arxiv.org/abs/1202.3878)

**The closest primary paper to what we run.** It is the kernel change-point problem with
the number of segments chosen by a penalty term — the same formulation our code solves,
including the penalty-based model selection that lets the method decide the step count on
its own instead of being told it.

**`changepoint-review-truong2020.pdf`** — Truong, Oudre, Vayatis, *Selective review of
offline change point detection methods*, Signal Processing 2020.
[arXiv:1801.00718](https://arxiv.org/abs/1801.00718)

The implementation reference, and the one to read first. Two things in it map directly
onto our code:

- **Cost function 11 (`c_kernel`)** is what `kernel_costs()` computes — the cost of a
  segment expressed through the Gram matrix, so the feature vectors are never mapped
  explicitly;
- **Algorithm 1 (Opt)** is what `penalised_segmentation()` implements — exact dynamic
  programming over segmentations, O(KT²).

This is also the paper behind the `ruptures` library, if you want a reference
implementation to compare against.

**`harchaoui-cappe-2007`** — Harchaoui & Cappé, *Retrospective multiple change-point
estimation with kernels*, IEEE/SP SSP 2007. **Not attached, IEEE paywall.** The origin of
kernel change-point detection; cited as [96] in the review above.

**Perochon & Oudre, *Unsupervised Action Segmentation of Untrimmed Egocentric Videos*,
ICASSP 2023.** [DOI](https://doi.org/10.1109/ICASSP49357.2023.10097216) —
**not attached, IEEE paywall; no author copy on HAL.** This is the paper that applies the
method above to video action segmentation, and it is the closest published validation of
our choice: on EPIC-KITCHENS-55 their unsupervised segmentation reaches f1@50 21.4 against
10.7 for a trained method.

**`pelt-killick2012.pdf`** — Killick, Fearnhead, Eckley, *Optimal detection of changepoints
with a linear computational cost*, JASA 2012.
[arXiv:1101.1438](https://arxiv.org/abs/1101.1438)

The other half of penalty-based selection: how a per-segment penalty picks the number of
change points, and how to do it in linear time. We use the exact O(KT²) version because
our clips are short, but this is the paper to read if the videos get long.

## Why a language model does not place boundaries

**`t-pivot-2024.pdf`** — *Open-Vocabulary Action Localization with Iterative Visual
Prompting*, [arXiv:2408.17422](https://arxiv.org/abs/2408.17422)

The published version of "let the VLM find boundaries by binary search": GPT-4o, window
halved, four iterations. IoU 40.8 on Breakfast against 52.1 for a trained method, and on
fine granularity 52.0 against 49.1 for a uniform split with no model at all. The authors
note that more iterations did not help. We reproduced the same failure independently.

**`numpro-cvpr2025.pdf`** — *Number Prompting for Video Temporal Grounding*, CVPR 2025

The one VLM trick that measurably helped us. Rendering the timestamp into the pixels
instead of writing it next to the image lifts Qwen2-VL-7B on Charades-STA from 7.9 to
38.5 mIoU with no training. Reproduced in our domain: F1@0.5 went 0.541 to 0.567. The
lesson is that the bottleneck is emitting time, not seeing.

## Training-free segmentation

**`uboco-cvpr2022.pdf`** — *UBoCo: Unsupervised Boundary Contrastive Learning*, CVPR 2022.
[arXiv:2111.14799](https://arxiv.org/abs/2111.14799)

Recursive parsing of the temporal self-similarity matrix, F1 0.703 on Kinetics-GEBD with
no trained parameters. Implemented as our `tsm-recursive`; measured and lost to
`tsm-kernel`, kept as a comparison point.

**`gebd-iccv2021.pdf`** — *Generic Event Boundary Detection*, ICCV 2021.
[arXiv:2101.10511](https://arxiv.org/abs/2101.10511)

Defines "find the moment one action ends and another begins" as a task separate from
classification. Important for us because the case may arrive without any class list, and
this shows boundaries can be found without a vocabulary at all.

**`twfinch-cvpr2021.pdf`** — *Temporally-Weighted Hierarchical Clustering for Unsupervised
Action Segmentation*, CVPR 2021. [arXiv:2103.11264](https://arxiv.org/abs/2103.11264)

Clustering frames with a temporal weight. Implemented as `tw-finch`; measured and lost.

## Supervised segmentation — the family we did not ship

**`mstcn-cvpr2019.pdf`** — MS-TCN, CVPR 2019. [arXiv:1903.01945](https://arxiv.org/abs/1903.01945)

The founding multi-stage temporal convolutional network. We took its architecture but
replaced the output: instead of a frame class it predicts the probability that the action
changes, which is the only way to use this family when there is no class list. See
`scripts/serve_tas.py`. Its smoothing loss term is also the standard cure for
over-segmentation.

**`asformer-bmvc2021.pdf`** — ASFormer, BMVC 2021. [arXiv:2110.08568](https://arxiv.org/abs/2110.08568)

The transformer nearly everything after 2021 builds on. Useful for the trick of growing
the attention window together with dilation, and for the finding that positional encoding
hurts on long videos.

**`diffact-iccv2023.pdf`** — DiffAct, ICCV 2023. [arXiv:2303.17959](https://arxiv.org/abs/2303.17959)

Segmentation as reverse diffusion, F1@50 83.7 on 50 Salads. The reference point for what
is achievable *with* training data on the target taxonomy.

## Background

**`tas-survey-tpami2024.pdf`** — Ding, Sener, Yao, *Temporal Action Segmentation: An
Analysis of Modern Techniques*, TPAMI 2024. [arXiv:2210.10352](https://arxiv.org/abs/2210.10352)

Read this first if you are new to the task. It also explains why the benchmarks freeze
their features on purpose: the tables compare heads, not systems.
