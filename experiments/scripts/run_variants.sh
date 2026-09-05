#!/usr/bin/env bash
# Все варианты детектора на одном корпусе, без подмены живой модели.
# Порог подбирается на отложенной десятой части обучающего корпуса (DEV), не на валидации.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH" PYTHONPATH="$PWD"
export PRAXIS_FEATURES=video PRAXIS_VIDEO_BASE_URL=http://127.0.0.1:8102
export PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104 PRAXIS_NAMER=none
export PRAXIS_VOCAB="$PWD/data/vocab_atomic.yaml" PRAXIS_WORK_DIR="$HOME/praxis/work"
export PRAXIS_MIN_SEGMENT_SEC=0.5 PRAXIS_IDLE_RATIO=0
PY="$HOME/praxis/venv/bin/python"
CORPUS=${CORPUS:-data/train_mix}
DEV=${DEV:-data/train_mix_dev}
VAL=${VAL:-$HOME/praxis/praxis-pool/validation}
ROBOT=${ROBOT:-}

train() {  # имя, MOTION, DIFF, доп. аргументы
  local name=$1 motion=$2 diff=$3; shift 3
  echo "### $(date +%H:%M) обучение $name (motion=$motion diff=$diff $*) ###"
  PRAXIS_TAS_MOTION=$motion PRAXIS_TAS_DIFF=$diff \
    $PY scripts/train_boundaries.py --train "$CORPUS" --epochs 60 --name "$name" --no-activate "$@"
}

train nomotion   0 0 --tolerance 2
train motion     1 0 --tolerance 2
train motiondiff 1 1 --tolerance 2
train augment    1 1 --tolerance 2 --augment
train sigma1     1 1 --tolerance 1
train sigma3     1 1 --tolerance 3

SETS="human=$VAL"
[ -n "$ROBOT" ] && SETS="$SETS robot=$ROBOT"
for thr in 0.35 0.5 0.7; do
  echo "### $(date +%H:%M) DEV, порог $thr ###"
  $PY scripts/eval_variants.py --variants base,nomotion,motion,motiondiff,augment,sigma1,sigma3 \
      --sets dev="$DEV" --threshold "$thr" --out experiments/results/dev
  echo "### $(date +%H:%M) валидация, порог $thr ###"
  $PY scripts/eval_variants.py --variants base,nomotion,motion,motiondiff,augment,sigma1,sigma3 \
      --sets $SETS --threshold "$thr" --out "experiments/results/thr$thr"
  echo "### $(date +%H:%M) валидация с TTA, порог $thr ###"
  PRAXIS_TAS_TTA=1 $PY scripts/eval_variants.py --variants motiondiff,augment,sigma3 \
      --sets $SETS --threshold "$thr" --out "experiments/results/tta$thr"
done
echo "### $(date +%H:%M) готово ###"
