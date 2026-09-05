#!/usr/bin/env bash
# Контроль архитектуры: базовая модель двухстадийная, варианты — четырёхстадийные на
# другом корпусе. Учим двухстадийные на том же корпусе (σ=1 с движением и разностями,
# движение+разности σ=2, голые признаки σ=1) и ансамбль σ=1 по трём инициализациям.
set -u
cd "$(dirname "$0")"
export PATH="$HOME/bin:$PATH" PYTHONPATH="$PWD"
export PRAXIS_FEATURES=video PRAXIS_VIDEO_BASE_URL=http://127.0.0.1:8102
export PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104 PRAXIS_NAMER=none
export PRAXIS_VOCAB="$PWD/data/vocab_atomic.yaml" PRAXIS_WORK_DIR="$HOME/praxis/work"
export PRAXIS_IDLE_RATIO=0 PRAXIS_TAS_MOTION=1 PRAXIS_TAS_DIFF=1 PRAXIS_MIN_SEGMENT_SEC=1.0 PRAXIS_VIDEO_MODEL=timesformer
PY="$HOME/praxis/venv/bin/python"
VAL="$HOME/praxis/praxis-pool/validation"
until curl -s --max-time 3 http://127.0.0.1:8104/health | grep -q '"ready":true'; do sleep 5; done
until curl -s --max-time 3 http://127.0.0.1:8102/health | grep -q timesformer; do sleep 10; done
echo "### $(date +%H:%M) двухстадийные варианты ###"
[ -f "$HOME/praxis/checkpoints/boundary-s2-sigma1.pt" ] || $PY scripts/train_boundaries.py --train data/train_mix --epochs 60 --tolerance 1 --stages 2 --name s2-sigma1 --no-activate
[ -f "$HOME/praxis/checkpoints/boundary-s2-motiondiff.pt" ] || $PY scripts/train_boundaries.py --train data/train_mix --epochs 60 --tolerance 2 --stages 2 --name s2-motiondiff --no-activate
[ -f "$HOME/praxis/checkpoints/boundary-s2-plain-sigma1.pt" ] || PRAXIS_TAS_MOTION=0 PRAXIS_TAS_DIFF=0 $PY scripts/train_boundaries.py --train data/train_mix --epochs 60 --tolerance 1 --stages 2 --name s2-plain-sigma1 --no-activate
echo "### $(date +%H:%M) ансамбль по инициализациям ###"
[ -f "$HOME/praxis/checkpoints/boundary-s2-sigma1-b.pt" ] || $PY scripts/train_boundaries.py --train data/train_mix --epochs 60 --tolerance 1 --stages 2 --name s2-sigma1-b --no-activate
[ -f "$HOME/praxis/checkpoints/boundary-s2-sigma1-c.pt" ] || $PY scripts/train_boundaries.py --train data/train_mix --epochs 60 --tolerance 1 --stages 2 --name s2-sigma1-c --no-activate
echo "### $(date +%H:%M) оценка ###"
$PY scripts/eval_variants.py \
    --variants base,sigma1,s2-sigma1,s2-motiondiff,s2-plain-sigma1,s2-sigma1+s2-sigma1-b+s2-sigma1-c,sigma1+s2-sigma1,base+s2-plain-sigma1 \
    --sets old="$VAL" clean=data/val_clean robot=data/robo_holdout --threshold 0.5 --out experiments/results/stages-prom0
echo "### $(date +%H:%M) контроль архитектуры готов ###"
