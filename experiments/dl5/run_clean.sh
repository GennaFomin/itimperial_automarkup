#!/usr/bin/env bash
# Режим кейса: чистая валидация (шаги ≥ 1.5 с). Меряет существующие варианты при двух
# минимальных длинах шага и двух порогах на чистой валидации, роботном держ-ауте и
# старом пуле; чистые обучающие варианты добавляет, только если корпус ≥ 30 роликов.
set -u
cd "$(dirname "$0")"
export PATH="$HOME/bin:$PATH" PYTHONPATH="$PWD"
export PRAXIS_FEATURES=video PRAXIS_VIDEO_BASE_URL=http://127.0.0.1:8102
export PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104 PRAXIS_NAMER=none
export PRAXIS_VOCAB="$PWD/data/vocab_atomic.yaml" PRAXIS_WORK_DIR="$HOME/praxis/work"
export PRAXIS_IDLE_RATIO=0 PRAXIS_TAS_MOTION=1 PRAXIS_TAS_DIFF=1 PRAXIS_VIDEO_MODEL=timesformer
PY="$HOME/praxis/venv/bin/python"
VAL="$HOME/praxis/praxis-pool/validation"
until grep -aq "ранжирование готово" "$HOME/praxis/logs/rank85.log" 2>/dev/null; do sleep 120; done
until curl -s --max-time 3 http://127.0.0.1:8104/health | grep -q '"ready":true'; do sleep 5; done
echo "### $(date +%H:%M) чистый режим: val_clean $(ls data/val_clean/gt | wc -l), train_clean $(ls data/train_clean/gt | wc -l) ###"
if [ "$(ls data/train_clean/gt | wc -l)" -ge 30 ]; then
  $PY scripts/make_mix.py --out data/clean_both --validation "$VAL" --source human=data/train_clean --source robot=data/robo_libero | tail -1
  $PY scripts/train_boundaries.py --train data/clean_both --epochs 60 --tolerance 2 --augment --name clean-both --no-activate
  VARIANTS=base,motiondiff,augment,clean-both,augment+clean-both
else
  VARIANTS=base,motion,motiondiff,augment,sigma3,augment+sigma3,emb-human,emb-robot,emb-human+emb-robot
fi
for minseg in 1.0 1.5; do for thr in 0.35 0.5; do
  echo "### $(date +%H:%M) валидация: min_seg $minseg, порог $thr ###"
  PRAXIS_MIN_SEGMENT_SEC=$minseg $PY scripts/eval_variants.py --variants "$VARIANTS" \
      --sets clean=data/val_clean robot=data/robo_holdout old="$VAL" --threshold "$thr" --out "experiments/results/clean-val-m$minseg-t$thr"
done; done
echo "### $(date +%H:%M) чистый режим готов ###"
