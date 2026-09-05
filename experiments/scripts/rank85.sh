#!/usr/bin/env bash
# Ночью, когда очередь на DL5 закончится: снять кривые всех вариантов на 85 атомарных
# роликах (они только на ноутбуке) и отранжировать по лучшему декодеру. Туннель
# поднимается заново при обрыве. В конце живая модель возвращается на базовую.
set -u
cd /mnt/data/praxis
PY=~/.venvs/praxis/bin/python
export PRAXIS_NAMER=none PRAXIS_FEATURES=video PRAXIS_VIDEO_BASE_URL=http://127.0.0.1:8102 PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104 PRAXIS_VOCAB=data/pool_atomic/vocab_atomic.yaml PRAXIS_VIDEO_MODEL=timesformer
tunnel() {
  curl -s --max-time 3 http://127.0.0.1:8104/health | grep -q ready && return 0
  ps -eo pid,args --no-headers | grep -F 'ssh -N -L 8100' | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null; sleep 1
  setsid nohup ssh -N -L 8100:127.0.0.1:8100 -L 8102:127.0.0.1:8102 -L 8104:127.0.0.1:8104 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes dl5 >/dev/null 2>&1 < /dev/null &
  for i in $(seq 1 15); do curl -s --max-time 3 http://127.0.0.1:8104/health | grep -q ready && return 0; sleep 2; done; return 1
}
until timeout 20 ssh -o ConnectTimeout=8 -n dl5 'grep -aq "контроль архитектуры готов" ~/praxis/logs/run_stages.log' 2>/dev/null; do sleep 600; done
tunnel || { echo "туннель не поднялся"; exit 1; }
OUT=experiments/results/2026-09-05-ranking85.md
echo "# Ранжирование вариантов на 85 атомарных роликах (лучший декодер каждому)" > $OUT
echo "" >> $OUT; echo "| вариант | F1@0.5 | Δшагов | декодер |" >> $OUT; echo "| --- | --- | --- | --- |" >> $OUT
rank() {  # имя MOTION DIFF
  tunnel || return 1
  curl -s --max-time 60 -X POST http://127.0.0.1:8104/load -H 'Content-Type: application/json' -d "{\"name\":\"$1\"}" | grep -q loaded || { echo "| $1 | — | — | нет чекпоинта |" >> $OUT; return 0; }
  PRAXIS_TAS_MOTION=$2 PRAXIS_TAS_DIFF=$3 $PY scripts/dump_scores.py --clips data/pool_atomic/clips --gt data/pool_atomic/gt --out work/scores/$1_atomic85.npz >/dev/null 2>&1 || { echo "| $1 | — | — | ошибка снятия |" >> $OUT; return 0; }
  line=$($PY scripts/decode_sweep.py work/scores/$1_atomic85.npz 2>/dev/null | sed -n '/лучшие/{n;p}')
  echo "| $1 | $(echo $line | awk '{print $1}') | $(echo $line | awk '{print $2}') | $(echo $line | cut -d' ' -f3-) |" >> $OUT
}
rank "" 0 0; sed -i 's/^|  |/| base |/' $OUT
rank nomotion 0 0; rank motion 1 0; rank motiondiff 1 1; rank augment 1 1; rank sigma1 1 1; rank sigma3 1 1; rank ms4-416 0 0
rank emb-human 1 1; rank emb-robot 1 1; rank emb-both 1 1
rank s2-sigma1 1 1; rank s2-motiondiff 1 1; rank s2-plain-sigma1 0 0; rank s2-sigma1-b 1 1; rank s2-sigma1-c 1 1
curl -s --max-time 60 -X POST http://127.0.0.1:8104/load -H 'Content-Type: application/json' -d '{"name":""}' >/dev/null
echo "готово: $OUT"; cat $OUT
