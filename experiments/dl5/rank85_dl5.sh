#!/usr/bin/env bash
# Ранжирование всех вариантов на 85 атомарных роликах — на DL5, чтобы не зависеть от
# ноутбука. Ждёт контроль архитектуры и полную копию роликов, снимает кривые каждого
# варианта, подбирает лучший декодер, пишет одну таблицу. В конце — живая модель базовая.
set -u
cd "$(dirname "$0")"
export PATH="$HOME/bin:$PATH" PYTHONPATH="$PWD"
export PRAXIS_FEATURES=video PRAXIS_VIDEO_BASE_URL=http://127.0.0.1:8102 PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104
export PRAXIS_NAMER=none PRAXIS_VOCAB="$PWD/data/vocab_atomic.yaml" PRAXIS_WORK_DIR="$HOME/praxis/work" PRAXIS_VIDEO_MODEL=timesformer
PY="$HOME/praxis/venv/bin/python"
until grep -aq "контроль архитектуры готов" "$HOME/praxis/logs/run_stages.log" 2>/dev/null; do sleep 120; done
# Ролики копируются с ноутбука, который могут выключить: ждём либо все 85, либо
# пока число не перестанет расти десять минут — и меряем на том, что доехало.
prev=-1; stable=0
until [ "$(ls data/pool_atomic/clips/*.mp4 2>/dev/null | wc -l)" -ge 85 ]; do
  have=$(ls data/pool_atomic/clips/*.mp4 2>/dev/null | wc -l)
  if [ "$have" -eq "$prev" ]; then stable=$((stable+1)); [ "$stable" -ge 10 ] && break; else stable=0; fi
  prev=$have; sleep 60
done
echo "### роликов для ранжирования: $(ls data/pool_atomic/clips/*.mp4 2>/dev/null | wc -l) ###"
until curl -s --max-time 3 http://127.0.0.1:8102/health | grep -q timesformer; do sleep 10; done
mkdir -p work/scores experiments/results
OUT=experiments/results/2026-09-05-ranking85.md
{ echo "# Ранжирование вариантов на атомарных роликах pool_atomic (лучший декодер каждому)"; echo; echo "| вариант | F1@0.5 | Δшагов | декодер |"; echo "| --- | --- | --- | --- |"; } > "$OUT"
rank() {  # имя MOTION DIFF
  local label=${1:-base}
  curl -s --max-time 60 -X POST http://127.0.0.1:8104/load -H 'Content-Type: application/json' -d "{\"name\":\"$1\"}" | grep -q loaded || { echo "| $label | — | — | нет чекпоинта |" >> "$OUT"; return 0; }
  PRAXIS_TAS_MOTION=$2 PRAXIS_TAS_DIFF=$3 $PY scripts/dump_scores.py --clips data/pool_atomic/clips --gt data/pool_atomic/gt --out "work/scores/${label}_85.npz" >/dev/null 2>&1 || { echo "| $label | — | — | ошибка снятия |" >> "$OUT"; return 0; }
  local line; line=$($PY scripts/decode_sweep.py "work/scores/${label}_85.npz" 2>/dev/null | sed -n '/лучшие/{n;p}')
  echo "| $label | $(echo "$line" | awk '{print $1}') | $(echo "$line" | awk '{print $2}') | $(echo "$line" | cut -d' ' -f3-) |" >> "$OUT"
}
rank "" 0 0
for v in nomotion ms4-416 s2-plain-sigma1; do rank $v 0 0; done
rank motion 1 0
for v in motiondiff augment sigma1 sigma3 emb-human emb-robot emb-both s2-sigma1 s2-motiondiff s2-sigma1-b s2-sigma1-c; do rank $v 1 1; done
curl -s --max-time 60 -X POST http://127.0.0.1:8104/load -H 'Content-Type: application/json' -d '{"name":""}' >/dev/null
echo "### $(date +%H:%M) ранжирование готово ###"; cat "$OUT"
