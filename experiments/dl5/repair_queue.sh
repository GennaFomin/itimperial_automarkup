#!/usr/bin/env bash
# Диагностика и починка очереди одним заходом: сервис признаков обратно на timesformer,
# устаревшие маркеры убраны, четыре очереди перезапущены в правильной зависимости.
cd "$HOME/praxis/app"
echo "8102 до: $(curl -s --max-time 4 http://127.0.0.1:8102/health)"
echo "backbones лог: $(grep -a '###' $HOME/praxis/logs/run_backbones.log 2>/dev/null | tail -1 | cut -c1-80)"
echo "clean лог маркер: $(grep -ac 'чистый режим готов' $HOME/praxis/logs/run_clean.log 2>/dev/null)"
for j in run_clea run_backbone run_idl run_stage rank85_dl; do pgrep -f "${j}[a-z0-9]*\.sh" | xargs -r kill; done
pgrep -f "eval_variant[s].py" | xargs -r kill; pgrep -f "train_boundarie[s].py" | xargs -r kill; sleep 2
# сервис признаков — только timesformer
if ! curl -s --max-time 4 http://127.0.0.1:8102/health | grep -q timesformer; then
  pgrep -f "serve_vide[o].py" | xargs -r kill; sleep 3
  (cd "$HOME/praxis" && setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HOME=$HOME/praxis/hf "$HOME/praxis/venv/bin/python" serve_video.py --port 8102 --model timesformer > "$HOME/praxis/logs/video.log" 2>&1 < /dev/null &)
  for i in $(seq 1 40); do curl -s --max-time 3 http://127.0.0.1:8102/health | grep -q timesformer && break; sleep 5; done
fi
echo "8102 после: $(curl -s --max-time 4 http://127.0.0.1:8102/health)"
# стереть устаревшие маркеры, чтобы очереди ждали настоящих
for f in run_clean run_backbones run_idle rank85; do : > "$HOME/praxis/logs/$f.log"; done
cp run_clean.new run_clean.sh; chmod +x *.sh
for f in run_stages.sh rank85_dl5.sh run_clean.sh run_backbones.sh run_idle.sh; do bash -n $f || { echo "синтаксис $f"; exit 1; }; done
setsid nohup ./run_stages.sh    > "$HOME/praxis/logs/run_stages.log"    2>&1 < /dev/null &
setsid nohup ./rank85_dl5.sh    > "$HOME/praxis/logs/rank85.log"        2>&1 < /dev/null &
setsid nohup ./run_clean.sh     > "$HOME/praxis/logs/run_clean.log"     2>&1 < /dev/null &
setsid nohup ./run_backbones.sh > "$HOME/praxis/logs/run_backbones.log" 2>&1 < /dev/null &
setsid nohup ./run_idle.sh      > "$HOME/praxis/logs/run_idle.log"      2>&1 < /dev/null &
sleep 15
echo "stages $(pgrep -fc 'run_stage[s].sh') rank85 $(pgrep -fc 'rank85_dl[5]') clean $(pgrep -fc 'run_clea[n].sh') bb $(pgrep -fc 'run_backbone[s]') idle $(pgrep -fc 'run_idl[e]') | обучение $(pgrep -fc 'train_boundarie[s]') | роликов85 $(ls data/pool_atomic/clips 2>/dev/null | wc -l)"
tail -2 "$HOME/praxis/logs/run_stages.log" | cut -c1-90
