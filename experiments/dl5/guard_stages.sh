#!/usr/bin/env bash
cd "$HOME/praxis/app"
tp=$(pgrep -f "train_boundarie[s].py" | head -1)
if [ -n "$tp" ]; then pp=$(ps -o ppid= -p "$tp" | tr -d ' '); echo "обучение pid $tp, родитель $pp: $(ps -o args= -p "$pp" 2>/dev/null | cut -c1-60)"; fi
cp run_stages.new run_stages.sh; chmod +x run_stages.sh; bash -n run_stages.sh || exit 1
if pgrep -f "run_stage[s].sh" >/dev/null; then echo "run_stages жив — не трогаю"; exit 0; fi
# родителя нет: дождаться текущего обучения и запустить заново (готовое не переучится)
(while pgrep -f "train_boundarie[s].py" >/dev/null; do sleep 30; done; setsid nohup ./run_stages.sh > "$HOME/praxis/logs/run_stages.log" 2>&1 < /dev/null &) > /dev/null 2>&1 &
echo "ожидание конца обучения и перезапуск run_stages поставлены"
