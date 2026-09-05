#!/usr/bin/env bash
# Супервизор очереди: этап за этапом, по маркерам. Если скрипт этапа не бежит, его
# маркер не появился, а предыдущий этап закончен — перезапускаем. Обучения в скриптах
# идемпотентны (готовый чекпоинт не переучивается), оценки просто повторяются.
cd "$HOME/praxis/app"
L="$HOME/praxis/logs"
stage() {  # скрипт, лог, маркер, [маркер-предыдущего лог] [маркер-предыдущего текст]
  local script=$1 log=$2 marker=$3 plog=${4:-} pmarker=${5:-}
  grep -aq "$marker" "$L/$log" 2>/dev/null && return 0
  [ -n "$plog" ] && ! grep -aq "$pmarker" "$L/$plog" 2>/dev/null && return 0
  pgrep -f "${script%.sh}" >/dev/null && return 0
  pgrep -f "train_boundarie[s].py" >/dev/null && return 0
  echo "$(date +%H:%M) перезапуск $script"
  setsid nohup "./$script" > "$L/$log" 2>&1 < /dev/null &
}
while true; do
  stage run_stages.sh    run_stages.log    "контроль архитектуры готов"
  stage rank85_dl5.sh    rank85.log        "ранжирование готово"   run_stages.log    "контроль архитектуры готов"
  stage run_clean.sh     run_clean.log     "чистый режим готов"    rank85.log        "ранжирование готово"
  stage run_backbones.sh run_backbones.log "A/B энкодеров готов"   run_clean.log     "чистый режим готов"
  stage run_idle.sh      run_idle.log      "пустые окна готовы"    run_backbones.log "A/B энкодеров готов"
  grep -aq "пустые окна готовы" "$L/run_idle.log" 2>/dev/null && { echo "$(date +%H:%M) очередь завершена"; exit 0; }
  sleep 300
done
