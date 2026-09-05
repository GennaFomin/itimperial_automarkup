#!/usr/bin/env bash
# Сравнение признаков для сегментации на одной сетке параметров.
#
# Смысл: подача признаков — единственное место, где мы отличаемся от литературы по
# temporal action segmentation, а сам механизм разбиения стандартный. Значит выбирать
# признаки надо измерением, а не по названию модели.
#
# Каждая модель поднимается по очереди на своей карте, прогоняется одна и та же сетка
# параметров, и берётся лучшая строка. Признаки считаются заново на каждую модель —
# это и есть основная стоимость прогона.
#
#   bash scripts/compare_features.sh vjepa2 videomae timesformer dinov2 swin3d mvit

set -u
MODELS=${@:-"vjepa2 videomae timesformer dinov2 swin3d mvit"}
CLIPS=${CLIPS:-data/devset/clips}
GT=${GT:-data/devset/gt}
PORT=8102
GPU=${GPU:-4}
PYTHON=~/.venvs/praxis/bin/python

printf '%-14s %8s %8s %8s %9s %8s %s\n' модель F1@0.1 F1@0.25 F1@0.5 границы шагов параметры

for model in $MODELS; do
    ssh dl5 "pgrep -f '[s]erve_video' | xargs -r kill" 2>/dev/null
    sleep 2
    ssh dl5 "cd ~/praxis && setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
        HF_HOME=\$HOME/praxis/hf ~/praxis/venv/bin/python serve_video.py --port $PORT --model $model \
        > ~/praxis/logs/video-$model.log 2>&1 < /dev/null &" >/dev/null 2>&1

    ready=""
    for _ in $(seq 60); do
        if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q true; then
            ready=1
            break
        fi
        sleep 5
    done
    if [ -z "$ready" ]; then
        printf '%-14s %s\n' "$model" "не поднялся, см. ~/praxis/logs/video-$model.log"
        continue
    fi

    best=$(PRAXIS_VIDEO_BASE_URL="http://127.0.0.1:$PORT" PRAXIS_FEATURES=video PRAXIS_IDLE_RATIO=0 \
        $PYTHON scripts/tune.py --clips "$CLIPS" --gt "$GT" \
        --penalties 0.1 0.2 0.3 --weights 0.0 0.1 0.2 --minimums 1.5 --components 0 8 \
        --max-segments 6 2>/dev/null | sed -n '4p')

    if [ -z "$best" ]; then
        printf '%-14s %s\n' "$model" "прогон не дал результата"
        continue
    fi
    set -- $best
    printf '%-14s %8s %8s %8s %9s %8s штраф=%s физика=%s сжатие=%s\n' \
        "$model" "$5" "$6" "$7" "$8" "$9" "$1" "$2" "$4"
done

ssh dl5 "pgrep -f '[s]erve_video' | xargs -r kill" 2>/dev/null
