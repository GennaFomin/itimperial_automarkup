#!/usr/bin/env bash
# Сравнение энкодеров признаков на одном наборе и одном методе нарезки.
#
# Литература называет признаки главным рычагом: EAST (ICCVW 2025) поднял Assembly101 с
# 22 до 32.8 F1@50, поменяв замороженный I3D на дообученный VideoMAEv2, при той же
# голове. Проверяем это на своём наборе — голова у нас одна и та же, меняется только
# энкодер.
#
#   bash scripts/compare_encoders.sh timesformer videomae dinov2 vjepa2
set -u
MODELS=${@:-"timesformer videomae dinov2 vjepa2"}
CLIPS=${CLIPS:-data/pool_val/clips}
GT=${GT:-data/pool_val/gt}
METHOD=${METHOD:-"tsm-kernel:PRAXIS_TSM_PENALTY=8"}
PORT=8102
GPU=${GPU:-4}
PYTHON=~/.venvs/praxis/bin/python

for model in $MODELS; do
    echo "=== $model ==="
    timeout 40 ssh -n dl5 "pgrep -f '[s]erve_video' | xargs -r kill" 2>/dev/null
    sleep 3
    # timeout обязателен: ssh с фоновым запуском не отпускает канал и висит вечно,
    # даже когда удалённый процесс уже отвязан через setsid и nohup.
    timeout 30 ssh -n dl5 "cd ~/praxis && setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
        HF_HOME=\$HOME/praxis/hf ~/praxis/venv/bin/python serve_video.py --port $PORT --model $model \
        > ~/praxis/logs/video-$model.log 2>&1 < /dev/null &" >/dev/null 2>&1

    ready=""
    for _ in $(seq 90); do
        if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q true; then
            ready=1; break
        fi
        sleep 5
    done
    if [ -z "$ready" ]; then
        echo "  не поднялся, см. ~/praxis/logs/video-$model.log"
        continue
    fi

    PRAXIS_VOCAB=data/pool_val/vocab_charades.yaml PRAXIS_NAMER=none PRAXIS_FEATURES=video \
    PRAXIS_VIDEO_BASE_URL="http://127.0.0.1:$PORT" PRAXIS_MIN_SEGMENT_SEC=1.5 PRAXIS_IDLE_RATIO=0.3 \
    $PYTHON scripts/sweep.py --clips "$CLIPS" --gt "$GT" --methods "$METHOD" 2>&1 | tail -3
done
