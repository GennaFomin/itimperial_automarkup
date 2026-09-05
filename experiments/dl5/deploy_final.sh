#!/usr/bin/env bash
# Выкатить итоговый детектор на DL5 отдельным сервисом, не трогая очередь экспериментов:
# чекпоинт → ~/praxis/checkpoints/boundary-final.pt, детектор на 8108 (GPU 6), приложение
# на 8000 перезапускается с новым адресом детектора; признаки берёт с 8109 (timesformer на
# GPU 7, не трогается очередью экспериментов, которая меняет модель на 8102).
#
#     experiments/dl5/deploy_final.sh checkpoints/boundary.pt [PRAXIS_VIDEO_FPS PRAXIS_VIDEO_STRIDE]
set -eu
CKPT=${1:?чекпоинт}
FPS=${2:-16}
STRIDE=${3:-4}
scp -q -o ConnectTimeout=8 "$CKPT" dl5:~/praxis/checkpoints/boundary-final.pt
scp -q -o ConnectTimeout=8 scripts/serve_tas.py dl5:~/praxis/serve_tas.py
rsync -a --timeout=60 --exclude __pycache__ praxis/ dl5:~/praxis/app/praxis/
ssh -o ConnectTimeout=8 -n dl5 'pgrep -f "serve_tas.py --port 810[8]" | xargs -r kill' || true
sleep 2
ssh -o ConnectTimeout=8 -n dl5 "cd ~/praxis && setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 ~/praxis/venv/bin/python serve_tas.py --port 8108 --checkpoint checkpoints/boundary-final.pt > ~/praxis/logs/tas-final.log 2>&1 < /dev/null & sleep 8; curl -s --max-time 5 http://127.0.0.1:8108/health"
echo
ssh -o ConnectTimeout=8 -n dl5 "cd ~/praxis/app && sed -e 's|PRAXIS_TAS_BASE_URL=http://127.0.0.1:8104|PRAXIS_TAS_BASE_URL=http://127.0.0.1:8108|' -e 's|PRAXIS_VIDEO_BASE_URL=http://127.0.0.1:8102|PRAXIS_VIDEO_BASE_URL=http://127.0.0.1:8109|' -e 's|PRAXIS_MIN_SEGMENT_SEC=0.8|PRAXIS_MIN_SEGMENT_SEC=1.0|' -e 's|^export PRAXIS_TAS_THRESHOLD=0.7|export PRAXIS_TAS_THRESHOLD=0.7 PRAXIS_TAS_MOTION=0 PRAXIS_TAS_DIFF=0|' run.sh > run_final.sh && sed -i 's|^export PRAXIS_FEATURES=video|export PRAXIS_FEATURES=video PRAXIS_VIDEO_FPS=$FPS PRAXIS_VIDEO_STRIDE=$STRIDE|' run_final.sh && chmod +x run_final.sh && grep -n 'TAS_BASE_URL\|MIN_SEGMENT\|VIDEO_FPS' run_final.sh"
ssh -o ConnectTimeout=8 -n dl5 'pgrep -f "uvicorn praxis.api:ap[p]" | xargs -r kill' || true
sleep 3
ssh -o ConnectTimeout=8 -n dl5 'cd ~/praxis/app && setsid nohup ./run_final.sh > ~/praxis/logs/app.log 2>&1 < /dev/null & sleep 6; curl -s --max-time 5 http://127.0.0.1:8000/api/v1/limits'
echo
