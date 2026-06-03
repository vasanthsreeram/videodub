#!/bin/bash
# GPU verification wrapper - runs check_tts_gpu.py inside qwen-dub conda env
# Usage: docker exec videodub /app/videodub/scripts/check_gpu.sh [--load-model]

set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen-dub

exec python /app/videodub/scripts/check_tts_gpu.py "$@"
