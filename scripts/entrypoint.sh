#!/bin/bash
set -e

# Activate qwen-dub environment and run the Gradio app
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen-dub

echo "======================================"
echo "Video Dubbing Pipeline - Starting..."
echo "======================================"
echo "Environment: qwen-dub (Python 3.12)"
echo "LatentSync: ${LATENTSYNC_REPO}"
echo "Work directory: ${DUB_WORKDIR}"
echo "HuggingFace cache: ${HF_HOME}"
echo "======================================"

# Check for required API key
if [ -z "${DEEPSEEK_API_KEY}" ]; then
    echo "WARNING: DEEPSEEK_API_KEY not set. Translation will fail."
    echo "Set it with: -e DEEPSEEK_API_KEY=your_key"
fi

# Run the Gradio app
cd /app/videodub
exec python app/gradio_app.py "$@"
