# Video Dubbing Pipeline Docker Image
# Combines Qwen ASR/TTS (Python 3.12) + LatentSync lip-sync (Python 3.10)
# Requires NVIDIA GPU with ~24GB+ VRAM for full pipeline

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    sox \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda with mamba)
ENV CONDA_ROOT=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_ROOT} && \
    rm /tmp/miniforge.sh && \
    ${CONDA_ROOT}/bin/conda init bash && \
    ${CONDA_ROOT}/bin/conda clean -afy

ENV PATH=${CONDA_ROOT}/bin:$PATH

# Create Python 3.12 environment for Qwen ASR/TTS (main app)
RUN conda create -y -n qwen-dub python=3.12 && \
    conda clean -afy

# Create Python 3.10 environment for LatentSync
RUN conda create -y -n latentsync python=3.10 && \
    conda clean -afy

# Clone LatentSync repository
WORKDIR /app
RUN git clone https://github.com/bytedance/LatentSync.git

# Install LatentSync dependencies in latentsync env
SHELL ["/opt/conda/bin/conda", "run", "-n", "latentsync", "/bin/bash", "-c"]
WORKDIR /app/LatentSync
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flash-attn --no-build-isolation || echo "flash-attn optional"

# Download LatentSync checkpoints from HuggingFace
# The LatentSync repo doesn't include a download script; use huggingface-cli directly.
RUN huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt whisper/tiny.pt --local-dir checkpoints

# Switch back to base shell for qwen-dub setup
SHELL ["/bin/bash", "-c"]

# Install Qwen ASR/TTS dependencies in qwen-dub env.
# Keep UI/runtime dependencies separate so a model package resolver issue cannot
# produce a "successful" image that cannot even start Gradio.
RUN /opt/conda/bin/conda run -n qwen-dub pip install --no-cache-dir \
    torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

RUN /opt/conda/bin/conda run -n qwen-dub pip install --no-cache-dir \
    'gradio>=5.24.0' \
    'openai>=1.0.0' \
    soundfile \
    numpy \
    python-dotenv

# Install qwen-asr/qwen-tts dependencies first, then packages with --no-deps
# to avoid resolver conflicts while ensuring all runtime deps are present.
# These are the key dependencies for the Qwen ASR/TTS models:
RUN /opt/conda/bin/conda run -n qwen-dub pip install --no-cache-dir \
    transformers \
    accelerate \
    einops \
    librosa \
    sox

RUN /opt/conda/bin/conda run -n qwen-dub pip install --no-cache-dir --no-deps \
    qwen-asr \
    qwen-tts

RUN /opt/conda/bin/conda run -n qwen-dub pip install --no-cache-dir flash-attn --no-build-isolation || echo "flash-attn optional"

RUN /opt/conda/bin/conda run -n qwen-dub python - <<'PY'
import gradio, openai, soundfile, numpy
print('Core UI/runtime imports OK')
PY

# Copy application code
WORKDIR /app/videodub
COPY app/ ./app/
COPY requirements.txt ./
COPY scripts/ ./scripts/

# Create directories for runs and caches
RUN mkdir -p /app/runs /app/cache /tmp/matplotlib

# Environment variables
ENV CONDA_ROOT=/opt/conda
ENV LATENTSYNC_REPO=/app/LatentSync
ENV LATENTSYNC_CONDA=latentsync
ENV DUB_WORKDIR=/app/runs
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch

# Gradio port
EXPOSE 7860

# Entrypoint script
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "7860"]
