# Video Dubbing Pipeline

Single-container GPU-powered video dubbing with voice cloning and lip sync.

**Pipeline**: Upload video → ASR transcription → Translation → Voice-cloned TTS → Lip-sync video

**Models used**:
- [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) - Speech recognition
- [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) - Voice-cloned text-to-speech
- [DeepSeek](https://api.deepseek.com) - Translation API
- [LatentSync](https://github.com/bytedance/LatentSync) - Lip-sync video generation

## Requirements

- NVIDIA GPU with 24GB+ VRAM (A6000, A100, L40, etc.)
- Docker with NVIDIA Container Toolkit
- DeepSeek API key for translation

## Quick Start

1. **Set your API key**:
   ```bash
   export DEEPSEEK_API_KEY=your_api_key_here
   ```

2. **Build and run**:
   ```bash
   docker compose up --build
   ```

3. **Open Gradio UI**: http://localhost:7860

4. **Upload a video**, select target language, click "Dub video"

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | (required) | DeepSeek API key for translation |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek API base URL |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model name |
| `QWEN_ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | HuggingFace ASR model |
| `QWEN_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace TTS model |
| `LATENTSYNC_STEPS` | `20` | LatentSync inference steps (lower = faster) |
| `LATENTSYNC_GUIDANCE` | `1.5` | LatentSync guidance scale |
| `QWEN_TTS_MAX_NEW_TOKENS` | `2048` | Max tokens for TTS generation |
| `QWEN_TTS_X_VECTOR_ONLY` | `1` | Use x-vector-only voice cloning (prevents reference text leakage) |
| `ATEMPO_WARN_THRESHOLD` | `1.20` | Log warning if TTS duration exceeds this ratio of source duration |
| `ATEMPO_MAX_THRESHOLD` | `1.50` | Log critical warning if TTS duration exceeds this ratio |

## Supported Languages

**TTS-supported** (full voice clone): Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

**ASR-supported** (transcription): All above plus Hindi, Malay, Indonesian, Thai, Vietnamese

## Volume Mounts

The compose file mounts:
- `videodub_cache` - HuggingFace model cache (persisted)
- `videodub_ckpts` - LatentSync checkpoints (persisted)
- `./runs` - Output runs directory (local)

First run will download models (~10GB total). Subsequent runs use cached models.

## Build Only

```bash
docker build -t videodub:latest .
```

## Run Manually

```bash
docker run --gpus all -p 7860:7860 \
  -e DEEPSEEK_API_KEY=your_key \
  -v videodub_cache:/app/cache \
  -v videodub_ckpts:/app/LatentSync/checkpoints \
  -v $(pwd)/runs:/app/runs \
  --shm-size=8g \
  videodub:latest
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                      │
│  ┌─────────────────────────────────────────────────────┐│
│  │ qwen-dub env (Python 3.12)                          ││
│  │  ├─ Gradio UI (port 7860)                           ││
│  │  ├─ Qwen3-ASR (transcription)                       ││
│  │  ├─ Qwen3-TTS (voice clone synthesis)               ││
│  │  └─ DeepSeek API (translation)                      ││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │ latentsync env (Python 3.10)                        ││
│  │  └─ LatentSync (lip-sync video generation)          ││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │ Volumes                                              ││
│  │  ├─ /app/cache (HuggingFace models)                 ││
│  │  ├─ /app/LatentSync/checkpoints (LatentSync weights)││
│  │  └─ /app/runs (job outputs)                         ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## GPU Verification

The pipeline requires NVIDIA GPU with CUDA. To verify GPU is correctly configured:

```bash
# Check nvidia-smi inside container
docker exec videodub nvidia-smi

# Quick CUDA check via qwen-dub env (no model download)
docker exec videodub /app/videodub/scripts/check_gpu.sh

# Full model load test (downloads model if needed, ~10GB first time)
docker exec videodub /app/videodub/scripts/check_gpu.sh --load-model
```

**Note**: GPU checks must run inside the `qwen-dub` conda environment where PyTorch is installed. The `check_gpu.sh` wrapper handles this automatically. Do not use `docker exec videodub python ...` directly as base Python does not have torch.

**Important Docker flags:**
- Must use `--gpus all` or NVIDIA runtime
- Example: `docker run --gpus all ...`

The TTS will fail loudly at startup if CUDA is not available, unless `ALLOW_CPU_TTS=1` is set (not recommended, very slow).

## Troubleshooting

**TTS fails with "CUDA is not available"**: Docker was not started with GPU support. Use `--gpus all` flag or ensure NVIDIA Container Toolkit is installed.

**Out of VRAM**: Reduce `LATENTSYNC_STEPS` (try 12) or use smaller TTS model.

**Slow first run**: Models are downloading. Subsequent runs use cache.

**Translation fails**: Check `DEEPSEEK_API_KEY` is set correctly.

**Face not detected**: LatentSync requires clear face visibility. Use videos with consistent face shots.

## Demo / Benchmark Outputs

See [docs/demo/README.md](docs/demo/README.md) for demo videos and benchmark results.

### Quick Benchmark Summary (RTX A6000)

| Metric | Value |
|--------|-------|
| Input | 44s English video (1440x2560) |
| Output | 44s Chinese dubbed video |
| Total E2E time | ~15 minutes |
| TTS generation | ~4 minutes |
| LatentSync inference | ~9 minutes |
| Realtime factor | ~20x (15 min for 44s video) |

### Comparison with Other Pipelines

See [docs/benchmark-existing-pipelines.md](docs/benchmark-existing-pipelines.md) for detailed comparison against:
- **pyvideotrans** - Subtitle-level TTS, most mature community project
- **video-dubbing-translator** - Sentence-level TTS with XTTS voice cloning
- **KrillinAI**, **SoniTranslate**, **Linly-Dubbing** - Other notable alternatives

Key differentiators of this pipeline:
- Single Docker container with all dependencies
- Qwen3-TTS (2025) for state-of-art Chinese/multilingual voice cloning
- LatentSync for lip-sync without separate face detection
- x-vector-only mode prevents source language leakage

## Running Benchmarks

```bash
# Inside Docker container
docker exec -it videodub bash
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen-dub

# Run benchmark with ASR verification
python /app/videodub/scripts/run_benchmark.py \
  --input /path/to/video.mp4 \
  --target Chinese \
  --verify \
  --output-json benchmark.json
```

## License

- Qwen models: Apache 2.0
- LatentSync: MIT
- This wrapper: MIT
