# Demo Videos and Benchmark Results

This directory contains demo outputs from the videodub pipeline and benchmark comparisons against other open-source video dubbing pipelines.

## Demo Outputs

### English to Chinese Dubbing

| Demo | Input | Output | Duration | Size |
|------|-------|--------|----------|------|
| [demo_en_to_zh_preview.mp4](demo_en_to_zh_preview.mp4) | 44s English | 44s Chinese | 15s preview | 1.5 MB |

*The preview is a 15-second, 720p compressed clip. Full-resolution outputs (~55MB) are available as GitHub Release assets.*

## Benchmark Results (2026-06-03)

### Test Configuration

- **GPU**: NVIDIA RTX A6000 (48GB VRAM)
- **Input**: 44.12s English video (1440x2560, portrait)
- **Direction**: English -> Chinese
- **Pipeline**: Qwen3-ASR + DeepSeek translation + Qwen3-TTS (x-vector voice clone) + LatentSync lip-sync
- **Settings**: QWEN_TTS_X_VECTOR_ONLY=1, LATENTSYNC_STEPS=20

### Timing Breakdown

| Stage | Time (seconds) | Notes |
|-------|---------------|-------|
| Audio extraction | 0.3s | FFmpeg |
| ASR transcription | 25.5s | Qwen3-ASR with FlashAttention |
| Translation | 3.6s | DeepSeek API |
| TTS synthesis | 162.9s | Qwen3-TTS voice clone |
| LatentSync inference + face restore | 779.9s | 20 steps, guidance 1.5 |
| **Total E2E** | **973.2s** | 16.2 minutes for 44s video |

### Quality Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Output duration | 44.16s | Match input | **PASS** |
| TTS raw duration | 40.96s | <53s (within 20%) | **PASS** |
| TTS fit ratio | 0.93x | <1.20x | **PASS** |
| ASR detected language | Chinese | Chinese | **PASS** |
| Language purity | 96.9% | >95% | **PASS** |
| English leakage | 6 words | <10 ideal | **PASS** |

### Analysis

1. **TTS Duration**: The Chinese translation spoken by Qwen TTS (40.96s) is now **shorter** than the English source (44.12s). This is optimal - the audio plays at natural speed without time compression.

2. **Language Purity**: ASR verification detected 190 Chinese characters and only 6 English words (proper nouns like "Facebook", "Meta", "AI"). This represents 96.9% language purity - a major improvement from the previous 70% with significant English leakage.

3. **Fixes Applied**:
   - `x_vector_only=True`: Prevents source language (English) ref_text from being passed to TTS
   - Explicit audio remux after LatentSync: Ensures intended Chinese TTS audio is used
   - Duration-aware translation prompt: DeepSeek produces concise translations

## Comparison with Other Pipelines

See [benchmark-existing-pipelines.md](../benchmark-existing-pipelines.md) for detailed comparison against:
- pyvideotrans (subtitle-level TTS, most mature)
- video-dubbing-translator (sentence-level TTS + XTTS)
- KrillinAI, SoniTranslate, Linly-Dubbing

### Key Differentiators

| Feature | videodub (ours) | pyvideotrans | kadirb4rut |
|---------|-----------------|--------------|------------|
| TTS Approach | Single-blob | Per-subtitle | Per-sentence |
| Voice Clone | Qwen3-TTS x-vector | Configurable (32 backends) | XTTS |
| Lip Sync | LatentSync | Optional | LatentSync (opt) |
| Containerization | Single Docker | Desktop app | Scripts |
| Language Purity | 97% verified | Not measured | Not measured |

## How to Run Your Own Benchmark

```bash
# Inside Docker container
docker exec -it videodub bash

# Activate qwen-dub environment
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen-dub

# Run benchmark with ASR verification
python /app/videodub/scripts/run_benchmark.py \
  --input /path/to/video.mp4 \
  --target Chinese \
  --verify \
  --output-json benchmark_results.json
```

## Output Files

The benchmark generates these files in the job directory (`/app/runs/<timestamp>/`):

| File | Description |
|------|-------------|
| `dubbed_video.mp4` | Final output with lip-sync |
| `metadata.json` | Full metadata including timing, transcripts, ASR verification |
| `run.log` | Detailed pipeline execution log |
| `source_16k.wav` | Extracted source audio (16kHz mono) |
| `reference_15s.wav` | 15-second reference for voice cloning |
| `translated_raw.wav` | Raw TTS output before time-fitting |
| `translated_fit.wav` | Time-fitted TTS audio |
| `output_16k.wav` | Extracted audio from final video (for ASR verification) |

## File Size Notes

GitHub has a 100MB file size limit for individual files. For full-resolution demo videos:
- Compressed preview clips (<25MB) are included directly in the repo
- Full outputs (~55MB) can be linked from GitHub Releases

To create a compressed preview:
```bash
ffmpeg -i full_video.mp4 -ss 0 -t 15 -c:v libx264 -crf 28 -preset fast \
  -vf "scale=720:-2" -c:a aac -b:a 128k -movflags +faststart preview.mp4
```
