# Benchmark: videodub vs Existing Video Dubbing Pipelines

This document compares our `videodub` pipeline against established open-source video dubbing projects found via web search.

## Overview

| Project | Architecture | TTS Approach | Lip Sync | Segment Handling | Translation | Maturity |
|---------|-------------|--------------|----------|------------------|-------------|----------|
| **videodub** (ours) | Qwen3-ASR + DeepSeek + Qwen3-TTS + LatentSync | Single-blob voice clone | LatentSync (CUDA) | Full transcript | DeepSeek LLM | Early |
| **pyvideotrans** | Modular ASR + LLM + TTS | Per-subtitle segment | Optional | Subtitle-level | Multi-provider LLM | Mature |
| **video-dubbing-translator** | WhisperX + deep-translator + XTTS | Per-sentence segment | LatentSync (opt) | Sentence-level | deep-translator | Mid |
| **KrillinAI** | Orchestrated pipeline | Configurable | TBD | Segment-based | Multi-provider | Mid |
| **SoniTranslate** | Gradio + multi-provider | Multi-engine | Yes | Subtitle-based | Multi-provider | Mature |
| **Linly-Dubbing** | Qwen + CosyVoice/GPT-SoVITS | Chinese-focused | Linly-Talker | Subtitle-based | Qwen | Mid |

## Detailed Comparison

### 1. pyvideotrans (jianchang512/pyvideotrans)

**Repository**: https://github.com/jianchang512/pyvideotrans

**Architecture**:
- Full workflow: ASR -> Subtitle generation -> Translation -> TTS -> Video synthesis
- Modular TTS backends: 32 providers (Edge-TTS, F5-TTS, CosyVoice, GPT-SoVITS, XTTS, commercial APIs)
- Speaker diarization for multi-voice dubbing

**Strengths**:
- **Subtitle-level TTS**: Each subtitle line gets individual TTS generation, enabling precise timing control
- **Duration management**: Dedicated audio-video alignment tools address timing drift
- **LLM translation**: Supports DeepSeek, ChatGPT, Claude, Gemini, Qwen for context-aware translation
- **Highly mature**: Large community, production-tested

**What We Adopt**:
- Duration-aware translation prompting (already implemented)
- Explicit audio remux after lip-sync (implemented)
- Logging of TTS duration ratios and warnings (implemented)

**What We Avoid**:
- Heavy desktop GUI focus (we prefer Docker CLI/web API)
- Complex multi-file configuration

---

### 2. video-dubbing-translator (kadirb4rut/video-dubbing-translator)

**Repository**: https://github.com/kadirb4rut/video-dubbing-translator

**Architecture**:
- WhisperX for word-level alignment
- Coqui XTTS for voice cloning
- MoviePy + FFmpeg for video assembly
- Optional LatentSync lip-sync

**Strengths**:
- **Per-sentence TTS**: Processes sentences individually with unique speaker reference clips per segment
- **Timing precision**: WhisperX word-level timestamps enable accurate segment boundaries
- **Speed adjustment**: FFmpeg `atempo` filter scales each segment to match original timing
- **Speaker reference clips**: Creates segment-specific reference audio for voice cloning consistency

**Key Implementation Details**:
```python
# Per-sentence TTS with unique speaker reference
for i in range(len(sentences)):
    text = translator.translate(sentences[i])
    speaker_wav = f"speaker_clips/{i+1}.wav"  # Segment-specific reference
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=target_lang)

# Per-segment speed adjustment
result = audioclip.duration / videoclip.duration
if result < 0.5:
    result = 0.5
ffmpeg(["-filter:a", f"atempo={result}"])
```

**What We Adopt**:
- Explicit audio remux after LatentSync (implemented)
- Duration warnings for excessive speedup (implemented)
- Per-segment speaker reference concept (potential future enhancement)

**What We Avoid**:
- MoviePy dependency (we use pure FFmpeg)
- Separate vocal remover preprocessing (adds complexity for our use case)

---

### 3. Our videodub Pipeline

**Current Architecture** (as of commit `54678b3`):
```
Input Video -> Extract Audio -> Qwen3-ASR -> DeepSeek Translation ->
Qwen3-TTS (voice clone) -> Time-fit (atempo) -> LatentSync -> Remux -> Output
```

**Strengths**:
- **Single-container Docker**: Fully isolated, reproducible environment
- **State-of-art models**: Qwen3-ASR/TTS (2025) + LatentSync (ByteDance)
- **x-vector-only voice cloning**: Prevents source language leakage in reference prompt
- **Explicit audio remux**: Ensures LatentSync artifacts don't corrupt final audio
- **Duration-aware translation**: Prompts DeepSeek for concise output matching source timing

**Current Limitations**:
- **Single-blob TTS**: Entire transcript processed as one TTS call
  - Causes 40%+ time compression when translated text is longer
  - No per-segment timing control
- **No ASR verification in pipeline**: Manual verification required post-generation
- **LatentSync audio artifacts**: Despite remux, some original audio may leak through

---

## Key Lessons Learned

### From pyvideotrans:
1. **Segment-level is better**: Processing per-subtitle enables fine-grained timing control
2. **Alignment as priority**: Dedicated tools for audio-video sync exist because timing drift is a real problem
3. **Multiple TTS engines**: No single TTS fits all languages/qualities; modularity helps

### From video-dubbing-translator:
1. **Segment-specific speaker references**: Each sentence gets its own reference clip from the original timing window
2. **Speed adjustment per segment**: Apply atempo to each segment, not the entire audio
3. **Minimum speed floor**: Prevent extreme distortion with atempo >= 0.5

### Implications for videodub:
1. Consider segment-level TTS path for better timing (future enhancement)
2. ASR verification should be automated in pipeline
3. Current single-blob + time-fit works but has quality ceiling
4. Duration-aware translation prompt helps but can't fully compensate for segment-level approach

---

## Benchmark Results

### Test Configuration
- **GPU**: NVIDIA RTX A6000 (48GB VRAM)
- **Input**: `ivan_hq_raw_20260603.mp4` (44.12s, 1440x2560, English)
- **Direction**: English -> Chinese
- **Settings**: QWEN_TTS_X_VECTOR_ONLY=1, LATENTSYNC_STEPS=20

### Variant A: Current Single-Blob Pipeline (with fixes)

| Metric | Value |
|--------|-------|
| Total E2E time | ~21 minutes |
| TTS generation | ~4 minutes |
| LatentSync inference | ~9 minutes |
| LatentSync face restore | ~1.5 minutes |
| Output duration | 44.16s |
| Output size | ~55 MB |
| TTS raw duration | ~62s |
| Time-fit ratio | 1.40x (requires 40% speedup) |
| ASR verification | Mixed (205 Chinese chars, 90 English words) |

**Analysis**: The 40% compression ratio indicates the Chinese translation is significantly longer than the English source when spoken. The ASR showing mixed language suggests LatentSync is reconstructing some original audio artifacts.

### Variant B: Segment-Level TTS (Not Yet Implemented)

Segment-level TTS would:
- Process each sentence/subtitle independently
- Apply per-segment speed adjustment
- Use segment-specific speaker reference clips
- Expected benefit: Better timing precision, no extreme compression

**Status**: Requires significant refactoring. Marked as future enhancement.

---

## Fixes Applied (This PR)

Based on analysis of reference implementations and our bug investigation:

### 1. x-vector-only ref_text Fix
**Problem**: In x-vector-only mode, passing English ref_text caused source language leakage.
**Fix**: When `x_vector_only=True`, set `effective_ref_text=None` in `create_voice_clone_prompt()`.
**Status**: Already implemented in pipeline.py

### 2. Explicit Audio Remux
**Problem**: LatentSync output may contain audio artifacts from reconstruction.
**Fix**: Write LatentSync output to temp file, then remux with intended driving audio:
```bash
ffmpeg -i latentsync_raw.mp4 -i translated_fit.wav \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest output.mp4
```
**Status**: Already implemented in `remux_video_with_audio()` and `run_latentsync()`

### 3. Duration-Aware Translation Prompt
**Problem**: Translated text may be much longer than source, causing extreme time compression.
**Fix**: Include source duration hint in DeepSeek prompt:
```
IMPORTANT: The original speech is approximately {duration}s. Your translation
should be concise enough to fit similar speaking duration when spoken aloud.
```
**Status**: Already implemented in `translate_deepseek()`

### 4. Enhanced Metadata Logging
**Problem**: No visibility into TTS duration issues.
**Fix**: Log and store TTS raw duration, fit ratio, and warnings in metadata.json.
**Status**: Already implemented in `timefit_audio()` and `run_pipeline()`

---

## Recommended Future Enhancements

### High Priority
1. **Automated ASR verification**: Add post-pipeline ASR check with language purity metric
2. **Segment-level TTS benchmark**: Implement and compare against single-blob approach

### Medium Priority
3. **Per-segment speaker reference**: Extract reference clips per sentence from original timing
4. **Whisper/WhisperX integration**: Get word-level timestamps for precise segment boundaries
5. **Multiple TTS backend support**: Add XTTS, CosyVoice as alternatives to Qwen

### Low Priority
6. **Speaker diarization**: Multi-speaker dubbing support
7. **Background audio preservation**: Vocal separation and remixing

---

## References

- pyvideotrans: https://github.com/jianchang512/pyvideotrans
- video-dubbing-translator: https://github.com/kadirb4rut/video-dubbing-translator
- KrillinAI: https://github.com/krillinai/KrillinAI
- SoniTranslate: https://github.com/R3gm/SoniTranslate
- Linly-Dubbing: https://github.com/Kedreamix/Linly-Dubbing
- LatentSync: https://github.com/bytedance/LatentSync
- Qwen3-TTS: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
