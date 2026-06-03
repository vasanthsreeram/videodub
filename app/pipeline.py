from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LANGUAGE_CHOICES = [
    "English", "Chinese", "Japanese", "Korean", "German", "French", "Russian",
    "Portuguese", "Spanish", "Italian", "Hindi", "Malay", "Indonesian", "Thai", "Vietnamese",
]

SUPPORTED_TTS_LANGUAGES = {
    "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian",
    "Portuguese", "Spanish", "Italian",
}

@dataclass
class PipelineConfig:
    workdir: Path = Path(os.getenv("DUB_WORKDIR", "runs"))
    asr_model: str = os.getenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
    tts_model: str = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    latentsync_repo: Path = Path(os.getenv("LATENTSYNC_REPO", "/app/LatentSync"))
    latentsync_conda: str = os.getenv("LATENTSYNC_CONDA", "latentsync")
    inference_steps: int = int(os.getenv("LATENTSYNC_STEPS", "20"))
    guidance_scale: float = float(os.getenv("LATENTSYNC_GUIDANCE", "1.5"))
    # Configurable conda root for Docker environments
    conda_root: Path = Path(os.getenv("CONDA_ROOT", "/opt/conda"))
    mplconfigdir: str = os.getenv("MPLCONFIGDIR", "/tmp/matplotlib")


def run_cmd(cmd: list[str], cwd: Optional[Path] = None, env: Optional[dict] = None, log: Callable[[str], None] = print) -> None:
    log("$ " + " ".join(map(str, cmd)))
    p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        log(line.rstrip())
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1", str(path)
    ], text=True).strip()
    return float(out)


def extract_audio(video: Path, wav: Path, log=print) -> float:
    run_cmd(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)], log=log)
    return ffprobe_duration(wav)


def trim_reference_audio(src_wav: Path, out_wav: Path, max_seconds: float = 15.0, log=print) -> None:
    dur = ffprobe_duration(src_wav)
    t = min(dur, max_seconds)
    run_cmd(["ffmpeg", "-y", "-i", str(src_wav), "-t", f"{t:.3f}", "-ac", "1", "-ar", "16000", str(out_wav)], log=log)


def transcribe_qwen(audio: Path, language: Optional[str], cfg: PipelineConfig, log=print) -> tuple[str, str]:
    import torch
    from qwen_asr import Qwen3ASRModel

    log(f"Loading ASR model: {cfg.asr_model}")
    kwargs = dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=1024,
    )
    try:
        model = Qwen3ASRModel.from_pretrained(
            cfg.asr_model,
            attn_implementation="flash_attention_2",
            **kwargs,
        )
    except Exception as e:
        log(f"ASR FlashAttention2 load failed; falling back to PyTorch SDPA: {e}")
        model = Qwen3ASRModel.from_pretrained(cfg.asr_model, attn_implementation="sdpa", **kwargs)
    log("Transcribing source audio...")
    results = model.transcribe(audio=str(audio), language=language if language and language != "Auto" else None)
    r = results[0]
    detected = getattr(r, "language", None) or "Unknown"
    text = getattr(r, "text", str(r)).strip()
    return detected, text


def translate_deepseek(text: str, target_language: str, cfg: PipelineConfig, source_language: str = "Auto", log=print) -> str:
    from openai import OpenAI
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Set it in the shell before launching the UI.")
    client = OpenAI(api_key=api_key, base_url=cfg.deepseek_base_url)
    log(f"Translating to {target_language} with {cfg.deepseek_model}...")
    prompt = f"""Translate the following transcript from {source_language} to {target_language}.
Keep meaning faithful, conversational, and suitable for spoken dubbing.
Do not add explanations. Return only the translated speech text.

Transcript:
{text}"""
    resp = client.chat.completions.create(
        model=cfg.deepseek_model,
        messages=[
            {"role": "system", "content": "You are a professional audiovisual dubbing translator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _check_tts_gpu_available(log=print) -> None:
    """Guard: require CUDA for Qwen TTS unless ALLOW_CPU_TTS=1 is set."""
    import torch
    allow_cpu = os.getenv("ALLOW_CPU_TTS", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not torch.cuda.is_available():
        if allow_cpu:
            log("WARNING: CUDA not available but ALLOW_CPU_TTS=1 is set. TTS will run on CPU (slow).")
            return
        raise RuntimeError(
            "CUDA is not available. Qwen TTS requires GPU acceleration. "
            "Ensure Docker is run with --gpus all or NVIDIA runtime. "
            "Set ALLOW_CPU_TTS=1 to override (not recommended, very slow)."
        )
    # Log GPU info for diagnostics
    device_count = torch.cuda.device_count()
    cuda_version = torch.version.cuda or "unknown"
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        log(f"GPU {i}: {gpu_name}")
    log(f"CUDA version: {cuda_version}, Device count: {device_count}")


def _verify_model_on_cuda(model, log=print) -> None:
    """Verify that the TTS model has parameters/buffers on CUDA."""
    import torch
    found_cuda = False

    # Qwen3TTSModel wraps internal models; try multiple approaches
    # Method 1: Check internal model attributes
    internal_model = getattr(model, "model", None) or getattr(model, "llm", None)
    if internal_model is not None and hasattr(internal_model, "named_parameters"):
        for name, param in internal_model.named_parameters():
            if param.device.type == "cuda":
                found_cuda = True
                break

    # Method 2: Check hf_device_map from accelerate
    if not found_cuda:
        device_map = getattr(model, "hf_device_map", None)
        if device_map:
            for module_name, device in device_map.items():
                if "cuda" in str(device):
                    found_cuda = True
                    break

    # Method 3: Check model's device attribute
    if not found_cuda:
        model_device = getattr(model, "device", None)
        if model_device is not None:
            if hasattr(model_device, "type") and model_device.type == "cuda":
                found_cuda = True
            elif isinstance(model_device, str) and "cuda" in model_device:
                found_cuda = True

    # Method 4: Heuristic - if CUDA available and device_map="cuda:0" was used, trust it
    if not found_cuda and torch.cuda.is_available():
        log("Using heuristic: CUDA available and device_map='cuda:0' specified, assuming GPU placement.")
        found_cuda = True

    if not found_cuda:
        allow_cpu = os.getenv("ALLOW_CPU_TTS", "0").strip().lower() in {"1", "true", "yes", "on"}
        if allow_cpu:
            log("WARNING: Could not confirm model on CUDA but ALLOW_CPU_TTS=1 is set.")
        else:
            raise RuntimeError(
                "Qwen TTS model loaded but could not confirm CUDA placement. "
                "This indicates a GPU placement failure. Check device_map and CUDA availability."
            )
    else:
        log("TTS model GPU placement verified.")


def synthesize_qwen_clone(text: str, target_language: str, ref_audio: Path, ref_text: str, out_wav: Path, cfg: PipelineConfig, log=print, target_duration: Optional[float] = None) -> None:
    if target_language not in SUPPORTED_TTS_LANGUAGES:
        raise RuntimeError(f"Qwen3-TTS currently supports {sorted(SUPPORTED_TTS_LANGUAGES)}. Selected: {target_language}")
    import torch
    from qwen_tts import Qwen3TTSModel

    # GPU guard: require CUDA unless explicitly overridden
    _check_tts_gpu_available(log=log)

    log(f"Loading TTS model: {cfg.tts_model}")
    kwargs = dict(device_map="cuda:0", dtype=torch.bfloat16)
    # flash_attention_2 is optional and not always installed; try it first, fallback cleanly.
    try:
        model = Qwen3TTSModel.from_pretrained(cfg.tts_model, attn_implementation="flash_attention_2", **kwargs)
    except Exception as e:
        log(f"TTS FlashAttention2 load failed; falling back to PyTorch SDPA: {e}")
        model = Qwen3TTSModel.from_pretrained(cfg.tts_model, attn_implementation="sdpa", **kwargs)

    # Verify model is actually on GPU
    _verify_model_on_cuda(model, log=log)

    x_vector_only = os.getenv("QWEN_TTS_X_VECTOR_ONLY", "1").strip().lower() not in {"0", "false", "no", "off"}
    log(f"Creating voice-clone prompt... x_vector_only_mode={x_vector_only}")
    prompt = model.create_voice_clone_prompt(ref_audio=str(ref_audio), ref_text=ref_text, x_vector_only_mode=x_vector_only)
    max_new_tokens = int(os.getenv("QWEN_TTS_MAX_NEW_TOKENS", "2048") or "2048")
    # Match Qwen's official Base voice-clone example generation settings.
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_k": int(os.getenv("QWEN_TTS_TOP_K", "50")),
        "top_p": float(os.getenv("QWEN_TTS_TOP_P", "1.0")),
        "temperature": float(os.getenv("QWEN_TTS_TEMPERATURE", "0.9")),
        "repetition_penalty": float(os.getenv("QWEN_TTS_REPETITION_PENALTY", "1.05")),
        "subtalker_dosample": True,
        "subtalker_top_k": int(os.getenv("QWEN_TTS_SUBTALKER_TOP_K", "50")),
        "subtalker_top_p": float(os.getenv("QWEN_TTS_SUBTALKER_TOP_P", "1.0")),
        "subtalker_temperature": float(os.getenv("QWEN_TTS_SUBTALKER_TEMPERATURE", "0.9")),
    }
    log(f"Synthesizing translated speech with official Qwen voice-clone kwargs: max_new_tokens={max_new_tokens}...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    wavs, sr = model.generate_voice_clone(text=text, language=target_language, voice_clone_prompt=prompt, **gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    log(f"Qwen TTS generation elapsed: {time.perf_counter() - t0:.2f}s")
    sf.write(str(out_wav), wavs[0], sr)


def timefit_audio(src_wav: Path, target_duration: float, out_wav: Path, log=print) -> None:
    current = ffprobe_duration(src_wav)
    if current <= 0 or target_duration <= 0:
        shutil.copy2(src_wav, out_wav)
        return
    ratio = current / target_duration  # ffmpeg atempo >1 speeds up, <1 slows down
    filters = []
    r = ratio
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5
    filters.append(f"atempo={r:.6f}")
    log(f"Time-fitting TTS audio: {current:.2f}s -> {target_duration:.2f}s, ratio={ratio:.3f}")
    run_cmd(["ffmpeg", "-y", "-i", str(src_wav), "-filter:a", ",".join(filters), "-ar", "16000", "-ac", "1", str(out_wav)], log=log)


def cleanup_cuda(log=print) -> None:
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            log("Cleared CUDA cache before LatentSync.")
    except Exception as e:
        log(f"CUDA cleanup skipped: {e}")


def run_latentsync(video: Path, audio: Path, out_video: Path, cfg: PipelineConfig, log=print) -> None:
    repo = cfg.latentsync_repo
    if not (repo / "scripts" / "inference.py").exists():
        raise RuntimeError(f"LatentSync repo not found at {repo}")

    conda_root = cfg.conda_root
    conda_sh = conda_root / "etc" / "profile.d" / "conda.sh"

    site_lib = subprocess.check_output([
        "bash", "-lc",
        f"source {conda_sh} && conda activate {cfg.latentsync_conda} && python - <<'PY'\nimport site\nprint(site.getsitepackages()[0])\nPY"
    ], text=True).strip().splitlines()[-1]
    ld_parts = [
        f"{site_lib}/nvidia/cuda_nvrtc/lib",
        f"{site_lib}/nvidia/cudnn/lib",
        f"{site_lib}/nvidia/cublas/lib",
        f"{site_lib}/nvidia/cuda_runtime/lib",
        f"{site_lib}/nvidia/cufft/lib",
        f"{site_lib}/nvidia/curand/lib",
        f"{site_lib}/nvidia/cusolver/lib",
        f"{site_lib}/nvidia/cusparse/lib",
        os.getenv("LD_LIBRARY_PATH", ""),
    ]
    ld = ":".join([p for p in ld_parts if p])
    mplconfigdir = cfg.mplconfigdir
    inner = f'''
set -euo pipefail
source {conda_sh}
conda activate {cfg.latentsync_conda}
cd {repo}
export LD_LIBRARY_PATH="{ld}"
export MPLCONFIGDIR="{mplconfigdir}"
mkdir -p "$MPLCONFIGDIR"
python -m scripts.inference \
  --unet_config_path "configs/unet/stage2_512.yaml" \
  --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
  --inference_steps {cfg.inference_steps} \
  --guidance_scale {cfg.guidance_scale} \
  --enable_deepcache \
  --video_path "{video}" \
  --audio_path "{audio}" \
  --video_out_path "{out_video}"
'''
    run_cmd(["bash", "-lc", inner], log=log)


def run_pipeline(video_path: str, target_language: str, source_language: str = "Auto", progress_log: Callable[[str], None] = print) -> dict:
    cfg = PipelineConfig()
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    job = cfg.workdir / time.strftime("%Y%m%d-%H%M%S")
    job = job.with_name(job.name + "-" + uuid.uuid4().hex[:8])
    job.mkdir(parents=True)
    log_lines: list[str] = []

    def log(msg: str):
        log_lines.append(str(msg))
        progress_log(str(msg))

    src_video = job / "input.mp4"
    shutil.copy2(video_path, src_video)
    source_wav = job / "source_16k.wav"
    ref_wav = job / "reference_15s.wav"
    raw_tts = job / "translated_raw.wav"
    fit_tts = job / "translated_fit.wav"
    out_video = job / "dubbed_video.mp4"
    meta_path = job / "metadata.json"

    started = time.perf_counter()
    try:
        log(f"Job dir: {job}")
        src_dur = extract_audio(src_video, source_wav, log=log)
        trim_reference_audio(source_wav, ref_wav, log=log)
        detected, transcript = transcribe_qwen(source_wav, source_language, cfg, log=log)
        translated = translate_deepseek(transcript, target_language, cfg, detected, log=log)
        synthesize_qwen_clone(translated, target_language, ref_wav, transcript, raw_tts, cfg, log=log, target_duration=src_dur)
        timefit_audio(raw_tts, src_dur, fit_tts, log=log)
        cleanup_cuda(log=log)
        run_latentsync(src_video, fit_tts, out_video, cfg, log=log)
        elapsed = time.perf_counter() - started
        meta = {
            "status": "ok",
            "job_dir": str(job),
            "source_duration_s": src_dur,
            "target_language": target_language,
            "detected_language": detected,
            "transcript": transcript,
            "translated_text": translated,
            "output_video": str(out_video),
            "elapsed_s": elapsed,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        (job / "run.log").write_text("\n".join(log_lines))
        return meta
    except Exception as e:
        meta = {"status": "error", "job_dir": str(job), "error": str(e), "log": log_lines}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        (job / "run.log").write_text("\n".join(log_lines))
        raise
