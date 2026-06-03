#!/usr/bin/env python3
"""Benchmark runner for videodub pipeline.

Usage (inside Docker container with qwen-dub env):
    python scripts/run_benchmark.py --input /path/to/video.mp4 --target Chinese --verify

Outputs benchmark results to stdout as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from pipeline import run_pipeline, PipelineConfig, ffprobe_duration


def main():
    parser = argparse.ArgumentParser(description="Run videodub pipeline benchmark")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--target", default="Chinese", help="Target language (default: Chinese)")
    parser.add_argument("--source", default="Auto", help="Source language (default: Auto)")
    parser.add_argument("--verify", action="store_true", help="Run ASR verification on output")
    parser.add_argument("--output-json", help="Write results to JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"=== videodub Benchmark ===", file=sys.stderr)
    print(f"Input: {args.input}", file=sys.stderr)
    print(f"Direction: {args.source} -> {args.target}", file=sys.stderr)
    print(f"ASR verify: {args.verify}", file=sys.stderr)
    print("", file=sys.stderr)

    input_duration = ffprobe_duration(args.input)
    print(f"Input duration: {input_duration:.2f}s", file=sys.stderr)

    start = time.perf_counter()
    result = run_pipeline(
        video_path=args.input,
        target_language=args.target,
        source_language=args.source,
        verify_output=args.verify,
    )
    wall_time = time.perf_counter() - start

    # Add benchmark metadata
    result["benchmark"] = {
        "input_path": args.input,
        "input_duration_s": input_duration,
        "wall_time_s": wall_time,
        "realtime_factor": wall_time / input_duration if input_duration > 0 else None,
        "verify_enabled": args.verify,
    }

    # Get output file info
    if result.get("output_video") and os.path.exists(result["output_video"]):
        out_path = result["output_video"]
        out_duration = ffprobe_duration(out_path)
        out_size = os.path.getsize(out_path)
        result["output"] = {
            "path": out_path,
            "duration_s": out_duration,
            "size_bytes": out_size,
            "size_mb": out_size / (1024 * 1024),
        }

    # Print summary to stderr
    print("", file=sys.stderr)
    print("=== Benchmark Results ===", file=sys.stderr)
    print(f"Status: {result.get('status')}", file=sys.stderr)
    print(f"Total time: {wall_time:.2f}s", file=sys.stderr)
    print(f"Realtime factor: {result['benchmark']['realtime_factor']:.2f}x", file=sys.stderr)
    if result.get("timing"):
        t = result["timing"]
        print(f"  ASR: {t.get('asr_s', 0):.2f}s", file=sys.stderr)
        print(f"  Translation: {t.get('translation_s', 0):.2f}s", file=sys.stderr)
        print(f"  TTS: {t.get('tts_s', 0):.2f}s", file=sys.stderr)
        print(f"  LatentSync: {t.get('latentsync_s', 0):.2f}s", file=sys.stderr)
    if result.get("tts_fit_ratio"):
        ratio = result["tts_fit_ratio"]
        print(f"TTS fit ratio: {ratio:.2f}x (raw {result.get('tts_raw_duration_s', 0):.2f}s -> {input_duration:.2f}s)", file=sys.stderr)
    if result.get("output"):
        o = result["output"]
        print(f"Output: {o['path']}", file=sys.stderr)
        print(f"  Duration: {o['duration_s']:.2f}s", file=sys.stderr)
        print(f"  Size: {o['size_mb']:.2f} MB", file=sys.stderr)
    if result.get("asr_verification"):
        v = result["asr_verification"]
        print(f"ASR verification:", file=sys.stderr)
        print(f"  Detected: {v.get('detected_language')}", file=sys.stderr)
        print(f"  Purity: {v.get('language_purity', 0):.1%}", file=sys.stderr)
        print(f"  Chinese chars: {v.get('chinese_chars', 0)}", file=sys.stderr)
        print(f"  English words: {v.get('english_words', 0)}", file=sys.stderr)
        if v.get("warning"):
            print(f"  WARNING: {v['warning']}", file=sys.stderr)

    # Output JSON to stdout
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Optionally write to file
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results written to: {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
