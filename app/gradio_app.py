from __future__ import annotations

import argparse
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from pipeline import LANGUAGE_CHOICES, run_pipeline

load_dotenv()


def run_ui(video, target_language, source_language):
    if video is None:
        raise gr.Error("Upload a video first.")
    logs = []

    def log(msg: str):
        logs.append(msg)

    meta = run_pipeline(video, target_language, source_language, progress_log=log)
    summary = (
        f"Status: {meta['status']}\n"
        f"Detected source: {meta.get('detected_language')}\n"
        f"Target: {meta.get('target_language')}\n"
        f"Elapsed: {meta.get('elapsed_s', 0):.1f}s\n"
        f"Job dir: {meta.get('job_dir')}\n\n"
        f"Transcript:\n{meta.get('transcript','')}\n\n"
        f"Translation:\n{meta.get('translated_text','')}\n"
    )
    return meta["output_video"], summary, "\n".join(logs)


def build_app():
    with gr.Blocks(title="Qwen + DeepSeek + LatentSync Dubbing") as demo:
        gr.Markdown("# Video Dubbing UI\nUpload a video, choose target language, get a cloned-voice lipsynced dubbed video.")
        with gr.Row():
            video = gr.Video(label="Input video", sources=["upload"])
            with gr.Column():
                source = gr.Dropdown(["Auto"] + LANGUAGE_CHOICES, value="Auto", label="Source language")
                target = gr.Dropdown(LANGUAGE_CHOICES[:10], value="English", label="Target language")
                run = gr.Button("Dub video", variant="primary")
        output = gr.Video(label="Dubbed output")
        summary = gr.Textbox(label="Summary", lines=12)
        logs = gr.Textbox(label="Logs", lines=20)
        run.click(run_ui, inputs=[video, target, source], outputs=[output, summary, logs])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port, share=args.share)
