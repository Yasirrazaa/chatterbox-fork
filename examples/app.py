import sys, os
import threading
import datetime
import random
import gc

# Ensure the src/ directory is in the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torchaudio as ta
import numpy as np
import gradio as gr

from chatterbox_serverless.inference import ChatterboxInference
from chatterbox.vc import ChatterboxVC

# ─────────────────────────────────────────────
# 1. SHARED INFRASTRUCTURE
# ─────────────────────────────────────────────

# Auto-detect best device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"🚀 Starting Chatterbox Studio on {DEVICE}")

# Lazy model singletons
_models = {}
_model_lock = threading.Lock()

def free_vram():
    """Best-effort VRAM/RAM cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_pipeline(model_type: str):
    """Lazily load the pipeline for a specific model type in a thread-safe way."""
    with _model_lock:
        if model_type not in _models:
            print(f"Loading {model_type} model...")
            if model_type == "vc":
                _models[model_type] = ChatterboxVC.from_pretrained(device=DEVICE)
            else:
                _models[model_type] = ChatterboxInference.from_pretrained(model_type=model_type, device=DEVICE)
        return _models[model_type]

def set_seed(seed: int):
    if seed != 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

# ─────────────────────────────────────────────
# 2. GRADIO UI
# ─────────────────────────────────────────────
with gr.Blocks(title="Chatterbox Studio") as demo:
    gr.Markdown("# 🎧 Chatterbox Studio\nUnified interface for TTS, Voice Cloning, and Voice Conversion.")

    with gr.Tabs():
        
        # --- TAB 1: TURBO TTS ---
        with gr.Tab("⚡ Turbo TTS"):
            gr.Markdown("Fastest generation. Supports event tags like `[chuckle]`, `[sigh]`, `[gasp]`.")
            with gr.Row():
                with gr.Column():
                    turbo_text = gr.Textbox(value="Oh, that's hilarious! [chuckle] Anyway, we have a new model.", label="Text", lines=4)
                    
                    with gr.Row():
                        gr.Button("[chuckle]").click(lambda t: t + " [chuckle]", inputs=[turbo_text], outputs=[turbo_text])
                        gr.Button("[sigh]").click(lambda t: t + " [sigh]", inputs=[turbo_text], outputs=[turbo_text])
                        gr.Button("[gasp]").click(lambda t: t + " [gasp]", inputs=[turbo_text], outputs=[turbo_text])
                        gr.Button("[clears throat]").click(lambda t: t + " [clears throat]", inputs=[turbo_text], outputs=[turbo_text])

                    turbo_ref = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio (Optional)")
                    
                    with gr.Accordion("Advanced Options", open=False):
                        turbo_exagg = gr.Slider(0.0, 2.0, value=0.0, label="Exaggeration (Requires ref audio)")
                        turbo_temp = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
                        turbo_minp = gr.Slider(0.0, 1.0, value=0.05, label="Min-P")
                        turbo_topp = gr.Slider(0.0, 1.0, value=0.95, label="Top-P")
                        turbo_rep = gr.Slider(1.0, 2.0, value=1.2, label="Repetition Penalty")
                        turbo_seed = gr.Number(value=0, label="Random Seed (0 for random)")
                        
                    turbo_btn = gr.Button("Generate Fast", variant="primary")
                    
                with gr.Column():
                    turbo_out = gr.Audio(label="Output")
                    
            def _gen_turbo(text, ref, exagg, temp, minp, topp, rep, seed):
                free_vram()
                pipe = get_pipeline("turbo")
                set_seed(int(seed))
                # Generate using fast path
                wav = pipe.generate_fast(
                    text,
                    audio_prompt_path=ref,
                    exaggeration=exagg,
                    temperature=temp,
                    min_p=minp,
                    top_p=topp,
                    repetition_penalty=rep
                )
                return (pipe.sr, wav.squeeze(0).numpy())

            turbo_btn.click(_gen_turbo, [turbo_text, turbo_ref, turbo_exagg, turbo_temp, turbo_minp, turbo_topp, turbo_rep, turbo_seed], turbo_out)

        # --- TAB 2: STANDARD TTS ---
        with gr.Tab("🎙️ Standard TTS"):
            gr.Markdown("Base English model. Best for high-fidelity voice cloning.")
            with gr.Row():
                with gr.Column():
                    base_text = gr.Textbox(value="Now let's make my mum's favourite. So three mars bars into the pan.", label="Text", lines=4)
                    base_ref = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio (Optional)")
                    
                    with gr.Row():
                        base_exagg = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="Exaggeration")
                        base_cfg = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="CFG / Pace")
                    
                    with gr.Accordion("Advanced Options", open=False):
                        base_temp = gr.Slider(0.05, 5.0, value=0.8, label="Temperature")
                        base_seed = gr.Number(value=0, label="Random Seed")
                        
                    base_btn = gr.Button("Generate", variant="primary")
                    
                with gr.Column():
                    base_out = gr.Audio(label="Output")
                    
            def _gen_base(text, ref, exagg, cfg, temp, seed):
                free_vram()
                pipe = get_pipeline("base")
                set_seed(int(seed))
                wav = pipe.generate(
                    text,
                    audio_prompt_path=ref,
                    exaggeration=exagg,
                    cfg_weight=cfg,
                    temperature=temp,
                )
                return (pipe.sr, wav.squeeze(0).numpy())

            base_btn.click(_gen_base, [base_text, base_ref, base_exagg, base_cfg, base_temp, base_seed], base_out)

        # --- TAB 3: MULTILINGUAL ---
        with gr.Tab("🌍 Multilingual"):
            gr.Markdown("Supports 23 languages with accent preservation.")
            with gr.Row():
                with gr.Column():
                    mtl_lang = gr.Dropdown(
                        choices=["en", "fr", "de", "es", "it", "pt", "pl", "nl", "ru", "zh", "ja", "ko"],
                        value="fr", label="Language"
                    )
                    mtl_text = gr.Textbox(value="Bonjour, je m'appelle Chatterbox.", label="Text", lines=4)
                    mtl_ref = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio (Optional)")
                    
                    with gr.Row():
                        mtl_exagg = gr.Slider(0.0, 2.0, value=0.0, label="Exaggeration")
                        mtl_cfg = gr.Slider(0.0, 1.0, value=0.0, label="CFG (Set to 0 for language transfer)")
                        
                    mtl_btn = gr.Button("Generate", variant="primary")
                    
                with gr.Column():
                    mtl_out = gr.Audio(label="Output")
                    
            def _gen_mtl(lang, text, ref, exagg, cfg):
                free_vram()
                pipe = get_pipeline("multilingual")
                wav = pipe.generate(
                    text,
                    language_id=lang,
                    audio_prompt_path=ref,
                    exaggeration=exagg,
                    cfg_weight=cfg,
                )
                return (pipe.sr, wav.squeeze(0).numpy())

            mtl_btn.click(_gen_mtl, [mtl_lang, mtl_text, mtl_ref, mtl_exagg, mtl_cfg], mtl_out)

        # --- TAB 4: LONG-FORM ---
        with gr.Tab("📄 Long-Form (Pipeline)"):
            gr.Markdown("Auto-chunks long text, supports Whisper validation and batching.")
            with gr.Row():
                with gr.Column():
                    long_model = gr.Radio(["base", "turbo", "multilingual"], value="base", label="Model")
                    long_text = gr.Textbox(label="Long Text Input", lines=6, placeholder="Paste a long article here...")
                    long_ref = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio (Optional)")
                    
                    with gr.Accordion("Pipeline Settings", open=False):
                        long_cands = gr.Slider(1, 5, value=1, step=1, label="Whisper Candidates (Requires faster-whisper)")
                        long_batch = gr.Checkbox(value=True, label="Enable Chunking")
                        
                    long_btn = gr.Button("Generate Full Audio", variant="primary")
                    
                with gr.Column():
                    long_out = gr.Audio(label="Output")
                    
            def _gen_long(mtype, text, ref, cands, batch):
                free_vram()
                pipe = get_pipeline(mtype)
                wav = pipe.generate(
                    text,
                    audio_prompt_path=ref,
                    num_candidates=cands,
                    sentence_split=batch,
                )
                return (pipe.sr, wav.squeeze(0).numpy())

            long_btn.click(_gen_long, [long_model, long_text, long_ref, long_cands, long_batch], long_out)

        # --- TAB 5: VOICE CONVERSION ---
        with gr.Tab("🔄 Voice Conversion"):
            gr.Markdown("Convert an audio file into another voice.")
            with gr.Row():
                with gr.Column():
                    vc_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input Audio (to convert)")
                    vc_target = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target Voice Audio")
                    vc_pitch = gr.Number(value=0, step=0.5, label="Pitch Shift")
                    vc_btn = gr.Button("Run Conversion", variant="primary")
                with gr.Column():
                    vc_out = gr.Audio(label="Output")

            def _gen_vc(inp, target, pitch):
                if not inp or not target:
                    raise gr.Error("Please provide both input and target audio files.")
                free_vram()
                model = get_pipeline("vc")
                wav = model.generate(audio=inp, target_voice_path=target, pitch_shift=pitch)
                return (model.sr, wav.squeeze(0).numpy())

            vc_btn.click(_gen_vc, [vc_input, vc_target, vc_pitch], vc_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
