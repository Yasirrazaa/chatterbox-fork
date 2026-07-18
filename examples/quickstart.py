"""
Chatterbox TTS — Quickstart Examples
=====================================
Run any section directly or import as a reference.

Sections:
  1. Basic TTS (base model)
  2. Turbo TTS (fast, with event tags)
  3. Multilingual TTS (23 languages)
  4. Long-Form TTS (pipeline wrapper with chunking)
  5. Voice Cloning (reference audio)
  6. Voice Conversion (VC model)
"""

import sys, os
from pathlib import Path

# Ensure the src/ directory is in the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torchaudio as ta

# Auto-detect best device (works on CUDA, MPS/Mac, and CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. BASIC TTS
# ─────────────────────────────────────────────
from chatterbox.tts import ChatterboxTTS
print("\nLoading Basic TTS...")
base_model = ChatterboxTTS.from_pretrained(device=DEVICE)

text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning."
print("Generating Basic TTS...")
wav = base_model.generate(text, exaggeration=1.5, cfg_weight=0.5)
ta.save(OUTPUT_DIR / "01_base_tts.wav", wav, base_model.sr)
print(f"Saved to {OUTPUT_DIR / '01_base_tts.wav'}")

# ─────────────────────────────────────────────
# 2. TURBO TTS (fastest, use generate_fast())
# ─────────────────────────────────────────────
from chatterbox.tts_turbo import ChatterboxTurboTTS
print("\nLoading Turbo TTS...")
turbo_model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)

# Turbo model supports event tags like [chuckle], [sigh]
turbo_text = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store."
print("Generating Turbo TTS...")
# generate_fast is the optimized path
wav_turbo = turbo_model.generate_fast(turbo_text)
ta.save(OUTPUT_DIR / "02_turbo_tts.wav", wav_turbo, turbo_model.sr)
print(f"Saved to {OUTPUT_DIR / '02_turbo_tts.wav'}")

# ─────────────────────────────────────────────
# 3. MULTILINGUAL TTS
# ─────────────────────────────────────────────
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
print("\nLoading Multilingual TTS...")
mtl_model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)

# Supported languages: en, fr, de, es, it, pt, pl, nl, ru, zh, ja, ko, etc.
mtl_text = "Bonjour, je m'appelle Chatterbox. Je peux parler plusieurs langues!"
print("Generating Multilingual TTS...")
wav_mtl = mtl_model.generate(mtl_text, language="fr")
ta.save(OUTPUT_DIR / "03_multilingual_tts.wav", wav_mtl, mtl_model.sr)
print(f"Saved to {OUTPUT_DIR / '03_multilingual_tts.wav'}")

# ─────────────────────────────────────────────
# 4. LONG-FORM TTS (via pipeline wrapper)
# ─────────────────────────────────────────────
from chatterbox_serverless.inference import ChatterboxInference
print("\nLoading Long-Form Inference Pipeline...")
# The inference wrapper handles chunking, batching, and Whisper validation
pipeline = ChatterboxInference.from_model(base_model, language="en")

long_text = "This is a longer piece of text. " * 10
print("Generating Long-Form TTS (auto-chunked)...")
wav_long = pipeline.generate(long_text)
ta.save(OUTPUT_DIR / "04_long_form_tts.wav", wav_long, pipeline.sr)
print(f"Saved to {OUTPUT_DIR / '04_long_form_tts.wav'}")

# ─────────────────────────────────────────────
# 5. VOICE CLONING (reference audio)
# ─────────────────────────────────────────────
print("\nVoice Cloning...")
# Provide a 5-10s clear audio sample of the target voice
REFERENCE_AUDIO = None  # e.g., "my_voice.wav"

if REFERENCE_AUDIO and Path(REFERENCE_AUDIO).exists():
    print(f"Generating cloned voice using {REFERENCE_AUDIO}...")
    wav_clone = base_model.generate("Hello, I am speaking with your voice.", audio_prompt_path=REFERENCE_AUDIO)
    ta.save(OUTPUT_DIR / "05_voice_cloned.wav", wav_clone, base_model.sr)
    print(f"Saved to {OUTPUT_DIR / '05_voice_cloned.wav'}")
else:
    print("Skipping voice cloning (set REFERENCE_AUDIO to a valid file to test).")

# ─────────────────────────────────────────────
# 6. VOICE CONVERSION
# ─────────────────────────────────────────────
from chatterbox.vc import ChatterboxVC
print("\nVoice Conversion...")
# Convert an existing audio file into a target voice
INPUT_AUDIO = None  # "input.wav"
TARGET_VOICE = None # "target.wav"

if INPUT_AUDIO and TARGET_VOICE and Path(INPUT_AUDIO).exists() and Path(TARGET_VOICE).exists():
    print("Loading VC Model...")
    vc_model = ChatterboxVC.from_pretrained(device=DEVICE)
    wav_vc = vc_model.generate(audio=INPUT_AUDIO, target_voice_path=TARGET_VOICE)
    ta.save(OUTPUT_DIR / "06_voice_conversion.wav", wav_vc, vc_model.sr)
    print(f"Saved to {OUTPUT_DIR / '06_voice_conversion.wav'}")
else:
    print("Skipping Voice Conversion (set INPUT_AUDIO and TARGET_VOICE to test).")

print("\nDone! Check the examples/output/ directory for generated files.")
