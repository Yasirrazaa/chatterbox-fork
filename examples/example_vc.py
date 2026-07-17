import torch
import os
import sys

# Ensure the src/ directory is in the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torchaudio as ta

from chatterbox.vc import ChatterboxVC

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

AUDIO_PATH = "YOUR_FILE.wav"
TARGET_VOICE_PATH = "YOUR_FILE.wav"

model = ChatterboxVC.from_pretrained(device)
wav = model.generate(
    audio=AUDIO_PATH,
    target_voice_path=TARGET_VOICE_PATH,
)
ta.save("testvc.wav", wav, model.sr)
