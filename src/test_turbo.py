import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Try instantiating it
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Assuming models are stored in some default place, wait I don't have the weights
