"""Optional RNNoise denoising (pyrnnoise).

Port of Chatterbox-TTS-Extended's pyrnnoise integration. Removes most TTS
artifacts. Requires the optional ``denoise`` extra (``pip install ".[denoise]"``).

The function works on a numpy float32 mono array at the model sample rate and
returns a denoised numpy array. If pyrnnoise is unavailable, ``denoise_audio``
returns the input unchanged and logs a warning.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import os
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyrnnoise
    _PYRNNOISE_AVAILABLE = True
except Exception:
    _PYRNNOISE_AVAILABLE = False


def denoise_audio(
    wav_np: np.ndarray,
    sr: int,
    use_cli: bool = True,
) -> np.ndarray:
    """Denoise a mono float32 waveform using RNNoise.

    Args:
        wav_np: Mono float32 waveform in [-1, 1].
        sr: Sample rate (model sr, typically 24000).
        use_cli: If True, prefer the ``denoise`` CLI when available (faster /
            more stable); otherwise use the Python API.

    Returns:
        Denoised mono float32 waveform (or the original array if RNNoise is
        unavailable).
    """
    if not _PYRNNOISE_AVAILABLE:
        logger.warning("pyrnnoise not installed; skipping denoising. "
                       "Install with: pip install '.[denoise]'")
        return wav_np

    if wav_np.size == 0:
        return wav_np

    # RNNoise operates at 48 kHz mono s16 internally; resample for best results.
    target_sr = 48000
    if sr != target_sr:
        try:
            import librosa
            wav_48k = librosa.resample(wav_np, orig_sr=sr, target_sr=target_sr)
        except Exception as e:
            logger.warning(f"Resampling failed ({e}); denoising skipped.")
            return wav_np
    else:
        wav_48k = wav_np

    # Try CLI first (most stable per Extended)
    if use_cli:
        cli = _run_denoise_cli(wav_48k, target_sr)
        if cli is not None:
            return _resample_back(cli, target_sr, sr)

    # Python API fallback
    try:
        from pyrnnoise import RNNoise
        denoiser = RNNoise()
        # RNNoise processes int16; convert
        int16 = np.clip(wav_48k * 32767.0, -32768, 32767).astype(np.int16)
        denoised = np.array([denoiser.filter_frame(frame) for frame in int16], dtype=np.float32) / 32767.0
        return _resample_back(denoised, target_sr, sr)
    except Exception as e:
        logger.warning(f"pyrnnoise Python API failed ({e}); returning original.")
        return wav_np


def _run_denoise_cli(wav_48k: np.ndarray, sr: int) -> np.ndarray | None:
    if subprocess.run(["which", "denoise"], capture_output=True).returncode != 0:
        return None
    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
            sf.write(fin.name, wav_48k, sr)
            in_path = fin.name
        out_path = in_path.replace(".wav", ".denoised.wav")
        res = subprocess.run(["denoise", "-i", in_path, "-o", out_path],
                             capture_output=True, timeout=120)
        if res.returncode != 0 or not os.path.exists(out_path):
            return None
        data, _ = sf.read(out_path)
        os.unlink(in_path)
        os.unlink(out_path)
        return data.astype(np.float32)
    except Exception as e:
        logger.warning(f"denoise CLI failed ({e})")
        return None


def _resample_back(wav: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return wav.astype(np.float32)
    try:
        import librosa
        return librosa.resample(wav, orig_sr=from_sr, target_sr=to_sr).astype(np.float32)
    except Exception:
        return wav.astype(np.float32)
