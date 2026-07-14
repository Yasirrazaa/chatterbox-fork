"""Optional FFmpeg loudness normalization.

Port of Chatterbox-TTS-Extended's FFmpeg normalization step. Applies EBU R128
or peak normalization to the generated audio. Requires ``ffmpeg`` on PATH.

If ffmpeg is unavailable, ``normalize_audio`` returns the input unchanged and
logs a warning.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import os
import numpy as np

logger = logging.getLogger(__name__)


def normalize_audio(
    wav_np: np.ndarray,
    sr: int,
    mode: str = "ebr",
    target_lufs: float = -23.0,
    peak: float = -1.0,
) -> np.ndarray:
    """Normalize a mono float32 waveform using ffmpeg.

    Args:
        wav_np: Mono float32 waveform in [-1, 1].
        sr: Sample rate.
        mode: "ebr" (EBU R128) or "peak".
        target_lufs: Target integrated loudness for EBU R128.
        peak: Target true peak (dBFS) for EBU R128, or peak ceiling for "peak".

    Returns:
        Normalized mono float32 waveform (or original if ffmpeg missing).
    """
    if wav_np.size == 0:
        return wav_np
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        logger.warning("ffmpeg not found on PATH; skipping normalization.")
        return wav_np

    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
            sf.write(fin.name, wav_np, sr)
            in_path = fin.name
        out_path = in_path.replace(".wav", ".norm.wav")

        if mode == "peak":
            filter_str = f"loudnorm=I={target_lufs}:TP={peak}:LRA=7:linear=true"
        else:
            filter_str = f"loudnorm=I={target_lufs}:TP={peak}:LRA=11"

        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-af", filter_str,
            "-ar", str(sr), out_path,
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=120)
        if res.returncode != 0 or not os.path.exists(out_path):
            logger.warning("ffmpeg normalization failed; returning original.")
            return wav_np
        data, _ = sf.read(out_path)
        os.unlink(in_path)
        os.unlink(out_path)
        return data.astype(np.float32)
    except Exception as e:
        logger.warning(f"Normalization error ({e}); returning original.")
        return wav_np
