"""Optional Whisper validation of generated audio.

Port of Chatterbox-TTS-Extended's Whisper sync step. Transcribes the generated
audio and compares it to the input text to flag mismatches (useful for
quality assurance / automated eval).

Requires the optional ``validate`` extra:
    pip install ".[validate]"

If no Whisper backend is available, ``validate_audio`` returns a result dict
with ``status="skipped"``.
"""

from __future__ import annotations

import logging
import re
import difflib
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _normalize_for_compare(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_audio(
    wav_np,
    sr: int,
    expected_text: str,
    backend: str = "faster-whisper",
    model_size: str = "base",
    language: Optional[str] = None,
) -> Dict:
    """Transcribe ``wav_np`` and compare to ``expected_text``.

    Args:
        wav_np: Mono float32 waveform.
        sr: Sample rate.
        expected_text: The text that should have been spoken.
        backend: "faster-whisper" or "whisper".
        model_size: Whisper model size (tiny/base/small/...).
        language: Optional language code for the transcriber.

    Returns:
        Dict with keys: status, transcript, similarity (0-1), wer (word error
        rate), backend.
    """
    transcript = _transcribe(wav_np, sr, backend, model_size, language)
    if transcript is None:
        return {
            "status": "skipped",
            "transcript": None,
            "similarity": None,
            "wer": None,
            "backend": backend,
        }

    ref = _normalize_for_compare(expected_text)
    hyp = _normalize_for_compare(transcript)
    similarity = difflib.SequenceMatcher(None, ref, hyp).ratio()

    ref_words = ref.split()
    hyp_words = hyp.split()
    if ref_words:
        sm = difflib.SequenceMatcher(None, ref_words, hyp_words)
        wer = 1.0 - sm.ratio()
    else:
        wer = 0.0

    return {
        "status": "ok",
        "transcript": transcript,
        "similarity": round(similarity, 4),
        "wer": round(wer, 4),
        "backend": backend,
    }


def _transcribe(wav_np, sr, backend: str, model_size: str, language) -> Optional[str]:
    try:
        import soundfile as sf
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav_np, sr)
            path = f.name
        try:
            if backend == "faster-whisper":
                from faster_whisper import WhisperModel
                model = WhisperModel(model_size, device="cpu")
                segs, _ = model.transcribe(path, language=language, beam_size=5)
                text = " ".join(s.text for s in segs)
            else:
                import whisper
                model = whisper.load_model(model_size)
                result = model.transcribe(path, language=language)
                text = result["text"]
            return text.strip()
        finally:
            os.unlink(path)
    except Exception as e:
        logger.warning(f"Whisper validation unavailable ({e}); skipping.")
        return None
