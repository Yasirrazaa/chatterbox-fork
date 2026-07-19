import runpod
import torch
import torchaudio
import nltk
import os
import time
import base64
import tempfile
import logging
from typing import Optional, Dict, Any, List
import traceback

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from chatterbox_serverless.inference import ChatterboxInference
    from chatterbox_serverless import text_preprocessing, validation as _validation
except ImportError as e:
    print(f"Warning: Could not import chatterbox_serverless: {e}")
    ChatterboxInference = None
    text_preprocessing = None
    _validation = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model caching
inference_model = None
device = None


def decode_audio_prompt_base64(audio_base64: str) -> Optional[str]:
    """Decode base64 audio data and save to temporary file."""
    try:
        audio_data = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        logger.info(f"Audio prompt saved to: {tmp_path}")
        return tmp_path
    except Exception as e:
        logger.error(f"Failed to decode audio prompt: {e}")
        return None


def ensure_nltk_data():
    """Ensure NLTK punkt tokenizer is available."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)


def initialize_model(
    model_type: str = "multilingual",
    language: str = "en",
    normalize_text: bool = True,
    sentence_split: bool = True,
    inter_sentence_silence_ms: int = 100,
    use_fast: bool = False,
    compile_dtype: str | None = None,
    repo_id: str | None = None,
):
    """Initialize the ChatterboxInference model. Called once when worker starts.

    Supports all three variants (base / multilingual / turbo). If ``use_fast`` is
    set, attempts ``model.compile()`` (rsxdalv-style torch.compile fast path);
    this is a safe no-op on the base pip package and transparently falls back.
    """
    global inference_model, device

    if inference_model is not None:
        logger.info("Model already initialized")
        return inference_model

    if ChatterboxInference is None:
        raise RuntimeError("ChatterboxInference module not available")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"CUDA available - using GPU: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA not available - using CPU (slower)")

    try:
        if device == "cpu":
            old_load = torch.load
            def new_load_on_cpu(*args, **kwargs):
                kwargs['map_location'] = 'cpu'
                return old_load(*args, **kwargs)
            torch.load = new_load_on_cpu

        inference_model = ChatterboxInference.from_pretrained(
            model_type=model_type,
            language=language,
            device=device,
            repo_id=repo_id,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
            inter_sentence_silence_ms=inter_sentence_silence_ms,
        )

        if device == "cpu":
            torch.load = old_load

        compiled = False
        if use_fast:
            if compile_dtype is None and device == "cuda":
                compile_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
            
            # Use 560 max_cache_len for optimal speed as benchmarked
            compiled = inference_model.compile(dtype=compile_dtype, max_cache_len=560)
            if not compiled:
                logger.warning("use_fast requested but model lacks compile hook; "
                               "generation will run at standard speed.")
            else:
                logger.info(f"Fast path compiled (dtype={compile_dtype or 'default'}, max_cache_len=560).")

        logger.info(f"Model loaded on {device} | sr={inference_model.sr} | "
                     f"variant={type(inference_model.model).__name__} | "
                     f"fast={'yes' if compiled else 'no'}")
        ensure_nltk_data()
        return inference_model

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        logger.error(traceback.format_exc())
        if device == "cpu":
            logger.info("Retrying model loading with explicit CPU mapping...")
            try:
                with torch.no_grad():
                    torch.set_default_tensor_type('torch.FloatTensor')
                    inference_model = ChatterboxInference.from_pretrained(
                        model_type=model_type, language=language, device=device,
                        repo_id=repo_id, normalize_text=normalize_text,
                        sentence_split=sentence_split,
                        inter_sentence_silence_ms=inter_sentence_silence_ms,
                    )
                if use_fast:
                    inference_model.compile(dtype=compile_dtype)
                logger.info(f"Model loaded on {device} (retry)")
                ensure_nltk_data()
                return inference_model
            except Exception as e2:
                logger.error(f"Retry also failed: {e2}")
        raise


def text_to_speech_pipeline(
    text: str,
    model: ChatterboxInference,
    language_id: str | None = None,
    inter_sentence_silence_ms: int | None = None,
    audio_prompt_path: Optional[str] = None,
    use_fast: bool = False,
    preprocess_text: bool = False,
    denoise: bool = False,
    normalize: bool = False,
    normalize_mode: str = "ebr",
    num_candidates: int = 1,
    **generation_kwargs,
) -> Optional[torch.Tensor]:
    """Generate audio via the ChatterboxInference wrapper.

    Pipeline:
      1. (optional) text preprocessing (sound-word removal, dot-letter fix, ...)
      2. generation (standard or compiled fast path) with normalize + split
      3. (optional) denoise / FFmpeg normalization
    """
    try:
        if preprocess_text and text_preprocessing is not None:
            text = text_preprocessing.preprocess_text(text)
            logger.info(f"Preprocessed text -> {text[:80]}...")

        if use_fast and hasattr(model.model, 'generate_fast'):
            logger.info("Using fast (compiled) generation")
            audio_tensor = model.generate_fast(
                text, language_id=language_id,
                inter_sentence_silence_ms=inter_sentence_silence_ms,
                audio_prompt_path=audio_prompt_path, num_candidates=num_candidates, **generation_kwargs,
            )
        else:
            audio_tensor = model.generate(
                text, language_id=language_id,
                inter_sentence_silence_ms=inter_sentence_silence_ms,
                audio_prompt_path=audio_prompt_path, num_candidates=num_candidates, **generation_kwargs,
            )

        if denoise or normalize:
            audio_tensor = model.postprocess(
                audio_tensor, denoise=denoise, normalize=normalize,
                normalize_mode=normalize_mode,
            )

        wav = audio_tensor.cpu().float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim > 2:
            wav = wav.squeeze()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
        return wav
    except Exception as e:
        logger.error(f"Error in text-to-speech pipeline: {e}")
        logger.error(traceback.format_exc())
        return None


def audio_to_base64(audio_tensor: torch.Tensor, sample_rate: int,
                    output_format: str = "wav") -> str:
    """Encode a (1, N) waveform tensor as base64 in the requested format."""
    fmt = output_format.lower()
    suffix = f".{fmt}" if fmt != "wav" else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        if fmt in ("wav", "flac", "mp3"):
            torchaudio.save(tmp.name, audio_tensor, sample_rate)
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")
        tmp_path = tmp.name
    with open(tmp_path, 'rb') as f:
        data = f.read()
    os.unlink(tmp_path)
    return base64.b64encode(data).decode('utf-8')


def validate_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize input parameters."""
    if 'text' not in job_input:
        raise ValueError("Missing required parameter: 'text'")
    text = job_input['text'].strip()
    if not text:
        raise ValueError("Text parameter cannot be empty")

    v = {
        'text': text,
        'audio_prompt_base64': job_input.get('audio_prompt_base64'),
        'language_id': job_input.get('language_id'),
        'inter_sentence_silence_ms': job_input.get('inter_sentence_silence_ms', 100),
        'temperature': job_input.get('temperature', 0.8),
        'cfg_weight': job_input.get('cfg_weight', 0.5),
        'exaggeration': job_input.get('exaggeration', 0.5),
        'repetition_penalty': job_input.get('repetition_penalty', 1.2),
        'min_p': job_input.get('min_p', 0.05),
        'top_p': job_input.get('top_p', 1.0),
        'use_fast': job_input.get('use_fast', False),
        'preprocess_text': job_input.get('preprocess_text', False),
        'denoise': job_input.get('denoise', False),
        'normalize': job_input.get('normalize', False),
        'normalize_mode': job_input.get('normalize_mode', 'ebr'),
        'validate': job_input.get('validate', False),
        'num_candidates': job_input.get('num_candidates', 1),
        'output_format': job_input.get('output_format', 'wav'),
    }

    ranges = {
        'temperature': (0.0, 2.0), 'cfg_weight': (0.0, 1.0),
        'exaggeration': (0.0, 1.0), 'repetition_penalty': (0.0, 2.0),
        'min_p': (0.0, 1.0), 'top_p': (0.0, 1.0),
    }
    for k, (lo, hi) in ranges.items():
        if not (lo <= v[k] <= hi):
            raise ValueError(f"{k} must be between {lo} and {hi}")
    if v['inter_sentence_silence_ms'] < 0:
        raise ValueError("inter_sentence_silence_ms must be non-negative")
    if v['output_format'] not in ('wav', 'mp3', 'flac'):
        raise ValueError("output_format must be wav, mp3, or flac")
    return v


def handler(job):
    """RunPod serverless handler. Also callable directly for local testing."""
    start_time = time.time()
    try:
        job_input_raw = job.get('input', {})
        model_config = job_input_raw.get('model_config', {})

        model = initialize_model(
            model_type=model_config.get('model_type', 'multilingual'),
            language=model_config.get('language', 'en'),
            normalize_text=model_config.get('normalize_text', True),
            sentence_split=model_config.get('sentence_split', True),
            inter_sentence_silence_ms=model_config.get('inter_sentence_silence_ms', 100),
            use_fast=model_config.get('use_fast', False),
            compile_dtype=model_config.get('compile_dtype'),
            repo_id=model_config.get('repo_id'),
        )

        generation_input = {k: v for k, v in job_input_raw.items() if k != 'model_config'}
        v = validate_input(generation_input)

        audio_prompt_path = None
        if v['audio_prompt_base64']:
            audio_prompt_path = decode_audio_prompt_base64(v['audio_prompt_base64'])
            if audio_prompt_path is None:
                logger.warning("Failed to decode audio prompt, proceeding without it")

        gen_kwargs = {
            'temperature': v['temperature'], 'cfg_weight': v['cfg_weight'],
            'exaggeration': v['exaggeration'], 'repetition_penalty': v['repetition_penalty'],
            'min_p': v['min_p'], 'top_p': v['top_p'],
        }

        logger.info("Starting generation...")
        audio_tensor = text_to_speech_pipeline(
            text=v['text'], model=model, language_id=v['language_id'],
            inter_sentence_silence_ms=v['inter_sentence_silence_ms'],
            audio_prompt_path=audio_prompt_path, use_fast=v['use_fast'],
            preprocess_text=v['preprocess_text'], denoise=v['denoise'],
            normalize=v['normalize'], normalize_mode=v['normalize_mode'],
            num_candidates=v['num_candidates'],
            **gen_kwargs,
        )
        if audio_tensor is None:
            raise RuntimeError("Failed to generate audio")

        audio_base64 = audio_to_base64(audio_tensor, model.sr, v['output_format'])

        duration_seconds = audio_tensor.shape[1] / model.sr
        processing_time = time.time() - start_time
        realtime_factor = round(duration_seconds / processing_time, 4) if processing_time > 0 else None

        metadata = {
            "duration_seconds": round(duration_seconds, 2),
            "sample_rate": model.sr,
            "processing_time_seconds": round(processing_time, 2),
            "realtime_factor": realtime_factor,
            "text_length": len(v['text']),
            "audio_shape": list(audio_tensor.shape),
            "model_type": type(model.model).__name__,
            "model_variant": model_config.get('model_type', 'multilingual'),
            "fast_mode": v['use_fast'] and getattr(model, '_compiled', False),
            "preprocess_text": v['preprocess_text'],
            "denoise": v['denoise'],
            "normalize": v['normalize'],
            "output_format": v['output_format'],
        }

        # Optional Whisper validation
        if v['validate'] and _validation is not None:
            import numpy as np
            arr = audio_tensor.squeeze(0).numpy().astype(np.float32)
            val = _validation.validate_audio(arr, model.sr, v['text'],
                                             language=v['language_id'])
            metadata['validation'] = val
            logger.info(f"Validation: {val}")

        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.unlink(audio_prompt_path)

        logger.info(f"=== Handler done in {processing_time:.2f}s (RTF={realtime_factor}) ===")
        return {"status": "success", "audio_base64": audio_base64, "metadata": metadata}

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Handler failed after {processing_time:.2f}s: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e),
                "metadata": {"processing_time_seconds": round(processing_time, 2)}}


if __name__ == '__main__':
    logger.info("Starting ChatterboxTTS Serverless Worker...")
    try:
        initialize_model()
        runpod.serverless.start({'handler': handler, 'return_aggregate_stream': False})
    except Exception as e:
        logger.error(f"Failed to start worker: {e}")
        logger.error(traceback.format_exc())
        raise
