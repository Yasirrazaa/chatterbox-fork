"""ChatterboxInference: Thin wrapper adding text normalization, sentence splitting,
streaming, and CUDA graph acceleration to Chatterbox TTS models.

Ported from coral_chatterbox/src/chatterbox/inference.py.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import warnings
from pathlib import Path
from typing import AsyncGenerator, Generator, Literal

import torch
from huggingface_hub import snapshot_download

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.utils.normalizer import normalize_text as normalize_text_content
from chatterbox.utils.splitter import chunk_text
from chatterbox.utils.device import resolve_device


ModelType = ChatterboxTTS | ChatterboxMultilingualTTS | ChatterboxTurboTTS


class ChatterboxInference:
    """Thin inference wrapper around existing Chatterbox TTS models.

    Not thread-safe. Speaker conditioning state (conds, CUDA graph cache) is stored
    on the instance. Use one instance per process or worker; do not share across
    threads or concurrent requests.
    """

    # File patterns to download from Hugging Face Hub per model variant
    MODEL_ALLOW_PATTERNS = {
        "base": [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ],
        "multilingual": [
            "ve.pt",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt",
            "Cangjie5_TC.json",
        ],
        "turbo": ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    }

    def __init__(
        self,
        model: ModelType,
        language: str = "en",
        normalize_text: bool = True,
        sentence_split: bool = True,
        inter_sentence_silence_ms: int = 100,
        max_chunk_chars: int = 300,
        min_chunk_chars: int = 20,
    ):
        self.model = model
        self.language = language
        self.normalize_text = normalize_text
        self.sentence_split = sentence_split
        self.inter_sentence_silence_ms = inter_sentence_silence_ms
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.sr = getattr(model, "sr", 24000)
        self._last_audio_prompt_path: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_type: Literal["base", "multilingual", "turbo"] = "multilingual",
        language: str = "en",
        device=None,
        repo_id: str | None = None,
        normalize_text: bool = True,
        sentence_split: bool = True,
        inter_sentence_silence_ms: int = 100,
        max_chunk_chars: int = 300,
        min_chunk_chars: int = 20,
    ) -> "ChatterboxInference":
        """Load a pretrained model from Hugging Face Hub or default repo.

        Args:
            model_type: One of "base", "multilingual", "turbo".
            language: Default language for multilingual model (ISO 639-1).
            device: Target device ("cuda", "mps", "cpu", or None for auto).
            repo_id: Optional custom Hugging Face repo ID (e.g., a finetuned model).
            normalize_text: Enable language-aware number normalization.
            sentence_split: Enable NLTK-based sentence splitting.
            inter_sentence_silence_ms: Silence between sentences in ms.
        """
        if repo_id is not None:
            model = cls._load_model_from_repo(model_type=model_type, repo_id=repo_id, device=device)
        else:
            model = cls._load_default_model(model_type=model_type, device=device)

        return cls(
            model=model,
            language=language,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
            inter_sentence_silence_ms=inter_sentence_silence_ms,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        )

    @classmethod
    def from_local(
        cls,
        ckpt_dir: str | Path,
        model_type: Literal["base", "multilingual", "turbo"] = "multilingual",
        language: str = "en",
        device=None,
        normalize_text: bool = True,
        sentence_split: bool = True,
        inter_sentence_silence_ms: int = 100,
        max_chunk_chars: int = 300,
        min_chunk_chars: int = 20,
    ) -> "ChatterboxInference":
        """Load a model from a local checkpoint directory."""
        model = cls._load_model_from_local(model_type=model_type, ckpt_dir=ckpt_dir, device=device)

        return cls(
            model=model,
            language=language,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
            inter_sentence_silence_ms=inter_sentence_silence_ms,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        )

    @classmethod
    def from_model(
        cls,
        model: ModelType,
        language: str = "en",
        normalize_text: bool = True,
        sentence_split: bool = True,
        inter_sentence_silence_ms: int = 100,
        max_chunk_chars: int = 300,
        min_chunk_chars: int = 20,
    ) -> "ChatterboxInference":
        """Wrap an already-instantiated model."""
        return cls(
            model=model,
            language=language,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
            inter_sentence_silence_ms=inter_sentence_silence_ms,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        )

    @classmethod
    def _load_default_model(cls, model_type: str, device=None) -> ModelType:
        device = resolve_device(device)
        if model_type == "multilingual":
            return ChatterboxMultilingualTTS.from_pretrained(device=device)
        if model_type == "turbo":
            return ChatterboxTurboTTS.from_pretrained(device=device)
        return ChatterboxTTS.from_pretrained(device=device)

    @classmethod
    def _load_model_from_local(cls, model_type: str, ckpt_dir: str | Path, device=None) -> ModelType:
        device = resolve_device(device)
        if model_type == "multilingual":
            return ChatterboxMultilingualTTS.from_local(ckpt_dir, device=device)
        if model_type == "turbo":
            return ChatterboxTurboTTS.from_local(ckpt_dir, device=device)
        return ChatterboxTTS.from_local(ckpt_dir, device=device)

    @classmethod
    def _load_model_from_repo(cls, model_type: str, repo_id: str, device=None) -> ModelType:
        device = resolve_device(device)
        ckpt_dir = snapshot_download(
            repo_id=repo_id,
            token=os.getenv("HF_TOKEN") or True,
            allow_patterns=cls.MODEL_ALLOW_PATTERNS[model_type],
        )
        return cls._load_model_from_local(model_type=model_type, ckpt_dir=ckpt_dir, device=device)

    def compile(
        self,
        backend: str = "cudagraphs",
        dtype: str | None = None,
        max_cache_len: int = 600,
    ) -> bool:
        """Optimize the T3 decoder with torch.compile (rsxdalv-style fast path).

        This activates the ~2-4x speedup described in the rsxdalv Chatterbox fork
        when the underlying model exposes a compilable step target
        (``model.t3._step_compilation_target``). The base ``chatterbox-tts==0.1.7``
        pip package does NOT expose this hook, so this is a safe no-op there and
        ``generate_fast`` transparently falls back to ``generate()``.

        To get the speedup on a compatible model (rsxdalv/coral forks, or a
        locally vendored T3 with ``_step_compilation_target``):

            inference.compile(dtype="bfloat16")
            wav = inference.generate_fast(text, ...)

        Args:
            backend: torch.compile backend (default "cudagraphs").
            dtype: Optional dtype to cast the T3 (and its conditionals) to,
                e.g. "bfloat16" or "float16", to cut memory bandwidth.
            max_cache_len: Max KV cache length for the static cache.

        Returns:
            True if compilation was applied, False if the model lacks the hook
            (fast path unavailable; falls back to standard generation).
        """
        target = getattr(self.model.t3, "_step_compilation_target", None)

        if dtype is not None:
            dt = getattr(torch, dtype, None)
            if dt is None:
                raise ValueError(f"Unknown dtype: {dtype}")
            self.model.t3.to(dtype=dt)
            if hasattr(self.model, "conds") and self.model.conds is not None:
                self.model.conds.t3.to(dtype=dt)

        if target is not None:
            self.model.t3._step_compilation_target = torch.compile(
                target, fullgraph=True, backend=backend
            )
            self._compiled = True
        elif hasattr(self.model, "generate_fast"):
            # Model uses manual bucketed CUDA graphs (e.g. coral approach)
            self._compiled = True
        else:
            warnings.warn(
                "Model does not expose _step_compilation_target or generate_fast; "
                "fast path unavailable. Falling back to standard generation.",
                stacklevel=2,
            )
            self._compiled = False
            return False

        # Some forks read max_cache_len from the model; set if present.
        if hasattr(self.model.t3, "max_cache_len"):
            self.model.t3.max_cache_len = max_cache_len
        return True

    def prepare_conditionals(self, audio_prompt_path: str, **kwargs) -> None:
        """Pre-compute and cache speaker embeddings from a reference audio file.

        Call this once before generate() to avoid re-encoding on every sentence
        when reusing the same voice across multiple calls.

        Accepts the same kwargs as the underlying model's prepare_conditionals()
        (e.g., exaggeration, norm_loudness for turbo).
        """
        valid_params = set(inspect.signature(self.model.prepare_conditionals).parameters) - {"self", "wav_fpath"}
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        self.model.prepare_conditionals(audio_prompt_path, **filtered)
        self._last_audio_prompt_path = audio_prompt_path

    def _prepare_text(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        sentence_split: bool | None = None,
    ) -> list[str]:
        language = language_id or self.language
        use_normalization = self.normalize_text if normalize_text is None else normalize_text
        use_sentence_split = self.sentence_split if sentence_split is None else sentence_split

        processed_text = text
        if use_normalization:
            processed_text = normalize_text_content(processed_text, language=language)

        processed_text = processed_text.strip()
        if not processed_text:
            return []

        if not use_sentence_split:
            return [processed_text]

        return chunk_text(
            processed_text,
            language=language,
            max_chars=self.max_chunk_chars,
            min_chars=self.min_chunk_chars,
        )

    def _silence_chunk(self, inter_sentence_silence_ms: int | None = None) -> torch.Tensor | None:
        silence_ms = self.inter_sentence_silence_ms if inter_sentence_silence_ms is None else inter_sentence_silence_ms
        if silence_ms <= 0:
            return None

        samples = int(self.sr * silence_ms / 1000)
        if samples <= 0:
            return None

        return torch.zeros((1, samples), dtype=torch.float32)

    def postprocess(
        self,
        wav: torch.Tensor,
        denoise: bool = False,
        normalize: bool = False,
        normalize_mode: str = "ebr",
    ) -> torch.Tensor:
        """Apply optional post-processing to a generated waveform tensor.

        Args:
            wav: (1, N) float32 waveform tensor.
            denoise: Apply RNNoise artifact removal (requires ``[denoise]`` extra).
            normalize: Apply FFmpeg loudness normalization (requires ffmpeg on PATH).
            normalize_mode: "ebr" (EBU R128) or "peak".

        Returns:
            Possibly-modified (1, N) waveform tensor.
        """
        if not (denoise or normalize):
            return wav

        import numpy as np
        from . import denoise as _denoise
        from . import normalize_audio as _norm

        arr = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if denoise:
            arr = _denoise.denoise_audio(arr, self.sr)
        if normalize:
            arr = _norm.normalize_audio(arr, self.sr, mode=normalize_mode)
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)

    def generate(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        sentence_split: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        num_candidates: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Generate audio from text with normalization, sentence splitting, and silence.

        Args:
            text: Input text to synthesize.
            language_id: Language code for multilingual model (defaults to self.language).
            normalize_text: Override instance default for number normalization.
            sentence_split: Override instance default for sentence splitting.
            inter_sentence_silence_ms: Override instance default for silence between sentences.
            num_candidates: Number of candidates to generate per sentence (best one is selected).
            **kwargs: Passed to underlying model.generate() (e.g., audio_prompt_path,
                     exaggeration, cfg_weight, temperature, etc.).

        Returns:
            Audio tensor of shape (1, num_samples).
        """
        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
        )
        if not sentences:
            return torch.zeros((1, 0), dtype=torch.float32)

        chunks = []
        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        # Pre-compute speaker embeddings once for the full generate call.
        # Only re-run if the audio prompt path has changed since the last call.
        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate() does not accept: {sorted(dropped)}. "
                f"These kwargs were ignored.",
                stacklevel=2,
            )

        # Phase 1: Generate first candidate for all chunks sequentially
        best_chunks = []
        for sentence in sentences:
            best_chunks.append(self.model.generate(sentence, **filtered_kwargs))

        if num_candidates > 1:
            import concurrent.futures
            import os
            from .validation import validate_audio

            sr = getattr(self.model, "sr", 24000)
            
            def _validate(idx, text, chunk_tensor):
                wav_np = chunk_tensor.squeeze().cpu().numpy()
                res = validate_audio(wav_np, sr, text, backend="faster-whisper", language=language_id)
                return idx, (res.get("wer", float('inf')) if res.get("status") == "ok" else float('inf'))

            # Phase 2: Parallel Whisper validation
            wers = [float('inf')] * len(sentences)
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sentences), os.cpu_count() or 4)) as pool:
                futs = [pool.submit(_validate, i, sentences[i], best_chunks[i]) for i in range(len(sentences))]
                for fut in concurrent.futures.as_completed(futs):
                    idx, wer = fut.result()
                    wers[idx] = wer

            # Phase 3: Retry Queue
            for i, sentence in enumerate(sentences):
                best_wer = wers[i]
                if best_wer <= getattr(self, "wer_threshold", 0.0):
                    continue
                
                for _ in range(1, num_candidates):
                    chunk = self.model.generate(sentence, **filtered_kwargs)
                    _, wer = _validate(i, sentence, chunk)
                    if wer < best_wer:
                        best_wer = wer
                        best_chunks[i] = chunk
                    if best_wer <= getattr(self, "wer_threshold", 0.0):
                        break

        chunks = []
        for index, chunk in enumerate(best_chunks):
            chunks.append(chunk)
            if silence is not None and index < len(best_chunks) - 1:
                chunks.append(silence)

        return torch.cat(chunks, dim=-1)

    def generate_fast(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        sentence_split: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        num_candidates: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Fast inference using CUDA graphs. Falls back to generate() on non-CUDA devices.

        See generate() for parameter documentation.
        """
        if not hasattr(self.model, "generate_fast"):
            return self.generate(
                text,
                language_id=language_id,
                normalize_text=normalize_text,
                sentence_split=sentence_split,
                inter_sentence_silence_ms=inter_sentence_silence_ms,
                num_candidates=num_candidates,
                **kwargs,
            )

        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
        )
        if not sentences:
            return torch.zeros((1, 0), dtype=torch.float32)

        chunks = []
        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate_fast).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate_fast() does not accept: {sorted(dropped)}. "
                f"These kwargs were ignored.",
                stacklevel=2,
            )

        # Phase 1: Generate first candidate for all chunks sequentially
        best_chunks = []
        for sentence in sentences:
            best_chunks.append(self.model.generate_fast(sentence, **filtered_kwargs))

        if num_candidates > 1:
            import concurrent.futures
            import os
            from .validation import validate_audio

            sr = getattr(self.model, "sr", 24000)
            
            def _validate(idx, text, chunk_tensor):
                wav_np = chunk_tensor.squeeze().cpu().numpy()
                res = validate_audio(wav_np, sr, text, backend="faster-whisper", language=language_id)
                return idx, (res.get("wer", float('inf')) if res.get("status") == "ok" else float('inf'))

            # Phase 2: Parallel Whisper validation
            wers = [float('inf')] * len(sentences)
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sentences), os.cpu_count() or 4)) as pool:
                futs = [pool.submit(_validate, i, sentences[i], best_chunks[i]) for i in range(len(sentences))]
                for fut in concurrent.futures.as_completed(futs):
                    idx, wer = fut.result()
                    wers[idx] = wer

            # Phase 3: Retry Queue
            for i, sentence in enumerate(sentences):
                best_wer = wers[i]
                if best_wer <= getattr(self, "wer_threshold", 0.0):
                    continue
                
                for _ in range(1, num_candidates):
                    chunk = self.model.generate_fast(sentence, **filtered_kwargs)
                    _, wer = _validate(i, sentence, chunk)
                    if wer < best_wer:
                        best_wer = wer
                        best_chunks[i] = chunk
                    if best_wer <= getattr(self, "wer_threshold", 0.0):
                        break

        chunks = []
        for index, chunk in enumerate(best_chunks):
            chunks.append(chunk)
            if silence is not None and index < len(best_chunks) - 1:
                chunks.append(silence)

        return torch.cat(chunks, dim=-1)

    def generate_stream_sync(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        num_candidates: int = 1,
        **kwargs,
    ) -> Generator[torch.Tensor, None, None]:
        """Sync generator yielding one wav tensor per sentence (plus silence tensors between).

        Sentence splitting is always enabled — each yielded chunk corresponds to one sentence.
        Concatenating all yielded tensors produces the same result as generate().
        """
        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=True,
        )
        if not sentences:
            return

        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate() does not accept: {sorted(dropped)}. "
                f"These kwargs were ignored.",
                stacklevel=2,
            )

        for i, sentence in enumerate(sentences):
            best_chunk = None
            best_wer = float('inf')

            for _ in range(num_candidates):
                chunk = self.model.generate(sentence, **filtered_kwargs)
                if num_candidates == 1:
                    best_chunk = chunk
                    break
                
                from .validation import validate_audio
                sr = getattr(self.model, "sr", 24000)
                wav_np = chunk.squeeze().cpu().numpy()
                val_result = validate_audio(wav_np, sr, sentence, backend="faster-whisper", language=language_id)
                wer = val_result.get("wer", 0.0) if val_result.get("status") == "ok" else 0.0
                
                if wer == 0.0:
                    best_chunk = chunk
                    break
                elif wer < best_wer:
                    best_wer = wer
                    best_chunk = chunk

            if best_chunk is None:
                best_chunk = chunk

            yield best_chunk
            if silence is not None and i < len(sentences) - 1:
                yield silence

    async def generate_stream_async(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        num_candidates: int = 1,
        **kwargs,
    ) -> AsyncGenerator[torch.Tensor, None]:
        """Async generator yielding one wav tensor per sentence (plus silence tensors between).

        Wraps each model.generate() call in asyncio.to_thread() to avoid blocking
        the event loop. Sentence splitting is always enabled.
        """
        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=True,
        )
        if not sentences:
            return

        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate() does not accept: {sorted(dropped)}. "
                f"These kwargs were ignored.",
                stacklevel=2,
            )

        for i, sentence in enumerate(sentences):
            best_chunk = None
            best_wer = float('inf')

            for _ in range(num_candidates):
                chunk = await asyncio.to_thread(self.model.generate, sentence, **filtered_kwargs)
                if num_candidates == 1:
                    best_chunk = chunk
                    break
                
                from .validation import validate_audio
                sr = getattr(self.model, "sr", 24000)
                wav_np = chunk.squeeze().cpu().numpy()
                val_result = await asyncio.to_thread(
                    validate_audio, wav_np, sr, sentence, backend="faster-whisper", language=language_id
                )
                wer = val_result.get("wer", 0.0) if val_result.get("status") == "ok" else 0.0
                
                if wer == 0.0:
                    best_chunk = chunk
                    break
                elif wer < best_wer:
                    best_wer = wer
                    best_chunk = chunk

            if best_chunk is None:
                best_chunk = chunk

            yield best_chunk
            if silence is not None and i < len(sentences) - 1:
                yield silence

    def generate_stream_fast_sync(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        num_candidates: int = 1,
        **kwargs,
    ) -> Generator[torch.Tensor, None, None]:
        """Sync streaming variant using generate_fast() per sentence.

        Sentence splitting is always enabled — each yielded chunk corresponds to one sentence.
        Falls back to generate_stream_sync() on non-CUDA devices or models without generate_fast.
        """
        if not hasattr(self.model, "generate_fast"):
            yield from self.generate_stream_sync(
                text,
                language_id=language_id,
                normalize_text=normalize_text,
                inter_sentence_silence_ms=inter_sentence_silence_ms,
                num_candidates=num_candidates,
                **kwargs,
            )
            return

        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=True,
        )
        if not sentences:
            return

        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate_fast).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate_fast() does not accept: {sorted(dropped)}. "
                f"These kwargs were ignored.",
                stacklevel=2,
            )

        for i, sentence in enumerate(sentences):
            best_chunk = None
            best_wer = float('inf')

            for _ in range(num_candidates):
                chunk = self.model.generate_fast(sentence, **filtered_kwargs)
                if num_candidates == 1:
                    best_chunk = chunk
                    break
                
                from .validation import validate_audio
                sr = getattr(self.model, "sr", 24000)
                wav_np = chunk.squeeze().cpu().numpy()
                val_result = validate_audio(wav_np, sr, sentence, backend="faster-whisper", language=language_id)
                wer = val_result.get("wer", 0.0) if val_result.get("status") == "ok" else 0.0
                
                if wer == 0.0:
                    best_chunk = chunk
                    break
                elif wer < best_wer:
                    best_wer = wer
                    best_chunk = chunk

            if best_chunk is None:
                best_chunk = chunk

            yield best_chunk
            if silence is not None and i < len(sentences) - 1:
                yield silence

    async def generate_stream_fast_async(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        num_candidates: int = 1,
        **kwargs,
    ) -> AsyncGenerator[torch.Tensor, None]:
        """Async streaming variant using generate_fast() per sentence.

        Sentence splitting is always enabled — each yielded chunk corresponds to one sentence.
        Falls back to generate_stream_async() on non-CUDA devices or models without generate_fast.
        """
        if not hasattr(self.model, "generate_fast"):
            async for chunk in self.generate_stream_async(
                text,
                language_id=language_id,
                normalize_text=normalize_text,
                inter_sentence_silence_ms=inter_sentence_silence_ms,
                num_candidates=num_candidates,
                **kwargs,
            ):
                yield chunk
            return

        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=True,
        )
        if not sentences:
            return

        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate_fast).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate_fast() does not accept: {sorted(dropped)}. "
                f"These kwargs were ignored.",
                stacklevel=2,
            )

        for i, sentence in enumerate(sentences):
            best_chunk = None
            best_wer = float('inf')

            for _ in range(num_candidates):
                chunk = await asyncio.to_thread(self.model.generate_fast, sentence, **filtered_kwargs)
                if num_candidates == 1:
                    best_chunk = chunk
                    break
                
                from .validation import validate_audio
                sr = getattr(self.model, "sr", 24000)
                wav_np = chunk.squeeze().cpu().numpy()
                val_result = await asyncio.to_thread(
                    validate_audio, wav_np, sr, sentence, backend="faster-whisper", language=language_id
                )
                wer = val_result.get("wer", 0.0) if val_result.get("status") == "ok" else 0.0
                
                if wer == 0.0:
                    best_chunk = chunk
                    break
                elif wer < best_wer:
                    best_wer = wer
                    best_chunk = chunk

            if best_chunk is None:
                best_chunk = chunk

            yield best_chunk
            if silence is not None and i < len(sentences) - 1:
                yield silence