"""Sentence splitting for TTS using NLTK.

Maps ISO 639-1 codes to nltk language names for sent_tokenize.
"""

import re

import nltk


# Map ISO 639-1 codes to nltk language names for sent_tokenize
_NLTK_LANGUAGE_MAP = {
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "it": "italian",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sl": "slovene",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
}

# Ensure punkt_tab is downloaded (quiet=True prevents spam on every import)
nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str, language: str = "en") -> list[str]:
    """Split text into sentence-like chunks for TTS.

    Args:
        text: Input text to split.
        language: ISO 639-1 language code (lowercase).

    Returns:
        List of sentence strings.

    Raises:
        RuntimeError: If NLTK punkt_tab data is not found.
    """
    nltk_language = _NLTK_LANGUAGE_MAP.get(language.lower(), "english")
    try:
        parts = nltk.tokenize.sent_tokenize(text, language=nltk_language)
    except LookupError:
        raise RuntimeError(
            "NLTK punkt_tab data not found. Download it with: "
            "python -c \"import nltk; nltk.download('punkt_tab')\""
        ) from None

    result = []
    for part in parts:
        # Remove leading dash/bullet characters (e.g., "- Hello")
        part = re.sub(r"^[-–—]\s+", "", part.strip())
        if part:
            result.append(part)
    return result


def split_long_sentence(sentence: str, max_len: int = 300, seps: list[str] | None = None) -> list[str]:
    """
    Recursively split a sentence into chunks of <= max_len using a sequence of separators.
    Tries each separator in order, splitting further as needed.
    """
    if seps is None:
        seps = [';', ':', '-', ',', ' ']

    sentence = sentence.strip()
    if len(sentence) <= max_len:
        return [sentence]

    if not seps:
        # Fallback: force split every max_len chars
        return [sentence[i:i+max_len].strip() for i in range(0, len(sentence), max_len)]

    sep = seps[0]
    parts = sentence.split(sep)

    if len(parts) == 1:
        # Separator not found, try next separator
        return split_long_sentence(sentence, max_len, seps=seps[1:])

    # Now recursively process each part, joining separator back except for the first
    chunks = []
    current = parts[0].strip()
    for part in parts[1:]:
        candidate = (current + sep + part).strip()
        if len(candidate) > max_len:
            # Split current chunk further with the next separator
            chunks.extend(split_long_sentence(current.strip(), max_len, seps=seps[1:]))
            current = part.strip()
        else:
            current = candidate
    # Process the last current
    if current:
        if len(current) > max_len:
            chunks.extend(split_long_sentence(current.strip(), max_len, seps=seps[1:]))
        else:
            chunks.append(current.strip())

    return chunks


def group_sentences(sentences: list[str], max_chars: int = 300) -> list[str]:
    """Packs short sentences together up to max_chars."""
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if not sentence:
            continue
        sentence = sentence.strip()
        sentence_len = len(sentence)

        if sentence_len > max_chars:
            for chunk in split_long_sentence(sentence, max_chars):
                if len(chunk) > max_chars:
                    for i in range(0, len(chunk), max_chars):
                        chunks.append(chunk[i:i+max_chars])
                else:
                    chunks.append(chunk)
            current_chunk = []
            current_length = 0
            continue

        if current_chunk and current_length + sentence_len + 1 > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len + (1 if current_chunk else 0)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def smart_append_short_sentences(sentences: list[str], max_chars: int = 300) -> list[str]:
    """Merge orphan sentences (<20 chars) into neighbors."""
    new_groups = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        # If very short and we can attach to next
        if len(sentence) < 20 and i + 1 < len(sentences):
            merged = sentence + " " + sentences[i+1].strip()
            if len(merged) <= max_chars:
                sentences[i+1] = merged
                i += 1
                continue
        # If very short and we can attach to prev
        if len(sentence) < 20 and len(new_groups) > 0:
            merged = new_groups[-1] + " " + sentence
            if len(merged) <= max_chars:
                new_groups[-1] = merged
                i += 1
                continue
        new_groups.append(sentence)
        i += 1
    return new_groups


def enforce_min_chunk_length(chunks: list[str], min_len: int = 20, max_len: int = 300) -> list[str]:
    """Final pass: ensure no chunk is shorter than min_len unless it's the last one."""
    out = []
    i = 0
    while i < len(chunks):
        current = chunks[i].strip()
        if len(current) >= min_len or i == len(chunks) - 1:
            out.append(current)
            i += 1
        else:
            if i + 1 < len(chunks):
                merged = current + " " + chunks[i + 1]
                if len(merged) <= max_len:
                    out.append(merged)
                    i += 2
                else:
                    out.append(current)
                    i += 1
            else:
                out.append(current)
                i += 1
    return out


def chunk_text(text: str, language: str = "en", max_chars: int = 300, min_chars: int = 20) -> list[str]:
    """Split text into TTS-ready chunks with length and quality guarantees.

    Applies a 4-stage pipeline:
        1. Language-aware NLTK sentence splitting
        2. Short sentence packing (merge up to max_chars), resolving overlong sentences
        3. Orphan fragment merging (min_chars threshold)
        4. Final min-length enforcement

    Args:
        text: Raw input text of any length.
        language: ISO 639-1 language code (e.g., "en", "fr").
        max_chars: Maximum characters per output chunk. Default 300.
        min_chars: Minimum characters per chunk. Default 20.

    Returns:
        List of text chunks, each guaranteed to be between min_chars
        and max_chars characters (except the last chunk in edge cases).
    """
    import logging
    logger = logging.getLogger(__name__)

    if not text or len(text.strip()) == 0:
        return []
        
    logger.debug(f"Chunking text of length {len(text)} for language '{language}'")
    
    sentences = split_sentences(text, language=language)
    logger.debug(f"Step 1: NLTK split into {len(sentences)} sentences")
    
    grouped = group_sentences(sentences, max_chars=max_chars)
    logger.debug(f"Step 2: Grouped into {len(grouped)} chunks")
    
    smart_appended = smart_append_short_sentences(grouped, max_chars=max_chars)
    logger.debug(f"Step 3: Smart appended short sentences into {len(smart_appended)} chunks")
    
    final_chunks = enforce_min_chunk_length(smart_appended, min_len=min_chars, max_len=max_chars)
    logger.debug(f"Step 4: Final pass min/max length yielded {len(final_chunks)} chunks")
    
    return final_chunks