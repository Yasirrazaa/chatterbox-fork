"""Text preprocessing utilities ported from Chatterbox-TTS-Extended.

These pure-Python helpers clean up input text before TTS synthesis to reduce
artifacts and improve prosody. They are applied *before* the language-aware
number normalization in ``normalizer.py``.

Functions:
- ``remove_sound_words`` — strip/replace filler words (um, ahh, ...)
- ``fix_dot_letters``    — "J.R.R." -> "J R R"
- ``remove_inline_ref_numbers`` — strip trailing reference numbers (".188")
- ``normalize_whitespace`` — collapse spaces / strip
- ``preprocess_text``     — run all of the above in a sensible order
"""

import re
from typing import Dict, Optional

# Default filler / sound-word list. Maps a word to its replacement (or "" to drop).
DEFAULT_SOUND_WORDS: Dict[str, str] = {
    "um": "",
    "uh": "",
    "uhm": "",
    "ahh": "",
    "err": "",
    "er": "",
    "mm": "",
    "hmm": "",
    "mmm": "",
    "zzz": "sigh",
}


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", text).strip()


def fix_dot_letters(text: str) -> str:
    """Convert dotted initials to spaced letters: ``J.R.R.`` -> ``J R R``.

    Handles both ``J.R.R.`` and ``J. R. R.`` forms.
    """
    # Pattern: a letter, a dot (optional space), repeated. e.g. "J.R.R." or "J. R. R."
    pattern = r"\b([A-Za-z])\.(\s?)([A-Za-z])\.(\s?)([A-Za-z])\.?\b"

    def _repl(m):
        letters = [m.group(1), m.group(3), m.group(5)]
        return " ".join(letters)

    # Handle 2-letter and 3+ letter cases
    text = re.sub(r"\b([A-Za-z])\.(\s?)([A-Za-z])\.(\s?)([A-Za-z])\.(\s?)([A-Za-z])\.?\b",
                  lambda m: " ".join([m.group(1), m.group(3), m.group(5), m.group(7)]), text)
    text = re.sub(pattern, _repl, text)
    return text


def remove_inline_ref_numbers(text: str) -> str:
    """Remove trailing reference numbers after sentence punctuation.

    Examples: ``"word.188"`` -> ``"word."`` ; ``"word.”3"`` -> ``"word.”"``
    """
    # After a sentence-ending punctuation, drop digits (e.g. footnote refs)
    return re.sub(r"([.!?\"”’'])\d+", r"\1", text)


def remove_sound_words(
    text: str,
    sound_words: Optional[Dict[str, str]] = None,
    case_sensitive: bool = False,
) -> str:
    """Remove or replace filler / sound words.

    Args:
        text: Input text.
        sound_words: Mapping of word -> replacement ("" to delete). Defaults to
            :data:`DEFAULT_SOUND_WORDS`.
        case_sensitive: If False, matching is case-insensitive.
    """
    if sound_words is None:
        sound_words = DEFAULT_SOUND_WORDS

    # Build a regex that matches whole words (with optional surrounding punctuation)
    flags = 0 if case_sensitive else re.IGNORECASE
    # Sort by length (longest first) so "ahh" beats "ah"
    items = sorted(sound_words.items(), key=lambda kv: len(kv[0]), reverse=True)

    for word, replacement in items:
        # word boundaries, allow leading/trailing punctuation/quotes
        pattern = rf"(?<!\w)([”'\"\s(]*){re.escape(word)}([”'\"\s.,!?;:)]*)"
        compiled = re.compile(pattern, flags)

        def _make_repl(repl):
            def _repl(m):
                lead, trail = m.group(1), m.group(2)
                if repl == "":
                    # Drop the word; keep leading space only if there is trailing content
                    return lead.rstrip()
                return f"{lead}{repl}{trail}"
            return _repl

        text = compiled.sub(_make_repl(replacement), text)

    return normalize_whitespace(text)


def preprocess_text(
    text: str,
    sound_words: Optional[Dict[str, str]] = None,
    do_fix_dot_letters: bool = True,
    do_remove_ref_numbers: bool = True,
    do_remove_sound_words: bool = True,
) -> str:
    """Run the full preprocessing pipeline.

    Order: dot-letter fix -> inline ref number removal -> sound-word removal ->
    whitespace normalization. (Number normalization happens later in the
    language-aware normalizer.)
    """
    if do_fix_dot_letters:
        text = fix_dot_letters(text)
    if do_remove_ref_numbers:
        text = remove_inline_ref_numbers(text)
    if do_remove_sound_words:
        text = remove_sound_words(text, sound_words)
    text = normalize_whitespace(text)
    return text


if __name__ == "__main__":
    samples = [
        "J.R.R. Tolkien wrote many books. um this is a test ahh okay.",
        "The score was 3.188 according to the report.”3",
        "Hello uh world, this is a mmm demonstration.",
    ]
    for s in samples:
        print("IN :", s)
        print("OUT:", preprocess_text(s))
        print()
