"""Cheap token estimates so prompts and research stay inside a budget.

Uses a 4-characters-per-token rule. That is enough to cap context without
calling a tokenizer model during scenario building or document research.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 4

# Narrative sent to the shock model, not the whole desk transcript.
NARRATIVE_TOKEN_BUDGET = 160
# System instruction for narrative → shocks. Kept short on purpose.
SHOCK_PROMPT_TOKEN_BUDGET = 120
# Completion cap passed to Ollama as num_predict.
SHOCK_COMPLETION_TOKEN_BUDGET = 160
# Methodology research pasted into an explanation.
RAG_RESEARCH_TOKENS = 240
RAG_RESEARCH_CHUNKS = 2
# One streamed answer chunk. Fewer events than one word at a time.
STREAM_CHUNK_TOKENS = 48
# Cap on any string copied into a local or exported trace.
TRACE_META_TOKENS = 40


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def clip_text(text: str, max_tokens: int) -> str:
    """Trim text to a token budget, on a word boundary when possible."""
    if max_tokens <= 0 or not text:
        return ""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    if not trimmed:
        trimmed = text[:max_chars].rstrip()
    return trimmed + "…"


def iter_token_chunks(text: str, max_tokens: int) -> list:
    """Split text into chunks that concatenate back to the original."""
    if not text:
        return []
    max_chars = max(1, max_tokens * CHARS_PER_TOKEN)
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        if end < length:
            space = text.rfind(" ", start, end)
            if space > start:
                end = space + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def clip_payload(value, max_tokens: int = TRACE_META_TOKENS, max_items: int = 8):
    """Clip strings inside trace payloads. Numbers stay intact."""
    if isinstance(value, str):
        return clip_text(value, max_tokens)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        clipped = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                break
            clipped[str(key)[:40]] = clip_payload(item, max_tokens, max_items)
        return clipped
    if isinstance(value, (list, tuple)):
        return [clip_payload(item, max_tokens, max_items) for item in list(value)[:max_items]]
    return clip_text(str(value), max_tokens)
