"""Shared text transformations for datasets and workload planning."""


def split_chunks(text: str, max_chars: int = 3000) -> list[str]:
    """Split a long string into <=max_chars chunks on word boundaries."""
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_len = len(normalized)
    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            split_at = normalized.rfind(" ", start, end)
            if split_at > start + (max_chars // 2):
                end = split_at
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks
