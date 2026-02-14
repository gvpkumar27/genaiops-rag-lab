def chunk_text(text: str, chunk_size: int, overlap: int):
    text = text.replace("\r\n", "\n").strip()
    chunks = []
    i = 0
    while i < len(text):
        j = min(i + chunk_size, len(text))
        if j < len(text):
            # Prefer breaking at natural boundaries instead of hard cutting words.
            window_start = max(i, j - 220)
            window = text[window_start:j]
            split_points = [window.rfind("\n\n"), window.rfind(". "), window.rfind(" ")]
            best = max(split_points)
            if best > 0:
                j = window_start + best + (2 if best == split_points[0] else 1)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks
