def count_words(text: str) -> int:
    if text is None:
        return 0
    text = text.strip()
    if not text:
        return 0
    return sum(1 for token in text.split(' ') if token != '')
