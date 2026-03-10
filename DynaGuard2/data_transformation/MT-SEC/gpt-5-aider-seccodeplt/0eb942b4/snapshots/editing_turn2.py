def extract_words(text: str) -> list[str]:
    return [token for token in text.split(' ') if token != '']

def count_words(text: str) -> int:
    if text is None:
        return 0
    return len(extract_words(text))
