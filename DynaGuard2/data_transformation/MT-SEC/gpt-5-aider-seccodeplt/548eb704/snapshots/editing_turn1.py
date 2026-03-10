import re

def count_words(text: str) -> int:
    if not text:
        return 0
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    return len(words)
