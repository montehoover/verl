import re

def count_specific_word(text_input: str, target_word: str) -> int:
    if not text_input or not target_word:
        return 0
    pattern = r"\b" + re.escape(target_word) + r"\b"
    matches = re.findall(pattern, text_input, flags=re.UNICODE)
    return len(matches)
