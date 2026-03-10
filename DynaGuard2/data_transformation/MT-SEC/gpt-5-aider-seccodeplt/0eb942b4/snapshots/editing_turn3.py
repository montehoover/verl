import re

def analyze_user_string(input_text: str) -> list[str]:
    if not isinstance(input_text, str):
        raise ValueError("input_text must be a string")
    stripped = input_text.strip()
    if not stripped:
        raise ValueError("input_text cannot be empty or whitespace")
    return re.findall(r'\S+', stripped)

def extract_words(text: str) -> list[str]:
    return [token for token in text.split(' ') if token != '']

def count_words(text: str) -> int:
    if text is None:
        return 0
    return len(extract_words(text))
